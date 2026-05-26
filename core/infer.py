import os
import sys
from pathlib import Path

import colorama
import hydra
import torch
import torch.nn.functional as F
from colorama import Fore
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.utils.core_utils import (
    copy_config_file,
    get_collator,
    get_dataset,
    is_movable,
    load_model,
    set_seed,
)
from core.utils.metrics_utils import MetricFactory


class Inferencer:
    def __init__(self, cfg: DictConfig):
        """
        Initialize inferencer.

        Args:
            cfg: Hydra config containing model, dataset, data parameters.
                 cfg.ckpt_path: Optional path to model checkpoint (.pt file). If None, uses original model parameters.
        """
        self.cfg = cfg
        ckpt_path = cfg.get("ckpt_path")
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None

        # Validate checkpoint exists if provided
        if self.ckpt_path and not self.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = cfg.model
        self.dataset_name = cfg.dataset
        self.batch_size = cfg.batch_size
        self.split = cfg.data.get("split", "test")
        self.statis = cfg.get("statis", None)
        self.missing_type = cfg.data.get("missing_type")
        self.missing_rate = cfg.data.get("missing_rate")

        # Initialize evaluator
        self.evaluator = MetricFactory.get_metric(cfg.dataset, self.device)

        # Initialize collator
        self.collator = get_collator(cfg.model, cfg.dataset, **cfg.data)

    def _load_model_and_checkpoint(self):
        """Load model architecture and checkpoint weights."""
        # Load model architecture
        self.model = load_model(self.model_name, **dict(self.cfg.para), cfg=self.cfg)

        # Load checkpoint weights if provided
        if self.ckpt_path:
            logger.info(f"Loading checkpoint from {self.ckpt_path}")
            if str(self.ckpt_path).endswith(".pt"):
                state_dict = torch.load(str(self.ckpt_path), map_location=self.device)
            else:
                state_dict = load_file(str(self.ckpt_path), device=str(self.device))
            self.model.load_state_dict(state_dict)
        else:
            logger.info("Model loaded with original parameters (no checkpoint)")

        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()

    def _load_dataset(self):
        """Load dataset for the specified split."""
        cpu_count = int(os.cpu_count() / 2)
        dataset = get_dataset(self.model_name, self.dataset_name, **self.cfg.data)

        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.collator,
            num_workers=min(cpu_count, self.batch_size // 2),
            shuffle=True,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
        )

        logger.info(f"Loaded {len(dataset)} samples from {self.split} split")

    def run(self):
        """Run inference for one epoch and compute metrics."""
        self._load_model_and_checkpoint()
        self._load_dataset()

        logger.info(f"{Fore.CYAN}Starting inference on {self.split} split...")

        loss_list = []

        # Inference loop
        for batch in tqdm(self.dataloader, bar_format=f"{Fore.CYAN}{{l_bar}}{{bar}}{{r_bar}}"):
            # Move inputs to device
            inputs = {
                key: value.to(self.device) if is_movable(value) else value for key, value in batch.items()
            }
            labels = inputs.pop("labels")

            # Forward pass
            with torch.no_grad():
                output = self.model(**inputs)
                logits = output["logits"]

                # Compute loss
                if len(labels.shape) == 1:
                    # Multi-class classification (Food101, HateMemes)
                    loss = F.cross_entropy(logits, labels)
                else:
                    # Multi-label classification (MMIMDB)
                    loss = F.binary_cross_entropy_with_logits(logits, labels.float())

            # Update evaluator
            self.evaluator.update(logits, labels)
            loss_list.append(loss.item())

        # Compute metrics
        metrics = self.evaluator.compute()
        fmt_metrics = metrics.pop("fmt_text")

        # Print results
        logger.info(f"{Fore.GREEN}{fmt_metrics}")

        # Save collected tokens if statis == "collect_token"
        if self.statis == "collect_token" and hasattr(self.model, "token_collector"):
            from core.utils.stats_utils import run_statis

            logger.info(f"{Fore.CYAN}Saving collected tokens...")

            # Call run_statis with parameters passed directly
            run_statis(
                statis_type=self.statis,
                model_name=self.model_name,
                save_dir="./cache/collect_token",
                token_collector=self.model.token_collector,
                dataset_name=self.dataset_name,
                missing_type=self.missing_type,
                missing_rate=self.missing_rate,
                pooling_method=getattr(self.model, "token_pooling", "mean"),
            )

            logger.info(f"{Fore.GREEN}Tokens saved successfully")

        # Save collected features if statis == "collect_features"
        if self.statis == "collect_features" and hasattr(self.model, "feature_collector"):
            from core.utils.stats_utils import run_statis

            logger.info(f"{Fore.CYAN}Saving collected features...")

            # Call run_statis with parameters passed directly
            run_statis(
                statis_type=self.statis,
                model_name=self.model_name,
                feature_collector=self.model.feature_collector,
                save_dir="./cache/collect_feature",
                dataset_name=self.dataset_name,
                top_percent=1.0,  # Save 100% of samples
                selection_seed=42,  # Deterministic selection
            )

            logger.info(f"{Fore.GREEN}Features saved successfully")

        # Save collected prompts if statis == "collect_prompts"
        if self.statis == "collect_prompts" and hasattr(self.model, "prompt_collector"):
            from core.utils.stats_utils import run_statis

            logger.info(f"{Fore.CYAN}Saving collected prompts...")

            # Get ablation from model if available
            ablation = getattr(self.model, "ablation", None)

            run_statis(
                statis_type=self.statis,
                model_name=self.model_name,
                save_dir="./cache/collect_prompt",
                prompt_collector=self.model.prompt_collector,
                dataset_name=self.dataset_name,
                missing_type=self.missing_type,
                missing_rate=self.missing_rate,
                top_percent=1.0,  # Save 100% of samples
                selection_seed=42,  # Deterministic selection (same as features)
                ablation=ablation,
            )

            logger.info(f"{Fore.GREEN}Prompts saved successfully")

        return metrics


@hydra.main(version_base=None, config_path="config", config_name="")
def main(cfg: DictConfig):
    """
    Main inference function.

    Config file can optionally specify ckpt_path parameter.
    If not provided, uses original model parameters.
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.info(OmegaConf.to_yaml(cfg))

    # Initialize colorama
    colorama.init()

    # Set seed for reproducibility
    set_seed(cfg.seed)

    # Run inference
    inferencer = Inferencer(cfg)
    inferencer.run()


if __name__ == "__main__":
    copy_config_file()
    main()
