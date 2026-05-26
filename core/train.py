import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import colorama
import hydra
import icecream
import numpy as np
import pandas as pd
import swanlab
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorama import Back, Fore, Style
from icecream import ic
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from core.utils.core_utils import (
    EarlyStopping,
    calculate_md5,
    copy_config_file,
    get_collator,
    get_dataset,
    get_optimizer,
    get_scheduler,
    is_movable,
    load_model,
    set_seed,
    set_worker_seed,
)
from core.utils.metrics_utils import (
    MetricFactory,
)
from core.utils.stats_utils import (
    get_model_params,
    get_model_params_info,
    run_statis,
)


class Trainer:
    def __init__(self, cfg: DictConfig, config_md5: str):
        self.cfg = cfg

        self.device = "cuda"
        self.task = cfg.task

        self.evaluator = MetricFactory.get_metric(cfg.dataset, self.device)

        self.type = cfg.type
        self.model_name = cfg.model
        self.dataset_name = cfg.dataset
        self.batch_size = cfg.batch_size
        self.num_epoch = cfg.num_epoch
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.statis = cfg.get("statis")

        # Parse and validate stats array
        self.stats_to_log = []
        available_stats = ["trainable_para", "total_para", "train_time", "gpu_memory"]
        stats_config = cfg.get("stats", None)

        if stats_config is not None:
            for stat in stats_config:
                if stat in available_stats:
                    self.stats_to_log.append(stat)
                else:
                    logger.warning(f"Unknown stat '{stat}' requested. Available stats: {available_stats}")
            if len(self.stats_to_log) > 0:
                logger.info(f"Stats tracking enabled: {self.stats_to_log}")

        # Initialize batch time tracking for entire training
        self.all_batch_times = []

        # Add "-Full" suffix to save path if full_finetune is enabled
        model_dir_name = f"{self.model_name}-{self.dataset_name}"
        self.save_path = Path("model") / model_dir_name / f"{config_md5}"
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.collator = get_collator(cfg.model, cfg.dataset, **cfg.data)

        self.global_step = 0
        self.val_int: Any = cfg.get("val_int", -1)

    def _reset(self, cfg, type):
        cpu_count = int(os.cpu_count()) // 2
        
        # Create a copy of cfg.data and remove 'split' to avoid multiple values error
        data_cfg = dict(cfg.data)
        data_cfg.pop("split", None)
        
        train_dataset = get_dataset(cfg.model, cfg.dataset, split="train", **data_cfg)
        valid_dataset = get_dataset(cfg.model, cfg.dataset, split="val", **data_cfg)
        test_dataset = get_dataset(cfg.model, cfg.dataset, split="test", **data_cfg)
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            collate_fn=self.collator,
            num_workers=min(cpu_count, cfg.batch_size // 2),
            shuffle=True,
            generator=self.generator,
            worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed),
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
        )
        self.valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=cfg.batch_size,
            collate_fn=self.collator,
            num_workers=min(cpu_count, cfg.batch_size // 2),
            shuffle=False,
            generator=self.generator,
            worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed),
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            collate_fn=self.collator,
            num_workers=min(cpu_count, cfg.batch_size // 2),
            shuffle=False,
            generator=self.generator,
            worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed),
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
        )

        steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)

        # Convert val_int from percentage to actual steps
        if self.val_int == -1:
            self.val_step_interval = -1
            logger.info("Validation mode: epoch-based")
        elif 0 < self.val_int < 1:
            self.val_step_interval = max(1, int(self.val_int * steps_per_epoch))
            logger.info(
                f"Validation mode: step-based (every {self.val_step_interval} steps, {self.val_int:.1%} of epoch)"
            )
        else:
            raise ValueError(f"val_int must be -1 or a fraction between 0 and 1, got {self.val_int}")

        self.model = load_model(cfg.model, **dict(cfg.para), cfg=cfg)

        self.model.to(self.device)
        # self.model = torch.compile(self.model)

        total_params, trainable_params, trainable_ratio = get_model_params_info(self.model)
        logger.info(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
        logger.info(f"Trainable ratio: {trainable_ratio:.2%}")

        # Store parameter stats for logging if needed
        self.total_params_m = total_params / 1e6
        self.trainable_params_m = trainable_params / 1e6

        self.optimizer = get_optimizer(self.model, **dict(cfg.opt))
        self.scheduler = get_scheduler(
            self.optimizer,
            steps_per_epoch=steps_per_epoch,
            total_steps=steps_per_epoch * cfg.num_epoch,
            **dict(cfg.sche),
        )
        self.earlystopping = EarlyStopping(
            patience=cfg.patience, path=str(self.save_path / "ckpt.safetensors")
        )

        # Reset GPU memory tracking if needed
        if "gpu_memory" in self.stats_to_log:
            torch.cuda.reset_peak_memory_stats()

    def run(self):
        self._reset(self.cfg, self.type)
        for epoch in range(self.num_epoch):
            logger.info(f"Current Epoch: {epoch}")
            early_stopped = self._train(epoch=epoch)

            # Only do epoch-based validation if val_step_interval == -1
            if self.val_step_interval == -1:
                self._valid(split="val", epoch=epoch, use_earlystop=True)
                if self.earlystopping.early_stop:
                    logger.info(f"{Fore.GREEN}Early stopping at epoch {epoch}")
                    break
                self._valid(split="test", epoch=epoch)
            elif early_stopped:
                # Step-based validation triggered early stop
                break
        self.model.load_state_dict(load_file(str(self.save_path / "ckpt.safetensors")))

        # Clear collectors before final test to ensure only test set samples
        if hasattr(self.model, "prompt_collector") and self.statis == "collect_prompts":
            self.model.prompt_collector.clear()
        if hasattr(self.model, "feature_collector") and self.statis == "collect_features":
            self.model.feature_collector.clear()

        # Use global_step for logging if using step-based validation
        final_step = self.global_step + 1 if self.val_step_interval > 0 else (epoch + 1)
        best_metrics = self._valid(split="test", epoch=epoch + 1, final=True, step=final_step)

        # Log best metrics to wandb
        best_metrics_summary = {f"{key}": value for key, value in best_metrics.items()}
        wandb.log(best_metrics_summary, step=final_step)

        state_dict = {
            k: v.contiguous() if hasattr(v, "contiguous") else v for k, v in self.model.state_dict().items()
        }
        save_file(state_dict, str(self.save_path / "ckpt.safetensors"))
        shutil.copy(str(self.save_path / "ckpt.safetensors"), str(self.save_path.parent / "ckpt.safetensors"))
        logger.info(f"Model saved at {self.save_path / 'ckpt.safetensors'}")

        # Log final stats once at the end of training
        if len(self.stats_to_log) > 0:
            stats_dict = {}

            if "trainable_para" in self.stats_to_log:
                stats_dict["stats/trainable_para"] = self.trainable_params_m

            if "total_para" in self.stats_to_log:
                stats_dict["stats/total_para"] = self.total_params_m

            if "train_time" in self.stats_to_log:
                avg_batch_time = np.mean(self.all_batch_times)
                stats_dict["stats/train_time"] = avg_batch_time

            if "gpu_memory" in self.stats_to_log:
                max_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                stats_dict["stats/gpu_memory"] = max_memory_mb

            wandb.log(stats_dict, step=final_step)
            logger.info(f"Final stats logged: {stats_dict}")

    def _train(self, epoch: int) -> bool:
        loss_list = []
        loss_pre_list = []
        grad_norm_list = []
        self.model.train()
        setattr(self.model, "current_epoch", epoch)
        pbar = tqdm(self.train_dataloader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}")
        epoch_start_time = time.time()
        for batch in pbar:
            batch_start_time = time.time() if "train_time" in self.stats_to_log else None
            _ = batch.pop("ids")
            inputs = {
                key: value.to(self.device) if is_movable(value) else value for key, value in batch.items()
            }
            labels = inputs.pop("labels")

            output = self.model(**inputs)
            probs = output["probs"]
            logits = output["logits"]

            if hasattr(self.model, "cal_loss"):
                match self.model.name:
                    case _:
                        loss, loss_pred = self.model.cal_loss(**output, label=labels)
            else:
                if self.task == "regression":
                    # Squeeze labels to match logits shape for regression
                    logits_squeezed = logits.squeeze()
                    labels_squeezed = labels.squeeze()
                    loss = loss_pred = F.mse_loss(logits_squeezed, labels_squeezed)
                elif len(labels.shape) == 1:
                    loss = loss_pred = F.cross_entropy(logits, labels)
                else:
                    # Handle labels with one-hot complement format [B, C, 2] -> [B, C]
                    if labels.ndim == 3 and labels.shape[-1] == 2:
                        labels_squeezed = labels[..., 0]
                    elif labels.shape != logits.shape:
                        labels_squeezed = labels.squeeze()
                    else:
                        labels_squeezed = labels
                    loss = loss_pred = F.binary_cross_entropy_with_logits(logits, labels_squeezed.float())

            self.evaluator.update(logits, labels)
            loss_list.append(loss.item())
            loss_pre_list.append(loss_pred.item())

            loss.backward()
            grad_norm = torch.norm(
                torch.stack([p.grad.detach().norm(2) for p in self.model.parameters() if p.grad is not None]),
                2,
            ).item()
            grad_norm_list.append(grad_norm)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            # Track batch time for stats
            if batch_start_time is not None:
                batch_time = time.time() - batch_start_time
                self.all_batch_times.append(batch_time)

            self.global_step += 1

            # Step-based validation
            if self.val_step_interval > 0 and self.global_step % self.val_step_interval == 0:
                # Validate on val set with early stopping
                self._valid(split="val", epoch=epoch, use_earlystop=True, step=self.global_step, leave=False)
                if self.earlystopping.early_stop:
                    logger.info(f"{Fore.GREEN}Early stopping at step {self.global_step}")
                    return True  # Signal early stop
                # Validate on test set
                self._valid(split="test", epoch=epoch, step=self.global_step, leave=False)
                # Set model back to training mode
                self.model.train()
        if self.cfg.get("exp.eff", None) is not None:
            total_params, trainable_params = get_model_params(self.model)
            max_memory_allocated = torch.cuda.max_memory_allocated()
            epoch_time = time.time() - epoch_start_time
            # Use global_step for step-based validation, otherwise use epoch
            log_step = self.global_step if self.val_step_interval > 0 else epoch
            wandb.log(
                {
                    "tot_params": total_params,
                    "train_params": trainable_params,
                    "max_gpu_memory": max_memory_allocated / 1024 / 1024,
                    "epoch_time": epoch_time,
                },
                step=log_step,
            )

        metrics = self.evaluator.compute()
        fmt_metrics = metrics.pop("fmt_text")
        # add train/ to each key of metrics
        metrics = {f"train/{key}": value for key, value in metrics.items()}
        # print
        logger.info(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list)}, Grad Norm: {np.mean(grad_norm_list)}")
        # Use global_step for step-based validation, otherwise use epoch
        log_step = self.global_step if self.val_step_interval > 0 else epoch
        wandb.log(
            {
                "train/loss": np.mean(loss_list),
                "train/loss_pred": np.mean(loss_pre_list),
                "train/grad_norm": np.mean(grad_norm_list),
                **metrics,
            },
            step=log_step,
        )
        logger.info(f"{Fore.BLUE}Train: {fmt_metrics}")
        return False  # No early stop

    def _valid(self, split: str, epoch: int, use_earlystop=False, final=False, step=None, leave=True):
        loss_list = []
        evaluator = MetricFactory.get_metric(self.cfg.dataset, self.device)
        self.model.eval()
        setattr(self.model, "current_epoch", epoch)
        if split == "val" and final:
            raise ValueError("print_wrong only support test split")
        if split == "val":
            dataloader = self.valid_dataloader
            split_name = "Valid"
            fcolor = Fore.YELLOW
        elif split == "test":
            dataloader = self.test_dataloader
            split_name = "Test"
            fcolor = Fore.RED
        else:
            raise ValueError("split not supported")

        # Use tqdm with leave parameter to control progress bar persistence
        iterator = tqdm(dataloader, bar_format=f"{fcolor}{{l_bar}}{{bar}}{{r_bar}}", leave=leave)

        last_output = None
        for batch in iterator:
            inputs = {
                key: value.to(self.device) if is_movable(value) else value for key, value in batch.items()
            }
            labels = inputs.pop("labels")

            with torch.no_grad():
                output = self.model(**inputs)
                probs = output["probs"]
                logits = output["logits"]
                if self.task == "regression":
                    # Squeeze labels to match logits shape for regression
                    logits_squeezed = logits.squeeze()
                    labels_squeezed = labels.squeeze()
                    loss = F.mse_loss(logits_squeezed, labels_squeezed)
                elif len(labels.shape) == 1:
                    loss = F.cross_entropy(logits, labels)
                else:
                    # Squeeze labels if needed for binary classification
                    labels_squeezed = labels.squeeze() if labels.shape != logits.shape else labels
                    loss = F.binary_cross_entropy_with_logits(logits, labels_squeezed.float())

            evaluator.update(logits, labels)
            loss_list.append(loss.item())
            last_output = output
        metrics = evaluator.compute()
        fmt_metrics = metrics.pop("fmt_text")
        # add train/ to each key of metrics
        metrics = {f"{split}/{key}": value for key, value in metrics.items()}
        logger.info(f"{fcolor}{split_name}: Loss: {np.mean(loss_list):.5f}")
        logger.info(f"{fcolor}{split_name}: {fmt_metrics}")
        # Use step for logging if provided, otherwise use epoch
        log_step = step if step is not None else epoch
        wandb.log(
            {**metrics},
            step=log_step,
        )
        if use_earlystop:
            self.earlystopping(-next(iter(metrics.values())), self.model)

        # Run statistics only once at the end of training
        if final and last_output is not None:
            run_statis(
                statis_type=self.statis,
                model_name=self.model.name,
                dataset_name=self.dataset_name,
                missing_type=self.cfg.data.get("missing_type", "unknown"),
                missing_rate=self.cfg.data.get("missing_rate", 0.0),
                ablation=self.cfg.para.get("ablation", None),
                **last_output,
            )

        return metrics


@hydra.main(version_base=None, config_path="config", config_name="")
def main(cfg: DictConfig):
    config_str = OmegaConf.to_yaml(cfg)
    config_md5 = calculate_md5(config_str)[:6]

    swanlab.sync_wandb()
    run = wandb.init(
        project="AOEPT",
        name=config_md5,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="online",
    )

    logger.remove()
    log_path = Path(f"log/{datetime.now().strftime('%m%d-%H%M%S')}") / config_md5
    logger.add(log_path / "log.log", retention="1 days", level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    logger.info(OmegaConf.to_yaml(cfg))
    pd.set_option("future.no_silent_downcasting", True)
    colorama.init()
    icecream.install()
    set_seed(cfg.seed)

    trainer = Trainer(cfg, config_md5)
    trainer.run()


if __name__ == "__main__":
    copy_config_file()
    main()
