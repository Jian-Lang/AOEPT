import sys
import traceback
from datetime import datetime
from pathlib import Path

import colorama
import hydra
import icecream
import pandas as pd
import swanlab
from colorama import Fore
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from swanlab.data.run import SwanLabRunState  # if this import fails, try: from swanlab import SwanLabRunState

import wandb
from core.infer import Inferencer
from core.train import Trainer
from core.utils.core_utils import calculate_md5, copy_config_file, set_seed


def send_notification(title: str, body: str) -> None:
    """Send desktop notification using OSC 777 escape sequence."""
    print(f"\033]777;notify;{title};{body}\007", end="", file=sys.stderr, flush=True)


@hydra.main(version_base=None, config_path="config", config_name="")
def main(cfg: DictConfig):
    """
    Unified entry point for both training and inference.

    Dispatches to Trainer or Inferencer based on cfg.mode parameter.

    Args:
        cfg: Hydra config containing mode ('train' or 'infer') and other parameters
    """
    # Calculate config MD5 for experiment tracking
    config_str = OmegaConf.to_yaml(cfg)
    config_md5 = calculate_md5(config_str)[:6]

    # Setup logging
    logger.remove()
    log_path = Path(f"log/{datetime.now().strftime('%m%d-%H%M%S')}") / config_md5
    logger.add(log_path / "log.log", retention="1 days", level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    logger.info(OmegaConf.to_yaml(cfg))

    # Initialize environment
    pd.set_option("future.no_silent_downcasting", True)
    colorama.init()
    icecream.install()
    set_seed(cfg.seed)

    # Get mode from config (default to train if not specified)
    mode = cfg.get("mode", "train")
    logger.info(f"{Fore.CYAN}Running in {mode.upper()} mode")

    # Mode-based dispatch
    try:
        if mode == "train":
            # Extract notes from config before wandb init
            notes = cfg.get("notes", None)
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            config_dict.pop("notes", None)

            # Initialize wandb for training
            swanlab.sync_wandb()
            run = wandb.init(
                project="AOEPT",
                name=config_md5,
                config=config_dict,
                notes=notes,
                mode="online",
            )

            # Run training
            trainer = Trainer(cfg, config_md5)
            trainer.run()

            # Send success notification
            send_notification("Training Complete", f"{cfg.model} {cfg.dataset}")

        elif mode == "infer":
            # Run inference
            inferencer = Inferencer(cfg)
            metrics = inferencer.run()

            # Log final metrics
            logger.info(f"{Fore.GREEN}Inference completed successfully")
            logger.info(f"{Fore.GREEN}Metrics: {metrics}")

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'train' or 'infer'")

    except Exception as e:
        # Send failure notification
        send_notification("Training Failed", f"{cfg.model} {cfg.dataset}: {type(e).__name__}")
        if mode == "train":
            swanlab.finish(state=SwanLabRunState.CRASHED, error=traceback.format_exc())
        raise e


if __name__ == "__main__":
    copy_config_file()
    main()
