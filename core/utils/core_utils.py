import hashlib
import importlib
import os
import random
import shutil

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from safetensors.torch import save_file
from torch import is_tensor
from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR, _LRScheduler
from transformers import BatchEncoding, BatchFeature


class WarmUpStepLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_rate, last_step=-1, **kargs):
        if warmup_rate < 0 or warmup_rate > 1:
            raise ValueError("warmup_rate should be between 0 and 1")
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_rate)
        self.last_step = last_step
        self.last_epoch = last_step
        super(WarmUpStepLR, self).__init__(optimizer, last_step)

    def get_lr(self):
        if self.last_step < self.warmup_steps:
            return [base_lr * ((self.last_step + 1) / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            factor = max(0, 1 - (self.last_step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
            return [base_lr * factor for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        self.last_epoch = step
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class WarmupConstantLR(_LRScheduler):
    """
    Warmup scheduler that linearly increases LR during warmup period,
    then maintains constant LR for the rest of training.
    """

    def __init__(self, optimizer, total_steps, warmup_rate, last_step=-1, **kargs):
        if warmup_rate < 0 or warmup_rate > 1:
            raise ValueError("warmup_rate should be between 0 and 1")
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_rate)
        self.last_step = last_step
        self.last_epoch = last_step
        super(WarmupConstantLR, self).__init__(optimizer, last_step)

    def get_lr(self):
        if self.last_step < self.warmup_steps:
            # Linear warmup: 0 to base_lr
            return [base_lr * ((self.last_step + 1) / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # Constant LR after warmup
            return [base_lr for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        self.last_epoch = step
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class PolynomialDecayWithWarmup(_LRScheduler):
    """
    Polynomial decay scheduler with linear warmup.
    Matches the DCP reference implementation.
    """

    def __init__(self, optimizer, total_steps, warmup_rate, end_lr=0, power=1.0, last_step=-1, **kargs):
        if warmup_rate < 0 or warmup_rate > 1:
            raise ValueError("warmup_rate should be between 0 and 1")
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_rate)
        self.end_lr = end_lr
        self.power = power
        self.last_step = last_step
        self.last_epoch = last_step
        super(PolynomialDecayWithWarmup, self).__init__(optimizer, last_step)

    def get_lr(self):
        if self.last_step < self.warmup_steps:
            # Linear warmup: 0 to base_lr
            return [base_lr * ((self.last_step + 1) / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # Polynomial decay from base_lr to end_lr
            progress = (self.last_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            return [
                self.end_lr + (base_lr - self.end_lr) * ((1.0 - progress) ** self.power)
                for base_lr in self.base_lrs
            ]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        self.last_epoch = step
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class CosineDecayWithWarmup(_LRScheduler):
    """
    Cosine decay scheduler with linear warmup.
    """

    def __init__(self, optimizer, total_steps, warmup_rate, end_lr=0, last_step=-1, **kargs):
        if warmup_rate < 0 or warmup_rate > 1:
            raise ValueError("warmup_rate should be between 0 and 1")
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_rate)
        self.end_lr = end_lr
        self.last_step = last_step
        self.last_epoch = last_step
        super(CosineDecayWithWarmup, self).__init__(optimizer, last_step)

    def get_lr(self):
        if self.last_step < self.warmup_steps:
            # Linear warmup: 0 to base_lr
            return [base_lr * ((self.last_step + 1) / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # Cosine decay from base_lr to end_lr
            progress = (self.last_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return [self.end_lr + (base_lr - self.end_lr) * cosine_decay for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        self.last_epoch = step
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_worker_seed(worker_id, worker_seed):
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


def load_model(model_name: str, **kargs):
    """
    Load model with architecture-specific module import.

    Args:
        model_name: Model name (e.g., 'DCP', 'MAPs', 'CLIP')
        **kargs: Must include 'cfg' with 'para.arch' parameter

    Returns:
        Instantiated model
    """
    # Extract config
    cfg = kargs.get("cfg")
    if cfg is None:
        raise ValueError("load_model requires 'cfg' parameter with architecture information")

    # Get architecture from para section
    arch = cfg.get("para", {}).get("arch")
    if arch is None:
        raise ValueError(f"Config must have 'arch' parameter in 'para' section for model '{model_name}'")

    # Special case: Base model has no architecture
    if model_name == "Base":
        module_name = f"core.model.{model_name}.base_model"
    else:
        module_name = f"core.model.{model_name}.{model_name}_{arch}_model"

    try:
        # Attempt to import the module
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import model module '{module_name}'. "
            f"Expected file: core/model/{model_name}/{model_name}_{arch}_model.py. "
            f"Original error: {str(e)}"
        )

    try:
        # Attempt to get the model class (class name remains unchanged, e.g., 'DCP', not 'DCP_CLIP')
        model_class = getattr(module, model_name)
    except AttributeError:
        raise ValueError(
            f"Model class '{model_name}' not found in module '{module_name}'. "
            f"Ensure the class is named '{model_name}' (not '{model_name}_{arch}')."
        )

    try:
        # Attempt to instantiate the model
        model = model_class(**kargs)
    except TypeError as e:
        raise TypeError(
            f"Error instantiating model '{model_name}': {str(e)}. Please check the provided arguments."
        )

    # Set the model name
    model.name = model_name

    return model


def get_dataset(model_name: str, dataset_name: str, **kargs):
    try:
        # Attempt to import the module
        module = importlib.import_module(f"core.model.{model_name}.{model_name}_data")
    except ImportError:
        raise ImportError(
            "Failed to import the 'data' module. Please ensure it exists and is in the correct path."
        )

    try:
        # Attempt to get the dataset class
        dataset_class = getattr(module, f"{dataset_name}_{model_name}_Dataset")
    except AttributeError:
        raise ValueError(f"Dataset '{dataset_name}' not found in the 'data' module.")

    try:
        # Attempt to instantiate the dataset
        dataset = dataset_class(**kargs)
    except TypeError as e:
        raise TypeError(
            f"Error instantiating dataset '{dataset_name}': {str(e)}. Please check the provided arguments."
        )

    # Set the dataset name
    dataset.name = dataset_name

    return dataset


def get_collator(model_name: str, dataset_name: str, **kargs):
    try:
        # Attempt to import the module
        module = importlib.import_module(f"core.model.{model_name}.{model_name}_data")
    except ImportError:
        raise ImportError(
            "Failed to import the 'data' module. Please ensure it exists and is in the correct path."
        )

    try:
        # Attempt to get the model class
        collator_class = getattr(module, f"{dataset_name}_{model_name}_Collator")
    except AttributeError:
        raise ValueError(f"Collator '{dataset_name}' not found in the 'data' module.")

    # Add dataset_name to kargs so collator can access it
    kargs["dataset_name"] = dataset_name

    try:
        # Attempt to instantiate the model
        collator = collator_class(**kargs)
    except TypeError as e:
        raise TypeError(
            f"Error instantiating collator '{dataset_name}': {str(e)}. Please check the provided arguments."
        )

    # Set the model name
    collator.name = dataset_name

    return collator


def get_optimizer(model: nn.Module, **kargs):
    optimizer_name = kargs.pop("name")
    optimizer = None
    match optimizer_name:
        case "AdamW":
            optimizer = torch.optim.AdamW
        case "Adam":
            optimizer = torch.optim.Adam
        case _:
            raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")
    return optimizer(model.parameters(), **kargs)


def get_scheduler(optimizer, **kargs):
    scheduler_name = kargs.pop("name")
    scheduler = None
    match scheduler_name:
        case "LinearDecayWithWarmup":
            total_steps = kargs.pop("total_steps")
            warmup_rate = kargs.pop("warmup_rate")
            warmup_steps = int(total_steps * warmup_rate)
            decay_steps = total_steps - warmup_steps

            # 1. Warmup: 0 -> 1.0 (relative to base LR)
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-8,  # Avoid true 0 to prevent division errors in some optimizers
                end_factor=1.0,
                total_iters=warmup_steps,
            )

            # 2. Decay: 1.0 -> 0.0
            # The decay should happen over the REMAINING steps
            decay_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=decay_steps)

            # Combine
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
            )

        case "WarmUpStepLR":
            scheduler = WarmUpStepLR(optimizer, **kargs)
        case "WarmupConstantLR":
            scheduler = WarmupConstantLR(optimizer, **kargs)
        case "PolynomialDecayWithWarmup":
            scheduler = PolynomialDecayWithWarmup(optimizer, **kargs)
        case "PolynomialLR":
            scheduler = PolynomialDecayWithWarmup(optimizer, **kargs)
        case "CosineDecayWithWarmup":
            scheduler = CosineDecayWithWarmup(optimizer, **kargs)
        case "DummyLR":
            scheduler = LambdaLR(optimizer, lambda x: 1)
        case _:
            raise NotImplementedError(f"Scheduler {scheduler_name} not implemented")
    return scheduler


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.safetensors", trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.safetensors'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        state_dict = {
            k: v.contiguous() if hasattr(v, "contiguous") else v for k, v in model.state_dict().items()
        }
        save_file(state_dict, self.path)
        self.val_loss_min = val_loss


def is_movable(obj):
    if is_tensor(obj):
        return True
    elif isinstance(obj, BatchEncoding):
        return True
    elif isinstance(obj, BatchFeature):
        return True
    else:
        return False


def copy_config_file():
    src_dir = "core/model"
    dest_dir = "core/config"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".yaml"):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_dir, file)
                shutil.copy2(src_file_path, dest_file_path)


def calculate_md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()
