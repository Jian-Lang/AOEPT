import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Convert to M (millions)
    def to_millions(num):
        return num / 1_000_000

    total_params_m = to_millions(total_params)
    trainable_params_m = to_millions(trainable_params)

    # Return total and trainable in M
    return total_params_m, trainable_params_m


def get_model_params_info(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0

    return total_params, trainable_params, trainable_ratio


def get_clip_token(
    epoch: int,
    layer_idx: int,
    text_hidden_states: torch.Tensor,
    vision_hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    eos_token_id: int,
    sample_ids: list[str],
    token_collector: dict,
    attention_mask: torch.Tensor = None,
    pooling_method: str = "mean",
    missing_masks: torch.Tensor = None,
) -> None:
    """
    Collect CLIP layer-wise tokens using specified pooling method, plus complete tokens.

    Args:
        epoch: Current training epoch
        layer_idx: Layer index in the transformer
        text_hidden_states: Text embeddings [batch, seq_len, embed_dim]
        vision_hidden_states: Vision embeddings [batch, seq_len, embed_dim]
        input_ids: Token IDs to locate EOS positions
        eos_token_id: EOS token ID from config
        sample_ids: List of sample identifiers
        token_collector: Mutable dict to accumulate tokens
        attention_mask: Attention mask for text [batch, seq_len], required for mean pooling
        pooling_method: Pooling strategy - "cls" (EOS/CLS tokens) or "mean" (average pooling)
        missing_masks: Missing modality masks [batch, 2] ([:, 0] for text, [:, 1] for vision)
    """
    if epoch not in token_collector:
        token_collector[epoch] = {}
    if layer_idx not in token_collector[epoch]:
        token_collector[epoch][layer_idx] = {
            "text_eos": [],
            "vision_cls": [],
            "complete_token": [],
            "sample_ids": [],
        }

    batch_size = text_hidden_states.shape[0]

    if pooling_method == "cls":
        # Extract text EOS token
        eos_positions = (input_ids == eos_token_id).int().argmax(dim=-1)
        text_token = text_hidden_states[
            torch.arange(batch_size, device=text_hidden_states.device), eos_positions
        ]

        # Extract vision CLS token (first token)
        vision_token = vision_hidden_states[:, 0, :]

    elif pooling_method == "mean":
        # Text mean pooling using attention mask
        if attention_mask is None:
            raise ValueError("attention_mask is required for mean pooling")

        # Expand attention mask to match hidden state dimensions
        text_mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]

        # Compute mean over valid tokens
        text_sum = (text_hidden_states * text_mask_expanded).sum(dim=1)  # [batch, embed_dim]
        text_count = text_mask_expanded.sum(dim=1).clamp(min=1)  # [batch, 1]
        text_token = text_sum / text_count  # [batch, embed_dim]

        # Vision mean pooling (all tokens are valid, no padding)
        vision_token = vision_hidden_states.mean(dim=1)  # [batch, embed_dim]

    elif pooling_method == "max":
        # Text max pooling using attention mask
        if attention_mask is None:
            raise ValueError("attention_mask is required for max pooling")

        # Expand attention mask to match hidden state dimensions
        text_mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]

        # Mask padding tokens with -inf before taking max
        masked_text = text_hidden_states.clone()
        masked_text[text_mask_expanded.squeeze(-1) == 0] = float("-inf")
        text_token = masked_text.max(dim=1)[0]  # [batch, embed_dim]

        # Vision max pooling (all tokens are valid, no padding)
        vision_token = vision_hidden_states.max(dim=1)[0]  # [batch, embed_dim]

    else:
        raise ValueError(f"pooling_method must be 'cls', 'mean', or 'max', got '{pooling_method}'")

    # For CLIP: complete_token is not implemented, use zero tensor
    embed_dim = text_hidden_states.shape[-1]
    complete_token = torch.zeros(batch_size, embed_dim, device=text_hidden_states.device)

    # Store tokens
    token_collector[epoch][layer_idx]["text_eos"].append(text_token.detach().cpu())
    token_collector[epoch][layer_idx]["vision_cls"].append(vision_token.detach().cpu())
    token_collector[epoch][layer_idx]["complete_token"].append(complete_token.detach().cpu())
    token_collector[epoch][layer_idx]["sample_ids"].extend(sample_ids)


def get_vilt_token(
    epoch: int,
    layer_idx: int,
    text_hidden_states: torch.Tensor,
    vision_hidden_states: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    sample_ids: list[str],
    token_collector: dict,
    missing_masks: torch.Tensor,
    combined_hidden_states: torch.Tensor,
    pooling_method: str = "mean",
) -> None:
    """
    Collect ViLT layer-wise tokens for text and vision modalities, plus complete tokens.

    Args:
        epoch: Current training epoch
        layer_idx: Layer index in the transformer
        text_hidden_states: Text hidden states (already masked) [batch, seq_len, embed_dim]
        vision_hidden_states: Vision hidden states (already masked) [batch, seq_len, embed_dim]
        token_type_ids: Token type IDs [batch, seq_len] (0 for text, 1 for vision)
        attention_mask: Attention mask [batch, seq_len]
        sample_ids: List of sample identifiers
        token_collector: Mutable dict to accumulate tokens
        missing_masks: Missing modality masks [batch, 2] ([:, 0] for text, [:, 1] for vision)
        combined_hidden_states: Unmasked hidden states [batch, seq_len, embed_dim] for complete_token
        pooling_method: Pooling strategy - "mean" (average pooling) or "max" (max pooling)
    """
    if epoch not in token_collector:
        token_collector[epoch] = {}
    if layer_idx not in token_collector[epoch]:
        token_collector[epoch][layer_idx] = {
            "text_token": [],
            "vision_token": [],
            "complete_token": [],
            "sample_ids": [],
        }

    # Create masks for text and vision tokens
    # token_type_ids: 0 for text, 1 for vision
    text_mask = (token_type_ids == 0) & (attention_mask == 1)  # [batch, seq_len]
    vision_mask = (token_type_ids == 1) & (attention_mask == 1)  # [batch, seq_len]

    # Expand masks for broadcasting: [batch, seq_len, 1]
    text_mask_expanded = text_mask.unsqueeze(-1).float()
    vision_mask_expanded = vision_mask.unsqueeze(-1).float()

    if pooling_method == "mean":
        # Calculate mean pooling for text tokens
        text_sum = (text_hidden_states * text_mask_expanded).sum(dim=1)  # [batch, embed_dim]
        text_count = text_mask_expanded.sum(dim=1).clamp(min=1)  # [batch, 1], avoid division by zero
        text_token = text_sum / text_count  # [batch, embed_dim]

        # Calculate mean pooling for vision tokens
        vision_sum = (vision_hidden_states * vision_mask_expanded).sum(dim=1)  # [batch, embed_dim]
        vision_count = vision_mask_expanded.sum(dim=1).clamp(min=1)  # [batch, 1]
        vision_token = vision_sum / vision_count  # [batch, embed_dim]

    elif pooling_method == "max":
        # Calculate max pooling for text tokens
        masked_text = text_hidden_states.clone()
        masked_text[text_mask_expanded.squeeze(-1) == 0] = float("-inf")
        text_token = masked_text.max(dim=1)[0]  # [batch, embed_dim]

        # Calculate max pooling for vision tokens
        masked_vision = vision_hidden_states.clone()
        masked_vision[vision_mask_expanded.squeeze(-1) == 0] = float("-inf")
        vision_token = masked_vision.max(dim=1)[0]  # [batch, embed_dim]

    elif pooling_method == "cls":
        # Use CLS token (index 0) from combined_hidden_states
        cls_token = combined_hidden_states[:, 0]  # [batch, embed_dim]

        # Mask out if modality is missing
        text_keep = (missing_masks[:, 0] == 0).float().unsqueeze(-1)  # [batch, 1]
        vision_keep = (missing_masks[:, 1] == 0).float().unsqueeze(-1)  # [batch, 1]

        text_token = cls_token * text_keep
        vision_token = cls_token * vision_keep

    else:
        raise ValueError(f"pooling_method must be 'mean', 'max', or 'cls', got '{pooling_method}'")

    # Calculate complete_token: pooling across all tokens for complete samples only
    # Identify complete samples (both modalities present)
    complete_mask = ~(missing_masks[:, 0] | missing_masks[:, 1])  # [batch], True if complete

    batch_size = combined_hidden_states.shape[0]
    embed_dim = combined_hidden_states.shape[-1]
    complete_token = torch.zeros(batch_size, embed_dim, device=combined_hidden_states.device)

    if complete_mask.any():
        attention_mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]

        if pooling_method == "mean":
            # For complete samples, compute mean across all valid tokens
            token_sum = (combined_hidden_states * attention_mask_expanded).sum(dim=1)  # [batch, embed_dim]
            token_count = attention_mask_expanded.sum(dim=1).clamp(min=1)  # [batch, 1]
            complete_pooled = token_sum / token_count  # [batch, embed_dim]

        elif pooling_method == "max":
            # For complete samples, compute max across all valid tokens
            masked_combined = combined_hidden_states.clone()
            masked_combined[attention_mask_expanded.squeeze(-1) == 0] = float("-inf")
            complete_pooled = masked_combined.max(dim=1)[0]  # [batch, embed_dim]

        elif pooling_method == "cls":
            complete_pooled = combined_hidden_states[:, 0]  # [batch, embed_dim]

        # Only keep values for complete samples, zero out incomplete ones
        complete_token[complete_mask] = complete_pooled[complete_mask]

    # Store tokens
    token_collector[epoch][layer_idx]["text_token"].append(text_token.detach().cpu())
    token_collector[epoch][layer_idx]["vision_token"].append(vision_token.detach().cpu())
    token_collector[epoch][layer_idx]["complete_token"].append(complete_token.detach().cpu())
    token_collector[epoch][layer_idx]["sample_ids"].extend(sample_ids)


def get_sample_specific_prompts(
    epoch: int,
    missing_aware_text_prompt: torch.Tensor,
    missing_aware_vision_prompt: torch.Tensor,
    sample_ids: list,
    prompt_collector: dict,
) -> None:
    """
    Collect sample-specific prompts that were selected during forward pass.

    This function accumulates the prompts actually used for each sample based on
    their missing modality pattern. Unlike global prompt collection, this tracks
    which specific prompt was selected from the prompt pool for each sample.

    Args:
        epoch: Current training epoch
        missing_aware_text_prompt: Selected text prompts for this batch [batch, prompt_num, prompt_len, 512]
        missing_aware_vision_prompt: Selected vision prompts for this batch [batch, prompt_num, prompt_len, 768]
        sample_ids: List of sample IDs for this batch
        prompt_collector: Dictionary to accumulate collected prompts
    """
    if epoch not in prompt_collector:
        prompt_collector[epoch] = {
            "text_prompts": [],
            "vision_prompts": [],
            "sample_ids": [],
        }

    # Append batch data to lists
    prompt_collector[epoch]["text_prompts"].append(
        missing_aware_text_prompt.detach().half().cpu()
    )
    prompt_collector[epoch]["vision_prompts"].append(
        missing_aware_vision_prompt.detach().half().cpu()
    )

    # Extend flat list with sample IDs
    prompt_collector[epoch]["sample_ids"].extend(sample_ids)


def save_features(
    feature_collector: dict,
    model_name: str,
    dataset_name: str,
    save_dir: Path | str = "./cache/collect_feature",
    top_percent: float = 1.0,
    selection_seed: int = 42,
) -> None:
    """
    Save collected features to .pt file with stacked format.

    Directory structure:
        save_dir/
            {model}-{dataset}.pt

    Saved file contains:
        For CLIP: 'text': [num_samples, num_layers, seq_len, embed_dim]
                  'vision': [num_samples, num_layers, seq_len, embed_dim]
                  'sample_ids': [num_samples]
        For ViLT: 'combined': [num_samples, num_layers, seq_len, embed_dim]
                  'sample_ids': [num_samples]

    Args:
        feature_collector: Dictionary containing collected features
        model_name: Model name (CLIP or ViLT)
        dataset_name: Dataset name
        save_dir: Base directory to save features
    """
    from loguru import logger

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get the first available epoch data
    epoch_data = next(iter(feature_collector.values()))

    # Get number of layers
    num_layers = len(epoch_data)

    # Stack features by layer
    if "text" in epoch_data[0]:
        # CLIP structure
        sample_ids_list = epoch_data[0]["sample_ids"]

        text_by_layer = []
        vision_by_layer = []

        for layer_idx in range(num_layers):
            text_tensors = epoch_data[layer_idx]["text"]
            vision_tensors = epoch_data[layer_idx]["vision"]

            # Concatenate batch tensors: [num_samples, seq_len, embed_dim]
            text_layer = torch.cat(text_tensors, dim=0)
            vision_layer = torch.cat(vision_tensors, dim=0)

            text_by_layer.append(text_layer)
            vision_by_layer.append(vision_layer)

        # Stack layers: [num_samples, num_layers, seq_len, embed_dim]
        text_features = torch.stack(text_by_layer, dim=1)
        vision_features = torch.stack(vision_by_layer, dim=1)

        # Get attention mask if available (only from layer 0)
        attention_mask = None
        if "attention_mask" in epoch_data[0] and len(epoch_data[0]["attention_mask"]) > 0:
            attention_mask = torch.cat(epoch_data[0]["attention_mask"], dim=0)  # [num_samples, seq_len]

        output = {
            "text": text_features,
            "vision": vision_features,
            "sample_ids": sample_ids_list,
        }
        if attention_mask is not None:
            output["attention_mask"] = attention_mask

    elif "combined" in epoch_data[0]:
        # ViLT structure
        sample_ids_list = epoch_data[0]["sample_ids"]

        combined_by_layer = []

        for layer_idx in range(num_layers):
            combined_tensors = epoch_data[layer_idx]["combined"]

            # Concatenate batch tensors: [num_samples, seq_len, embed_dim]
            combined_layer = torch.cat(combined_tensors, dim=0)

            combined_by_layer.append(combined_layer)

        # Stack layers: [num_samples, num_layers, seq_len, embed_dim]
        combined_features = torch.stack(combined_by_layer, dim=1)

        output = {
            "combined": combined_features,
            "sample_ids": sample_ids_list,
        }

    # Save to file
    filename = f"{model_name}-{dataset_name}.pt"
    save_path = save_dir / filename
    torch.save(output, save_path)

    logger.info(f"Saved feature collection to {save_path}")


def save_token_collection(
    token_collector: dict,
    model_name: str,
    dataset_name: str,
    missing_type: str,
    missing_rate: float,
    save_dir: Path | str = "./cache/collect_token",
    pooling_method: str = "mean",
) -> None:
    """
    Save collected tokens to single .pt file with stacked format.

    Directory structure:
        save_dir/
            {model}-{dataset}-{missing_type}-{missing_rate}-{pooling_method}.pt

    Saved file contains:
        'text_token': [num_samples, num_layers, embed_dim]
        'vision_token': [num_samples, num_layers, embed_dim]
        'complete_token': [num_samples, num_layers, embed_dim] - Mean pooled tokens from complete samples only
        'sample_ids': [num_samples]

    Args:
        token_collector: Dictionary containing collected tokens
        model_name: Model name (CLIP or ViLT)
        dataset_name: Dataset name
        missing_type: Missing scenario type
        missing_rate: Missing rate
        save_dir: Base directory to save tokens
        pooling_method: Token pooling method ("cls" or "mean")
    """
    from loguru import logger

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Standard format for CLIP/ViLT
    # Assume epoch 0 for inference
    epoch = list(token_collector.keys())[0]
    layer_data = token_collector[epoch]

    # Get number of layers
    num_layers = len(layer_data)

    # Stack tokens by layer
    if model_name == "CLIP":
        # Get sample_ids from first layer
        sample_ids = layer_data[0]["sample_ids"]

        # Stack text EOS tokens: [num_samples, num_layers, embed_dim]
        text_eos_by_layer = []
        vision_cls_by_layer = []
        complete_by_layer = []

        for layer_idx in range(num_layers):
            text_eos_tensors = layer_data[layer_idx]["text_eos"]
            vision_cls_tensors = layer_data[layer_idx]["vision_cls"]
            complete_tensors = layer_data[layer_idx]["complete_token"]

            # Concatenate batch tensors
            text_eos_layer = torch.cat(text_eos_tensors, dim=0)  # [num_samples, embed_dim]
            vision_cls_layer = torch.cat(vision_cls_tensors, dim=0)
            complete_layer = torch.cat(complete_tensors, dim=0)

            text_eos_by_layer.append(text_eos_layer)
            vision_cls_by_layer.append(vision_cls_layer)
            complete_by_layer.append(complete_layer)

        # Stack layers: [num_samples, num_layers, embed_dim]
        text_token = torch.stack(text_eos_by_layer, dim=1)
        vision_token = torch.stack(vision_cls_by_layer, dim=1)
        complete_token = torch.stack(complete_by_layer, dim=1)

        # Create output dict
        output = {
            "text_token": text_token,
            "vision_token": vision_token,
            "complete_token": complete_token,
            "sample_ids": sample_ids,
        }

    elif model_name == "ViLT":
        # Get sample_ids from first layer
        sample_ids = layer_data[0]["sample_ids"]

        # Stack text and vision tokens: [num_samples, num_layers, embed_dim]
        text_by_layer = []
        vision_by_layer = []
        complete_by_layer = []

        for layer_idx in range(num_layers):
            text_tensors = layer_data[layer_idx]["text_token"]
            vision_tensors = layer_data[layer_idx]["vision_token"]
            complete_tensors = layer_data[layer_idx]["complete_token"]

            # Concatenate batch tensors
            text_layer = torch.cat(text_tensors, dim=0)  # [num_samples, embed_dim]
            vision_layer = torch.cat(vision_tensors, dim=0)
            complete_layer = torch.cat(complete_tensors, dim=0)

            text_by_layer.append(text_layer)
            vision_by_layer.append(vision_layer)
            complete_by_layer.append(complete_layer)

        # Stack layers: [num_samples, num_layers, embed_dim]
        text_token = torch.stack(text_by_layer, dim=1)
        vision_token = torch.stack(vision_by_layer, dim=1)
        complete_token = torch.stack(complete_by_layer, dim=1)

        # Create output dict
        output = {
            "text_token": text_token,
            "vision_token": vision_token,
            "complete_token": complete_token,
            "sample_ids": sample_ids,
        }

    # Save to file
    filename = f"{model_name}-{dataset_name}-{missing_type}-{missing_rate}-{pooling_method}.pt"
    save_path = save_dir / filename
    torch.save(output, save_path)

    logger.info(f"Saved token collection to {save_path}")


def save_prompt_collection(
    prompt_collector: dict,
    model_name: str,
    dataset_name: str,
    missing_type: str,
    missing_rate: float,
    save_dir: Path | str = "./cache/collect_prompt",
    top_percent: float = 1.0,
    selection_seed: int = 42,
    ablation: str | None = None,
) -> None:
    """
    Save sample-specific collected prompts to .pt file.

    Saved file contains:
        'text_prompts': [num_samples, prompt_num, prompt_len, 512]
        'vision_prompts': [num_samples, prompt_num, prompt_len, 768]
        'sample_ids': [num_samples] - sample identifiers
        'metadata': dict with configuration information

    Args:
        prompt_collector: Dictionary containing collected prompts
        model_name: Model name (e.g., "MAPs")
        dataset_name: Dataset name (e.g., "Food101")
        missing_type: Missing scenario type ("text", "image", or "both")
        missing_rate: Missing rate (e.g., 0.7)
        save_dir: Base directory to save prompts
    """
    from datetime import datetime
    from loguru import logger

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get data from first (and likely only) epoch
    epoch_data = next(iter(prompt_collector.values()))

    # Stack text prompts: List of [batch, prompt_num, prompt_len, 512] -> [num_samples, prompt_num, prompt_len, 512]
    text_prompts = torch.cat(epoch_data["text_prompts"], dim=0)

    # Stack vision prompts: List of [batch, prompt_num, prompt_len, 768] -> [num_samples, prompt_num, prompt_len, 768]
    vision_prompts = torch.cat(epoch_data["vision_prompts"], dim=0)

    # Get sample_ids
    sample_ids_list = epoch_data["sample_ids"]

    # Get metadata from tensor shapes
    num_samples = text_prompts.shape[0]
    prompt_num = text_prompts.shape[1]
    prompt_len = text_prompts.shape[2]

    # Create metadata
    metadata = {
        "missing_type": missing_type,
        "prompt_num": prompt_num,
        "prompt_len": prompt_len,
        "num_samples": num_samples,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "missing_rate": missing_rate,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Create output dict
    output = {
        "text_prompts": text_prompts,
        "vision_prompts": vision_prompts,
        "sample_ids": sample_ids_list,
        "metadata": metadata,
    }

    # Save to file
    if ablation:
        # Replace / with - in ablation name for safe filenames
        safe_ablation = ablation.replace("/", "-")
        filename = f"{model_name}-{dataset_name}-{missing_type}-{missing_rate}-{safe_ablation}-prompts.pt"
    else:
        filename = f"{model_name}-{dataset_name}-{missing_type}-{missing_rate}-prompts.pt"
    save_path = save_dir / filename
    torch.save(output, save_path)

    logger.info(f"Saved sample-specific prompt collection to {save_path}")
    logger.info(f"  Text prompts shape: {text_prompts.shape}")
    logger.info(f"  Vision prompts shape: {vision_prompts.shape}")
    logger.info(f"  Number of samples: {num_samples}")
    logger.info(f"  Sample IDs: {len(sample_ids_list)} samples")


def run_statis(
    statis_type: str | None = None,
    model_name: str = "model",
    save_dir: str | Path = None,
    top_percent: float = 1.0,
    **kwargs,
) -> None:
    """
    Run statistics collection based on model output and requested type.

    Args:
        statis_type: Statistics option from config
        model_name: Name of the model for organizing saved features
        save_dir: Base directory for saving statistics
        top_percent: Percentage of samples to randomly select for saving (default: 1.0 for 100%)
        **kwargs: Additional parameters passed directly (token_collector, dataset_name, missing_type, etc.)
    """
    match statis_type:
        case None:
            return

        case "collect_token":
            token_collector = kwargs.get("token_collector")
            if token_collector is None:
                return

            dataset_name = kwargs.get("dataset_name", "unknown")
            missing_type = kwargs.get("missing_type", "unknown")
            missing_rate = kwargs.get("missing_rate", 0.0)
            pooling_method = kwargs.get("pooling_method", "mean")

            # Only pass save_dir if it's not None, to allow default value
            token_args = {
                "token_collector": token_collector,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "missing_type": missing_type,
                "missing_rate": missing_rate,
                "pooling_method": pooling_method,
            }
            if save_dir is not None:
                token_args["save_dir"] = save_dir

            save_token_collection(**token_args)

        case "collect_features":
            feature_collector = kwargs.get("feature_collector")
            if feature_collector is None or len(feature_collector) == 0:
                return

            dataset_name = kwargs.get("dataset_name", "unknown")
            selection_seed = kwargs.get("selection_seed", 42)

            # Save features to disk
            feature_args = {
                "feature_collector": feature_collector,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "top_percent": top_percent,
                "selection_seed": selection_seed,
            }
            if save_dir is not None:
                feature_args["save_dir"] = save_dir

            save_features(**feature_args)

        case "collect_prompts":
            prompt_collector = kwargs.get("prompt_collector")
            if prompt_collector is None or len(prompt_collector) == 0:
                return

            dataset_name = kwargs.get("dataset_name", "unknown")
            missing_type = kwargs.get("missing_type", "unknown")
            missing_rate = kwargs.get("missing_rate", 0.0)
            selection_seed = kwargs.get("selection_seed", 42)
            ablation = kwargs.get("ablation", None)

            # Only pass save_dir if it's not None, to allow default value
            prompt_args = {
                "prompt_collector": prompt_collector,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "missing_type": missing_type,
                "missing_rate": missing_rate,
                "top_percent": top_percent,
                "selection_seed": selection_seed,
                "ablation": ablation,
            }
            if save_dir is not None:
                prompt_args["save_dir"] = save_dir

            save_prompt_collection(**prompt_args)

        case _:
            # Unknown statis_type, do nothing
            return
