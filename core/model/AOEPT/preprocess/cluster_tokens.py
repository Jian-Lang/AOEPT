#!/usr/bin/env python
"""
Token Clustering Script for AOEPT Model

Clusters collected tokens from cache/collect_token/ into N clusters per layer
using K-Means clustering, saving results to cache/collect_token_cluster/.

Usage:
    python core/model/AOEPT/preprocess/cluster_tokens.py \
        --input cache/collect_token/CLIP-Food101-text-0.7-mean.pt \
        --output cache/collect_token_cluster/CLIP-Food101-text-0.7-mean.pt \
        --n_clusters 32
"""

import argparse
from pathlib import Path

import torch
from loguru import logger
from sklearn.cluster import KMeans


def load_token_file(path: str) -> dict:
    """
    Load and validate token file.

    Args:
        path: Path to token file

    Returns:
        Dictionary containing token tensors and optionally 'sample_ids'

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If no keys ending with '_token' are found
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Token file not found: {path}")

    logger.info(f"Loading token file from {path}")
    data = torch.load(path, map_location="cpu")

    # Validate that at least one token key exists
    token_keys = [k for k in data.keys() if k.endswith("_token")]
    if not token_keys:
        raise KeyError("No keys ending with '_token' found in token file")

    # Log dimensions
    for key in token_keys:
        logger.info(f"  {key} shape: {data[key].shape}")

    if "sample_ids" in data:
        logger.info(f"  Number of samples: {len(data['sample_ids'])}")

    return data


def filter_zero_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Filter out samples with all-zero values.

    Args:
        tokens: Token tensor of shape [Samples, Layers, D]

    Returns:
        Tuple of (filtered_tokens, non_zero_mask)
    """
    # Sum across layers and embedding dimensions
    non_zero_mask = tokens.abs().sum(dim=(1, 2)) > 0
    filtered_tokens = tokens[non_zero_mask]

    num_original = tokens.shape[0]
    num_filtered = filtered_tokens.shape[0]
    num_removed = num_original - num_filtered

    if num_removed > 0:
        logger.info(f"  Filtered out {num_removed} zero samples ({num_removed/num_original*100:.1f}%)")
    else:
        logger.info(f"  No zero samples found")

    return filtered_tokens, non_zero_mask


def cluster_per_layer(
    tokens: torch.Tensor,
    n_clusters: int,
    random_state: int,
    modality: str = "text",
    l2_normalize: bool = False,
    target_norm: float | None = None,
) -> torch.Tensor:
    """
    Cluster tokens for each layer separately using K-Means.

    Args:
        tokens: [Samples, Layers, D] feature tensor.
        n_clusters: number of clusters per layer.
        random_state: random seed.
        modality: label for logging.
        l2_normalize: L2-normalize each sample to the unit hypersphere before
            k-means. Produces rank-1 centers when features are already collinear.
        target_norm: optional L2-norm rescale of each cluster center.

    Returns:
        Cluster centers of shape [Layers, N, D].
    """
    num_samples, num_layers, embed_dim = tokens.shape

    actual_n_clusters = min(n_clusters, num_samples)
    if actual_n_clusters < n_clusters:
        logger.warning(
            f"{modality}: Samples ({num_samples}) < n_clusters ({n_clusters}), "
            f"using {actual_n_clusters} clusters"
        )

    logger.info(
        f"{modality}: Clustering {num_samples} samples into {actual_n_clusters} clusters per layer "
        f"(l2_normalize={l2_normalize})"
    )

    clustered = torch.zeros(num_layers, actual_n_clusters, embed_dim, dtype=torch.float32)

    for layer_idx in range(num_layers):
        layer = tokens[:, layer_idx, :].float()

        if l2_normalize:
            layer = torch.nn.functional.normalize(layer, p=2, dim=-1)

        if layer.abs().sum().item() == 0:
            logger.warning(f"{modality}: Layer {layer_idx} has all zeros, skipping clustering")
            continue

        features_np = layer.cpu().numpy()

        kmeans = KMeans(
            n_clusters=actual_n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300,
            verbose=0,
        )
        kmeans.fit(features_np)
        centers = torch.from_numpy(kmeans.cluster_centers_).float()

        if target_norm is not None:
            # Rescale each center to the requested L2 norm. Preserves the
            # angular / principal-direction structure while bringing per-element
            # magnitudes to a scale compatible with prompt-init expectations.
            norms = centers.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            centers = centers * (target_norm / norms)

        clustered[layer_idx] = centers

        center_mean = clustered[layer_idx].mean().item()
        center_std = clustered[layer_idx].std().item()
        logger.info(
            f"{modality}: Layer {layer_idx:2d} - "
            f"inertia: {kmeans.inertia_:10.2f}, "
            f"mean: {center_mean:7.4f}, std: {center_std:7.4f}"
        )

    return clustered


def main():
    parser = argparse.ArgumentParser(
        description="Cluster collected tokens into N clusters per layer using K-Means"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input token file path (e.g., cache/collect_token/CLIP-Food101-text-0.7-mean.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output clustered token file path (e.g., cache/collect_token_cluster/CLIP-Food101-text-0.7-mean.pt)"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=32,
        help="Number of clusters per layer (default: 32)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for tensor operations (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--l2_normalize",
        dest="l2_normalize",
        action="store_true",
        help="L2-normalize features before k-means (legacy behavior).",
    )
    parser.add_argument(
        "--no_l2_normalize",
        dest="l2_normalize",
        action="store_false",
        help="Do not L2-normalize features (default).",
    )
    parser.set_defaults(l2_normalize=False)
    parser.add_argument(
        "--target_norm",
        type=float,
        default=None,
        help=(
            "Optional. If set, rescale every cluster center to have this L2 norm. "
            "Preserves angular structure while controlling per-element magnitude."
        ),
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Token Clustering Script")
    logger.info("=" * 80)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Number of clusters: {args.n_clusters}")
    logger.info(f"Random state: {args.random_state}")
    logger.info(f"Device: {args.device}")
    logger.info(f"L2-normalize features: {args.l2_normalize}")
    logger.info(f"Target center L2 norm: {args.target_norm}")
    logger.info("=" * 80)

    # Load token file
    data = load_token_file(args.input)

    # Identify token keys
    token_keys = [k for k in data.keys() if k.endswith("_token")]
    output_data = {}

    for key in token_keys:
        token_tensor = data[key]
        modality_name = key.replace("_token", "").capitalize()

        # Filter zero tokens
        logger.info(f"\nProcessing {modality_name} modality ({key})...")
        token_filtered, _ = filter_zero_tokens(token_tensor)

        if token_filtered.shape[0] == 0:
            logger.warning(f"No valid tokens found for {modality_name}, skipping clustering.")
            continue

        # Cluster tokens
        logger.info("\n" + "=" * 80)
        logger.info(f"Clustering {modality_name} Tokens")
        logger.info("=" * 80)
        clustered = cluster_per_layer(
            token_filtered,
            args.n_clusters,
            args.random_state,
            modality=modality_name,
            l2_normalize=args.l2_normalize,
            target_norm=args.target_norm,
        )

        output_data[key] = clustered

    # Include sample_ids for reference if present
    if "sample_ids" in data:
        output_data["sample_ids"] = data["sample_ids"]

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save clustered tokens
    logger.info("\n" + "=" * 80)
    logger.info(f"Saving clustered tokens to {args.output}")
    torch.save(output_data, args.output)

    for key, val in output_data.items():
        if isinstance(val, torch.Tensor):
            logger.info(f"  {key}: {val.shape}")

    logger.info("=" * 80)
    logger.info("Clustering complete!")
    logger.info("=" * 80)



if __name__ == "__main__":
    main()
