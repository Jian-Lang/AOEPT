from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

from core.model.Base.base_data import Dataset_Factory


def generate_missing_table(dataset_name: str, missing_type: str, missing_rate: float) -> Path:
    """Create a missing modality table and store it as a JSONL file."""
    dataset = Dataset_Factory.get_dataset(dataset_name)
    all_ids = list(dataset._get_ids())
    id_count = len(all_ids)

    rng = np.random.default_rng()

    table = pd.DataFrame({"id": all_ids})
    table["text_missing"] = False
    table["image_missing"] = False

    missing_total = int(id_count * missing_rate)

    if missing_type == "text":
        selected = rng.choice(id_count, size=missing_total, replace=False) if missing_total else np.array([], dtype=int)
        table.loc[selected, "text_missing"] = True
    elif missing_type == "image":
        selected = rng.choice(id_count, size=missing_total, replace=False) if missing_total else np.array([], dtype=int)
        table.loc[selected, "image_missing"] = True
    else:  # Assumes other values mean removing both modalities.
        selected = rng.choice(id_count, size=missing_total, replace=False) if missing_total else np.array([], dtype=int)
        selected = rng.permutation(selected)
        text_count = missing_total // 2
        text_indices = selected[:text_count]
        image_indices = selected[text_count:]
        table.loc[text_indices, "text_missing"] = True
        table.loc[image_indices, "image_missing"] = True

    rate_str = f"{missing_rate}".rstrip("0").rstrip(".")
    output_dir = Path("data") / dataset_name / "missing_tbl"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{missing_type}-{rate_str}.jsonl"
    table.to_json(output_path, orient="records", lines=True)
    return output_path


def main() -> None:
    parser = ArgumentParser(description="Generate a missing table for multimodal datasets.")
    parser.add_argument("--dataset_name", help="Dataset identifier registered in Dataset_Factory.")
    parser.add_argument("--missing_type", help="One of text, image, or both to mark missing modalities.")
    parser.add_argument("--missing_rate", type=float, help="Fraction of samples to mark as missing.")
    args = parser.parse_args()

    output_path = generate_missing_table(args.dataset_name, args.missing_type, args.missing_rate)
    print(output_path)


if __name__ == "__main__":
    main()
