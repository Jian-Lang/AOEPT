#!/usr/bin/env python3
"""
Preprocessing script to generate data.jsonl and split CSV files for each dataset.
This script creates standardized data files using DataLoader for faster processing.
"""

import argparse
import json
from pathlib import Path
from threading import Lock
from typing import List

import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# Copy dataset classes from core/model/Base/base_data.py to avoid dependencies
class Base_Dataset(Dataset):
    def __init__(self, **kargs):
        super().__init__()
        self.missing_type = kargs.get("missing_type", None)
        self.missing_rate = kargs.get("missing_rate", None)
        self.data_path = Path("data")
        self._initialized_missing_tbl = False

    def _get_data(self, split: str, task: str = "binary"):
        raise NotImplementedError

    def get_split(self, split: str):
        raise NotImplementedError

    def _get_ids(self):
        # fetch from get_split with train/val/test
        return self.get_split("train") + self.get_split("val") + self.get_split("test")

    def _get_missing_mask(self, id) -> list[bool]:
        # return text_missing, image_missing
        if not self._initialized_missing_tbl:
            self._initialized_missing_tbl = True
            self.missing_tbl = pd.read_json(
                self.data_path / "missing_tbl" / f"{self.missing_type}-{self.missing_rate}.jsonl", lines=True
            )
        text_missing = self.missing_tbl[self.missing_tbl["id"] == id]["text_missing"].values[0]
        image_missing = self.missing_tbl[self.missing_tbl["id"] == id]["image_missing"].values[0]
        return [text_missing, image_missing]


class Food101_Dataset(Base_Dataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.data_path = Path("data/food101")
        self._initialized_text = False

    def _get_image(self, id):
        image = self.data_path / "image" / f"{id}.jpg"
        return image

    def _get_text(self, id):
        if not self._initialized_text:
            self._initialized_text = True
            train_text_file = pd.read_csv(
                self.data_path / "raw/meta_data/train_titles.csv", header=None, names=["id", "text", "label"]
            )
            test_text_file = pd.read_csv(
                self.data_path / "raw/meta_data/test_titles.csv", header=None, names=["id", "text", "label"]
            )
            self.text_data = pd.concat([train_text_file, test_text_file], axis=0)
            # Load label mapping once
            with open(self.data_path / "raw/class_idx.json", "r") as f:
                self.label_map = json.load(f)
        text = self.text_data[self.text_data["id"] == f"{id}.jpg"]["text"].values[0]
        return text

    def _get_label(self, id):
        if not self._initialized_text:
            self._initialized_text = True
            train_text_file = pd.read_csv(
                self.data_path / "raw/meta_data/train_titles.csv", header=None, names=["id", "text", "label"]
            )
            test_text_file = pd.read_csv(
                self.data_path / "raw/meta_data/test_titles.csv", header=None, names=["id", "text", "label"]
            )
            self.text_data = pd.concat([train_text_file, test_text_file], axis=0)
            # Load label mapping once
            with open(self.data_path / "raw/class_idx.json", "r") as f:
                self.label_map = json.load(f)
        label_name = self.text_data[self.text_data["id"] == f"{id}.jpg"]["label"].values[0]
        label = self.label_map[label_name]
        return label

    def get_split(self, split) -> List[str]:
        with open(self.data_path / "raw/split.json", "r") as f:
            split_data = json.load(f)
        split_map = {"train": "train", "val": "val", "test": "test"}
        # Remove .jpg extension from IDs to match the format expected by _get_text and _get_label
        return [id.split(".jpg")[0] for id in split_data[split_map[split]]]


class MMIMDB_Dataset(Base_Dataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.data_path = Path("data/mmimdb")
        self._initialized_text = False
        # Define the 23 genres to use (excluding rare genres: Adult, News, Reality-TV, Talk-Show)
        self.all_genres = [
            "Action",
            "Adventure",
            "Animation",
            "Biography",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Family",
            "Fantasy",
            "Film-Noir",
            "History",
            "Horror",
            "Music",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Short",
            "Sport",
            "Thriller",
            "War",
            "Western",
        ]

    def _get_image(self, id):
        # Try .jpeg extension first, then .jpg
        image_path = self.data_path / "image" / f"{id}.jpeg"
        if not image_path.exists():
            image_path = self.data_path / "image" / f"{id}.jpg"
        return image_path

    def _get_text(self, id):
        if not self._initialized_text:
            self._initialized_text = True
            text_file_path = Path(self.data_path) / "raw/meta_data"
            data_list = []
            for file in text_file_path.glob("*.json"):
                with open(file, "r") as f:
                    data = json.load(f)
                    data["id"] = file.stem  # Use filename without extension as ID
                    data_list.append(data)

            self.text_data = pd.DataFrame(data_list)

        # Find the row with matching ID
        row = self.text_data[self.text_data["id"] == id]
        if len(row) == 0:
            raise ValueError(f"ID {id} not found in MMIMDB dataset")

        plot = row["plot outline"].iloc[0]
        if isinstance(plot, list) and len(plot) > 0:
            return plot[0]  # Take the first plot description
        return str(plot)

    def _get_text_long(self, id):
        if not self._initialized_text:
            self._initialized_text = True
            text_file_path = Path(self.data_path) / "raw/meta_data"
            data_list = []
            for file in text_file_path.glob("*.json"):
                with open(file, "r") as f:
                    data = json.load(f)
                    data["id"] = file.stem  # Use filename without extension as ID
                    data_list.append(data)

            self.text_data = pd.DataFrame(data_list)

        # Find the row with matching ID
        row = self.text_data[self.text_data["id"] == id]
        if len(row) == 0:
            raise ValueError(f"ID {id} not found in MMIMDB dataset")

        plot = row["plot"].iloc[0]
        if isinstance(plot, list) and len(plot) > 0:
            return plot[0]  # Take the first plot description
        return str(plot)

    def _get_label(self, id):
        if not self._initialized_text:
            self._initialized_text = True
            text_file_path = Path(self.data_path) / "raw/meta_data"
            data_list = []
            for file in text_file_path.glob("*.json"):
                with open(file, "r") as f:
                    data = json.load(f)
                    data["id"] = file.stem  # Use filename without extension as ID
                    data_list.append(data)

            self.text_data = pd.DataFrame(data_list)

        # Find the row with matching ID
        row = self.text_data[self.text_data["id"] == id]
        if len(row) == 0:
            raise ValueError(f"ID {id} not found in MMIMDB dataset")

        genres = row["genres"].iloc[0]
        if not isinstance(genres, list):
            genres = []

        # Convert to multi-hot encoding (only use the 23 predefined genres)
        multi_hot = np.zeros(len(self.all_genres), dtype=int)
        for genre in genres:
            if genre in self.all_genres:
                index = self.all_genres.index(genre)
                multi_hot[index] = 1
        return multi_hot

    def get_split(self, split) -> List[str]:
        with open(self.data_path / "raw/split.json", "r") as f:
            split_data = json.load(f)
        split_map = {"train": "train", "val": "dev", "test": "test"}
        return split_data[split_map[split]]


class HateMemes_Dataset(Base_Dataset):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.data_path = Path("data/hatememes")
        self._initialized_text = False

    def _load_all_data(self):
        """Load all JSONL files from raw/original directory"""
        self._initialized_text = True
        original_path = self.data_path / "raw/original"

        # Read all JSONL files
        data_frames = []
        for jsonl_file in original_path.glob("*.jsonl"):
            df = pd.read_json(jsonl_file, lines=True)
            data_frames.append(df)

        self.text_data = pd.concat(data_frames, axis=0, ignore_index=True)

    def _get_image(self, id):
        # Get image path from the data
        if not self._initialized_text:
            self._load_all_data()
        img_path = self.text_data[self.text_data["id"] == id]["img"].values[0]
        # Extract just the filename (e.g., "42953.png" from "img/42953.png")
        image_filename = Path(img_path).name
        return image_filename

    def _get_text(self, id):
        if not self._initialized_text:
            self._load_all_data()
        text = self.text_data[self.text_data["id"] == id]["text"].values[0]
        return text

    def _get_label(self, id):
        if not self._initialized_text:
            self._load_all_data()
        label = self.text_data[self.text_data["id"] == id]["label"].values[0]
        return label

    def get_split(self, split) -> List[str]:
        original_path = self.data_path / "raw/original"

        # Map split names to files
        match split:
            case "train":
                data_file = original_path / "train.jsonl"
            case "val":
                data_file = original_path / "dev_seen.jsonl"
            case "test":
                data_file = original_path / "test_seen.jsonl"
            case _:
                raise ValueError(f"Invalid split: {split}")

        data_df = pd.read_json(data_file, lines=True)
        return data_df["id"].tolist()


class Dataset_Factory:
    @staticmethod
    def get_dataset(name: str, **kargs) -> Base_Dataset:
        match name:
            case "food101":
                return Food101_Dataset(**kargs)
            case "mmimdb":
                return MMIMDB_Dataset(**kargs)
            case "hatememes":
                return HateMemes_Dataset(**kargs)
            case _:
                raise ValueError(f"Dataset {name} not supported")


# Preprocessing dataset for DataLoader
class PreprocessingDataset(Dataset):
    def __init__(self, dataset_instance, split_name, ids):
        self.dataset = dataset_instance
        self.split_name = split_name
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_item = self.ids[idx]
        try:
            text = self.dataset._get_text(id_item)
            label = self.dataset._get_label(id_item)

            # Get image filename with extension
            image_path = self.dataset._get_image(id_item)
            image_filename = Path(image_path).name

            # Handle different label formats
            if isinstance(label, np.ndarray):
                label_str = json.dumps(label.tolist())
            elif isinstance(label, (list, tuple)):
                label_str = json.dumps(list(label))
            else:
                label_str = int(label)

            result = {
                "id": str(id_item),
                "text": str(text),
                "label": label_str,
                "image": image_filename,
                "split": self.split_name,
            }

            # Add text_long if the dataset has the method
            if hasattr(self.dataset, "_get_text_long"):
                text_long = self.dataset._get_text_long(id_item)
                result["text_long"] = str(text_long)

            return result
        except Exception:
            # Return None for failed items, will be filtered out
            return None


def collate_fn(batch):
    """Custom collate function to handle None values"""
    # Filter out None values
    batch = [item for item in batch if item is not None]
    return batch


def preprocess_dataset(dataset_name: str, output_dir: str):
    """
    Preprocess a single dataset using DataLoader for faster processing.

    Args:
        dataset_name: Name of the dataset (food101, mmimdb, hatememes)
        output_dir: Directory to save the processed files
    """
    print(f"Processing {dataset_name} dataset...")

    # Create dataset instance
    dataset = Dataset_Factory.get_dataset(dataset_name)

    # Create output directory
    dataset_output_dir = Path(output_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Thread-safe data collection
    all_data = []
    split_data = {"train": [], "valid": [], "test": []}

    # For HateMemes, add unseen splits
    if dataset_name == "hatememes":
        split_data["valid_unseen"] = []
        split_data["test_unseen"] = []

    data_lock = Lock()

    # Define splits to process
    splits_to_process = ["train", "val", "test"]
    if dataset_name == "hatememes":
        splits_to_process.extend(["val_unseen", "test_unseen"])

    # Process each split with DataLoader
    for split in splits_to_process:
        # Map split names to output names
        if split == "val":
            split_name = "valid"
        elif split == "val_unseen":
            split_name = "valid_unseen"
        elif split == "test_unseen":
            split_name = "test_unseen"
        else:
            split_name = split

        ids = dataset.get_split(split)

        print(f"  Processing {split} split with {len(ids)} samples...")

        # Create preprocessing dataset
        prep_dataset = PreprocessingDataset(dataset, split_name, ids)

        # Create DataLoader with 16 workers
        dataloader = DataLoader(
            prep_dataset,
            batch_size=32,  # Process in batches for efficiency
            num_workers=16,
            collate_fn=collate_fn,
            shuffle=False,
        )

        # Process batches
        processed_count = 0
        for batch in dataloader:
            with data_lock:
                for item in batch:
                    data_entry = {
                        "id": item["id"],
                        "text": item["text"],
                        "label": item["label"],
                        "image": item["image"],
                    }
                    # Add text_long if present
                    if "text_long" in item:
                        data_entry["text_long"] = item["text_long"]

                    all_data.append(data_entry)
                    split_data[item["split"]].append(item["id"])
                    processed_count += 1

        print(f"    Successfully processed {processed_count} items")

    # Save data.jsonl using pandas for consistency with requirements
    data_jsonl_path = dataset_output_dir / "data.jsonl"
    df = pd.DataFrame(all_data)
    df.to_json(data_jsonl_path, orient="records", lines=True, force_ascii=False)

    print(f"  Saved {len(all_data)} entries to {data_jsonl_path}")

    # Save split CSV files
    for split_name, ids in split_data.items():
        if ids:
            split_df = pd.DataFrame({"id": ids})
            split_csv_path = dataset_output_dir / f"{split_name}.csv"
            split_df.to_csv(split_csv_path, index=False)
            print(f"  Saved {len(ids)} IDs to {split_csv_path}")


def main():
    """Main function to preprocess a single dataset."""
    parser = argparse.ArgumentParser(description="Preprocess a dataset for multimodal classification")
    parser.add_argument(
        "--dataset",
        type=str,
        default="food101",
        choices=["food101", "mmimdb", "hatememes"],
        help="Dataset to preprocess (default: food101)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for processed data (default: data)",
    )

    args = parser.parse_args()

    # Create main output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Starting dataset preprocessing with DataLoader (16 workers)...")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output_dir}")

    # Process the dataset
    try:
        preprocess_dataset(args.dataset, args.output_dir)
        print(f"✓ Successfully processed {args.dataset}")
    except Exception as e:
        print(f"✗ Failed to process {args.dataset}: {e}")
        import traceback

        traceback.print_exc()

    print("Dataset preprocessing completed!")


if __name__ == "__main__":
    main()
