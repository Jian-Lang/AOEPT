from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Base_Dataset(Dataset):
    def __init__(self, **kargs):
        super().__init__()
        self.missing_type = kargs.get("missing_type", None)
        self.missing_rate = kargs.get("missing_rate", None)

        # Check if this is training split and missing_rate_train is specified
        missing_rate_train = kargs.get("missing_rate_train", None)
        if missing_rate_train is not None and kargs.get("split") == "train":
            self.missing_rate = missing_rate_train

        self.stage = kargs.get("stage", None)
        self.model = kargs.get("model", None)
        self.statis = kargs.get("statis", None)
        self.split = kargs.get("split", None)
        # Keep dataset-specific data_path set by subclasses.
        existing_data_path = getattr(self, "data_path", None)
        self.data_path = Path(existing_data_path) if existing_data_path is not None else Path("data")
        self._initialized_missing_tbl = False
        self._initialized_data = False

        if self.split:
            self._load_split_data()

    def get_data(self):
        self.data = pd.read_json(self.data_path / "data.jsonl", lines=True)

        # Skip loading missing_tbl if using dynamic missing
        if self.missing_type != "dynamic":
            try:
                self.missing_tbl = pd.read_json(
                    self.data_path / "missing_tbl" / f"{self.missing_type}-{self.missing_rate}.jsonl",
                    lines=True,
                )
            except ValueError:
                raise ValueError(
                    f"Missing table {self.data_path}/missing_tbl/{self.missing_type}-{self.missing_rate}.jsonl not found."
                )

            # Flip missing table for feature collection (ViLT/CLIP only)
            if self.statis == "collect_features" and self.model in ["ViLT", "CLIP"]:
                self.missing_tbl["text_missing"] = ~self.missing_tbl["text_missing"]
                self.missing_tbl["image_missing"] = ~self.missing_tbl["image_missing"]

            # merge data and missing_tbl on 'id'
            self.data = pd.merge(self.data, self.missing_tbl, on="id")

        return self.data

    def get_split(self, split: str):
        raise NotImplementedError

    def _get_ids(self):
        # fetch from get_split with train/val/test
        return self.get_split("train") + self.get_split("val") + self.get_split("test")

    def _load_split_data(self):
        all_data = pd.read_json(self.data_path / "data.jsonl", lines=True)
        split_ids = self.get_split(self.split)
        self.data = all_data[all_data["id"].isin(split_ids)].reset_index(drop=True)
        self._initialized_data = True

    def __len__(self):
        if not self._initialized_data:
            self._load_split_data()
        return len(self.data)

    def __getitem__(self, idx):
        if not self._initialized_data:
            self._load_split_data()

        row = self.data.iloc[idx]
        image_path = self.data_path / "image" / row["image"]
        image = Image.open(image_path).convert("RGB")

        return {
            "vids": row["id"],
            "text": row["text"],
            "image": image,
            "label": row["label"],
        }


class Food101_Dataset(Base_Dataset):
    def __init__(self, **kargs):
        self.data_path = Path("data/food101")
        super().__init__(**kargs)

    def get_split(self, split: str):
        """Get list of IDs for the specified split"""
        match split:
            case "train":
                split_file = self.data_path / "train.csv"
            case "val":
                split_file = self.data_path / "valid.csv"
            case "test":
                split_file = self.data_path / "test.csv"
            case _:
                raise ValueError(f"Invalid split: {split}")

        split_data = pd.read_csv(split_file)
        return split_data["id"].tolist()


class MMIMDB_Dataset(Base_Dataset):
    def __init__(self, **kargs):
        self.data_path = Path("data/mmimdb")
        super().__init__(**kargs)

    def get_split(self, split: str):
        """Get list of IDs for the specified split"""
        match split:
            case "train":
                split_file = self.data_path / "train.csv"
            case "val":
                split_file = self.data_path / "valid.csv"
            case "test":
                split_file = self.data_path / "test.csv"
            case _:
                raise ValueError(f"Invalid split: {split}")

        split_data = pd.read_csv(split_file)
        return split_data["id"].tolist()


class HateMemes_Dataset(Base_Dataset):
    def __init__(self, **kargs):
        self.data_path = Path("data/hatememes")
        self.type = kargs.get("type", "seen")
        super().__init__(**kargs)

    def get_split(self, split: str):
        """Get list of IDs for the specified split"""
        match split:
            case "train":
                split_file = self.data_path / "train.csv"
            case "val":
                if self.type == "unseen":
                    split_file = self.data_path / "valid_unseen.csv"
                else:
                    split_file = self.data_path / "valid.csv"
            case "test":
                if self.type == "unseen":
                    split_file = self.data_path / "test_unseen.csv"
                else:
                    split_file = self.data_path / "test.csv"
            case _:
                raise ValueError(f"Invalid split: {split}")

        split_data = pd.read_csv(split_file)
        return split_data["id"].tolist()

    def _get_ids(self):
        """Get all IDs from all splits (including unseen)"""
        all_ids = []
        for split_file in ["train.csv", "valid.csv", "test.csv"]:
            split_path = self.data_path / split_file
            if split_path.exists():
                split_data = pd.read_csv(split_path)
                all_ids.extend(split_data["id"].tolist())
        return all_ids


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
