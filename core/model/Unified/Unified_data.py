import json
import random
import warnings

import torch
from PIL import Image
from transformers import AutoProcessor

# Ignore decompression bomb warning for large images
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)

from core.model.Base.base_data import Base_Dataset


class Unified_Dataset(Base_Dataset):
    def __init__(self, split: str, **kargs):
        super().__init__(split=split, **kargs)

        self.arch = kargs.get("arch")
        self.ids = self.get_split(split)
        self.data = self.get_data()
        self.data = self.data[self.data["id"].isin(self.ids)].reset_index(drop=True)

        # Check if this is MMIMDB dataset and text_long column exists
        self.is_mmimdb = "mmimdb" in str(self.data_path)
        self.has_text_long = "text_long" in self.data.columns

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Same logic for both CLIP and ViLT
        id = self.ids[idx]
        item = self.data[self.data["id"] == id]

        # Use text_long for MMIMDB if available, otherwise use text
        if self.is_mmimdb and self.has_text_long:
            text = item["text_long"].values[0]
        else:
            text = item["text"].values[0]

        image = item["image"].values[0]
        image = Image.open(self.data_path / "image" / image).convert("RGB")
        label = item["label"].values[0]
        if isinstance(label, str):
            label = json.loads(label)
            label = [int(x) for x in label]

        # If missing_type='dynamic', return complete data; missing will be applied in collator
        if self.missing_type == "dynamic":
            return {
                "vids": id,
                "text": text,
                "image": image,
                "label": label,
                "missing_mask": [False, False],  # Placeholder
            }

        # Original logic: use pre-defined missing table
        text_missing, image_missing = item[["text_missing", "image_missing"]].values[0].tolist()

        # Make text empty if text_missing; make image empty if image_missing
        text = "" if text_missing else text
        if image_missing:
            if self.arch == "ViLT":
                image = Image.new("RGB", image.size, (1, 1, 1))
            else:
                image = Image.new("RGB", image.size, (0, 0, 0))

        missing_mask = [text_missing, image_missing]
        return {
            "vids": id,
            "text": text,
            "image": image,
            "label": label,
            "missing_mask": missing_mask,
        }


class Unified_Collator:
    def __init__(
        self,
        seq_len: int,
        arch: str = "ViLT",
        model_id: str = None,
        missing_type: str = None,
        missing_rate: float = 0.0,
        missing_seed: int = None,
        dataset_name: str = None,
        **kargs,
    ):
        # Remove arch from kargs to avoid duplicate keyword argument error
        kargs.pop("arch", None)
        self.arch = arch
        self.seq_len = seq_len
        self.missing_type = missing_type
        self.missing_rate = missing_rate

        # Store missing_seed for deterministic dynamic missing
        # Use 42 as default seed if not provided for reproducibility
        self.missing_seed = missing_seed if missing_seed is not None else 42

        # Store dataset_name for dataset-specific logic
        self.dataset_name = dataset_name

        # Set default model_id based on arch if not provided
        if model_id is None:
            model_id = "dandelin/vilt-b32-mlm" if self.arch == "ViLT" else "openai/clip-vit-base-patch16"

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=(self.arch == "ViLT"))

    def _apply_dynamic_missing(self, batch, batch_size, sample_ids):
        """
        Apply dynamic missing pattern to the batch in a deterministic way.
        Uses sample IDs to generate reproducible missing patterns.
        All samples in the batch share the same missing_type (text/image/both),
        but only (1-missing_rate) fraction of samples actually have missing modalities.

        Args:
            batch: List of batch items
            batch_size: Number of items in batch
            sample_ids: List of sample IDs for deterministic random generation

        Returns:
            tuple: (processed_texts, processed_images, missing_masks)
        """
        # Step 1: Deterministically choose missing type for this batch based on first sample ID
        batch_rng = random.Random(self.missing_seed + hash(str(sample_ids[0])))
        missing_type = batch_rng.choice(["text", "image", "both"])

        # Step 2: Determine which samples should have missing modalities
        # missing_rate=0.3 means (1-0.3)=70% samples have missing
        num_missing = int(batch_size * (1 - self.missing_rate))
        missing_indices = set(batch_rng.sample(range(batch_size), num_missing))

        texts = []
        images = []
        missing_masks = []

        # Step 3: Apply missing pattern
        if missing_type == "text":
            for i, item in enumerate(batch):
                if i in missing_indices:
                    # Apply text missing
                    texts.append("")
                    images.append(item["image"])
                    missing_masks.append([True, False])
                else:
                    # Keep complete
                    texts.append(item["text"])
                    images.append(item["image"])
                    missing_masks.append([False, False])

        elif missing_type == "image":
            for i, item in enumerate(batch):
                if i in missing_indices:
                    # Apply image missing
                    texts.append(item["text"])
                    if self.arch == "ViLT":
                        empty_image = Image.new("RGB", item["image"].size, (1, 1, 1))
                    else:
                        empty_image = Image.new("RGB", item["image"].size, (0, 0, 0))
                    images.append(empty_image)
                    missing_masks.append([False, True])
                else:
                    # Keep complete
                    texts.append(item["text"])
                    images.append(item["image"])
                    missing_masks.append([False, False])

        else:  # both
            # Split missing samples: half text_missing, half image_missing
            missing_list = list(missing_indices)
            batch_rng.shuffle(missing_list)
            mid = len(missing_list) // 2
            text_missing_indices = set(missing_list[:mid])
            image_missing_indices = set(missing_list[mid:])

            for i, item in enumerate(batch):
                if i in text_missing_indices:
                    # Apply text missing
                    texts.append("")
                    images.append(item["image"])
                    missing_masks.append([True, False])
                elif i in image_missing_indices:
                    # Apply image missing
                    texts.append(item["text"])
                    if self.arch == "ViLT":
                        empty_image = Image.new("RGB", item["image"].size, (1, 1, 1))
                    else:
                        empty_image = Image.new("RGB", item["image"].size, (0, 0, 0))
                    images.append(empty_image)
                    missing_masks.append([False, True])
                else:
                    # Keep complete
                    texts.append(item["text"])
                    images.append(item["image"])
                    missing_masks.append([False, False])

        return texts, images, missing_masks

    def _split_complete_samples_for_token_collection(self, batch):
        """
        Split complete samples into text-only and vision-only versions.

        Complete samples (missing_mask=[False, False]) are duplicated:
        - Text-only version: text preserved, image replaced with black, missing_mask=[False, True]
        - Vision-only version: text replaced with empty string, image preserved, missing_mask=[True, False]

        Already-incomplete samples pass through unchanged.

        Args:
            batch: List of dicts with keys {vids, text, image, label, missing_mask}

        Returns:
            Expanded batch with split samples
        """
        expanded_batch = []

        for item in batch:
            text_missing, vision_missing = item["missing_mask"]
            is_complete = (not text_missing) and (not vision_missing)

            if is_complete:
                # Create text-only version (vision missing)
                text_only_sample = {
                    "vids": f"{item['vids']}_text_only",
                    "text": item["text"],
                    "image": Image.new("RGB", item["image"].size, (0, 0, 0)),  # Black image
                    "label": item["label"],
                    "missing_mask": [False, True],  # Vision missing
                }

                # Create vision-only version (text missing)
                vision_only_sample = {
                    "vids": f"{item['vids']}_vision_only",
                    "text": "",  # Empty text
                    "image": item["image"],
                    "label": item["label"],
                    "missing_mask": [True, False],  # Text missing
                }

                expanded_batch.extend([text_only_sample, vision_only_sample])
            else:
                # Keep incomplete samples unchanged
                expanded_batch.append(item)

        return expanded_batch

    def __call__(self, batch):
        # Split complete samples for ViLT token collection
        if hasattr(self, "collect_token") and self.collect_token and self.arch == "ViLT":
            batch = self._split_complete_samples_for_token_collection(batch)

        batch_size = len(batch)
        ids = [item["vids"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.int64)

        # Apply dynamic missing or use pre-defined missing_mask
        if self.missing_type == "dynamic":
            texts, images, missing_masks = self._apply_dynamic_missing(batch, batch_size, ids)
            missing_masks = torch.tensor(missing_masks, dtype=torch.bool)
        else:
            # Use original logic with pre-defined missing from dataset
            texts = [item["text"] for item in batch]
            images = [item["image"] for item in batch]
            missing_masks = torch.tensor(
                [[bool(x) for x in item["missing_mask"]] for item in batch], dtype=torch.bool
            )

        # Use processor for both text and images
        if self.arch == "ViLT":
            encoding = self.processor(
                images=images,
                text=texts,
                padding="max_length",
                truncation=True,
                max_length=self.seq_len,
                return_special_tokens_mask=False,
                return_tensors="pt",
            )
            # Set pixel values to 1 for missing images (Food101 and MMIMDB)
            if self.dataset_name in ["Food101", "MMIMDB"] and missing_masks[:, 1].any():
                encoding["pixel_values"][missing_masks[:, 1]] = 1.0
        else:  # CLIP
            encoding = self.processor(
                images=images,
                text=texts,
                padding="max_length",
                truncation=True,
                max_length=self.seq_len,
                return_tensors="pt",
            )

        return {
            "ids": ids,
            **encoding,
            "labels": labels,
            "missing_masks": missing_masks,
        }
