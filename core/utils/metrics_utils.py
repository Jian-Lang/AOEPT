from typing import Iterable, Optional

import torch
from torchmetrics import Metric
from torchmetrics.classification import AUROC, Accuracy, F1Score, MultilabelF1Score, Precision, Recall


class MultilabelF1Samples(Metric):
    """
    Multilabel F1 with 'samples' averaging:
    - For each sample i: F1_i = 2*TP_i / (2*TP_i + FP_i + FN_i)
    - Return mean_i F1_i
    Args:
        num_labels: optional, for sanity checks; shape must be (N, C)
        threshold: float in [0,1], used if preds are probabilities/logits
        from_logits: if True, apply sigmoid before thresholding
        ignore_index: int or iterable[int], labels (column indices) to ignore
        zero_division: 0.0 or 1.0. When a sample has TP=FP=FN=0 (both preds
                       and target are all zeros), set F1_i to this value.
    Notes:
        - Works on CPU/GPU; states are reduced with 'sum' so it's DDP-safe.
        - Expects 2D tensors: preds, target shaped (N, C).
    """

    full_state_update = False

    def __init__(
        self,
        num_labels: Optional[int] = None,
        threshold: float = 0.5,
        from_logits: bool = True,
        ignore_index: Optional[Iterable[int]] = None,
        zero_division: float = 0.0,
        eps: float = 1e-8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert zero_division in (0.0, 1.0), "zero_division must be 0.0 or 1.0"
        self.num_labels = num_labels
        self.threshold = float(threshold)
        self.from_logits = bool(from_logits)
        self.zero_division = float(zero_division)
        self.eps = float(eps)

        if ignore_index is None:
            self._ignore_mask = None
        else:
            _idx = (
                list(ignore_index)
                if isinstance(ignore_index, Iterable) and not isinstance(ignore_index, (int,))
                else [int(ignore_index)]
            )
            self._ignore_idx = torch.tensor(sorted(set(_idx)), dtype=torch.long)
            self._ignore_mask = None  # lazily built on first update

        # accumulate sum of per-sample F1, and number of samples
        self.add_state("sum_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n_samples", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def _binarize(self, preds: torch.Tensor) -> torch.Tensor:
        if preds.dtype.is_floating_point:
            if self.from_logits:
                preds = preds.sigmoid()
            preds = preds >= self.threshold
        else:
            # already binary/multi-hot (bool, int)
            preds = preds.to(torch.bool)
        return preds

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # shape checks
        if preds.ndim != 2 or target.ndim != 2:
            raise ValueError(f"Expected 2D tensors (N,C), got {preds.shape=} {target.shape=}")
        if preds.shape != target.shape:
            raise ValueError(f"Shape mismatch: {preds.shape} vs {target.shape}")
        if self.num_labels is not None and preds.shape[1] != self.num_labels:
            raise ValueError(f"num_labels={self.num_labels} but C={preds.shape[1]}")

        # build ignore mask lazily (device-aware)
        if getattr(self, "_ignore_idx", None) is not None and self._ignore_mask is None:
            C = preds.shape[1]
            mask = torch.ones(C, dtype=torch.bool, device=preds.device)
            mask[self._ignore_idx.to(preds.device)] = False
            self._ignore_mask = mask

        # binarize preds; ensure target is bool
        preds_b = self._binarize(preds)
        target_b = target.to(torch.bool)

        if self._ignore_mask is not None:
            preds_b = preds_b[:, self._ignore_mask]
            target_b = target_b[:, self._ignore_mask]

        # per-sample counts along class dim
        tp = (preds_b & target_b).sum(dim=1).to(torch.float32)
        fp = (preds_b & ~target_b).sum(dim=1).to(torch.float32)
        fn = (~preds_b & target_b).sum(dim=1).to(torch.float32)

        denom = 2 * tp + fp + fn
        # handle zero-division per sample
        f1_i = torch.where(
            denom > 0,
            (2 * tp) / (denom + self.eps),
            torch.full_like(denom, self.zero_division),
        )

        self.sum_f1 += f1_i.sum()
        self.n_samples += torch.tensor(preds.shape[0], device=self.sum_f1.device, dtype=torch.long)

    def compute(self) -> torch.Tensor:
        if self.n_samples == 0:
            return torch.tensor(0.0, device=self.sum_f1.device)
        return self.sum_f1 / self.n_samples


class BinaryClassificationMetric:
    def __init__(self, device):
        self.accuracy = Accuracy(task="multiclass", num_classes=2).to(device)
        self.f1_score = F1Score(task="multiclass", average="macro", num_classes=2).to(device)
        self.precision = Precision(task="multiclass", average="macro", num_classes=2).to(device)
        self.recall = Recall(task="multiclass", average="macro", num_classes=2).to(device)

    def _reset(self):
        self.accuracy.reset()
        self.f1_score.reset()
        self.precision.reset()
        self.recall.reset()

    def update(self, preds, labels):
        preds = torch.softmax(preds, dim=-1)
        self.accuracy.update(preds, labels)
        self.f1_score.update(preds, labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)

    def compute(self):
        acc = self.accuracy.compute()
        macro_f1 = self.f1_score.compute()
        macro_prec = self.precision.compute()
        macro_rec = self.recall.compute()
        self._reset()
        return {
            "acc": acc.item(),
            "macro_f1": macro_f1.item(),
            "macro_prec": macro_prec.item(),
            "macro_rec": macro_rec.item(),
        }


class TernaryClassificationMetric:
    def __init__(self, device):
        self.accuracy = Accuracy(task="multiclass", num_classes=3).to(device)
        self.f1_score = F1Score(task="multiclass", average="macro", num_classes=3).to(device)
        self.precision = Precision(task="multiclass", average="macro", num_classes=3).to(device)
        self.recall = Recall(task="multiclass", average="macro", num_classes=3).to(device)

    def _reset(self):
        self.accuracy.reset()
        self.f1_score.reset()
        self.precision.reset()
        self.recall.reset()

    def update(self, preds, labels):
        preds = torch.softmax(preds, dim=-1)
        self.accuracy.update(preds, labels)
        self.f1_score.update(preds, labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)

    def compute(self):
        acc = self.accuracy.compute()
        macro_f1 = self.f1_score.compute()
        macro_prec = self.precision.compute()
        macro_rec = self.recall.compute()
        self._reset()
        return {
            "acc": acc.item(),
            "macro_f1": macro_f1.item(),
            "macro_prec": macro_prec.item(),
            "macro_rec": macro_rec.item(),
        }


class HateMemesMetric:
    def __init__(self, device):
        self.auroc = AUROC(task="binary").to(device)

    def _reset(self):
        self.auroc.reset()

    def update(self, logits, labels):
        probs = torch.softmax(logits, dim=-1)
        # For binary classification, AUROC expects positive class probabilities only
        probs = probs[:, 1] if probs.ndim > 1 else probs
        self.auroc.update(probs, labels)

    def compute(self):
        auroc = self.auroc.compute()
        fmt_text = f"auroc: {auroc.item():.4f}"
        self._reset()
        return {"auroc": auroc.item(), "fmt_text": fmt_text}


class MMIMDbMetric:
    def __init__(self, device):
        self.f1_micro = F1Score(task="multilabel", average="micro", num_labels=23).to(device)
        self.f1_macro = F1Score(task="multilabel", average="macro", num_labels=23).to(device)
        # self.f1_sample = F1Score(
        # task="multilabel", average="none", multidim_average="samplewise", num_labels=23
        # ).to(device)
        self.f1_sample = MultilabelF1Samples(num_labels=23, from_logits=True).to(device)

    def _reset(self):
        self.f1_micro.reset()
        self.f1_macro.reset()
        self.f1_sample.reset()

    def update(self, logits, labels):
        probs = torch.sigmoid(logits)
        self.f1_micro.update(probs, labels)
        self.f1_macro.update(probs, labels)
        self.f1_sample.update(logits, labels)

    def compute(self):
        f1_macro = self.f1_macro.compute()
        f1_sample = self.f1_sample.compute()
        f1_micro = self.f1_micro.compute()
        f1_ms = f1_macro + f1_sample
        fmt_text = f"f1_macro: {f1_macro.item():.4f}, f1_sample: {f1_sample.item():.4f}, f1_micro: {f1_micro.item():.4f}, f1_ms: {f1_ms.item():.4f}"
        self._reset()
        return {
            "f1_macro": f1_macro.item(),
            "f1_sample": f1_sample.item(),
            "f1_micro": f1_micro.item(),
            "f1_ms": f1_ms.item(),
            "fmt_text": fmt_text,
        }


class Food101Metric:
    def __init__(self, device):
        self.accuracy = Accuracy(task="multiclass", num_classes=101).to(device)

    def _reset(self):
        self.accuracy.reset()

    def update(self, logits, labels):
        probs = torch.softmax(logits, dim=-1)
        self.accuracy.update(probs, labels)

    def compute(self):
        accuracy = self.accuracy.compute()
        fmt_text = f"accuracy: {accuracy.item():.4f}"
        self._reset()
        return {
            "accuracy": accuracy.item(),
            "fmt_text": fmt_text,
        }


class MetricFactory:
    @staticmethod
    def get_metric(dataset_name: str, device: str):
        match dataset_name.lower():
            case "food101":
                return Food101Metric(device)
            case "mmimdb":
                return MMIMDbMetric(device)
            case "hatememes":
                return HateMemesMetric(device)
            case _:
                raise ValueError(f"Metric {dataset_name} not supported")
