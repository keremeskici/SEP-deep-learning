
from typing import Dict, List, Optional
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


class MetricsCalculator:

    def __init__(self, class_names: Optional[List[str]] = None, num_classes: int = 6):
        if class_names is None:
            class_names = ["happiness", "surprise", "sadness", "anger", "disgust", "fear"]

        self.class_names = class_names
        self.num_classes = int(num_classes)
        self.reset()

    def reset(self) -> None:
        self.predictions: List[int] = []
        self.targets: List[int] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        preds = torch.argmax(logits, dim=1)
        self.predictions.extend(preds.detach().cpu().numpy().tolist())
        self.targets.extend(labels.detach().cpu().numpy().tolist())

    def compute(self) -> Dict[str, float]:
        if len(self.predictions) == 0:
            return {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}

        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        acc = accuracy_score(targets, preds)

        precision, recall, f1, support = precision_recall_fscore_support(
            targets,
            preds,
            labels=list(range(self.num_classes)),
            average=None,
            zero_division=0,
        )

        _, _, macro_f1, _ = precision_recall_fscore_support(targets, preds, average="macro", zero_division=0)
        _, _, weighted_f1, _ = precision_recall_fscore_support(targets, preds, average="weighted", zero_division=0)

        out: Dict[str, any] = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class_precision": {},
            "per_class_recall": {},
            "per_class_f1": {},
            "per_class_support": {},
        }

        for i, name in enumerate(self.class_names):
            out["per_class_precision"][name] = float(precision[i])
            out["per_class_recall"][name] = float(recall[i])
            out["per_class_f1"][name] = float(f1[i])
            out["per_class_support"][name] = int(support[i])

        return out

    def get_confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(self.targets, self.predictions, labels=list(range(self.num_classes)))

    def get_classification_report(self) -> str:
        return classification_report(
            self.targets,
            self.predictions,
            target_names=self.class_names,
            labels=list(range(self.num_classes)),
            zero_division=0,
        )


def compute_confusion_matrix(
    predictions: List[int],
    targets: List[int],
    num_classes: int = 6,
    normalize: bool = False,
) -> np.ndarray:
    cm = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm.astype(np.float64) / (row_sums + 1e-8)
    return cm


def print_metrics(metrics: Dict[str, float], epoch: Optional[int] = None) -> None:
    header = "EVALUATION METRICS"
    if epoch is not None:
        header += f" (Epoch {epoch})"

    print("\n" + "=" * 60)
    print(header)
    print("=" * 60)

    print(f"\nOverall:")
    print(f"  Accuracy:    {metrics.get('accuracy', 0.0):.4f}")
    print(f"  Macro F1:    {metrics.get('macro_f1', 0.0):.4f}")
    print(f"  Weighted F1: {metrics.get('weighted_f1', 0.0):.4f}")

    if "per_class_f1" in metrics:
        print("\nPer-class F1:")
        for name, val in metrics["per_class_f1"].items():
            sup = metrics.get("per_class_support", {}).get(name, 0)
            print(f"  {name:<12}: {val:.4f} (n={sup})")

    print("=" * 60 + "\n")


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * int(n)
        self.count += int(n)
        self.avg = self.sum / self.count if self.count > 0 else 0.0
