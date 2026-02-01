
from typing import List, Optional, Union, Dict
import numpy as np
import torch


def compute_class_weights(labels: Union[List[int], np.ndarray], num_classes: int = 6) -> torch.Tensor:   
    if isinstance(labels, list):
        labels = np.array(labels)

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = float(len(labels))

    eps = 1e-8
    weights = total / (num_classes * (counts + eps))

   
    weights[counts == 0] = 0.0

    return torch.tensor(weights, dtype=torch.float32)


def get_class_distribution(
    labels: Union[List[int], np.ndarray],
    class_names: Optional[List[str]] = None,
    num_classes: int = 6,
) -> Dict[str, Dict[str, float]]:
    
    if class_names is None:
        class_names = ["happiness", "surprise", "sadness", "anger", "disgust", "fear"]

    if isinstance(labels, list):
        labels = np.array(labels)

    counts = np.bincount(labels, minlength=num_classes)
    total = max(1, len(labels))

    weights = compute_class_weights(labels, num_classes=num_classes).numpy()

    out = {}
    for i, name in enumerate(class_names):
        c = int(counts[i])
        out[name] = {
            "count": c,
            "frequency": float(c / total),
            "weight": float(weights[i]),
        }
    return out


def print_class_distribution(
    labels: Union[List[int], np.ndarray],
    class_names: Optional[List[str]] = None,
    num_classes: int = 6,
) -> None:    
    stats = get_class_distribution(labels, class_names=class_names, num_classes=num_classes)

    print("\n" + "=" * 55)
    print("CLASS DISTRIBUTION & WEIGHTS")
    print("=" * 55)
    print(f"{'Class':<12} {'Count':>8} {'Frequency':>12} {'Weight':>10}")
    print("-" * 55)

    total_count = 0
    for name, info in stats.items():
        total_count += info["count"]
        freq_str = f"{info['frequency'] * 100:.2f}%"
        print(f"{name:<12} {info['count']:>8} {freq_str:>12} {info['weight']:>10.4f}")

    print("-" * 55)
    print(f"{'TOTAL':<12} {total_count:>8}")
    print("=" * 55 + "\n")