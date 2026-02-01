
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim


def get_checkpoint_path(checkpoint_dir: str, epoch: Optional[int] = None, best: bool = False) -> str:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if best:
        return os.path.join(checkpoint_dir, "best_model.pth")
    if epoch is not None:
        return os.path.join(checkpoint_dir, f"checkpoint_epoch_{int(epoch):03d}.pth")
    return os.path.join(checkpoint_dir, "last_model.pth") 



def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    filepath: str,
    class_weights: Optional[torch.Tensor] = None,
    is_best: bool = False,
) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics,
        "config": config,
    }

    if class_weights is not None:
        ckpt["class_weights"] = class_weights.detach().cpu()

    torch.save(ckpt, filepath)

    if is_best:
        best_path = Path(filepath).parent / "best_model.pth"
        torch.save(ckpt, str(best_path))


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    ckpt = torch.load(filepath, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    out = {
        "epoch": ckpt.get("epoch", 0),
        "metrics": ckpt.get("metrics", {}),
        "config": ckpt.get("config", {}),
    }

    if "class_weights" in ckpt:
        out["class_weights"] = ckpt["class_weights"]

    return out


def load_model_for_inference(filepath: str, model: nn.Module, device: str = "cpu") -> nn.Module:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model checkpoint not found: {filepath}")

    ckpt = torch.load(filepath, map_location=device)

    # Falls voll gespeicherter Checkpoint
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # Falls nur state_dict gespeichert wurde
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model
