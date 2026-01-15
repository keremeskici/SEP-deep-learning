from __future__ import annotations

from typing import Any, Dict, Optional
import torch.nn as nn

from .resnet import ModifiedResNet18


def build_model(
    config: Optional[Dict[str, Any]] = None,
    *,
    architecture: Optional[str] = None,
    num_classes: Optional[int] = None,
    in_channels: Optional[int] = None,
    init_weights: bool = False,
) -> nn.Module:
    if config is not None:
        model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
        architecture = architecture or model_cfg.get("architecture", "modified_resnet18")
        num_classes = num_classes if num_classes is not None else model_cfg.get("num_classes", 6)
        in_channels = in_channels if in_channels is not None else model_cfg.get("in_channels", 3)
    else:
        architecture = architecture or "modified_resnet18"
        num_classes = 6 if num_classes is None else num_classes
        in_channels = 3 if in_channels is None else in_channels

    arch = str(architecture).lower()

    if arch == "modified_resnet18":
        model: nn.Module = ModifiedResNet18(num_classes=int(num_classes), in_channels=int(in_channels))
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    if init_weights:
        initialize_weights_(model)

    return model


def initialize_weights_(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


__all__ = ["ModifiedResNet18", "build_model", "initialize_weights_"]
