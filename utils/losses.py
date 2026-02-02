from typing import Dict, Any, Optional
import torch
import torch.nn as nn


def get_loss_function(
    config: Dict[str, Any],
    class_weights: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> nn.Module:
    training_cfg = config.get("training", {})
    use_weights = bool(training_cfg.get("use_class_weights", True))

    if use_weights and class_weights is not None:
        class_weights = class_weights.to(device)
        return nn.CrossEntropyLoss(weight=class_weights)

    return nn.CrossEntropyLoss()