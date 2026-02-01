
from typing import Any, Dict
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    opt_cfg = config.get("training", {}).get("optimizer", {})
    name = str(opt_cfg.get("name", "adam")).lower()

    lr = float(opt_cfg.get("lr", 0.001))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0001))
    betas = opt_cfg.get("betas", [0.9, 0.999])
    if isinstance(betas, list):
        betas = tuple(betas)

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)


def get_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]):
    sched_cfg = config.get("training", {}).get("scheduler", {})
    name = str(sched_cfg.get("name", "step_lr")).lower()

    if name == "cosine":
        T_0 = int(sched_cfg.get("T_0", 10))
        T_mult = int(sched_cfg.get("T_mult", 2))
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)

    step_size = int(sched_cfg.get("step_size", 15))
    gamma = float(sched_cfg.get("gamma", 0.1))
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


def get_current_lr(optimizer: optim.Optimizer) -> float:    
    return float(optimizer.param_groups[0]["lr"])
