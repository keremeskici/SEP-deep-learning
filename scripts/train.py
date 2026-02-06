
import os
import sys
import argparse
import wandb
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import load_config, get_default_config
from data.dataloader import get_dataloaders
from models import build_model
from utils.class_weights import compute_class_weights, print_class_distribution
from utils.losses import get_loss_function
from utils.optim import get_optimizer, get_scheduler, get_current_lr
from utils.metrics import MetricsCalculator, AverageMeter, print_metrics
from utils.early_stopping import EarlyStopping
from utils.checkpoint import save_checkpoint, load_checkpoint, get_checkpoint_path
from utils.device import get_device, print_device_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train FER model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--config", type=str, default="configs/default_config.yaml", help="YAML config path")
    p.add_argument("--data_root", type=str, default="data", help="Dataset root (contains training/validation/test)")
    p.add_argument("--output_dir", type=str, default="./outputs/models", help="Where to save run folders")

    p.add_argument("--epochs", type=int, default=None, help="Override epochs")
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def extract_labels_from_dataset(ds) -> list:
    if hasattr(ds, "samples"):
        return [int(lbl) for _, lbl in ds.samples]
    labels = []
    for i in range(len(ds)):
        _, y = ds[i]
        labels.append(int(y))
    return labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 10
) -> Tuple[float, float]:
    model.train()

    loss_meter = AverageMeter()
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)

    for step, (x, y) in enumerate(pbar, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loss_meter.update(float(loss.item()), n=x.size(0))

        preds = torch.argmax(logits, dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.numel())

        if step % log_interval == 0:
            acc = correct / max(total, 1)
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc:.4f}")

    return loss_meter.avg, (correct / max(total, 1))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list,
    epoch: int
) -> Dict[str, Any]:
    model.eval()

    loss_meter = AverageMeter()
    metrics_calc = MetricsCalculator(class_names=class_names, num_classes=len(class_names))

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]", leave=False)

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        loss_meter.update(float(loss.item()), n=x.size(0))
        metrics_calc.update(logits, y)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    metrics = metrics_calc.compute()
    metrics["loss"] = loss_meter.avg
    return metrics


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        config = get_default_config()

    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config.setdefault("data", {})["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        config.setdefault("training", {}).setdefault("optimizer", {})["lr"] = float(args.lr)

    config["seed"] = int(args.seed)
    config["device"] = args.device

    device = get_device(args.device)
    print_device_info(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(config, f, sort_keys=False)

    tb_writer = None
    if config.get("logging", {}).get("use_tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=str(log_dir))
        except Exception:
            tb_writer = None

    data_cfg = config.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False))

    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        seed=int(args.seed),
    )

    class_names = getattr(train_loader.dataset, "classes_list", None)
    if not class_names:
        class_names = config.get("classes", ["happiness", "surprise", "sadness", "anger", "disgust", "fear"])

    train_labels = extract_labels_from_dataset(train_loader.dataset)
    print_class_distribution(train_labels, class_names, num_classes=len(class_names))

    class_weights = compute_class_weights(train_labels, num_classes=len(class_names))

    model = build_model(config)
    model = model.to(device)

    criterion = get_loss_function(config, class_weights=class_weights, device=device.type)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    es_cfg = config.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", True))
    early_stopping = EarlyStopping(
        patience=int(es_cfg.get("patience", 10)),
        min_delta=float(es_cfg.get("min_delta", 0.001)),
        mode="max",
        verbose=True,
    )

    start_epoch = 0
    best_f1 = -1.0

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=scheduler, device=device.type)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_f1 = float(ckpt.get("metrics", {}).get("val_f1", best_f1))

    epochs = int(config.get("training", {}).get("epochs", 10))
    log_interval = int(config.get("logging", {}).get("log_interval", 10))
    save_interval = int(config.get("checkpoint", {}).get("save_interval", 5))

# Implementation of WandB for experiment tracking. Make sure to install wandb and login before running.
    wandb.init(
    project="fer-training", # The name can be changed to your liking, this will be the project name in your WandB dashboard
    name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config=config
)

    #Train loop:
    for epoch in range(start_epoch, epochs):
        epoch_id = epoch + 1
        lr_now = get_current_lr(optimizer)
        print(f"\nEpoch {epoch_id}/{epochs} | lr={lr_now:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch_id, log_interval=log_interval
        )

        val_metrics = validate(
            model, val_loader, criterion, device, class_names, epoch_id
        )
        
# WandB requires logging once per epoch, so we log the training and validation metrics here
# you will have to login preferably via lmu and after logging in, you can run the training script and it will automatically log the metrics to your WandB dashboard under the specified project name.
        wandb.log({
            "epoch": epoch_id,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
        })

        if hasattr(scheduler, "__class__") and scheduler.__class__.__name__ == "CosineAnnealingWarmRestarts":
            scheduler.step(epoch_id)
        else:
            scheduler.step()

        val_f1 = float(val_metrics.get("macro_f1", 0.0))

        print(f"  train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
        print(f"  val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_f1:.4f}")

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", train_loss, epoch_id)
            tb_writer.add_scalar("train/acc", train_acc, epoch_id)
            tb_writer.add_scalar("val/loss", float(val_metrics["loss"]), epoch_id)
            tb_writer.add_scalar("val/acc", float(val_metrics["accuracy"]), epoch_id)
            tb_writer.add_scalar("val/macro_f1", val_f1, epoch_id)
            tb_writer.add_scalar("lr", lr_now, epoch_id)

        is_best = False
        if es_enabled:
            is_best = early_stopping(val_f1, epoch=epoch_id)
        else:
            is_best = val_f1 > best_f1

        metrics_to_save = {
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["accuracy"]),
            "val_f1": float(val_f1),
        }

        if epoch_id % save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler,
                epoch=epoch,
                metrics=metrics_to_save,
                config=config,
                filepath=get_checkpoint_path(str(ckpt_dir), epoch=epoch_id),
                class_weights=class_weights,
                is_best=False
            )

        if is_best:
            best_f1 = max(best_f1, val_f1)
            save_checkpoint(
                model, optimizer, scheduler,
                epoch=epoch,
                metrics=metrics_to_save,
                config=config,
                filepath=get_checkpoint_path(str(ckpt_dir), best=True),
                class_weights=class_weights,
                is_best=True
            )

        if es_enabled and early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    best_model_path = get_checkpoint_path(str(ckpt_dir), best=True)
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model, device=device.type)

    print("\nFinal evaluation on test set:")
    test_metrics = validate(model, test_loader, criterion, device, class_names, epoch=0)
    print_metrics(test_metrics)

    if tb_writer is not None:
        tb_writer.close()

    print(f"\nRun saved to: {run_dir}")
    print(f"Best model: {best_model_path}")
    print(f"Best val macro_f1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
