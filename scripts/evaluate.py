
import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import local modules after path update
from configs import load_config
from data.dataloader import get_dataloaders
from utils.checkpoint import load_model_from_checkpoint
from utils.device import get_device, print_device_info
from utils.metrics import print_metrics
# Import validate from train script to reuse logic
from scripts.train import validate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    return parser.parse_args()

def main():
    args = parse_args()
    device = get_device(args.device)
    print_device_info(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    print(f"Loading model from {args.model_path}...")
    model = load_model_from_checkpoint(args.model_path, device=device.type)
    
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Warning: Config file {args.config} not found.")
        raise FileNotFoundError(f"Config file not found: {args.config}")

    data_cfg = config.get("data", {})
    batch_size = args.batch_size if args.batch_size is not None else int(data_cfg.get("batch_size", 64))
    num_workers = int(data_cfg.get("num_workers", 4))
    
    datasets_cfg = data_cfg.get("datasets", [])
    if not datasets_cfg:
        if data_cfg.get("dataset_type") == "rafdb":
             datasets_cfg.append({
                 "type": "rafdb",
                 "root": data_cfg.get("rafdb_root", "rafdb"),
                 "train_csv": data_cfg.get("rafdb_train_csv", "rafdb/train.csv"),
                 "val_csv": data_cfg.get("rafdb_val_csv", "rafdb/val.csv"),
                 "test_csv": data_cfg.get("rafdb_test_csv", "rafdb/test.csv"),
             })
        else:
             datasets_cfg.append({"type": "folder", "root": "data"})


    print(f"Loading data (split={args.split})...")
    train_loader, val_loader, test_loader = get_dataloaders(
        datasets_cfg=datasets_cfg,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True, 
        persistent_workers=True,
    )
    
    loader = None
    if args.split == "train":
        loader = train_loader
    elif args.split == "val":
        loader = val_loader
    else:
        loader = test_loader

    if loader is None:
        raise ValueError(f"{args.split} loader is None. Check if dataset has that split configured.")

    class_names = getattr(loader.dataset, "classes_list", None)
    if not class_names:
        class_names = config.get("classes", ["happiness", "surprise", "sadness", "anger", "disgust", "fear"])

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print("Starting evaluation...")
    metrics = validate(model, loader, criterion, device, class_names, epoch=0)
    
    print("\nEvaluation Results:")
    print_metrics(metrics)

if __name__ == "__main__":
    main()
