import sys
import os
import argparse
import csv
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import get_test_transforms
from utils.device import get_device

CLASS_NAMES = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

def process_image(image_path, model, transform, device):
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            
        return {name: probs[i].item() for i, name in enumerate(CLASS_NAMES)}
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_path',type=str)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    
    folder_path = Path(args.folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: Folder '{folder_path}' not found")
        return

    device = get_device(args.device)
    print(f"Using device: {device}")
    
    print(f"Loading model from {args.model_path}...")
    try:
        from utils.checkpoint import load_model_from_checkpoint
        model = load_model_from_checkpoint(args.model_path, device=device.type)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = get_test_transforms(mean, std)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [
        f for f in folder_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not images:
        print(f"No images found in {folder_path}")
        return
        
    print(f"Found {len(images)} images")
    
    output_dir = PROJECT_ROOT / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / 'predictions.csv'
    
    print(f"Processing images and saving to {output_csv}...")
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filepath'] + CLASS_NAMES
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for img_path in tqdm(sorted(images)):
            probs = process_image(img_path, model, transform, device)
            
            if probs:
                row = {'filepath': str(img_path)}
                row.update(probs)
                writer.writerow(row)
                
    print("Done.")

if __name__ == '__main__':
    main()
