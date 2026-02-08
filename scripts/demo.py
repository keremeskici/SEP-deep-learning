#!/usr/bin/env python3
import sys
import os
import argparse
import csv
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image

# add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# imports from project modules (must be after sys.path insert)
from data.transforms import get_test_transforms
from utils.device import get_device

# emotion classes matching the model output
CLASS_NAMES = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

def process_image(image_path, model, transform, device):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            
        # Return dictionary of probabilities
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
    
    # setup transform
    transform = get_test_transforms()
    
    # find images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = [
        f for f in folder_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not images:
        print(f"No images found in {folder_path}")
        return
        
    print(f"Found {len(images)} images")
    
    # prepare output csv path
    # Requirement: "should save the csv file under @[scripts] folder directly"
    output_csv = Path(__file__).parent / 'predictions.csv'
    
    print(f"Processing images and saving to {output_csv}...")
    
    with open(output_csv, 'w', newline='') as csvfile:
        # Header as shown in the requirement image: filepath, happiness, surprise, sadness, anger, disgust, fear
        fieldnames = ['filepath'] + CLASS_NAMES
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for img_path in tqdm(sorted(images)):
            probs = process_image(img_path, model, transform, device)
            
            if probs:
                # Requirement: Use truncated/relative path or full path? 
                # The image shows "/folder/img001.png". I will stick to full path or just filename.
                # Let's use the full path as requested by previous logic, allowing user to see exactly where it is.
                row = {'filepath': str(img_path)}
                row.update(probs)
                writer.writerow(row)
                
    print("Done.")

if __name__ == '__main__':
    main()
