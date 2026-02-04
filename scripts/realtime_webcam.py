#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from data.transforms import get_inference_transforms
from utils.checkpoint import load_model_for_inference
from utils.gradcam import GradCAM
from utils.device import get_device

# emotion classes
CLASS_NAMES = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

# colors for each emotion (BGR)
EMOTION_COLORS = {
    'happiness': (0, 215, 255),
    'surprise': (0, 165, 255),
    'sadness': (230, 150, 50),
    'anger': (60, 60, 220),
    'disgust': (80, 180, 80),
    'fear': (180, 80, 180),
}

class FaceDetector:
    # simple face detector
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.min_size = (60, 60)
    
    def detect(self, frame):
        # find face in frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=self.min_size
        )
        
        if len(faces) > 0:
            # return largest face
            largest = max(faces, key=lambda f: f[2] * f[3])
            return tuple(largest)
        return None

def preprocess_frame(frame, transform, roi=None):
    # convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # crop face if detected
    if roi is not None:
        x, y, w, h = roi
        # add padding
        pad = int(0.1 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        rgb = rgb[y1:y2, x1:x2]
    
    # convert to PIL and transform
    pil_image = Image.fromarray(rgb)
    tensor = transform(pil_image)
    
    return tensor.unsqueeze(0)

def draw_results(frame, emotion, confidence, probabilities, heatmap, roi, alpha=0.4):
    # draw gradcam heatmap and results with nice UI
    output = frame.copy()
    h, w = frame.shape[:2]
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))
    
    # draw heatmap overlay
    if heatmap is not None and roi is not None:
        x, y, fw, fh = roi
        # resize heatmap to face size
        heatmap_resized = cv2.resize(heatmap, (fw, fh))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # blend with face region
        face_region = output[y:y+fh, x:x+fw]
        blended = cv2.addWeighted(
            face_region, 1 - alpha,
            heatmap_colored, alpha,
            0
        )
        output[y:y+fh, x:x+fw] = blended
    
    # draw face box with corners
    if roi is not None:
        x, y, fw, fh = roi
        thickness = 3
        cv2.rectangle(output, (x, y), (x + fw, y + fh), color, thickness)
        
        # corner accents
        corner_len = 20
        cv2.line(output, (x, y), (x + corner_len, y), color, thickness + 2)
        cv2.line(output, (x, y), (x, y + corner_len), color, thickness + 2)
        cv2.line(output, (x + fw, y), (x + fw - corner_len, y), color, thickness + 2)
        cv2.line(output, (x + fw, y), (x + fw, y + corner_len), color, thickness + 2)
        cv2.line(output, (x, y + fh), (x + corner_len, y + fh), color, thickness + 2)
        cv2.line(output, (x, y + fh), (x, y + fh - corner_len), color, thickness + 2)
        cv2.line(output, (x + fw, y + fh), (x + fw - corner_len, y + fh), color, thickness + 2)
        cv2.line(output, (x + fw, y + fh), (x + fw, y + fh - corner_len), color, thickness + 2)
    
    # draw emotion panel (top-left)
    panel_h = 80
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (280, panel_h), (30, 30, 30), -1)
    output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)
    
    # emotion name
    cv2.putText(
        output, emotion.upper(), (15, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
    )
    
    # confidence
    cv2.putText(
        output, f"{confidence:.0%}", (200, 45),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2
    )
    
    # draw probability bars (right side)
    bar_width = 150
    bar_height = 18
    bar_x = w - bar_width - 20
    bar_y_start = 20
    bar_spacing = 28
    
    # background panel
    panel_h = len(CLASS_NAMES) * bar_spacing + 10
    overlay = output.copy()
    cv2.rectangle(
        overlay,
        (bar_x - 80, bar_y_start - 10),
        (w - 10, bar_y_start + panel_h),
        (30, 30, 30), -1
    )
    output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)
    
    # draw bars for each emotion
    for i, emo_name in enumerate(CLASS_NAMES):
        prob = probabilities.get(emo_name, 0.0)
        bar_color = EMOTION_COLORS.get(emo_name, (200, 200, 200))
        y = bar_y_start + i * bar_spacing
        
        # label
        label = emo_name[:3].upper()
        cv2.putText(
            output, label, (bar_x - 45, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1
        )
        
        # background bar
        cv2.rectangle(
            output,
            (bar_x, y),
            (bar_x + bar_width, y + bar_height),
            (60, 60, 60), -1
        )
        
        # filled bar
        fill_width = int(bar_width * prob)
        if fill_width > 0:
            cv2.rectangle(
                output,
                (bar_x, y),
                (bar_x + fill_width, y + bar_height),
                bar_color, -1
            )
        
        # percentage
        cv2.putText(
            output, f"{prob:.0%}", (bar_x + bar_width + 5, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1
        )
    
    # draw quit instruction
    cv2.putText(
        output, "Press 'q' to quit", (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1
    )
    
    return output

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Webcam emotion recognition with Grad-CAM")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    args = parser.parse_args()
    
    # get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # load model
    print("Loading model...")
    config = {
        'model': {
            'architecture': 'modified_resnet18',
            'num_classes': 6,
            'in_channels': 3
        }
    }
    model = build_model(config)
    model = load_model_for_inference(args.model_path, model, device=device.type)
    
    # setup gradcam
    gradcam = GradCAM(model, target_layer=model.layer4)
    
    # setup transform
    transform = get_inference_transforms(image_size=64)
    
    # setup face detector
    face_detector = FaceDetector()
    
    # open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting webcam. Press 'q' to quit.")
    
    # main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # detect face
        roi = face_detector.detect(frame)
        
        # preprocess
        input_tensor = preprocess_frame(frame, transform, roi)
        input_tensor = input_tensor.to(device)
        
        # run inference
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]
        
        # get prediction
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        emotion = CLASS_NAMES[pred_idx]
        
        # create probability dictionary
        probabilities = {name: probs[i].item() for i, name in enumerate(CLASS_NAMES)}
        
        # generate gradcam
        heatmap = None
        with torch.enable_grad():
            heatmap, _ = gradcam(input_tensor, target_class=pred_idx)
        
        # draw results
        output = draw_results(frame, emotion, confidence, probabilities, heatmap, roi)
        
        # show frame
        cv2.imshow('Emotion Recognition', output)
        
        # check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    gradcam.remove_hooks()
    print("Done.")

if __name__ == '__main__':
    main()
