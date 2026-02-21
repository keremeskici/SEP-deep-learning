#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import yaml  # PyYAML
except ImportError:
    yaml = None

# add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.transforms import get_test_transforms
from utils.checkpoint import load_model_from_checkpoint
from utils.gradcam import GradCAM
from utils.device import get_device

# emotion classes
CLASS_NAMES = ["anger", "fear", "disgust", "sadness", "happiness", "surprise"]

# colors for each emotion (BGR)
EMOTION_COLORS = {
    'happiness': (0, 215, 255),
    'surprise': (0, 165, 255),
    'sadness': (230, 150, 50),
    'anger': (60, 60, 220),
    'disgust': (80, 180, 80),
    'fear': (180, 80, 180),
}


def infer_run_dir_from_model_path(model_path: str) -> Path:
    """
    Expects something like:
      outputs/models/<run_name>/checkpoints/best_model.pth
    Returns:
      outputs/models/<run_name>
    """
    p = Path(model_path).resolve()
    if p.parent.name.lower() == "checkpoints":
        return p.parent.parent
    return p.parent


def load_mean_std_from_config(run_dir: Path) -> Optional[Tuple[list, list]]:
    """
    Tries to load mean/std from config_used.yaml inside run_dir.
    Returns (mean, std) or None if not found.
    """
    cfg_path = run_dir / "config_used.yaml"
    if not cfg_path.exists():
        return None

    # Prefer YAML parser if available
    if yaml is not None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Try common nested keys
        for root in ["data", "dataset", "transforms", "normalize"]:
            if isinstance(cfg, dict) and root in cfg and isinstance(cfg[root], dict):
                mean = cfg[root].get("mean")
                std = cfg[root].get("std")
                if mean is not None and std is not None:
                    return mean, std

        # Try flat keys
        if isinstance(cfg, dict):
            mean = cfg.get("mean")
            std = cfg.get("std")
            if mean is not None and std is not None:
                return mean, std

        return None

    # Fallback: tiny regex parse for lines like mean: [..] std: [..]
    text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    import re
    mean_m = re.search(r"mean\s*:\s*\[([^\]]+)\]", text)
    std_m = re.search(r"std\s*:\s*\[([^\]]+)\]", text)
    if not mean_m or not std_m:
        return None

    def parse_list(s: str) -> list:
        return [float(x.strip()) for x in s.split(",")]

    return parse_list(mean_m.group(1)), parse_list(std_m.group(1))


def parse_cli_mean_std(args) -> Tuple[Optional[list], Optional[list]]:
    mean = args.mean if args.mean is not None else None
    std = args.std if args.std is not None else None
    return mean, std


class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.min_size = (60, 60)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=self.min_size
        )

        if len(faces) > 0:
            largest = max(faces, key=lambda f: f[2] * f[3])
            return tuple(largest)
        return None


def preprocess_frame(frame, transform, roi=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if roi is not None:
        x, y, w, h = roi
        pad = int(0.1 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        rgb = rgb[y1:y2, x1:x2]

    pil_image = Image.fromarray(rgb)
    tensor = transform(pil_image)
    return tensor.unsqueeze(0)


def draw_results(frame, emotion, confidence, probabilities, heatmap, roi, alpha=0.4):
    output = frame.copy()
    h, w = frame.shape[:2]
    color = EMOTION_COLORS.get(emotion, (255, 255, 255))

    # heatmap overlay
    if heatmap is not None and roi is not None:
        x, y, fw, fh = roi
        heatmap_resized = cv2.resize(heatmap, (fw, fh))
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        face_region = output[y:y+fh, x:x+fw]
        blended = cv2.addWeighted(face_region, 1 - alpha, heatmap_colored, alpha, 0)
        output[y:y+fh, x:x+fw] = blended

    # face box
    if roi is not None:
        x, y, fw, fh = roi
        thickness = 3
        cv2.rectangle(output, (x, y), (x + fw, y + fh), color, thickness)

        corner_len = 20
        cv2.line(output, (x, y), (x + corner_len, y), color, thickness + 2)
        cv2.line(output, (x, y), (x, y + corner_len), color, thickness + 2)
        cv2.line(output, (x + fw, y), (x + fw - corner_len, y), color, thickness + 2)
        cv2.line(output, (x + fw, y), (x + fw, y + corner_len), color, thickness + 2)
        cv2.line(output, (x, y + fh), (x + corner_len, y + fh), color, thickness + 2)
        cv2.line(output, (x, y + fh), (x, y + fh - corner_len), color, thickness + 2)
        cv2.line(output, (x + fw, y + fh), (x + fw - corner_len, y + fh), color, thickness + 2)
        cv2.line(output, (x + fw, y + fh), (x + fw, y + fh - corner_len), color, thickness + 2)

    # top-left panel
    panel_h = 80
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (280, panel_h), (30, 30, 30), -1)
    output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)

    cv2.putText(output, emotion.upper(), (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.putText(output, f"{confidence:.0%}", (200, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    # right panel bars
    bar_width = 150
    bar_height = 18
    bar_x = w - bar_width - 20
    bar_y_start = 20
    bar_spacing = 28

    panel_h = len(CLASS_NAMES) * bar_spacing + 10
    overlay = output.copy()
    cv2.rectangle(overlay, (bar_x - 80, bar_y_start - 10),
                  (w - 10, bar_y_start + panel_h), (30, 30, 30), -1)
    output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)

    for i, emo_name in enumerate(CLASS_NAMES):
        prob = float(probabilities.get(emo_name, 0.0))
        bar_color = EMOTION_COLORS.get(emo_name, (200, 200, 200))
        y = bar_y_start + i * bar_spacing

        label = emo_name[:3].upper()
        cv2.putText(output, label, (bar_x - 45, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.rectangle(output, (bar_x, y), (bar_x + bar_width, y + bar_height),
                      (60, 60, 60), -1)

        fill_width = int(bar_width * prob)
        if fill_width > 0:
            cv2.rectangle(output, (bar_x, y), (bar_x + fill_width, y + bar_height),
                          bar_color, -1)

        cv2.putText(output, f"{prob:.0%}", (bar_x + bar_width + 5, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    cv2.putText(output, "Press 'q' to quit", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto')

    # optional: override mean/std manually
    parser.add_argument('--mean', type=float, nargs=3, default=None, help="Normalization mean (3 floats)")
    parser.add_argument('--std', type=float, nargs=3, default=None, help="Normalization std (3 floats)")

    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")

    print("Loading model...")
    try:
        model = load_model_from_checkpoint(args.model_path, device=device.type)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # setup gradcam
    gradcam = GradCAM(model, target_layer=model.layer4)

    # build transform with SAME normalization as training
    run_dir = infer_run_dir_from_model_path(args.model_path)

    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        if "classes" in cfg:
            CLASS_NAMES[:] = cfg["classes"]
            print("Loaded classes:", CLASS_NAMES)

    mean, std = parse_cli_mean_std(args)
    if mean is None or std is None:
        loaded = load_mean_std_from_config(run_dir)
        if loaded is not None:
            mean, std = loaded

    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        print("[WARN] mean/std not found in config_used.yaml and not provided via CLI. Falling back to ImageNet mean/std.")

    print(f"Using normalization mean={mean}, std={std}")
    transform = get_test_transforms(mean=mean, std=std)

    # face detector
    face_detector = FaceDetector()

    # open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        gradcam.remove_hooks()
        return

    print("Starting webcam. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = face_detector.detect(frame)

        input_tensor = preprocess_frame(frame, transform, roi).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]

        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())
        emotion = CLASS_NAMES[pred_idx]
        probabilities = {name: float(probs[i].item()) for i, name in enumerate(CLASS_NAMES)}

        heatmap = None
        with torch.enable_grad():
            heatmap, _ = gradcam(input_tensor, target_class=pred_idx)

        output = draw_results(frame, emotion, confidence, probabilities, heatmap, roi)

        cv2.imshow('Emotion Recognition', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    gradcam.remove_hooks()
    print("Done.")


if __name__ == '__main__':
    main()