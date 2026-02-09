# adapters are used to extract labels from the csv files and folder structures of the datasets and map them to the canonical classes we defined in labels.py


from __future__ import annotations

import os
import csv
from typing import List, Tuple, Optional

from data.labels import (
    RAF_ID_TO_CANONICAL,
    to_canonical,
    canonical_to_id,
)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def _is_image(fn: str) -> bool:
    return fn.lower().endswith(IMG_EXTS)

def folder_adapter(root_dir: str) -> List[Tuple[str, int]]:
    """
    Folder structure:
      anger/
        img1.jpg
        img2.jpg
      fear/
        img3.jpg
        ...
        Drops folders that don't match canonical classes, and files that don't look like images
    """
    samples: List[Tuple[str, int]] = []

    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        canonical = to_canonical(folder)
        if canonical is None:
            continue

        label_id = canonical_to_id(canonical)

        for fn in os.listdir(folder_path):
            if not _is_image(fn):
                continue
            samples.append((os.path.join(folder_path, fn), label_id))

    return samples

# RAF-Adapter
def rafdb_csv_adapter(images_root: str, csv_path: str) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    missing = 0
    kept = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_rel = row.get("image") or row.get("path") or row.get("filename")
            lab_raw = row.get("label")
            if not img_rel or lab_raw is None:
                continue

            try:
                lab_int = int(lab_raw)
            except ValueError:
                continue

            canonical = RAF_ID_TO_CANONICAL.get(lab_int, None)
            if canonical is None:
                continue

            label_id = canonical_to_id(canonical)

            img_rel = str(img_rel).strip().lstrip("/\\").replace("\\", "/")
            img_path = os.path.join(images_root, img_rel)

            if not os.path.exists(img_path):
                missing += 1
                continue

            samples.append((img_path, label_id))
            kept += 1

    print(f"[RAFDB adapter] kept={kept} missing_files={missing} from {csv_path}")
    return samples

# AffectNet Adapter
def affectnet_csv_adapter(images_root: str, csv_path: str) -> List[Tuple[str, int]]:
    """
    AffectNet CSV (typical: pth,label):
      pth,label
      10000003.jpg,Happy
      10000004.jpg,Neutral
      We use Columns: pth (or path or image), label
       - label is a string that maps to a canonical class via to_canonical (using ALIASES)
       - unknown labels will be dropped
    """
    samples: List[Tuple[str, int]] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pth = row.get("pth") or row.get("path") or row.get("image")
            lbl = row.get("label")
            if pth is None or lbl is None:
                continue

            canonical = to_canonical(str(lbl))
            if canonical is None:
                continue  # drop

            label_id = canonical_to_id(canonical)
            img_path = os.path.join(images_root, str(pth).strip().lstrip("/\\").replace("\\", "/"))
            if os.path.exists(img_path):
                samples.append((img_path, label_id))

    return samples