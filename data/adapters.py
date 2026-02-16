# adapters are used to extract labels from the csv files and folder structures of the datasets and map them to the canonical classes we defined in labels.py


from __future__ import annotations

import os
import csv
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

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

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"folder_adapter: root directory not found: {root_dir}")

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
            img_path = os.path.normpath(os.path.join(folder_path, fn))
            samples.append((img_path, label_id))

    logger.info(f"[folder_adapter] found {len(samples)} images under {root_dir}")
    return samples


# RAF-Adapter
def rafdb_csv_adapter(images_root: str, csv_path: str) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    missing = 0
    kept = 0
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"RAF-DB adapter: CSV file not found: {csv_path}")

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

            # normalize path (csv may contain backslashes)
            img_rel = str(img_rel).strip().lstrip("/\\").replace("\\", os.sep)

            # try 1: csv already contains label folder (e.g. "1/xxx.jpg") or direct relative path
            cand1 = os.path.normpath(os.path.join(images_root, img_rel))

            # try 2: label folder inferred from label id (e.g. ".../train/1/xxx.jpg")
            cand2 = os.path.normpath(os.path.join(images_root, str(lab_int), img_rel))

            if os.path.exists(cand1):
                img_path = cand1
            elif os.path.exists(cand2):
                img_path = cand2
            else:
                missing += 1
                continue  # <-- continue only if both candidates are missing

            samples.append((img_path, label_id))
            kept += 1

    logger.info(f"[RAFDB adapter] kept={kept} missing_files={missing} from {csv_path}")
    return samples

# AffectNet Adapter
def affectnet_csv_adapter(images_root: str, csv_path: str) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    missing = 0
    kept = 0
    dropped = 0

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"AffectNet adapter: CSV file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            pth = row.get("pth") or row.get("path") or row.get("image")
            lbl = row.get("label")

            # fallback for csvs that accidentally contain an index column:
            # row looks like: pth="0", label="anger/image....jpg", relFCs="surprise"
            if pth is not None and str(pth).isdigit():
                maybe_path = row.get("label")
                maybe_label = row.get("relFCs")
                if maybe_path and ("/" in maybe_path) and (".jpg" in maybe_path or ".png" in maybe_path):
                    pth = maybe_path
                    lbl = maybe_label

            if pth is None or lbl is None:
                dropped += 1
                continue

            canonical = to_canonical(str(lbl))
            if canonical is None:
                dropped += 1
                continue

            label_id = canonical_to_id(canonical)
            pth_clean = str(pth).strip().lstrip("/\\").replace("\\", os.sep)

            img_path = os.path.normpath(os.path.join(images_root, pth_clean))

            if not os.path.exists(img_path):
                img_path_train = os.path.normpath(os.path.join(images_root, "Train", pth_clean))
                img_path_test = os.path.normpath(os.path.join(images_root, "Test", pth_clean))

                if os.path.exists(img_path_train):
                    img_path = img_path_train
                elif os.path.exists(img_path_test):
                    img_path = img_path_test
                else:
                    missing += 1
                    continue

            samples.append((img_path, label_id))
            kept += 1

    logger.info(f"[AffectNet adapter] kept={kept} missing_files={missing} dropped={dropped} from {csv_path}")
    return samples