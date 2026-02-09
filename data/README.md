# Postum Dataloader (PyTorch)

This module provides a flexible PyTorch data-loading pipeline for FER with **multiple datasets mixed together** (FER-2013, RAF-DB, AffectNet) while ensuring **no data leakage** between train/val/test.

It is designed to:
- unify different dataset formats (folders + CSVs),
- map dataset-specific labels into a **single canonical label space**,
- compute dataset normalization statistics **only on the training split**,
- apply train/val/test transforms consistently,
- support reproducible splits and deterministic worker seeding.

---

## Key Features

### Multi-dataset mixing (train pool + test pool)
- **TRAIN-POOL**: only the official train data from each enabled dataset is collected, concatenated, and then split into **train/val**.
- **TEST-POOL**: only the official test data is used (never split into train/val).

This prevents the common mistake of accidentally mixing official test data into training.

### Canonical label mapping (6 classes)
All datasets are mapped into the same 6-class setup:

`anger, fear, disgust, sadness, happiness, surprise`

Samples that do not fit these classes are dropped (e.g., RAF-DB “neutral”, AffectNet “contempt” depending on your CSV labels).

### Mean/Std computed only on training subset
Normalization statistics are computed **after** train/val split, using **only training images**, to avoid leaking information from val/test into training.

### Transforms applied after splitting
Transforms are wrapped in a small dataset wrapper so that:
- the split is done on the raw dataset,
- and then transforms are applied to each subset.

### Reproducibility
- fixed random seed is used for splitting
- dataloader workers get deterministic seeds via `worker_init_fn`

---

## Project Structure (relevant files)

```text
data/
  dataloader.py          # main entrypoint: get_dataloaders(cfg)
  dataset.py             # PIL-based dataset that returns (image, label)
  adapters.py            # dataset-specific adapters (folder/RAF/AffectNet)
  labels.py              # canonical classes + mapping helpers
  transforms.py          # train/val/test augmentations
  calculate_mean_std.py  # mean/std computation on the train subset