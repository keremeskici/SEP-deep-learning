# AffectNet, RafDB and Fer2013 come with more than 6 Labels
# Since we only want to define 6 Classes: (anger, fear, disgust, sadness, happiness, surprise) we need to map the original labels to these 6 classes
# In some cases, we also need to drop some samples (e.g. neutral in RafDB)
# FER already uses Folder structure s

from __future__ import annotations

from typing import Dict, Optional

# Canonical classes defines the classes we want to use
CANONICAL_CLASSES = ["anger", "fear", "disgust", "sadness", "happiness", "surprise"]
CANONICAL_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

def _norm(s: str) -> str:
    return s.strip().lower().replace(" ", "").replace("_", "")

# RAF-DB usually contains 7 classes, so neutral will be dropped
RAF_ID_TO_CANONICAL: Dict[int, Optional[str]] = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happiness",
    5: "sadness",
    6: "anger",
    7: None, #-> neutral, drop
}

# AffectNet usually contains 8 classes, so contempt will be dropped
ALIASES: Dict[str, str] = {
    "anger": "anger",
    "angry": "anger",

    "fear": "fear",
    "fearful": "fear",

    "disgust": "disgust",
    "disgusted": "disgust",

    "sad": "sadness",
    "sadness": "sadness",

    "happy": "happiness",
    "happiness": "happiness",

    "surprise": "surprise",
    "surprised": "surprise",
}

def to_canonical(label: str) -> Optional[str]:
    """
    Maps a label string to a canonical class name or None (if the label should be dropped)
    """
    key = _norm(label)
    if key in CANONICAL_TO_ID:
        return key
    return ALIASES.get(key, None)

def canonical_to_id(canonical: str) -> int:
    return CANONICAL_TO_ID[canonical]