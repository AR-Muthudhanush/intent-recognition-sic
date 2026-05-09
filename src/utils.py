from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_NAME = "huawei-noah/TinyBERT_General_4L_312D"

LABEL_COLUMNS = [
    "intent",
    "target_type",
    "attribute",
    "spatial_relation",
    "spatial_reference",
    "position",
]

ENCODER_COLUMNS = [
    "intent",
    "target_type",
    "attribute",
    "spatial_relation",
    "position",
]


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_missing(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "none"
    text = str(value).strip()
    if not text or text.upper() == "NONE" or text.lower() == "nan":
        return "none"
    return text


def fit_label_encoders(records: list[dict[str, Any]]) -> dict[str, LabelEncoder]:
    encoders: dict[str, LabelEncoder] = {}
    for column in ENCODER_COLUMNS:
        encoder = LabelEncoder()
        encoder.fit([str(record[column]) for record in records])
        encoders[column] = encoder
    return encoders


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def pass_fail(value: float, threshold: float, greater: bool = True) -> str:
    passed = value > threshold if greater else value < threshold
    return "PASS" if passed else "FAIL"
