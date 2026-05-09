from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .utils import (
    DATA_DIR,
    LABEL_COLUMNS,
    MODEL_NAME,
    MODELS_DIR,
    SEED,
    ensure_dirs,
    fit_label_encoders,
    normalize_missing,
    save_pickle,
)


@dataclass
class SplitBundle:
    train: list[dict[str, Any]]
    val: list[dict[str, Any]]
    test: list[dict[str, Any]]
    encoders: dict[str, Any]
    tokenizer: PreTrainedTokenizerBase


class BilingualIntentDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 64,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        encoded = self.tokenizer(
            record["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["intent_label"] = torch.tensor(record["intent_label"], dtype=torch.long)
        item["target_label"] = torch.tensor(record["target_label"], dtype=torch.long)
        item["spatial_label"] = torch.tensor(record["spatial_label"], dtype=torch.long)
        item["row_id"] = record["row_id"]
        item["lang"] = record["lang"]
        item["text"] = record["text"]
        return item


def load_csv_rows(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(frames, ignore_index=True)
    required = {"english_command", "korean_command", *LABEL_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    for column in ["attribute", "spatial_relation", "spatial_reference", "position"]:
        df[column] = df[column].map(normalize_missing)
    for column in ["intent", "target_type"]:
        df[column] = df[column].map(lambda value: str(value).strip())

    return df


def expand_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        labels = {column: getattr(row, column) for column in LABEL_COLUMNS}
        records.append(
            {
                "row_id": int(row.row_id),
                "text": "[EN] " + str(row.english_command),
                "lang": "en",
                **labels,
            }
        )
        records.append(
            {
                "row_id": int(row.row_id),
                "text": "[KO] " + str(row.korean_command),
                "lang": "ko",
                **labels,
            }
        )
    return records


def split_original_rows(df: pd.DataFrame) -> tuple[set[int], set[int], set[int]]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["intent"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["intent"],
    )
    return set(train_df["row_id"]), set(val_df["row_id"]), set(test_df["row_id"])


def attach_encoded_labels(
    records: list[dict[str, Any]],
    encoders: dict[str, Any],
) -> list[dict[str, Any]]:
    encoded_records: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["intent_label"] = int(encoders["intent"].transform([record["intent"]])[0])
        item["target_label"] = int(encoders["target_type"].transform([record["target_type"]])[0])
        item["spatial_label"] = int(
            encoders["spatial_relation"].transform([record["spatial_relation"]])[0]
        )
        encoded_records.append(item)
    return encoded_records


def print_language_distribution(name: str, records: list[dict[str, Any]]) -> None:
    counts = pd.Series([record["lang"] for record in records]).value_counts().to_dict()
    print(f"{name} language distribution: en={counts.get('en', 0)}, ko={counts.get('ko', 0)}")


def prepare_data_splits(
    save_encoders: bool = True,
    encoders: dict[str, Any] | None = None,
) -> SplitBundle:
    ensure_dirs()
    df = load_csv_rows()
    all_records = expand_rows(df)
    if encoders is None:
        encoders = fit_label_encoders(all_records)
    if save_encoders:
        save_pickle(encoders, MODELS_DIR / "label_encoders.pkl")

    train_ids, val_ids, test_ids = split_original_rows(df)
    encoded_records = attach_encoded_labels(all_records, encoders)

    train = [record for record in encoded_records if record["row_id"] in train_ids]
    val = [record for record in encoded_records if record["row_id"] in val_ids]
    test = [record for record in encoded_records if record["row_id"] in test_ids]

    print_language_distribution("Train", train)
    print_language_distribution("Val", val)
    print_language_distribution("Test", test)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return SplitBundle(train=train, val=val, test=test, encoders=encoders, tokenizer=tokenizer)
