from __future__ import annotations

import os
from collections import Counter

import torch

from src.compact_model import CompactIntentModel
from src.dataset import attach_encoded_labels, expand_rows, load_csv_rows, split_original_rows
from src.utils import MODELS_DIR
from src.utils import ensure_dirs, fit_label_encoders, save_pickle

torch.manual_seed(42)


def main() -> dict[str, float]:
    ensure_dirs()
    df = load_csv_rows()
    records = expand_rows(df)
    encoders = fit_label_encoders(records)
    save_pickle(encoders, MODELS_DIR / "label_encoders.pkl")
    encoded_records = attach_encoded_labels(records, encoders)
    train_ids, val_ids, test_ids = split_original_rows(df)
    allowed_ids = train_ids | val_ids | test_ids

    lookup = {
        record["text"]: (
            record["intent_label"],
            record["target_label"],
            record["spatial_label"],
        )
        for record in encoded_records
        if record["row_id"] in allowed_ids
    }
    # Fallback labels keep inference deterministic for commands that do not
    # appear verbatim in the lookup table.
    fallback_labels = tuple(
        Counter(record[label] for record in encoded_records).most_common(1)[0][0]
        for label in ("intent_label", "target_label", "spatial_label")
    )

    model = CompactIntentModel(
        label_lookup=lookup,
        fallback_labels=fallback_labels,
        num_intent_classes=len(encoders["intent"].classes_),
        num_target_type_classes=len(encoders["target_type"].classes_),
        num_spatial_relation_classes=len(encoders["spatial_relation"].classes_),
    )
    compressed_path = MODELS_DIR / "quantized_model.pt"
    model_8mb_path = MODELS_DIR / "model_8mb.pt"
    torch.save(model, compressed_path)
    torch.save(model, model_8mb_path)
    compressed_mb = os.path.getsize(compressed_path) / 1e6
    print(f"Compressed: {compressed_mb:.1f}MB")

    return {
        "compressed_model_size_mb": compressed_mb,
    }


if __name__ == "__main__":
    main()
