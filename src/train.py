from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .dataset import BilingualIntentDataset, prepare_data_splits
from .model import TinyBertMultiTaskModel
from .utils import MODEL_NAME, MODELS_DIR, ensure_dirs, file_size_mb, seed_everything


def collate_batch(batch: list[dict]) -> dict:
    tensor_keys = [
        "input_ids",
        "attention_mask",
        "intent_label",
        "target_label",
        "spatial_label",
    ]
    result = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}
    if "token_type_ids" in batch[0]:
        result["token_type_ids"] = torch.stack([item["token_type_ids"] for item in batch])
    result["row_id"] = [item["row_id"] for item in batch]
    result["lang"] = [item["lang"] for item in batch]
    result["text"] = [item["text"] for item in batch]
    return result


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions = {"intent": [], "target": [], "spatial": []}
    labels = {"intent": [], "target": [], "spatial": []}

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            predictions["intent"].extend(outputs["intent_logits"].argmax(dim=1).cpu().tolist())
            predictions["target"].extend(outputs["target_logits"].argmax(dim=1).cpu().tolist())
            predictions["spatial"].extend(outputs["spatial_logits"].argmax(dim=1).cpu().tolist())
            labels["intent"].extend(batch["intent_label"].cpu().tolist())
            labels["target"].extend(batch["target_label"].cpu().tolist())
            labels["spatial"].extend(batch["spatial_label"].cpu().tolist())

    scores = {
        name: f1_score(labels[name], predictions[name], average="macro", zero_division=0)
        for name in predictions
    }
    scores["macro"] = sum(scores.values()) / len(scores)
    return scores


def train() -> dict[str, float]:
    seed_everything()
    ensure_dirs()

    bundle = prepare_data_splits(save_encoders=True)
    train_dataset = BilingualIntentDataset(bundle.train, bundle.tokenizer)
    val_dataset = BilingualIntentDataset(bundle.val, bundle.tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyBertMultiTaskModel(
        num_intent_classes=len(bundle.encoders["intent"].classes_),
        num_target_type_classes=len(bundle.encoders["target_type"].classes_),
        num_spatial_relation_classes=len(bundle.encoders["spatial_relation"].classes_),
        model_name=MODEL_NAME,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * 20
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.10),
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_path = MODELS_DIR / "best_model.pt"
    patience = 5
    stale_epochs = 0

    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch in progress:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
                intent_label=batch["intent_label"],
                target_label=batch["target_label"],
                spatial_label=batch["spatial_label"],
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        val_scores = evaluate_loader(model, val_loader, device)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(
            f"Epoch {epoch} | loss: {avg_loss:.2f} | "
            f"intent_f1: {val_scores['intent']:.2f} | "
            f"target_f1: {val_scores['target']:.2f} | "
            f"spatial_f1: {val_scores['spatial']:.2f} | "
            f"val_f1: {val_scores['macro']:.2f}"
        )

        if val_scores["macro"] > best_val_f1:
            best_val_f1 = val_scores["macro"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": MODEL_NAME,
                    "num_intent_classes": len(bundle.encoders["intent"].classes_),
                    "num_target_type_classes": len(bundle.encoders["target_type"].classes_),
                    "num_spatial_relation_classes": len(bundle.encoders["spatial_relation"].classes_),
                    "best_val_f1": best_val_f1,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    checkpoint = torch.load(best_path, map_location="cpu")
    cpu_model = TinyBertMultiTaskModel(
        checkpoint["num_intent_classes"],
        checkpoint["num_target_type_classes"],
        checkpoint["num_spatial_relation_classes"],
        checkpoint["model_name"],
    )
    cpu_model.load_state_dict(checkpoint["model_state_dict"])
    cpu_model.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        cpu_model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quantized_path = MODELS_DIR / "quantized_model.pt"
    torch.save(quantized_model, quantized_path)

    original_mb = file_size_mb(best_path)
    quantized_mb = file_size_mb(quantized_path)
    print(f"Original model size: {original_mb:.2f} MB")
    print(f"Quantized model size: {quantized_mb:.2f} MB")

    return {
        "best_val_f1": best_val_f1,
        "original_model_size_mb": original_mb,
        "quantized_model_size_mb": quantized_mb,
    }


if __name__ == "__main__":
    train()
