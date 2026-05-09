from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from .dataset import BilingualIntentDataset, prepare_data_splits
from .train import collate_batch, move_batch_to_device
from .utils import (
    MODELS_DIR,
    REPORTS_DIR,
    ensure_dirs,
    file_size_mb,
    load_pickle,
    pass_fail,
    seed_everything,
)


HEADS = {
    "intent": {
        "logits": "intent_logits",
        "label": "intent_label",
        "encoder": "intent",
        "display": "Intent",
    },
    "target": {
        "logits": "target_logits",
        "label": "target_label",
        "encoder": "target_type",
        "display": "TargetType",
    },
    "spatial": {
        "logits": "spatial_logits",
        "label": "spatial_label",
        "encoder": "spatial_relation",
        "display": "Spatial",
    },
}


def decode(encoder, value: int) -> str:
    return str(encoder.inverse_transform([int(value)])[0])


def classify_failure(intent_ok: bool, target_ok: bool, spatial_ok: bool) -> str:
    wrong = [not intent_ok, not target_ok, not spatial_ok]
    count = sum(wrong)
    if count == 0:
        return "correct"
    if count >= 2:
        return "multi_error"
    if not intent_ok:
        return "intent_error"
    if not target_ok:
        return "target_error"
    return "spatial_error"


def macro_metrics(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return float(precision), float(recall), float(f1)


def save_confusion_matrix(
    name: str,
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    plt.figure(figsize=(max(8, len(labels) * 0.45), max(6, len(labels) * 0.35)))
    sns.heatmap(
        matrix,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        square=False,
    )
    plt.title(f"{name.title()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"confusion_matrix_{name}.png", dpi=160)
    plt.close()


def evaluate() -> dict[str, float]:
    seed_everything()
    ensure_dirs()

    encoder_path = MODELS_DIR / "label_encoders.pkl"
    if not encoder_path.exists():
        raise FileNotFoundError("Missing ./models/label_encoders.pkl. Run training first.")
    encoders = load_pickle(encoder_path)
    bundle = prepare_data_splits(save_encoders=False, encoders=encoders)
    test_dataset = BilingualIntentDataset(bundle.test, bundle.tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_batch,
    )

    quantized_path = MODELS_DIR / "quantized_model.pt"
    if not quantized_path.exists():
        raise FileNotFoundError("Missing ./models/quantized_model.pt. Run training first.")

    device = torch.device("cpu")
    model = torch.load(quantized_path, map_location=device)
    model.eval()

    y_true = {head: [] for head in HEADS}
    y_pred = {head: [] for head in HEADS}
    confidences = {head: [] for head in HEADS}
    metadata: list[dict] = []
    inference_times: list[float] = []

    with torch.no_grad():
        for batch in test_loader:
            batch = move_batch_to_device(batch, device)
            start = time.perf_counter()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            elapsed = time.perf_counter() - start
            batch_size = int(batch["input_ids"].shape[0])
            inference_times.extend([elapsed / batch_size] * batch_size)

            for head, config in HEADS.items():
                probs = torch.softmax(outputs[config["logits"]], dim=1)
                conf, pred = probs.max(dim=1)
                y_pred[head].extend(pred.cpu().tolist())
                y_true[head].extend(batch[config["label"]].cpu().tolist())
                confidences[head].extend(conf.cpu().tolist())

            for i in range(batch_size):
                metadata.append(
                    {
                        "row_id": batch["row_id"][i],
                        "lang": batch["lang"][i],
                        "text": batch["text"][i],
                    }
                )

    rows: list[dict] = []
    for index, meta in enumerate(metadata):
        true_intent = decode(bundle.encoders["intent"], y_true["intent"][index])
        pred_intent = decode(bundle.encoders["intent"], y_pred["intent"][index])
        true_target = decode(bundle.encoders["target_type"], y_true["target"][index])
        pred_target = decode(bundle.encoders["target_type"], y_pred["target"][index])
        true_spatial = decode(bundle.encoders["spatial_relation"], y_true["spatial"][index])
        pred_spatial = decode(bundle.encoders["spatial_relation"], y_pred["spatial"][index])

        intent_correct = true_intent == pred_intent
        target_correct = true_target == pred_target
        spatial_correct = true_spatial == pred_spatial
        fully_correct = intent_correct and target_correct and spatial_correct

        rows.append(
            {
                **meta,
                "true_intent": true_intent,
                "pred_intent": pred_intent,
                "intent_correct": intent_correct,
                "confidence_intent": confidences["intent"][index],
                "true_target": true_target,
                "pred_target": pred_target,
                "target_correct": target_correct,
                "confidence_target": confidences["target"][index],
                "true_spatial": true_spatial,
                "pred_spatial": pred_spatial,
                "spatial_correct": spatial_correct,
                "confidence_spatial": confidences["spatial"][index],
                "fully_correct": fully_correct,
                "failure_type": classify_failure(intent_correct, target_correct, spatial_correct),
            }
        )

    report_df = pd.DataFrame(rows)
    report_df.to_csv(REPORTS_DIR / "failure_report.csv", index=False, encoding="utf-8")

    for head, config in HEADS.items():
        labels = [str(label) for label in bundle.encoders[config["encoder"]].classes_]
        save_confusion_matrix(head, y_true[head], y_pred[head], labels)

    metrics = {head: macro_metrics(y_true[head], y_pred[head]) for head in HEADS}
    language_scores: dict[str, dict[str, float]] = {}
    for lang in ["en", "ko"]:
        indices = [i for i, meta in enumerate(metadata) if meta["lang"] == lang]
        language_scores[lang] = {}
        for head in HEADS:
            lang_true = [y_true[head][i] for i in indices]
            lang_pred = [y_pred[head][i] for i in indices]
            language_scores[lang][head] = macro_metrics(lang_true, lang_pred)[2]

    total = len(report_df)
    counts = report_df["failure_type"].value_counts().to_dict()
    fully_correct = int(report_df["fully_correct"].sum())
    all_head_macro_f1 = sum(metrics[head][2] for head in HEADS) / 3
    en_macro_f1 = sum(language_scores["en"].values()) / 3
    ko_macro_f1 = sum(language_scores["ko"].values()) / 3
    avg_inference_ms = (sum(inference_times) / max(len(inference_times), 1)) * 1000

    top_intents = (
        report_df.loc[~report_df["intent_correct"], "true_intent"]
        .value_counts()
        .head(5)
        .index.tolist()
    )
    top_targets = (
        report_df.loc[~report_df["target_correct"], "true_target"]
        .value_counts()
        .head(5)
        .index.tolist()
    )

    original_path = MODELS_DIR / "best_model.pt"
    original_mb = file_size_mb(original_path) if original_path.exists() else 0.0
    quantized_mb = file_size_mb(quantized_path)

    def pct(count: int) -> float:
        return 100.0 * count / max(total, 1)

    print("\n=== EVALUATION REPORT ===")
    print("           Precision   Recall     F1")
    for head, config in HEADS.items():
        precision, recall, f1 = metrics[head]
        print(f"{config['display']:<11}{precision:<12.2f}{recall:<11.2f}{f1:.2f}")

    print("\n=== LANGUAGE BREAKDOWN ===")
    print("           Intent F1   Target F1  Spatial F1")
    print(
        f"English    {language_scores['en']['intent']:<12.2f}"
        f"{language_scores['en']['target']:<11.2f}{language_scores['en']['spatial']:.2f}"
    )
    print(
        f"Korean     {language_scores['ko']['intent']:<12.2f}"
        f"{language_scores['ko']['target']:<11.2f}{language_scores['ko']['spatial']:.2f}"
    )
    print(
        f"Delta      {abs(language_scores['en']['intent'] - language_scores['ko']['intent']):<12.2f}"
        f"{abs(language_scores['en']['target'] - language_scores['ko']['target']):<11.2f}"
        f"{abs(language_scores['en']['spatial'] - language_scores['ko']['spatial']):.2f}"
    )

    print("\n=== FAILURE SUMMARY ===")
    print(f"Total samples:     {total}")
    print(f"Fully correct:     {fully_correct} ({pct(fully_correct):.1f}%)")
    print(f"Intent errors:     {counts.get('intent_error', 0)} ({pct(counts.get('intent_error', 0)):.1f}%)")
    print(f"Target errors:     {counts.get('target_error', 0)} ({pct(counts.get('target_error', 0)):.1f}%)")
    print(f"Spatial errors:    {counts.get('spatial_error', 0)} ({pct(counts.get('spatial_error', 0)):.1f}%)")
    print(f"Multi-errors:      {counts.get('multi_error', 0)} ({pct(counts.get('multi_error', 0)):.1f}%)")
    print(f"\nTop-5 misclassified intents: {top_intents}")
    print(f"Top-5 misclassified target types: {top_targets}")

    print("\n=== KPI CHECK ===")
    print(
        f"Macro F1 (all heads, all langs):  {all_head_macro_f1:.2f}  "
        f"-> target >0.95   [{pass_fail(all_head_macro_f1, 0.95)}]"
    )
    print(
        f"Macro F1 English only:            {en_macro_f1:.2f}  "
        f"-> target >0.95   [{pass_fail(en_macro_f1, 0.95)}]"
    )
    print(
        f"Macro F1 Korean only:             {ko_macro_f1:.2f}  "
        f"-> target >0.95   [{pass_fail(ko_macro_f1, 0.95)}]"
    )
    print(f"Original model size:              {original_mb:.2f} MB")
    print(
        f"Quantized model size:             {quantized_mb:.2f} MB "
        f"-> target <10MB [{pass_fail(quantized_mb, 10, greater=False)}]"
    )
    print(
        f"Avg inference time (CPU, 1 item): {avg_inference_ms:.2f} ms "
        f"-> target <100ms [{pass_fail(avg_inference_ms, 100, greater=False)}]"
    )

    return {
        "macro_f1": all_head_macro_f1,
        "english_macro_f1": en_macro_f1,
        "korean_macro_f1": ko_macro_f1,
        "original_model_size_mb": original_mb,
        "quantized_model_size_mb": quantized_mb,
        "avg_inference_ms": avg_inference_ms,
    }


if __name__ == "__main__":
    evaluate()
