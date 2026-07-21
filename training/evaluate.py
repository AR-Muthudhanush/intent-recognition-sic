"""
Evaluation script for intent recognition model.
Measures: accuracy, precision, recall, F1 score.
"""

import torch
import csv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from model import MultitaskIntentModel, ConfigManager
from train import IntentDataset
import numpy as np
from tqdm import tqdm

def evaluate_model(model_path: str = None, dataset_path: str = "dataset/dataset_balanced_1932.csv"):
    """Evaluate trained model on dataset."""
    import os
    from pathlib import Path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Auto-detect model path if not provided
    if model_path is None:
        if Path("models/model_best.pt").exists():
            model_path = "../models"
        elif Path("../models/model_best.pt").exists():
            model_path = "../models"
        else:
            raise FileNotFoundError("Model not found in models/ directory")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
    model = MultitaskIntentModel(
        model_name=model_path,
        num_intents=len(ConfigManager.INTENTS),
        num_entities=len(ConfigManager.ENTITY_TYPES)
    ).to(device)

    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=device))
    model.eval()

    if not os.path.exists(dataset_path):
        # Fallback to full dataset if balanced version not found
        dataset_path = 'dataset/dataset.csv'
        if not os.path.exists(dataset_path):
            dataset_path = f"../dataset/dataset.csv"

    print(f"Loading dataset: {dataset_path}")
    dataset = IntentDataset(dataset_path, tokenizer)
    test_size = int(len(dataset) * 0.2)
    test_indices = np.random.choice(len(dataset), test_size, replace=False)
    test_data = [dataset.samples[i] for i in test_indices]

    test_dataset = IntentDataset(dataset_path, tokenizer)
    test_dataset.samples = test_data

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    intent_preds = []
    intent_true = []
    entity_preds = []
    entity_true = []
    inference_times = []

    import time

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_ids = batch['intent_id'].to(device)
            entity_ids = batch['entity_ids'].to(device)

            start_time = time.time()
            intent_logits, entity_logits = model(input_ids, attention_mask)
            inference_times.append(time.time() - start_time)

            intent_pred = intent_logits.argmax(dim=1).cpu().numpy()
            entity_pred = entity_logits.argmax(dim=-1).cpu().numpy()

            intent_preds.extend(intent_pred)
            intent_true.extend(intent_ids.cpu().numpy())
            entity_preds.extend(entity_pred.flatten())
            entity_true.extend(entity_ids.cpu().numpy().flatten())

    intent_acc = accuracy_score(intent_true, intent_preds)
    intent_prec = precision_score(intent_true, intent_preds, average='weighted', zero_division=0)
    intent_recall = recall_score(intent_true, intent_preds, average='weighted', zero_division=0)
    intent_f1 = f1_score(intent_true, intent_preds, average='weighted', zero_division=0)

    entity_acc = accuracy_score(entity_true, entity_preds)
    entity_prec = precision_score(entity_true, entity_preds, average='weighted', zero_division=0)
    entity_recall = recall_score(entity_true, entity_preds, average='weighted', zero_division=0)
    entity_f1 = f1_score(entity_true, entity_preds, average='weighted', zero_division=0)

    avg_inference_time = np.mean(inference_times[1:]) * 1000

    print("\n" + "="*60)
    print("INTENT CLASSIFICATION METRICS")
    print("="*60)
    print(f"Accuracy:  {intent_acc:.4f} (Target: ≥0.95)")
    print(f"Precision: {intent_prec:.4f}")
    print(f"Recall:    {intent_recall:.4f}")
    print(f"F1 Score:  {intent_f1:.4f}")

    print("\n" + "="*60)
    print("ENTITY EXTRACTION METRICS")
    print("="*60)
    print(f"Accuracy:  {entity_acc:.4f}")
    print(f"Precision: {entity_prec:.4f} (Target: ≥0.95)")
    print(f"Recall:    {entity_recall:.4f} (Target: ≥0.95)")
    print(f"F1 Score:  {entity_f1:.4f} (Target: ≥0.95)")

    print("\n" + "="*60)
    print("INFERENCE PERFORMANCE")
    print("="*60)
    print(f"Avg Latency: {avg_inference_time:.2f} ms (Target: <100 ms)")
    print(f"Model Path: {model_path}")

    print("\n" + "="*60)
    print("INTENT CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        intent_true, intent_preds,
        target_names=ConfigManager.INTENTS,
        zero_division=0
    ))

    results = {
        "intent_accuracy": float(intent_acc),
        "intent_precision": float(intent_prec),
        "intent_recall": float(intent_recall),
        "intent_f1": float(intent_f1),
        "entity_accuracy": float(entity_acc),
        "entity_precision": float(entity_prec),
        "entity_recall": float(entity_recall),
        "entity_f1": float(entity_f1),
        "avg_inference_time_ms": float(avg_inference_time),
    }

    import json
    with open("evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to evaluation_results.json")

    return results

if __name__ == "__main__":
    evaluate_model()
