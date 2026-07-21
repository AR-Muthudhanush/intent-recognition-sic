"""
Quick training for rapid iteration (5k samples, 2 epochs, skip validation).
Use this for development/testing. Use train.py for production.
"""

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from typing import List, Dict
import json
from model import MultitaskIntentModel, ConfigManager

class QuickIntentDataset(Dataset):
    """Lightweight dataset for quick training."""

    def __init__(self, csv_path: str, tokenizer, max_samples: int = 5000, max_seq_length: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []
        self.intent_to_id = {intent: idx for idx, intent in enumerate(ConfigManager.INTENTS)}

        count = 0
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if count >= max_samples:
                    break
                self.samples.append({
                    'command': row['command'],
                    'intent': row['intent'],
                    'target': row['target'],
                    'attribute': row['attribute'],
                    'label': row['label'],
                    'index': row['index'],
                    'relation': row['relation'],
                    'reference': row['reference'],
                })
                count += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        command = sample['command']

        encoding = self.tokenizer(
            command,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        intent_id = self.intent_to_id.get(sample['intent'], 0)

        entity_labels = ['O'] * len(encoding['input_ids'][0])
        entity_ids = [ConfigManager.get_entity_id(entity) for entity in entity_labels]

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'intent_id': torch.tensor(intent_id, dtype=torch.long),
            'entity_ids': torch.tensor(entity_ids, dtype=torch.long),
        }

def train_quick():
    """Quick training (5k samples, 2 epochs, no validation)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("⚡ Quick training mode (5k samples, 2 epochs, no validation)\n")

    tokenizer = AutoTokenizer.from_pretrained(ConfigManager.MODEL_CONFIG['model_name'])

    dataset_path = 'dataset/dataset.csv'
    if not os.path.exists(dataset_path):
        dataset_path = '../dataset/dataset.csv'

    print("Loading dataset...")
    dataset = QuickIntentDataset(dataset_path, tokenizer, max_samples=5000)
    print(f"✅ Loaded {len(dataset)} samples\n")

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    model = MultitaskIntentModel(
        model_name=ConfigManager.MODEL_CONFIG['model_name'],
        num_intents=len(ConfigManager.INTENTS),
        num_entities=len(ConfigManager.ENTITY_TYPES),
        dropout_rate=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * 2
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()

    print("Starting quick training...\n")

    for epoch in range(2):
        print(f"Epoch {epoch + 1}/2")
        model.train()
        total_loss = 0
        intent_correct = 0
        intent_total = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_ids = batch['intent_id'].to(device)
            entity_ids = batch['entity_ids'].to(device)

            optimizer.zero_grad()

            intent_logits, entity_logits = model(input_ids, attention_mask)

            intent_loss = criterion(intent_logits, intent_ids)
            entity_loss = criterion(entity_logits.view(-1, entity_logits.shape[-1]), entity_ids.view(-1))

            loss = intent_loss + 0.3 * entity_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            intent_pred = intent_logits.argmax(dim=1)
            intent_correct += (intent_pred == intent_ids).sum().item()
            intent_total += intent_ids.size(0)

        avg_loss = total_loss / len(train_loader)
        intent_acc = intent_correct / intent_total

        print(f"Loss: {avg_loss:.4f}, Intent Acc: {intent_acc:.4f}\n")

    print("✅ Quick training complete!")
    print("\nSaving model...")
    os.makedirs('bert_tiny_model', exist_ok=True)
    torch.save(model.state_dict(), 'bert_tiny_model/pytorch_model.bin')
    tokenizer.save_pretrained('bert_tiny_model')
    print("✅ Model saved to bert_tiny_model/")

if __name__ == "__main__":
    train_quick()
