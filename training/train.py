"""
Training script for multitask intent recognition model.
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

class IntentDataset(Dataset):
    """Dataset for intent recognition."""

    def __init__(self, csv_path: str, tokenizer, max_seq_length: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []
        self.intent_to_id = {intent: idx for idx, intent in enumerate(ConfigManager.INTENTS)}

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
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

def split_dataset(dataset: Dataset, train_ratio: float = 0.8):
    """Split dataset into train/eval."""
    train_size = int(len(dataset) * train_ratio)
    eval_size = len(dataset) - train_size

    indices = np.random.permutation(len(dataset))
    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]

    train_data = [dataset.samples[i] for i in train_indices]
    eval_data = [dataset.samples[i] for i in eval_indices]

    class DatasetSubset(Dataset):
        def __init__(self, samples, tokenizer, intent_to_id, max_seq_length=128):
            self.samples = samples
            self.tokenizer = tokenizer
            self.intent_to_id = intent_to_id
            self.max_seq_length = max_seq_length

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
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

    train_subset = DatasetSubset(train_data, dataset.tokenizer, dataset.intent_to_id)
    eval_subset = DatasetSubset(eval_data, dataset.tokenizer, dataset.intent_to_id)

    return train_subset, eval_subset

def train_epoch(model, train_loader, optimizer, scheduler, device, criterion):
    """Train one epoch."""
    model.train()
    total_loss = 0
    intent_correct = 0
    intent_total = 0

    for batch in tqdm(train_loader, desc="Training"):
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

    return avg_loss, intent_acc

def evaluate(model, eval_loader, device, criterion):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    intent_correct = 0
    intent_total = 0
    entity_correct = 0
    entity_total = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent_ids = batch['intent_id'].to(device)
            entity_ids = batch['entity_ids'].to(device)

            intent_logits, entity_logits = model(input_ids, attention_mask)

            intent_loss = criterion(intent_logits, intent_ids)
            entity_loss = criterion(entity_logits.view(-1, entity_logits.shape[-1]), entity_ids.view(-1))

            loss = intent_loss + 0.3 * entity_loss
            total_loss += loss.item()

            intent_pred = intent_logits.argmax(dim=1)
            intent_correct += (intent_pred == intent_ids).sum().item()
            intent_total += intent_ids.size(0)

            entity_pred = entity_logits.argmax(dim=-1)
            entity_correct += (entity_pred == entity_ids).sum().item()
            entity_total += entity_ids.numel()

    avg_loss = total_loss / len(eval_loader)
    intent_acc = intent_correct / intent_total
    entity_acc = entity_correct / entity_total

    return avg_loss, intent_acc, entity_acc

def train():
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(ConfigManager.MODEL_CONFIG['model_name'])

    import os
    dataset_path = 'dataset/dataset.csv'
    if not os.path.exists(dataset_path):
        dataset_path = '../dataset/dataset.csv'

    dataset = IntentDataset(dataset_path, tokenizer)

    train_dataset, eval_dataset = split_dataset(dataset, train_ratio=0.8)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    model = MultitaskIntentModel(
        model_name=ConfigManager.MODEL_CONFIG['model_name'],
        num_intents=len(ConfigManager.INTENTS),
        num_entities=len(ConfigManager.ENTITY_TYPES),
        dropout_rate=ConfigManager.MODEL_CONFIG['dropout_rate']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    best_intent_acc = 0

    for epoch in range(10):
        print(f"\nEpoch {epoch + 1}/10")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, criterion)
        eval_loss, eval_intent_acc, eval_entity_acc = evaluate(model, eval_loader, device, criterion)

        print(f"Train Loss: {train_loss:.4f}, Train Intent Acc: {train_acc:.4f}")
        print(f"Eval Loss: {eval_loss:.4f}, Eval Intent Acc: {eval_intent_acc:.4f}, Entity Acc: {eval_entity_acc:.4f}")

        if eval_intent_acc > best_intent_acc:
            best_intent_acc = eval_intent_acc
            os.makedirs('bert_tiny_model', exist_ok=True)
            torch.save(model.state_dict(), 'bert_tiny_model/pytorch_model.bin')
            tokenizer.save_pretrained('bert_tiny_model')
            print(f"Saved model with intent accuracy: {eval_intent_acc:.4f}")

    print("\nTraining complete!")
    print(f"Best Intent Accuracy: {best_intent_acc:.4f}")

if __name__ == "__main__":
    train()
