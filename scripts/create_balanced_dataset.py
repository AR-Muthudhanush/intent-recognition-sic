"""
Create balanced 5k-6k dataset covering ALL intents, targets, and spatial relations.
Uses stratified sampling to ensure comprehensive scenario coverage.
"""

import csv
from collections import defaultdict, Counter
import random

def create_balanced_dataset(target_size: int = 5500):
    """
    Create balanced dataset ensuring:
    - ALL intents represented
    - ALL targets represented
    - ALL spatial relations represented
    - Stratified sampling by intent
    """
    print(f"Creating balanced {target_size}-sample dataset with ALL scenarios...")
    print("Reading full dataset...")

    all_rows = []
    with open('dataset/dataset.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    print(f"Total samples: {len(all_rows)}")

    # Group by intent
    by_intent = defaultdict(list)
    for row in all_rows:
        intent = row['intent']
        by_intent[intent].append(row)

    print(f"\nIntent coverage: {len(by_intent)} intents")
    print(f"Intents: {sorted(by_intent.keys())[:10]}... (showing first 10)")

    # Calculate samples per intent to ensure ALL intents are included
    num_intents = len(by_intent)
    samples_per_intent = max(5, target_size // num_intents)

    print(f"\nSampling ~{samples_per_intent} samples per intent...")

    selected = []

    # First pass: ensure each intent has samples
    for intent, rows in sorted(by_intent.items()):
        # Sample from this intent
        n_samples = min(samples_per_intent, len(rows))
        sampled = random.sample(rows, n_samples)
        selected.extend(sampled)
        print(f"  {intent}: {len(sampled)} samples")

    print(f"\nTotal after intent pass: {len(selected)} samples")

    # Second pass: ensure target coverage
    selected_targets = set(row['target'] for row in selected if row['target'])
    by_target = defaultdict(list)
    for row in all_rows:
        if row['target']:
            by_target[row['target']].append(row)

    print(f"\nTarget coverage: {len(selected_targets)}/{len(by_target)} targets")

    missing_targets = set(by_target.keys()) - selected_targets
    if missing_targets:
        print(f"Missing targets: {sorted(missing_targets)[:10]}...")
        for target in missing_targets:
            if len(selected) < target_size:
                candidates = by_target[target]
                if candidates:
                    selected.append(random.choice(candidates))

    # Third pass: ensure spatial relation coverage
    selected_relations = set(row['relation'] for row in selected if row['relation'])
    by_relation = defaultdict(list)
    for row in all_rows:
        if row['relation']:
            by_relation[row['relation']].append(row)

    print(f"Spatial relation coverage: {len(selected_relations)}/{len(by_relation)} relations")

    missing_relations = set(by_relation.keys()) - selected_relations
    if missing_relations:
        print(f"Missing relations: {sorted(missing_relations)[:10]}...")
        for relation in missing_relations:
            if len(selected) < target_size:
                candidates = by_relation[relation]
                if candidates:
                    selected.append(random.choice(candidates))

    # Ensure we have target size
    if len(selected) < target_size:
        remaining = target_size - len(selected)
        candidates = [r for r in all_rows if r not in selected]
        if candidates:
            selected.extend(random.sample(candidates, min(remaining, len(candidates))))

    # Remove duplicates
    seen = set()
    unique_selected = []
    for row in selected:
        key = (row['command'], row['intent'])
        if key not in seen:
            seen.add(key)
            unique_selected.append(row)

    selected = unique_selected[:target_size]

    # Shuffle
    random.shuffle(selected)

    # Save
    output_path = f'dataset/dataset_balanced_{len(selected)}.csv'

    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        fieldnames = selected[0].keys()
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    # Print statistics
    print(f"\n{'='*60}")
    print(f"[OK] Created {output_path}")
    print(f"{'='*60}")
    print(f"Total samples: {len(selected)}")

    intent_counts = Counter(row['intent'] for row in selected)
    print(f"Intents: {len(intent_counts)} unique")
    print(f"  Most common: {intent_counts.most_common(3)}")
    print(f"  Least common: {intent_counts.most_common()[-3:]}")

    target_counts = Counter(row['target'] for row in selected if row['target'])
    print(f"\nTargets: {len(target_counts)} unique")

    relation_counts = Counter(row['relation'] for row in selected if row['relation'])
    print(f"Spatial relations: {len(relation_counts)} unique")

    attribute_counts = Counter(row['attribute'] for row in selected if row['attribute'])
    print(f"Attributes: {len(attribute_counts)} unique")

    label_counts = Counter(row['label'] for row in selected if row['label'])
    print(f"Labels: {len(label_counts)} unique")

    print(f"\n{'='*60}")
    print("USAGE IN TRAINING:")
    print(f"{'='*60}")
    print(f"""
Edit training/train.py, replace:
  dataset = IntentDataset('dataset/dataset.csv', tokenizer)
With:
  dataset = IntentDataset('dataset/dataset_balanced_{len(selected)}.csv', tokenizer)

Then run:
  python training/train.py
    """)

if __name__ == "__main__":
    random.seed(42)
    create_balanced_dataset(target_size=5500)
