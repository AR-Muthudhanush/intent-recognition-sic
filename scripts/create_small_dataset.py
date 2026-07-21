"""
Create a small dataset subset for quick testing/development.
Use: python create_small_dataset.py [size]
Default size: 5000 samples
"""

import csv
import sys

def create_small_dataset(size: int = 5000):
    """Extract first N samples from dataset."""
    print(f"Creating {size}-sample dataset...")

    with open('dataset/dataset.csv', 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)[:size]

    output_path = f'dataset/dataset_small_{size}.csv'

    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        if rows:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"✅ Created {output_path} with {len(rows)} samples")
    print(f"\nUsage in training script:")
    print(f"  dataset_path = 'dataset/dataset_small_{size}.csv'")

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    create_small_dataset(size)
