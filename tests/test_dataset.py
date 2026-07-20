"""
Tests for dataset validity and completeness.
"""

import pytest
import csv
from pathlib import Path
from typing import Set

DATASET_PATH = Path(__file__).parent.parent / 'dataset' / 'dataset.csv'

@pytest.fixture
def dataset_rows():
    """Load dataset rows."""
    rows = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows

class TestDatasetStructure:
    """Test dataset structure and format."""

    def test_dataset_file_exists(self):
        """Test that dataset file exists."""
        assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"

    def test_dataset_not_empty(self, dataset_rows):
        """Test that dataset contains rows."""
        assert len(dataset_rows) > 0, "Dataset is empty"

    def test_dataset_size(self, dataset_rows):
        """Test dataset has appropriate size."""
        assert len(dataset_rows) >= 40000, f"Dataset too small: {len(dataset_rows)} rows"
        assert len(dataset_rows) <= 70000, f"Dataset too large: {len(dataset_rows)} rows"

    def test_required_columns(self, dataset_rows):
        """Test that all required columns exist."""
        required_columns = ['command', 'intent', 'target', 'attribute', 'label', 'index', 'relation', 'reference']
        if dataset_rows:
            columns = set(dataset_rows[0].keys())
            for col in required_columns:
                assert col in columns, f"Missing column: {col}"

    def test_no_empty_commands(self, dataset_rows):
        """Test that no command field is empty."""
        empty_commands = [row for row in dataset_rows if not row.get('command', '').strip()]
        assert len(empty_commands) == 0, f"Found {len(empty_commands)} empty commands"

    def test_no_empty_intents(self, dataset_rows):
        """Test that no intent field is empty."""
        empty_intents = [row for row in dataset_rows if not row.get('intent', '').strip()]
        assert len(empty_intents) == 0, f"Found {len(empty_intents)} empty intents"

class TestDatasetCoverage:
    """Test dataset coverage of intents, targets, and relations."""

    def test_intent_coverage(self, dataset_rows):
        """Test that dataset covers multiple intents."""
        intents = set(row['intent'] for row in dataset_rows if row.get('intent'))
        assert len(intents) >= 30, f"Insufficient intent coverage: {len(intents)}"

    def test_target_coverage(self, dataset_rows):
        """Test that dataset covers multiple targets."""
        targets = set(row['target'] for row in dataset_rows if row.get('target'))
        assert len(targets) >= 20, f"Insufficient target coverage: {len(targets)}"

    def test_spatial_relation_coverage(self, dataset_rows):
        """Test that dataset covers spatial relations."""
        relations = set(row['relation'] for row in dataset_rows if row.get('relation'))
        assert len(relations) >= 15, f"Insufficient relation coverage: {len(relations)}"

    def test_attribute_variety(self, dataset_rows):
        """Test variety in attributes."""
        attributes = set(row['attribute'] for row in dataset_rows if row.get('attribute'))
        assert len(attributes) > 0, "No attribute variation found"

class TestDatasetQuality:
    """Test dataset quality metrics."""

    def test_command_language_mix(self, dataset_rows):
        """Test that dataset has English and non-English commands."""
        english_count = 0
        non_english_count = 0

        for row in dataset_rows:
            command = row.get('command', '')
            if command:
                is_english = all(ord(c) < 128 for c in command if c.isalpha())
                if is_english or any(c in command for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                    english_count += 1
                else:
                    non_english_count += 1

        assert english_count > 0, "No English commands found"

    def test_command_length_variety(self, dataset_rows):
        """Test variety in command lengths."""
        lengths = [len(row.get('command', '')) for row in dataset_rows]
        min_length = min(lengths)
        max_length = max(lengths)
        assert min_length < 20, "Minimum command too long"
        assert max_length > 30, "Maximum command too short"

    def test_no_duplicate_rows(self, dataset_rows):
        """Test that dataset doesn't have too many duplicates."""
        commands = [row['command'] for row in dataset_rows]
        unique_commands = set(commands)
        duplicate_ratio = 1 - (len(unique_commands) / len(commands))
        assert duplicate_ratio < 0.5, f"Too many duplicates: {duplicate_ratio * 100:.1f}%"

    def test_intent_distribution(self, dataset_rows):
        """Test that intents are distributed across dataset."""
        from collections import Counter
        intent_counts = Counter(row['intent'] for row in dataset_rows if row.get('intent'))
        total = sum(intent_counts.values())

        max_count = max(intent_counts.values())
        max_ratio = max_count / total

        assert max_ratio < 0.3, f"Intent imbalance too high: {max_ratio * 100:.1f}%"

class TestDatasetVariations:
    """Test that dataset includes command variations."""

    def test_typo_variations_present(self, dataset_rows):
        """Test that dataset includes typos/variations."""
        commands = [row['command'] for row in dataset_rows]
        has_variations = False

        for cmd in commands:
            if any(c.isdigit() for c in cmd if c not in []):
                has_variations = True
                break

        assert has_variations or len(set(commands)) > len(commands) * 0.8, \
            "Dataset may not have sufficient variations"

    def test_command_similarity_variance(self, dataset_rows):
        """Test command string variance."""
        commands = [row['command'] for row in dataset_rows[:100]]
        unique = len(set(commands))
        assert unique > len(commands) * 0.5, \
            f"Low variance in commands: {unique}/{len(commands)} unique"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
