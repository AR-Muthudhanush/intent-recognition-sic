"""
Training module for Intent Recognition System.
"""

from .model import MultitaskIntentModel, ConfigManager
from .train import train, IntentDataset
from .evaluate import evaluate_model

__all__ = [
    'MultitaskIntentModel',
    'ConfigManager',
    'train',
    'IntentDataset',
    'evaluate_model',
]
