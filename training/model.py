"""
Multitask Intent Recognition Model.
Tasks: Intent classification + Entity extraction.
Architecture: BERT-Tiny multilingual transformer.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple

class MultitaskIntentModel(nn.Module):
    """Multitask transformer for intent classification and entity extraction."""

    def __init__(self, model_name: str = "bert-base-multilingual-uncased",
                 num_intents: int = 62, num_entities: int = 50,
                 dropout_rate: float = 0.1):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_intents)
        )

        self.entity_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_entities)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns: (intent_logits, entity_logits)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        pooled_output = outputs.pooler_output
        sequence_output = outputs.last_hidden_state

        pooled_output = self.dropout(pooled_output)
        sequence_output = self.dropout(sequence_output)

        intent_logits = self.intent_classifier(pooled_output)
        entity_logits = self.entity_extractor(sequence_output)

        return intent_logits, entity_logits

class ConfigManager:
    """Centralized configuration."""

    INTENTS = [
        "click", "tap", "double_tap", "long_press", "scroll", "swipe",
        "drag", "drop", "delete", "copy", "paste", "highlight", "select",
        "open", "close", "play", "pause", "stop", "search", "zoom",
        "rotate", "move", "resize", "upload", "download", "share", "save",
        "refresh", "retry", "enable", "disable", "check", "uncheck",
        "expand", "collapse", "accept", "reject", "approve", "back",
        "next", "previous", "home", "settings", "login", "logout",
        "bookmark", "pin", "unpin", "start", "finish", "exit", "continue",
        "add", "remove", "hide", "show", "mute", "unmute", "install",
        "uninstall", "submit", "cancel", "type", "enter", "focus",
        "hover", "launch"
    ]

    TARGETS = [
        "button", "icon", "image", "checkbox", "switch", "radio_button",
        "dropdown", "textbox", "input", "text", "chart", "graph", "table",
        "card", "menu", "navigation_bar", "toolbar", "search_box", "video",
        "audio", "attachment", "file", "document", "message", "notification",
        "popup", "dialog", "tab", "list_item", "profile", "avatar",
        "calendar", "slider", "progress_bar", "link", "qr_code", "camera",
        "gallery", "email", "password", "phone_number"
    ]

    ENTITY_TYPES = [
        "intent", "target", "attribute", "label", "index", "relation", "reference",
        "O"  # Outside
    ]

    MODEL_CONFIG = {
        "model_name": "bert-base-multilingual-uncased",
        "num_intents": len(INTENTS),
        "num_entities": len(ENTITY_TYPES),
        "hidden_size": 256,
        "dropout_rate": 0.1,
        "max_seq_length": 128,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 10,
        "warmup_steps": 500,
    }

    @staticmethod
    def get_intent_id(intent: str) -> int:
        return ConfigManager.INTENTS.index(intent) if intent in ConfigManager.INTENTS else 0

    @staticmethod
    def get_intent_label(intent_id: int) -> str:
        return ConfigManager.INTENTS[intent_id] if 0 <= intent_id < len(ConfigManager.INTENTS) else "unknown"

    @staticmethod
    def get_entity_id(entity: str) -> int:
        return ConfigManager.ENTITY_TYPES.index(entity) if entity in ConfigManager.ENTITY_TYPES else len(ConfigManager.ENTITY_TYPES) - 1

    @staticmethod
    def get_entity_label(entity_id: int) -> str:
        return ConfigManager.ENTITY_TYPES[entity_id] if 0 <= entity_id < len(ConfigManager.ENTITY_TYPES) else "O"
