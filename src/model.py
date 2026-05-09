from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel

from .utils import MODEL_NAME


class TinyBertMultiTaskModel(nn.Module):
    def __init__(
        self,
        num_intent_classes: int,
        num_target_type_classes: int,
        num_spatial_relation_classes: int,
        model_name: str = MODEL_NAME,
    ) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.intent_head = nn.Linear(hidden_size, num_intent_classes)
        self.target_head = nn.Linear(hidden_size, num_target_type_classes)
        self.spatial_head = nn.Linear(hidden_size, num_spatial_relation_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        intent_label: torch.Tensor | None = None,
        target_label: torch.Tensor | None = None,
        spatial_label: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        outputs = self.bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        intent_logits = self.intent_head(cls_embedding)
        target_logits = self.target_head(cls_embedding)
        spatial_logits = self.spatial_head(cls_embedding)

        result = {
            "intent_logits": intent_logits,
            "target_logits": target_logits,
            "spatial_logits": spatial_logits,
        }

        if intent_label is not None and target_label is not None and spatial_label is not None:
            loss_intent = self.loss_fn(intent_logits, intent_label)
            loss_target = self.loss_fn(target_logits, target_label)
            loss_spatial = self.loss_fn(spatial_logits, spatial_label)
            result["loss"] = 0.5 * loss_intent + 0.3 * loss_target + 0.2 * loss_spatial
            result["loss_intent"] = loss_intent
            result["loss_target"] = loss_target
            result["loss_spatial"] = loss_spatial

        return result
