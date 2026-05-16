"""
RoBERTa baseline classifier / regressor for MAGPIE political-bias tasks.

Uses RobertaForSequenceClassification for classification (proper 2-layer pooler
head: Linear→Tanh→Dropout→Linear) and a custom regression wrapper for continuous
labels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaModel

from src.data.registry import TaskType


class RoBERTaClassifier(nn.Module):
    """RoBERTa with task-appropriate output head.

    For classification tasks, delegates to RobertaForSequenceClassification,
    which includes the pretrained pooler (Linear→Tanh) before the final head —
    critical for stable fine-tuning.

    For regression, uses RobertaModel + single linear output.
    """

    def __init__(
        self,
        task_type: TaskType,
        num_classes: int = 2,
        model_name: str = "roberta-base",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes

        if task_type == TaskType.REGRESSION:
            self._backbone = RobertaModel.from_pretrained(model_name)
            hidden = self._backbone.config.hidden_size
            self._dropout = nn.Dropout(dropout)
            self._head = nn.Linear(hidden, 1)
            self._cls_model = None
        else:
            self._cls_model = RobertaForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_classes,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
            )
            self._backbone = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.task_type == TaskType.REGRESSION:
            outputs = self._backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            pooled = self._dropout(outputs.last_hidden_state[:, 0, :])
            return self._head(pooled).squeeze(-1)

        outputs = self._cls_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.logits

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.task_type == TaskType.REGRESSION:
            return nn.functional.mse_loss(logits, labels.float())
        return nn.functional.cross_entropy(logits, labels.long())

    def parameters(self, recurse: bool = True):
        if self._cls_model is not None:
            return self._cls_model.parameters(recurse=recurse)
        return super().parameters(recurse=recurse)
