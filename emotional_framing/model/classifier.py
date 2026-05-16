"""
RoBERTa-base encoder with a single multi-label classification head.

This is intentionally single-task. The comment blocks marked
"MTL HOOK" show where a multi-task head would plug in for the
MTL model that follows this ablation baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class EmotionalFramingClassifier(RobertaPreTrainedModel):
    """
    RoBERTa encoder → pooled [CLS] → linear head → 11 sigmoid outputs.

    Loss: BCEWithLogitsLoss (numerically stable, no explicit sigmoid in forward).
    Threshold: applied externally at inference (default 0.5, tunable on dev).
    """

    def __init__(self, config: "RobertaConfig"):  # noqa: F821
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=True)

        # ── MTL HOOK ──────────────────────────────────────────────────────────
        # In the MTL model, self.classifier becomes a ModuleDict of task heads,
        # each Linear(config.hidden_size, task_num_labels). The shared encoder
        # (self.roberta) and pooled representation stay identical.
        # ─────────────────────────────────────────────────────────────────────
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # pooler_output is the [CLS] token representation after a dense + tanh
        pooled: torch.Tensor = outputs.pooler_output  # (B, hidden_size)

        # ── MTL HOOK ──────────────────────────────────────────────────────────
        # In MTL, pooled is passed to each task head independently here.
        # Task-specific logits are computed, each with its own loss, then
        # combined into a weighted sum before back-propagation.
        # ─────────────────────────────────────────────────────────────────────
        logits: torch.Tensor = self.classifier(pooled)  # (B, num_labels)

        loss: torch.Tensor | None = None
        if labels is not None:
            # BCEWithLogitsLoss = sigmoid + binary cross-entropy, numerically stable
            loss = nn.BCEWithLogitsLoss()(logits, labels.float())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
