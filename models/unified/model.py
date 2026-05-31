"""
model.py — UnifiedModel: shared RoBERTa encoder + three task-specific heads.

Task routing is by string constant:
    TASK_EPISTEMIC → SentenceHead (CLS→3-class) + TokenHead (token→binary)
    TASK_BIAS      → BiasHead (CLS→2-layer-pooler→num_classes, or regression scalar)
    TASK_EMOTION   → EmotionHead (CLS→1-layer-pooler→11 sigmoid)

The Encoder, SentenceHead, and TokenHead are imported directly from
models.epistemic.model so they are not duplicated. EmotionHead and BiasHead
replicate the pooler layers from the original single-task models so that
pre-trained checkpoints can be warm-started without weight key remapping.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.epistemic.model import Encoder, SentenceHead, TokenHead
from src.data.registry import TaskType

TASK_EPISTEMIC = "epistemic"
TASK_BIAS      = "bias"
TASK_EMOTION   = "emotion"


# ── Emotion head ──────────────────────────────────────────────────────────────

class EmotionPooler(nn.Module):
    """
    Replicates RobertaPooler used in EmotionalFramingClassifier:
        CLS token → Linear(hidden→hidden) → Tanh
    Weight key mapping from HF checkpoint:
        roberta.pooler.dense.{weight,bias} → pooler.dense.{weight,bias}
    """

    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls = hidden_states[:, 0, :]                   # (B, H)
        return torch.tanh(self.dense(cls))             # (B, H)


class EmotionHead(nn.Module):
    """
    Multi-label head: CLS → EmotionPooler → Dropout → Linear(hidden→11).
    Loss: BCEWithLogitsLoss (no explicit sigmoid; applied externally at inference).
    Weight key mapping from HF checkpoint:
        roberta.pooler.dense.*  → pooler.dense.*
        classifier.{weight,bias} → proj.{weight,bias}
    """

    def __init__(self, hidden_size: int = 768, num_labels: int = 11, dropout: float = 0.1) -> None:
        super().__init__()
        self.pooler = EmotionPooler(hidden_size)
        self.drop   = nn.Dropout(dropout)
        self.proj   = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, num_labels). Apply sigmoid + threshold at inference."""
        return self.proj(self.drop(self.pooler(hidden_states)))


# ── Bias head ─────────────────────────────────────────────────────────────────

class BiasPooler(nn.Module):
    """
    Replicates RobertaClassificationHead used in RoBERTaClassifier (cls path):
        CLS → Dropout → Linear(hidden→hidden) → Tanh → Dropout
    Weight key mapping from checkpoint saved by train_baseline.py:
        _cls_model.classifier.dense.{weight,bias}     → dense.{weight,bias}
    """

    def __init__(self, hidden_size: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop  = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.drop(hidden_states[:, 0, :])          # (B, H)
        x = torch.tanh(self.dense(x))                  # (B, H)
        return self.drop(x)                             # (B, H)


class BiasHead(nn.Module):
    """
    Supports both classification (CrossEntropy) and regression (MSE).

    Classification path:  CLS → BiasPooler → Linear(hidden→num_classes)
    Regression path:      CLS → Dropout → Linear(hidden→1)

    Weight key mapping from checkpoint saved by train_baseline.py:
        Classification:
            _cls_model.classifier.dense.*     → pooler.dense.*
            _cls_model.classifier.out_proj.*  → proj.*
            _cls_model.roberta.*              → (encoder, not loaded here)
        Regression:
            _backbone.*                       → (encoder, not loaded here)
            _head.*                           → proj.*
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_classes: int = 2,
        task_type: TaskType = TaskType.BINARY_CLS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task_type  = task_type
        self.num_classes = num_classes

        if task_type == TaskType.REGRESSION:
            self.pooler = None
            self.drop   = nn.Dropout(dropout)
            self.proj   = nn.Linear(hidden_size, 1)
        else:
            self.pooler = BiasPooler(hidden_size, dropout)
            self.drop   = None
            self.proj   = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.task_type == TaskType.REGRESSION:
            cls = self.drop(hidden_states[:, 0, :])    # (B, H)
            return self.proj(cls).squeeze(-1)           # (B,)
        pooled = self.pooler(hidden_states)             # (B, H)
        return self.proj(pooled)                        # (B, num_classes)


# ── Unified model ─────────────────────────────────────────────────────────────

class UnifiedModel(nn.Module):
    """
    Shared RoBERTa-base encoder with three task-specific heads trained jointly.

    Each forward pass specifies a single task; the encoder runs once and the
    appropriate head computes logits and optionally loss.

    Label keyword arguments per task:
        TASK_EPISTEMIC: sent_label (B,) int64, token_labels (B, L) int64 [-100=ignore]
        TASK_BIAS:      bias_label (B,) int64 for cls or float32 for regression
        TASK_EMOTION:   emotion_labels (B, num_labels) float32 multi-hot
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.1,
        lambda_token: float = 0.3,
        bias_task_type: TaskType = TaskType.BINARY_CLS,
        bias_num_classes: int = 2,
        emotion_num_labels: int = 11,
        sent_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(model_name)
        H = self.encoder.hidden_size

        self.sent_head    = SentenceHead(H, dropout)
        self.token_head   = TokenHead(H, dropout)
        self.bias_head    = BiasHead(H, bias_num_classes, bias_task_type, dropout)
        self.emotion_head = EmotionHead(H, emotion_num_labels, dropout)

        self.sent_loss_fn  = nn.CrossEntropyLoss(weight=sent_class_weights)
        self.token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.bias_cls_loss = nn.CrossEntropyLoss()
        self.bias_reg_loss = nn.MSELoss()
        self.emotion_loss  = nn.BCEWithLogitsLoss()

        self.lambda_token   = lambda_token
        self.bias_task_type = bias_task_type

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str,
        sent_label: torch.Tensor | None = None,
        token_labels: torch.Tensor | None = None,
        bias_label: torch.Tensor | None = None,
        emotion_labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)
            task:           TASK_EPISTEMIC | TASK_BIAS | TASK_EMOTION
            **label kwargs: task-specific label tensors (optional; skip to get logits only)

        Returns dict with task-specific keys:
            TASK_EPISTEMIC: sent_logits (B,3), token_logits (B,L,2), optionally loss
            TASK_BIAS:      bias_logits (B,C) or (B,),               optionally loss
            TASK_EMOTION:   emotion_logits (B,11),                    optionally loss
        """
        hidden = self.encoder(input_ids, attention_mask)
        out    = {"task": task}

        if task == TASK_EPISTEMIC:
            sent_logits  = self.sent_head(hidden)
            token_logits = self.token_head(hidden)
            out["sent_logits"]  = sent_logits
            out["token_logits"] = token_logits

            loss      = torch.tensor(0.0, device=input_ids.device)
            has_label = False

            if sent_label is not None:
                loss      = loss + self.sent_loss_fn(sent_logits, sent_label)
                has_label = True
            if token_labels is not None:
                B, L, _ = token_logits.shape
                tok_loss  = self.token_loss_fn(
                    token_logits.view(B * L, 2), token_labels.view(B * L)
                )
                loss      = loss + self.lambda_token * tok_loss
                has_label = True
            if has_label:
                out["loss"] = loss

        elif task == TASK_BIAS:
            bias_logits = self.bias_head(hidden)
            out["bias_logits"] = bias_logits
            if bias_label is not None:
                if self.bias_task_type == TaskType.REGRESSION:
                    out["loss"] = self.bias_reg_loss(bias_logits, bias_label.float())
                else:
                    out["loss"] = self.bias_cls_loss(bias_logits, bias_label.long())

        elif task == TASK_EMOTION:
            emotion_logits = self.emotion_head(hidden)
            out["emotion_logits"] = emotion_logits
            if emotion_labels is not None:
                out["loss"] = self.emotion_loss(emotion_logits, emotion_labels.float())

        else:
            raise ValueError(f"Unknown task: {task!r}. Use TASK_EPISTEMIC, TASK_BIAS, or TASK_EMOTION.")

        return out
