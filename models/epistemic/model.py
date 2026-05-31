"""
model.py — epistemic certainty model architecture.

Three-class sentence classifier (asserted / hedged / speculative) with an
auxiliary token-level binary head for hedge cue detection.  Both heads share
a RoBERTa-base encoder.

─────────────────────────────────────────────────────────────────────────────
TEAMMATE INTERFACE NOTE (2026-05-30)
─────────────────────────────────────────────────────────────────────────────
Proposed shared-encoder contract for the multi-task merge:

    class Encoder(nn.Module):
        def forward(self, input_ids, attention_mask) -> torch.Tensor:
            # returns last_hidden_state: (batch, seq_len, hidden_size)

Heads receive the full sequence and index into it themselves ([CLS] at 0,
all tokens for token-level tasks).  This is compatible with:
  - emotional framing (CLS → Linear(768→11))
  - epistemic certainty (CLS → Linear(768→3); tokens → Linear(768→2))
  - political bias regression (CLS path, already uses last_hidden_state[:,0,:])

ACTION for political-bias teammate: RoBERTaClassifier currently uses
RobertaForSequenceClassification for the classification path, which bakes a
2-layer pooler (Linear→Tanh→Dropout→Linear) into the backbone.  For the
multi-task merge the pooler must move into the classification head so the
encoder stays shareable.  The regression path is already compatible.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import RobertaModel


# ── Sub-modules ───────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Thin wrapper around RoBERTa-base that returns the full last_hidden_state.

    Keeping pooling out of the encoder lets the sentence head use CLS and the
    token head use the full sequence, and makes the encoder directly swappable
    into a multi-task shared backbone.
    """

    def __init__(self, model_name: str = "roberta-base") -> None:
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.hidden_size: int = self.roberta.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns last_hidden_state: (batch, seq_len, hidden_size)."""
        return self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state


class SentenceHead(nn.Module):
    """
    3-way classifier over the [CLS] token.

    Returns raw logits (batch, 3); apply softmax for probabilities or
    use cross_entropy loss directly.  The uncertainty_score at inference is
    computed in EpistemicModel.predict(), not here.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) from Encoder
        Returns:
            logits: (batch, 3)
        """
        cls = hidden_states[:, 0, :]   # CLS token
        return self.proj(self.drop(cls))


class TokenHead(nn.Module):
    """
    Binary classifier over every token for hedge-cue detection.

    Returns raw logits (batch, seq_len, 2).  Positions labelled -100 in
    the target (special tokens, padding) are ignored by the loss.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) from Encoder
        Returns:
            logits: (batch, seq_len, 2)
        """
        return self.proj(self.drop(hidden_states))


# ── Joint model ───────────────────────────────────────────────────────────────

class EpistemicModel(nn.Module):
    """
    Full epistemic certainty model: shared Encoder + SentenceHead + TokenHead.

    Joint loss = sentence_loss + lambda_token * token_loss

    Either head's loss term is skipped when the corresponding label tensor is
    absent from the batch (None), so sentence-only and token-only batches both
    work without masking.

    Inference output includes both the 3-class label and a continuous
    uncertainty_score = 0*P(asserted) + 0.5*P(hedged) + 1*P(speculative).
    """

    UNCERTAINTY_WEIGHTS = torch.tensor([0.0, 0.5, 1.0])

    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.1,
        lambda_token: float = 0.3,
        sent_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.encoder      = Encoder(model_name)
        hidden            = self.encoder.hidden_size
        self.sent_head    = SentenceHead(hidden, dropout)
        self.token_head   = TokenHead(hidden, dropout)
        self.lambda_token = lambda_token

        # Class-weighted CE for sentence head (handles asserted-dominated imbalance)
        self.sent_loss_fn = nn.CrossEntropyLoss(weight=sent_class_weights)
        # Token head: standard CE; -100 positions are ignored automatically
        self.token_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sent_label: torch.Tensor | None = None,
        token_labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            input_ids:     (batch, seq_len)
            attention_mask:(batch, seq_len)
            sent_label:    (batch,) int64  — None for token-only batches
            token_labels:  (batch, seq_len) int64, -100 for ignored positions
                           — None for sentence-only batches

        Returns dict with keys:
            sent_logits   (batch, 3)             always present
            token_logits  (batch, seq_len, 2)    always present
            loss          scalar                 only when any label provided
        """
        hidden       = self.encoder(input_ids, attention_mask)
        sent_logits  = self.sent_head(hidden)
        token_logits = self.token_head(hidden)

        out = {"sent_logits": sent_logits, "token_logits": token_logits}

        loss = torch.tensor(0.0, device=input_ids.device)
        computed = False

        if sent_label is not None:
            loss = loss + self.sent_loss_fn(sent_logits, sent_label)
            computed = True

        if token_labels is not None:
            # token_logits: (B, L, 2) → (B*L, 2); token_labels: (B, L) → (B*L,)
            B, L, _ = token_logits.shape
            tok_loss = self.token_loss_fn(
                token_logits.view(B * L, 2),
                token_labels.view(B * L),
            )
            loss = loss + self.lambda_token * tok_loss
            computed = True

        if computed:
            out["loss"] = loss

        return out

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        Inference-only forward.  Returns:
            label            (batch,) int64    argmax of sentence logits
            uncertainty_score(batch,) float32  continuous [0, 1]
            sent_probs       (batch, 3)        softmax probabilities
            token_probs      (batch, seq_len)  P(hedge cue) per token
        """
        self.eval()
        out = self.forward(input_ids, attention_mask)

        sent_probs = torch.softmax(out["sent_logits"], dim=-1)
        weights    = self.UNCERTAINTY_WEIGHTS.to(sent_probs.device)
        uncertainty_score = (sent_probs * weights).sum(dim=-1)

        token_probs = torch.softmax(out["token_logits"], dim=-1)[:, :, 1]

        return {
            "label":             sent_probs.argmax(dim=-1),
            "uncertainty_score": uncertainty_score,
            "sent_probs":        sent_probs,
            "token_probs":       token_probs,
        }
