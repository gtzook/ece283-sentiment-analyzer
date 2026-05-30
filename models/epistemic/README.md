# Epistemic Certainty Model

Single-dimension baseline for predicting epistemic certainty at the sentence level.
Designed as a modular head that can be plugged into the shared multi-task encoder
described in Horych et al. (2024) MAGPIE (LREC-COLING).

---

## Label Schema

**Three classes — do not change without re-running LLM validation.**

| Label | Int | Criteria |
|---|---|---|
| asserted | 0 | No qualifying hedge cue; or CT+/CT− in FactBank |
| hedged | 1 | `speculation_modal_probable_` Szeged cue; or PR+/PR− in FactBank |
| speculative | 2 | `hypo_doxastic` word in {allegedly, reportedly, supposedly, ostensibly, purportedly, apparently}; or PS+/PS−/Uu in FactBank |

**Continuous output:** `uncertainty_score = 0·P(asserted) + 0.5·P(hedged) + 1·P(speculative)` ∈ [0, 1]

### Why the schema is narrow

An LLM validation study (5 models, 250 sentences, Cohen's κ + Fleiss' κ) showed
that only `modal_probable` reliably corresponds to genuine hedging (58% LLM agreement).
The other Szeged cue types failed:

- `hypo_investigation` (4% agreement): whether-complement clauses, not epistemic hedges
- `hypo_doxastic` (24% agreement): collapses attribution with genuine hedges; only the 6-word list above is salvageable
- `hypo_condition` (40% agreement): mixed causal rules with epistemic conditionals; recovered speculative class using modal check (would/could/might in consequent)

A binary option (asserted vs. uncertain) remains an open alternative if 3-class performance is poor.

---

## Data Sources

| Corpus | Sentences | Head | Notes |
|---|---|---|---|
| `uncertainty/wiki.xml` | ~20,748 | sent + tok | Primary training source |
| `factbank_ldc/` | ~2,807 | sent + tok | Independent factuality annotations |
| `uncertainty/factbank.xml` | ~3,129 | sent + tok | Same NYT docs as LDC FactBank but Szeged annotations; kept separate |
| `uncertainty/bio_bmc/fly/hbc.xml` | ~19,500 | tok only | Biomedical; down-weighted (λ=0.5) |
| `bioscope/abstracts.xml` + `full_papers.xml` | ~14,500 | tok only | Biomedical; down-weighted |

All raw data lives in `/mldata/ece283-sentiment-analyzer/epistemic/raw/` — not in the repo.
FactBank LDC is LDC-licensed; never commit files under `factbank_ldc/`.

---

## Architecture

```
Input tokens
    │
    ▼
Encoder (roberta-base)
    │  last_hidden_state: (batch, seq_len, 768)
    ├──────────────────┐
    ▼                  ▼
SentenceHead        TokenHead
CLS → Linear(768→3) per-token → Linear(768→2)
3-class logits      binary logits (hedge cue / not)
```

Joint loss: `sentence_loss + 0.3 × token_loss`

**Multi-task encoder interface** (for merge with political-bias and emotional-framing models):
```python
class Encoder(nn.Module):
    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        # returns last_hidden_state: (batch, seq_len, 768)
```
Heads receive the full sequence and index into it themselves.
Note for political-bias teammate: `RobertaForSequenceClassification` bakes a
2-layer pooler into the backbone. Move it into the head before the multi-task merge.

---

## Training

```bash
# Default config
python -m models.epistemic.train

# Override any config value
python -m models.epistemic.train --set data.bio_token_weight=0.25
python -m models.epistemic.train --set training.epochs=5
```

Key hyperparameters (`config.yaml`):

| | |
|---|---|
| Model | roberta-base |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Epochs | 3 |
| Warmup | 10% of steps |
| Mixed precision | fp16 |
| λ (token loss weight) | 0.3 |
| Bio down-weight | 0.5 (ablate over 0.0, 0.25, 0.5, 1.0) |

Splits are document-level (80/10/10, seed=42) to prevent sentence leakage.
Best checkpoint saved by validation macro-F1 (sentence head).

---

## Evaluation

```bash
python -m models.epistemic.eval --checkpoint runs/20260530_071702/best.pt
```

Outputs: per-class P/R/F1, macro-F1, Cohen's κ, ECE, token macro-F1, confusion matrix PNG.

**Results (epoch 2 checkpoint, val split):**

| Metric | Value |
|---|---|
| Sentence macro-F1 | 0.820 |
| Token macro-F1 | 0.877 |
| LLM agreement ceiling (Fleiss' κ) | 0.545 |
| Rule vs. LLM majority κ | 0.243 |

---

## Inference

```bash
# Single sentence
python -m models.epistemic.predict \
    --checkpoint runs/20260530_071702/best.pt \
    "The drug may help some patients."

# Batch (stdin)
cat sentences.txt | python -m models.epistemic.predict \
    --checkpoint runs/20260530_071702/best.pt
```

Output (JSON, one object per line):
```json
{
  "text": "The drug may help some patients.",
  "label": 1,
  "label_name": "hedged",
  "uncertainty_score": 0.52,
  "sent_probs": [0.31, 0.55, 0.14],
  "cue_spans": [[9, 12]]
}
```

Python API:
```python
from models.epistemic.predict import load_predictor, predict

predictor = load_predictor("runs/20260530_071702/best.pt")
result    = predict(predictor, "The drug may help some patients.")
```

---

## Known Limitations

- **BABE transfer test pending:** Cross-domain evaluation on `/mldata/ece283-sentiment-analyzer/10_BABE/preprocessed.csv` requires LLM-ensemble epistemic labeling of BABE sentences (not yet done). The eval script will run this automatically once the `epistemic_label` column is present.
- **`modal_probable` 58% LLM agreement:** Bare modal auxiliaries ("may", "could") in generic/capacity contexts are mislabeled as hedged. A targeted filter (splitting modal auxiliaries from epistemic adverbs/verbs) is planned after test-set evaluation.
- **Label schema is narrow by design:** The three-class schema is the result of the LLM validation study ruling out most Szeged cue types. A binary (asserted / uncertain) fallback is documented if 3-class performance is insufficient.
