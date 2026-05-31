# Emotional Framing Floor Model — Project Summary

## Purpose

This is a **single-task baseline (floor model)** for emotional framing classification
in news text, built as part of a larger multi-task media bias detection project
(ECE 283). It will be used as an ablation baseline against a multi-head MTL model
built later. "Floor model" means intentionally minimal: no auxiliary tasks, no MTL
logic, just RoBERTa-base fine-tuned on one classification head.

---

## Task Definition

**Multi-label emotion classification** — given a sentence or short passage from a
news article, predict which of 11 emotions the text evokes:

```
anger, anticipation, disgust, fear, joy, love, optimism, pessimism,
sadness, surprise, trust
```

Schema: SemEval-2018 Task 1 E-c (multi-label, not mutually exclusive).

---

## Repository Location

```
/home/gzook/ece283-sentiment-analyzer/emotional_framing/
```

Part of the existing `ece283-sentiment-analyzer` project. Run all commands from
`/home/gzook/ece283-sentiment-analyzer/` using `/home/gzook/venv/bin/python`.

---

## File Structure

```
emotional_framing/
  config.py          # EmotionalFramingConfig dataclass — all hyperparams
  data/
    loader.py        # 4-source dataset loading + iterative-stratification splits
  model/
    classifier.py    # RoBERTa encoder + Linear(768→11) head
  train.py           # HF Trainer subclass, CLI entry point
  evaluate.py        # metrics (macro-F1, micro-F1, subset accuracy, Hamming loss)
  predict.py         # EmotionalFramingPredictor.predict(texts) inference API
  requirements.txt
  SUMMARY.md         # this file
```

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| Encoder | `roberta-base` (125M params) |
| Head | `Linear(768 → 11)` with sigmoid activation |
| Loss | `BCEWithLogitsLoss` (numerically stable multi-label) |
| Threshold | 0.5 per label at inference; tuned on dev set after training |
| Class | `EmotionalFramingClassifier(RobertaPreTrainedModel)` |

The model uses the `[CLS]` pooled representation from `roberta.pooler_output`.
Comments marked `# MTL HOOK` throughout the codebase show where multi-task heads
would plug in for the follow-on MTL model.

---

## Training Configuration (`config.py`)

```python
encoder_lr     = 2e-5     # separate LR for RoBERTa encoder layers
head_lr        = 1e-4     # higher LR for freshly-initialized classifier head
weight_decay   = 0.01
batch_size     = 32
max_epochs     = 10
patience       = 3        # early stopping on dev macro-F1
max_seq_length = 128
warmup_ratio   = 0.06     # 6% of total steps for linear warmup
seed           = 42
fp16           = True     # auto-disabled if no CUDA
```

**Custom optimizer**: `EmotionalFramingTrainer` subclasses HF `Trainer` and
overrides `create_optimizer()` to set per-param-group learning rates (encoder vs.
head) using `torch.optim.AdamW`. This is necessary because HF `TrainingArguments`
only accepts a single global LR.

**Warmup**: `warmup_steps` is computed at runtime as
`ceil(train_size / batch_size) × epochs × warmup_ratio`. The `warmup_ratio`
argument was deprecated in transformers ≥ v5.2.

---

## Dataset (`data/loader.py`)

Four sources merged before a single 80/10/10 stratified split.
Split uses `skmultilearn.iterative_train_test_split` to preserve per-label
prevalence across splits. All HF datasets are cached locally after first download.

| Source | Size | Labels covered | Notes |
|--------|------|----------------|-------|
| MAGPIE `84_emotion_tweets` | 195,744 | anger, anticipation, disgust, fear, joy, sadness, surprise, trust | Local: `/mldata/ece283-sentiment-analyzer/`. 8-class multiclass (0-indexed Plutchik) converted to 11-d multi-hot. love/optimism/pessimism always 0 for this source. |
| GoEmotions simplified | ~34,860 | all 11 (multi-label) | HF: `google-research-datasets/go_emotions`. 27 emotions → 11 via conservative mapping table. Neutral-only and fully-unmapped examples dropped. |
| dair-ai/emotion | 20,000 | sadness, joy, love, anger, fear, surprise | HF: `dair-ai/emotion`. 6-class single-label. Primary value: clean **love** signal. |
| TweetEval emotion | 5,052 | anger, joy, optimism, sadness | HF: `cardiffnlp/tweet_eval` config `emotion`. 4-class single-label. Primary value: clean **optimism** signal. |
| **Total** | **~255,656** | all 11 labels | |

**SemEval-2018 Task 1 E-c** is not used: HF datasets v5 dropped support for the
old dataset script it requires. All 11 labels are covered by the four sources above.

**Weakest label**: `trust` (GoEmotions `approval`/`admiration` mapping only).

### GoEmotions → SemEval mapping (conservative)

Only clear semantic overlaps are mapped. Ambiguous labels (`confusion`,
`disapproval`, `embarrassment`, `neutral`) are omitted.

```
anger/annoyance       → anger
curiosity/excitement/desire → anticipation
disgust               → disgust
fear/nervousness      → fear
joy/amusement/pride/relief → joy
admiration/caring/gratitude/love → love
optimism              → optimism
disappointment/grief  → pessimism
sadness/remorse       → sadness
surprise/realization  → surprise
approval              → trust
```

---

## Training (CLI)

```bash
cd /home/gzook/ece283-sentiment-analyzer

# Full run (~4 h, roberta-base, 256k samples, 10 epochs)
/home/gzook/venv/bin/python emotional_framing/train.py

# Fast test run (~20 min, distilroberta-base, seq_len=64, 3 epochs, 20k samples)
/home/gzook/venv/bin/python emotional_framing/train.py --fast

# Arbitrary sample cap
/home/gzook/venv/bin/python emotional_framing/train.py --samples 50000

# 100-sample smoke test (~2 min)
/home/gzook/venv/bin/python emotional_framing/train.py --debug
```

`--fast` saves to `./checkpoints/emotional_framing_floor_fast/` to avoid
overwriting full-run checkpoints. The distilroberta-base and roberta-base
checkpoints are compatible with `predict.py` (same hidden size 768, same head).

---

## Evaluation (`evaluate.py`)

Reported on dev and test sets:
- Per-label F1, Precision, Recall (via `sklearn.classification_report`)
- **Macro-F1** — primary metric, used for early stopping
- Micro-F1
- Subset Accuracy (exact match)
- Hamming Loss

Threshold tuning: `tune_threshold()` grid-searches [0.3, 0.4, 0.5, 0.6, 0.7]
on the dev set logits after training completes. The best threshold is saved to
`<checkpoint>/threshold.txt` and loaded automatically by `predict.py`.

---

## Inference (`predict.py`)

```python
from emotional_framing.predict import EmotionalFramingPredictor

predictor = EmotionalFramingPredictor("./checkpoints/emotional_framing_floor")
results = predictor.predict([
    "Protesters flooded the streets in outrage over the decision.",
    "The new policy brings hope for thousands of families.",
])
# [
#   {"anger": 1, "anticipation": 0, ..., "scores": {"anger": 0.87, ...}},
#   {"joy": 0, "optimism": 1, ..., "scores": {"optimism": 0.72, ...}},
# ]
```

Or from CLI:
```bash
/home/gzook/venv/bin/python emotional_framing/predict.py \
  --checkpoint ./checkpoints/emotional_framing_floor \
  --texts "Protesters flooded the streets in outrage."
```

---

## Known Issues Fixed During Development

| Issue | Fix |
|-------|-----|
| `AdamW` removed from `transformers` v5 | Use `torch.optim.AdamW` |
| `warmup_ratio` deprecated in transformers ≥ v5.2 | Compute `warmup_steps` at runtime |
| `accelerate>=1.1.0` required by Trainer | Installed in venv |
| wandb crashes on bad API key | `wandb.Api().viewer()` verifies key before enabling; falls back to no logging |
| HF datasets v5 dropped dataset scripts | SemEval-2018 E-c skipped; coverage filled by GoEmotions + dair-ai/emotion + TweetEval |

---

## Load Report (expected, not a bug)

When loading `roberta-base` into `EmotionalFramingClassifier`:

- **UNEXPECTED `lm_head.*`**: roberta-base was pretrained with Masked Language
  Modeling. These weights are discarded — we don't need them for classification.
- **MISSING `classifier.*`**: Our new Linear(768→11) head, freshly initialized.
- **MISSING `roberta.pooler.*`**: The [CLS] pooler dense layer; roberta-base does
  not include this in its checkpoint (task-specific). Freshly initialized.

All three are expected and correct. Training from these fresh initializations is
the intended fine-tuning workflow.

---

## Environment

```
Python:        3.12 (venv at /home/gzook/venv/)
PyTorch:       2.12.0
Transformers:  5.8.1
Datasets:      4.8.5
Accelerate:    1.13.0
scikit-learn:  1.8.0
wandb:         0.27.0
MAGPIE data:   /mldata/ece283-sentiment-analyzer/
```

---

## MTL Integration Notes

The floor model is deliberately minimal. When building the MTL model:

1. `model/classifier.py` — replace `self.classifier = nn.Linear(...)` with a
   `nn.ModuleDict` of per-task heads. The encoder (`self.roberta`) is shared.
2. `train.py` `create_optimizer()` — add a param group per task head.
3. `predict.py` `predict()` — add a `task_name` argument to route to the
   correct head.
4. `data/loader.py` — `load_and_split()` returns source metadata in the `source`
   column; use this to assign examples to tasks in the MTL data pipeline.
