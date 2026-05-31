# Unified Model

A single shared RoBERTa-base encoder trained jointly on three tasks: epistemic uncertainty, political bias, and emotional framing. All three heads run in one forward pass, unlike the ensemble in `src/unified_analyzer.py` which loads three independent models sequentially.

## Architecture

```
Input
  └── Encoder (RoBERTa-base, shared)
        ├── SentenceHead  → epistemic sentence label  (3-class: asserted / hedged / speculative)
        ├── TokenHead     → epistemic token label     (binary: certain / uncertain)
        ├── BiasHead      → political bias            (classification or regression)
        └── EmotionHead   → emotional framing         (11-label multi-hot)
```

Each forward pass specifies a single task via the `task=` argument. The encoder runs once; the appropriate head produces logits and, if labels are provided, a loss.

`SentenceHead` and `TokenHead` are shared directly from `models/epistemic/model.py`. `BiasHead` and `EmotionHead` replicate the pooler layers from the original single-task models so that pre-trained checkpoints can be warm-started without weight key remapping.

## Files

| File | Purpose |
|---|---|
| [model.py](model.py) | `UnifiedModel` + `BiasHead` + `EmotionHead` definitions |
| [train.py](train.py) | Multi-task training loop (round-robin over three DataLoaders) |
| [eval.py](eval.py) | Comparison table: unified model vs. three individual baselines |
| [predict.py](predict.py) | `load_unified_predictor`, `predict_all`, and per-task inference functions |
| [config.yaml](config.yaml) | All hyperparameters |

## Training

```bash
python -m models.unified.train --config models/unified/config.yaml

# Verify the data pipeline without running a full epoch
python -m models.unified.train --config models/unified/config.yaml --dry-run
```

Each epoch round-robins one batch from each of the three task loaders. The loader with the most batches sets the epoch length; shorter loaders cycle so no data is wasted.

Checkpoints are saved to `runs/unified/<timestamp>/best.pt` (best composite val score) and `last.pt`.

### Key Hyperparameters

| Param | Default | Description |
|---|---|---|
| `encoder_lr` | `2e-5` | Learning rate for the shared encoder |
| `head_lr` | `1e-4` | Learning rate for all task heads |
| `epochs` | `5` | Training epochs |
| `batch_size` | `16` | Per-task batch size (effective 32 with `grad_accum_steps: 2`) |
| `lambda_token` | `0.3` | Weight of epistemic token-head loss relative to sentence-head loss |
| `task_weights.emotion` | `0.5` | Down-weighted because the emotion dataset is ~10× larger |

The best checkpoint is selected on a composite score — the mean macro-F1 across all three validation tasks.

## Evaluation

Compares the unified model against three individual baseline checkpoints and prints a side-by-side table.

```bash
python -m models.unified.eval \
    --unified-checkpoint  runs/unified/20260601_120000/best.pt \
    --epistemic-checkpoint runs/20260530_071702/best.pt \
    --bias-checkpoint      runs/10_BABE/label/best.pt \
    --emotion-checkpoint   checkpoints/emotional_framing_floor \
    --config               models/unified/config.yaml
```

Baseline checkpoints are optional — omit any to show `nan` in that column. Results are also saved to `eval_comparison.json` next to the unified checkpoint.

Metrics per task:

| Task | Metrics |
|---|---|
| Epistemic | sent macro-F1, ECE |
| Bias | macro-F1, accuracy |
| Emotion | macro-F1, micro-F1, hamming loss |

## Inference

```python
from models.unified.predict import load_unified_predictor, predict_all

predictor = load_unified_predictor(
    checkpoint="runs/unified/20260601_120000/best.pt",
    config="models/unified/config.yaml",
)

results = predict_all(predictor, ["The drug may help some patients."])
# [{"text": ..., "epistemic": {...}, "bias": {...}, "emotion": {...}}]
```

Per-task functions are also available if you only need one head:

```python
from models.unified.predict import predict_epistemic, predict_bias, predict_emotion

predict_epistemic(predictor, ["text"])
# [{"label": 1, "label_name": "hedged", "uncertainty_score": 0.42, "sent_probs": [...]}]

predict_bias(predictor, ["text"])
# [{"prediction": "biased", "confidence": 0.87, "probabilities": {...}}]

predict_emotion(predictor, ["text"])
# [{"anger": 0, "joy": 1, ..., "scores": {"anger": 0.12, "joy": 0.91, ...}}]
```

**CLI:**
```bash
python -m models.unified.predict \
    --checkpoint runs/unified/20260601_120000/best.pt \
    --texts "Left-wing activists demand reform" \
    --task all   # or: epistemic | bias | emotion
```
