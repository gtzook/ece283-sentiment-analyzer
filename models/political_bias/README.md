# Political Bias Model

RoBERTa-based classifier/regressor for political bias detection on MAGPIE datasets.

## Files

| File | Purpose |
|---|---|
| [model.py](model.py) | `RoBERTaClassifier` — shared model for classification and regression tasks |
| [train.py](train.py) | Unified training entry point; delegates to `train_baseline` or `train_improved` |
| [train_baseline.py](train_baseline.py) | Standard AdamW fine-tuning |
| [train_improved.py](train_improved.py) | Layerwise LR decay + auxiliary pre-fine-tuning |
| [eval.py](eval.py) | Standalone evaluation against a saved checkpoint |
| [predict.py](predict.py) | Interactive inference REPL and `BiasPredictor` class |
| [config.yaml](config.yaml) | Default hyperparameters |

## Supported Datasets

| ID | Task | Labels |
|---|---|---|
| `10_BABE` | News sentence bias | not biased / biased |
| `72_LIAR` | PolitiFact truthfulness | true / false (or regression) |
| `03_CW_HARD` | Hyperpartisan news | mainstream / hyperpartisan |
| `75_RedditBias` | Reddit ideological bias | unbiased / biased |
| `80_DebateEffects` | Political debate persuasion | not persuasive / persuasive |
| `9_BASIL` | News bias type | lexical bias / informational bias / not biased |
| `19_MultiDimNews` | Multi-dimensional news bias | binary × 4 columns |

## Training

```bash
# Baseline (standard AdamW)
python -m models.political_bias.train --dataset 10_BABE

# Improved (layerwise LR decay + auxiliary pre-training on 25_FakeNewsNet)
python -m models.political_bias.train --mode improved --dataset 10_BABE

# Tune LR decay multiplier (default 0.9)
python -m models.political_bias.train --mode improved --dataset 10_BABE --lr-decay 0.8

# Skip auxiliary pre-training
python -m models.political_bias.train --mode improved --dataset 10_BABE --no-aux

# Specific label column (for multi-label datasets)
python -m models.political_bias.train --dataset 72_LIAR --label-col label_binary
```

Checkpoints are saved to `runs/<dataset>/<label_col>/best.pt` (baseline) or `runs_improved/...` (improved).

### Key Hyperparameters

| Param | Default | Description |
|---|---|---|
| `--lr` | `2e-5` | Base learning rate |
| `--epochs` | `15` | Max training epochs |
| `--batch-size` | `32` | Training batch size |
| `--patience` | `4` | Early stopping patience |
| `--lr-decay` | `0.9` | Per-layer LR multiplier (improved only) |
| `--aux-dataset` | `25_FakeNewsNet` | Auxiliary pre-training dataset (improved only) |
| `--aux-epochs` | `2` | Epochs for auxiliary pre-training (improved only) |

## Evaluation

```bash
python -m models.political_bias.eval \
    --checkpoint runs/10_BABE/label/best.pt \
    --dataset 10_BABE --label-col label

# Evaluate a specific split (train/val/test)
python -m models.political_bias.eval \
    --checkpoint runs/10_BABE/label/best.pt \
    --dataset 10_BABE --split val
```

Metrics are printed to stdout and saved to `eval_<split>.json` next to the checkpoint.

- **Classification**: accuracy, per-class F1, macro F1
- **Regression**: MSE, MAE, Pearson r

## Inference

**Interactive REPL** — paste headlines one at a time:
```bash
python -m models.political_bias.predict
```

**Single headline:**
```bash
python -m models.political_bias.predict \
    --text "Left-wing activists demand sweeping reforms" \
    --checkpoint runs/10_BABE/label/best.pt
```

**Programmatic use:**
```python
from models.political_bias.predict import BiasPredictor

predictor = BiasPredictor("runs/10_BABE/label/best.pt", dataset_id="10_BABE")
result = predictor.predict("Left-wing activists demand sweeping reforms")
# {"prediction": "biased", "confidence": 0.94, "probabilities": {"not biased": 0.06, "biased": 0.94}}

results = predictor.predict_batch(["headline 1", "headline 2"])
```

## Model Architecture

`RoBERTaClassifier` wraps `roberta-base` with a task-appropriate head:

- **Classification**: delegates to `RobertaForSequenceClassification` (pretrained Linear→Tanh pooler + classification head)
- **Regression**: `RobertaModel` + dropout + single linear output; trained with MSE loss

The improved trainer applies **layerwise LR decay** — the classifier head trains at `base_lr`, each successive transformer layer below it gets `base_lr × decay^depth`, down to the embeddings at `base_lr × decay^13`. This preserves general representations in lower layers while allowing task-specific adaptation at the top.
