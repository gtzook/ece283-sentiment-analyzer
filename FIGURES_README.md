# Performance Figure Generation

## Quick start

```bash
python generate_figures.py
```

Figures are written to `./figures/` as both `.png` (300 dpi) and `.pdf`.

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--data PATH` | `./results.csv` | Path to evaluation results CSV |
| `--output DIR` | `./figures/` | Output directory |
| `--tasks {epistemic,political,emotion} …` | all | Regenerate only a subset of tasks |
| `--models {epistemic,political,emotion,unified_warm,unified_cold} …` | all | Include only specific models |

**Example — regenerate only the epistemic task figures:**
```bash
python generate_figures.py --tasks epistemic
```

---

## Adding new checkpoint data

Append new rows to `results.csv` (one row per model / task / metric / checkpoint / split combination) and re-run the script. No code changes are needed.

```
model_id,task,metric,value,checkpoint_step,split
unified_warm,epistemic,accuracy,0.871,7500,val
...
```

The script automatically detects the latest checkpoint per model and updates all figures, including training curves and the "in-progress" annotations for unified models that have not yet reached the specialist checkpoint count.

---

## Figure descriptions

| File | Type | What it shows |
|---|---|---|
| `radar_per_task` | Radar / spider | All models across all task dimensions at their latest checkpoint. Unified models use dashed/dash-dot lines; a grey ring marks the 1.0 baseline. |
| `bar_latest_checkpoint` | Grouped bar | Side-by-side bars per task for every model at its latest checkpoint. Unified model bars are hatched while training is incomplete. Dashed line at y=1.0 marks the specialist baseline. |
| `training_curves_by_task` | Line chart (3 panels) | Learning curves over training steps, one panel per task. A shaded region and "ongoing" label indicate where unified models have not yet completed training. |
| `heatmap_model_x_task` | Heatmap | 5×3 grid (models × tasks) with a diverging colormap centred at 1.0. Cells for still-training models are marked with `*`. |
| `unified_vs_specialist_scatter` | Scatter | `unified_warm` vs. `unified_cold` relative performance per (task, metric) point. The `y = x` diagonal and the (1.0, 1.0) specialist star serve as reference. |

---

## Normalization

All values in the figures are **relative to the sub-task specialist** on each task:

- Higher-is-better metrics (`accuracy`, `f1`): `relative = model / specialist`
- Lower-is-better metrics (`calibration_error`, `bias_score`): `relative = specialist / model`

A value of 1.0 means the model matches the specialist. Values > 1.0 indicate improvement; values < 1.0 indicate degradation. Baselines are computed from the `test` split of each specialist's final checkpoint.
