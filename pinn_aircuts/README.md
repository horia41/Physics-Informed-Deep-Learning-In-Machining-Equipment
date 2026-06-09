# pinn_aircuts — t3_gated_top25_md30 + Taylor, WITH air-cut gating

> **Version lineage:** built on **`pinnV2`** (the canonical soft-constraint Taylor
> version). The single added variable is **air-cut removal** on the sensor features.
> Outcome: for the *fusion* model this did **not** beat the no-air-cut control
> (see the recorded numbers in `what_we_need_here_&_results.txt`). The follow-up
> **`pinn_aircutsV2`** revalidates the air-cut window with a diagnostic
> (`check_aircuts.py`) and a cleaner control ablation. See the root
> [`README.md`](../README.md) §5.3.1–5.3.2 for the full air-cut story.

Self-contained replica of the **fusion half** of `pinnV2/run_taylorV2_17ep.sh`
(model `t3_gated_top25_md30`, 17 epochs, 3 seeds), but every run computes sensor
features on the **tool-engaged window only** (air cuts removed). Everything
needed lives in this folder — nothing is imported from sibling folders.

## What "air cuts" means / what changed
Each ~50–60 s sensor recording starts with the tool approaching the workpiece
(no contact) and ends with it retracting; ~28–46 % of every CSV is this silent
"air cut". With `--gate-aircuts`, a **wavelet edge-detector** (`wavelet_cut_window`
— db4 level-4 DWT on the vibration + acoustic channels) finds the single cutting
window and trims the approach/retract phases before feature extraction. Band-energy
features are also the **relative** form (`band / total power`) so they are
length-invariant — required because gated windows vary in length. The 40-feature
layout, `top25` selection, and the model are otherwise unchanged, so this is the
**same experiment with exactly one variable added: air-cut removal.**

## Files
| File | Role |
|---|---|
| `DatasetClass_VisionSensors.py` | multimodal dataset **+ `gate_aircuts` flag** (`in_cut_mask`, relative band energy) |
| `modelVisionSensorV2.py` | gated-fusion model + modality dropout (unchanged from pinnV2) |
| `train_vision_sensorV2.py` | task-3 grid incl. `t3_gated_top25_md30` **+ `gate_aircuts` plumbing** |
| `taylor_physics_loss.py` | frozen-constant Taylor penalty (incl. GPU device fix) |
| `fit_taylor.py` | offline Taylor calibration → `taylor_constants.json` |
| `train_vision_sensor_taylor.py` | fusion + Taylor trainer **+ `--gate-aircuts`** |
| `run_taylor_aircuts_17ep.sh` | SLURM array (0–11): 4 cells × 3 seeds |

## How to run (Snellius)
```bash
# 0) Edit the paths in run_taylor_aircuts_17ep.sh (username, ROOT, DATA, venv).

# 1) ONCE on the login node — make the Taylor constants (CPU, ~1s):
cd $ROOT/pinn_aircuts
python fit_taylor.py \
    --labels-csv $ROOT/dataset/matwi/labels.csv \
    --sets-csv   $ROOT/dataset/matwi/sets.csv \
    --out        ./taylor_constants.json

# 2) Submit the 12-task array:
mkdir -p $ROOT/pinn_aircuts/runs/aircuts/logs
sbatch run_taylor_aircuts_17ep.sh
```

The array maps `i // 3` → cell, `i % 3` → seed (42/43/44):

| cell | run | λ | physics |
|---|---|---|---|
| 0 | `t3gated_ctrl_aircuts` | 0.0 | none (air-cut only) |
| 1 | `t3gated_taylor_ceil_aircuts` | 0.05 | one-sided ceiling, taylor-slope |
| 2 | `t3gated_taylor_ceil_aircuts` | 0.1 | one-sided ceiling, taylor-slope |
| 3 | `t3gated_taylor_ceil_aircuts` | 0.5 | one-sided ceiling, taylor-slope |

Results land in `runs/aircuts/stage3_fusion/<name>/results.json`. Compare the
`*_aircuts_*` runs against the matching non-air-cut runs from
`pinnV2/run_taylorV2_17ep.sh` to isolate the air-cut effect.

## To run a single config manually
```bash
python train_vision_sensor_taylor.py \
    --data-dir $DATA --labels-csv $DATA/labels.csv --sets-csv $DATA/sets.csv \
    --output-dir runs/aircuts/stage3_fusion --constants taylor_constants.json \
    --base-exp t3_gated_top25_md30 --gate-aircuts \
    --lambda-max 0.05 --warmup 2 --one-sided --apply-to all --use-taylor-slope \
    --epochs 17 --seed 42
```
Drop `--gate-aircuts` to get the ungated baseline from the same code.
