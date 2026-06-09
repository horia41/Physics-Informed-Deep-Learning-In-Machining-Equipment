# pinn_hard — HARD physics constraint (RQ3)

Hard-constraint counterpart to the soft `λ·L_physics` penalty in `pinn*/`. Instead
of *penalising* wear-law violations in the loss, the prediction is reparametrised
so violations are **mathematically impossible by construction** — directly
answering project RQ3 ("enforcing physical constraints through architectural
design … making violations of wear laws mathematically impossible").

## The idea
For an image `x` of set `s` at milling pass `t` (ImageID):

```
VB_pred(x,s,t) = VB_taylor(s,t)  +  C · tanh( g_theta(x) )
```

- `g_theta(x)` — the raw scalar output of the **same** EfficientNetV2 vision
  network used everywhere else (`out["wear"]`, an unbounded real number).
- `VB_taylor(s,t)` — a per-sample **physics anchor** (µm): the Taylor-predicted
  expected wear for that set/pass.
- `C` — the band **half-width** (µm). `tanh ∈ (−1,1)`, so the learned correction
  is bounded to `(−C, +C)`.

Because the correction is bounded, the prediction can only ever lie in the band
`[VB_taylor − C, VB_taylor + C]`. No weights `theta` can produce a wear estimate
outside it — that is the hard constraint. Loss is then just the plain data MSE on
`VB_pred`; **there is no λ and no soft penalty** (the physics moved from the loss
into the architecture; adding a penalty too would double-count).

Nice property: at init the head outputs ≈0 ⇒ `tanh≈0` ⇒ `VB_pred ≈ VB_taylor`.
Training **starts at the physics prediction** and learns bounded corrections.

## Three design decisions (why it's built this way)

1. **The anchor is computed from `Vc`, not from a fitted per-set line.**
   A hard constraint is part of the architecture, so `VB_taylor` must exist for
   *every* sample at train **and** eval — including val/test sets {3,4,6,9} and the
   excluded set 1, none of which have a fitted line in `taylor_constants.json`.
   So we use pure physics:
   ```
   k_s = W_f / T_taylor(Vc_s),   T_taylor(Vc_s) = (C_mat / Vc_s)^(1/n_mat)
   VB_taylor(s,t) = clip( k_s · t , 0 , W_f )           [µm],  W_f = 300
   ```
   `k_s` (Taylor wear rate, µm/pass) needs only the set's cutting speed `Vc_s`
   (from `sets.csv`) and the per-material constants `n_mat, C_mat` (fitted on the
   **train** sets). This anchor is **defined for every set** and **leakage-free**
   (it never uses a val/test set's own wear labels, unlike a fitted per-set
   intercept would). A set with no usable `Vc` (e.g. set 1) gets `k_s=0` ⇒ anchor
   0 ⇒ prediction falls back to the pure bounded output `C·tanh(·)`.
   *(Verified: `k_s` reproduces the `taylor_slope_um_per_pass` values in the
   constants file exactly.)*

2. **`C` is the key hyperparameter — choose it deliberately.**
   - Too small → the band can't reach genuine high wear (the anchor clips at
     `W_f=300`, so max reachable wear is `300+C`; you need **C ≥ 150 µm** just to
     reach the 450 µm cap) → underfits exactly the hard adhesion cases.
   - Too large → the band is so wide the constraint is vacuous (≈ unconstrained
     model + offset).
   - Recommended: `--auto-C` sets `C` to the 95th percentile of
     `|wear − VB_taylor|` over the **train** sets (leakage-free). The SLURM sweep
     also tries fixed `C ∈ {100,150,200}` µm.

3. **Two demonstrators — vision AND fusion.** The same `HardTaylorConstraint`
   (modality-agnostic: it only needs the raw output + set_id + ImageID) is applied
   to two `g_theta`:
   - **vision** (`train_vision_hard.py`): `g_theta(image)` — strongest baseline
     (vision-only, 19 µm), cleanest RQ3 demonstrator.
   - **fusion** (`train_vision_sensor_hard.py`): `g_theta(image, sensor)` — the
     gated `t3_gated_top25_md30` model, so the bounded correction is driven by
     both modalities. Consistent with the rest of Stage 3; directly comparable to
     the soft fusion result (pinnV2 ceil-T λ0.5 = −1.38). Uses the clean
     (non-air-cut) DatasetClass so the comparison is fair.
   In both, omit `--hard` to get the unconstrained control from the *same* code
   (reproduces `vision_only_ctrl` / `t3gated_ctrl`) for a clean in-folder A/B.

## Files
| File | Role |
|---|---|
| `taylor_hard.py` | `HardTaylorConstraint`: builds `k_s` from constants + `sets.csv`, reparametrises the output, `recommend_C()` (shared by both trainers) |
| `train_vision_hard.py` | **vision** trainer; loads the Stage-1 vision base from `../vision-only/` |
| `train_vision_sensor_hard.py` | **fusion** trainer; loads the local fusion base (`train_vision_sensorV2.py` + model + dataset) |
| `train_vision_sensorV2.py`, `modelVisionSensorV2.py`, `DatasetClass_VisionSensors.py` | fusion base (copied from `vision-sensor/improve_attempt/`, clean / no air-cut mods) |
| `fit_taylor.py`, `taylor_constants.json` | calibration (copied from pinnV2, identical) |
| `run_taylor_hard_17ep.sh` | SLURM 0–11: vision ctrl + C∈{100,150,200} × 3 seeds |
| `run_taylor_hard_fusion_17ep.sh` | SLURM 0–11: fusion ctrl + C∈{100,150,200} × 3 seeds |
| `aggregate_hard.py` | mean±std + Δ vs ctrl, grouped by C (handles vision + fusion) |

All trainers: `--hard`/`--C-um`/`--auto-C`; loss = MSE on the constrained output;
log correction size + tanh saturation (and gate value for fusion).

## Run (Snellius)
```bash
cd $ROOT/pinn_hard
# constants are already present (copy of pinnV2's); to refit:
# python fit_taylor.py --labels-csv ../dataset/matwi/labels.csv --sets-csv ../dataset/matwi/sets.csv --out ./taylor_constants.json

# vision
mkdir -p runs/hard/logs        && sbatch run_taylor_hard_17ep.sh
# fusion
mkdir -p runs/hard_fusion/logs && sbatch run_taylor_hard_fusion_17ep.sh

# when done:
python aggregate_hard.py runs/hard/stage3_vision
python aggregate_hard.py runs/hard_fusion/stage3_fusion
```

## What to read in the results
- **Δ vs ctrl** (in `aggregate_hard.py`) and **Δ vs the SOFT result**
  (pinnV2 vision sym-T λ0.5, overall −2.58) → the head-to-head for RQ3.
- **`sat` / `corr` columns in the per-epoch log**: if the tanh saturates often
  (`sat_frac` high) the band is binding hard ⇒ `C` is too small and the
  constraint is hurting; if `corr_abs_um` ≈ small and MAE ≈ ctrl, the constraint
  is inert ⇒ `C` is effectively too large. The sweet spot is in between.

## Caveats
- Pre-existing: the material constants were fitted including sets 12/13 (RVS uses
  literature-n anyway). Not introduced here; shared with the soft runs so the
  comparison stays fair.
- This is a symmetric band. A one-sided *hard ceiling* (forbid only
  overprediction, matching the soft ceiling that worked best) is a natural
  variant — replace `C·tanh(·)` with a non-positive bounded correction.
