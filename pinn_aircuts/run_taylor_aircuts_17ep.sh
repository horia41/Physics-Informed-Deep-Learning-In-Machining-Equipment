#!/bin/bash
# ============================================================================
# MATWI Stage 3 — t3_gated_top25_md30 + Taylor, WITH AIR-CUT GATING (17 epochs)
# ============================================================================
# Same experiment as pinnV2/run_taylorV2_17ep.sh, FUSION half only, but every
# run computes sensor features on the tool-engaged window only (--gate-aircuts:
# the tool-approach/retraction "air-cut" phases are removed, and band energy is
# the length-invariant relative form). Self-contained: every file it needs is
# in this folder (pinn_aircuts/).
#
#   model = t3_gated_top25_md30 (gated fusion, top25, modality dropout 0.3)
#   cells (i // 3):
#     0 fusion ctrl        l0.0   (no physics, air-cut only)
#     1 fusion ceil-T      l0.05  (one-sided ceiling, taylor-slope)
#     2 fusion ceil-T      l0.1
#     3 fusion ceil-T      l0.5
#   seed (i % 3): 0->42 1->43 2->44      4 cells x 3 seeds = 12 tasks (0-11)
#
# RUN ONCE FIRST (login node, CPU, ~1s) to make the constants:
#   cd /scratch-shared/hionescu/project_pinn/pinn_aircuts
#   python fit_taylor.py \
#       --labels-csv /scratch-shared/hionescu/project_pinn/dataset/matwi/labels.csv \
#       --sets-csv   /scratch-shared/hionescu/project_pinn/dataset/matwi/sets.csv \
#       --out        ./taylor_constants.json
#   mkdir -p /scratch-shared/hionescu/project_pinn/pinn_aircuts/runs/aircuts/logs
#   sbatch /scratch-shared/hionescu/project_pinn/pinn_aircuts/run_taylor_aircuts_17ep.sh
# ============================================================================
#
#SBATCH --job-name=taylor_aircuts
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=/scratch-shared/hionescu/project_pinn/pinn_aircuts/runs/aircuts/logs/ac_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/pinn_aircuts/runs/aircuts/logs/ac_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/pinn_aircuts
DATA=$ROOT/dataset/matwi
CONST=$CODE/taylor_constants.json
WARMUP=2; EP=17
if [ ! -f "$CONST" ]; then echo "ERROR: $CONST not found. Run fit_taylor.py first (see header)."; exit 1; fi

# Fusion-only cells (the t3_gated_top25_md30 model), mirroring the fusion half
# of run_taylorV2_17ep.sh. ROLE: ctrl (lambda 0) or ceilT (one-sided ceiling).
CELLS=(
  "ctrl  0.0"
  "ceilT 0.05"
  "ceilT 0.1"
  "ceilT 0.5"
)
SEEDS=(42 43 44)

i=$SLURM_ARRAY_TASK_ID
cell=$(( i / 3 )); s=$(( i % 3 ))
read -r ROLE LAM <<< "${CELLS[$cell]}"
SEED=${SEEDS[$s]}
cd "$CODE"

if [ "$ROLE" = ctrl ]; then
    NAME="t3gated_ctrl_aircuts_e${EP}_s${SEED}"
    PHYS="--lambda-max 0.0"
else
    NAME="t3gated_taylor_ceil_aircuts_l${LAM}_tslope_e${EP}_s${SEED}"
    PHYS="--lambda-max ${LAM} --warmup ${WARMUP} --one-sided --apply-to all --use-taylor-slope"
fi

echo "=== $i | $NAME (gate_aircuts=ON) ==="
srun python train_vision_sensor_taylor.py \
    --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
    --output-dir "$CODE/runs/aircuts/stage3_fusion" --constants "$CONST" \
    --base-exp t3_gated_top25_md30 \
    --name "$NAME" $PHYS \
    --gate-aircuts \
    --epochs "$EP" --seed "$SEED" --num-workers 4
echo "done: $NAME"
