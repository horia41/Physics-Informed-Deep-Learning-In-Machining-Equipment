#!/bin/bash
# ============================================================================
# MATWI Stage 3 (pinnV3) — 17 epochs
# ============================================================================
# Tests use_taylor_slope=True across lambda {0.05, 0.1, 0.5}, 3 seeds, vs
# matched controls. SAME models / sweep as pinnV2 (vision sym-T, fusion ceil-T,
# base t3_gated_top25_md30) — only the pinnV3 constants (data-fitted, 90um
# anchor) and code differ, so V2-vs-V3 is apples-to-apples on the models.
#
#   cells (i//3):
#     0 vision ctrl (l0)          4 fusion ctrl (l0)
#     1 vision sym-T  l0.05       5 fusion ceil-T l0.05
#     2 vision sym-T  l0.1        6 fusion ceil-T l0.1
#     3 vision sym-T  l0.5        7 fusion ceil-T l0.5
#   seed (i%3): 0->42 1->43 2->44        8 cells x 3 seeds = 24 tasks (0-23)
#
#   # generate constants FIRST, then:
#   mkdir -p /scratch-shared/hionescu/project_pinn/pinnV3/runs/confirm/logs
#   sbatch /scratch-shared/hionescu/project_pinn/pinnV3/run_taylorV3_17ep.sh
# ============================================================================
#
#SBATCH --job-name=taylorV3_17
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-23
#SBATCH --output=/scratch-shared/hionescu/project_pinn/pinnV3/runs/confirm/logs/v3_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/pinnV3/runs/confirm/logs/v3_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/pinnV3
DATA=$ROOT/dataset/matwi
CONST=$CODE/taylor_constants.json
WARMUP=2; EP=17
if [ ! -f "$CONST" ]; then echo "ERROR: $CONST not found. Run fit_taylor.py first."; exit 1; fi

CELLS=(
  "vision ctrl  0.0"
  "vision symT  0.05"
  "vision symT  0.1"
  "vision symT  0.5"
  "fusion ctrl  0.0"
  "fusion ceilT 0.05"
  "fusion ceilT 0.1"
  "fusion ceilT 0.5"
)
SEEDS=(42 43 44)

i=$SLURM_ARRAY_TASK_ID
cell=$(( i / 3 )); s=$(( i % 3 ))
read -r FAM ROLE LAM <<< "${CELLS[$cell]}"
SEED=${SEEDS[$s]}
cd "$CODE"

if [ "$FAM" = vision ]; then
    if [ "$ROLE" = ctrl ]; then
        NAME="vision_only_ctrl_e${EP}_s${SEED}"; PHYS="--lambda-max 0.0"
    else
        NAME="vis_taylor_sym_l${LAM}_tslope_e${EP}_s${SEED}"
        PHYS="--lambda-max ${LAM} --warmup ${WARMUP} --apply-to all --use-taylor-slope"
    fi
    echo "=== $i | $NAME ==="
    python train_vision_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/confirm/stage3_vision" --constants "$CONST" \
        --name "$NAME" $PHYS \
        --backbone efficientnetv2_s --set-range 1-13 --data-loss mse \
        --epochs "$EP" --lr 3e-4 --batch-size 32 --seed "$SEED" --num-workers 4
else
    if [ "$ROLE" = ctrl ]; then
        NAME="t3gated_ctrl_e${EP}_s${SEED}"; PHYS="--lambda-max 0.0"
    else
        NAME="t3gated_taylor_ceil_l${LAM}_tslope_e${EP}_s${SEED}"
        PHYS="--lambda-max ${LAM} --warmup ${WARMUP} --one-sided --apply-to all --use-taylor-slope"
    fi
    echo "=== $i | $NAME ==="
    python train_vision_sensor_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/confirm/stage3_fusion" --constants "$CONST" \
        --base-exp t3_gated_top25_md30 \
        --name "$NAME" $PHYS \
        --seed "$SEED" --num-workers 4
fi
echo "done: $NAME"