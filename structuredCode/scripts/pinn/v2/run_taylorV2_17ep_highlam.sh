#!/bin/bash
# ============================================================================
# MATWI Stage 3 (pinnV2, fixed constants) — HIGH-lambda add-on, 17 epochs
# ============================================================================
# Extends the lambda ladder upward for the two Taylor-slope winners. No
# controls (already have them from run_taylorV2_17ep.sh). Writes into the SAME
# pinnV2 confirm/ dirs so aggregate_confirm.py folds these into the existing
# 0.05/0.1/0.5 rows.
#
#   cells (i//3):
#     0 vision sym-T  l0.75     3 fusion ceil-T l0.75
#     1 vision sym-T  l1.0      4 fusion ceil-T l1.0
#     2 vision sym-T  l1.5      5 fusion ceil-T l1.5
#   seed (i%3): 0->42 1->43 2->44      6 cells x 3 seeds = 18 tasks (0-17)
#
#   sbatch /scratch-shared/hionescu/project_pinn/pinnV2/run_taylorV2_17ep_highlam.sh
# (runs/confirm/logs already exists from the previous batch)
# ============================================================================
#
#SBATCH --job-name=taylorV2_hi
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-17
#SBATCH --output=/scratch-shared/hionescu/project_pinn/pinnV2/runs/confirm/logs/v2hi_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/pinnV2/runs/confirm/logs/v2hi_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/pinnV2
DATA=$ROOT/dataset/matwi
CONST=$CODE/taylor_constants.json
WARMUP=2; EP=17
if [ ! -f "$CONST" ]; then echo "ERROR: $CONST not found."; exit 1; fi

LAMBDAS=(0.75 1.0 1.5)
SEEDS=(42 43 44)

i=$SLURM_ARRAY_TASK_ID
cell=$(( i / 3 )); s=$(( i % 3 ))
fam=$(( cell / 3 ))          # 0 = vision sym-T, 1 = fusion ceil-T
l=$(( cell % 3 ))
LAM=${LAMBDAS[$l]}; SEED=${SEEDS[$s]}
cd "$CODE"

if [ "$fam" -eq 0 ]; then
    NAME="vis_taylor_sym_l${LAM}_tslope_e${EP}_s${SEED}"
    echo "=== $i | $NAME ==="
    python train_vision_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/confirm/stage3_vision" --constants "$CONST" \
        --name "$NAME" --lambda-max "$LAM" --warmup "$WARMUP" \
        --apply-to all --use-taylor-slope \
        --backbone efficientnetv2_s --set-range 1-13 --data-loss mse \
        --epochs "$EP" --lr 3e-4 --batch-size 32 --seed "$SEED" --num-workers 4
else
    NAME="t3gated_taylor_ceil_l${LAM}_tslope_e${EP}_s${SEED}"
    echo "=== $i | $NAME ==="
    python train_vision_sensor_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/confirm/stage3_fusion" --constants "$CONST" \
        --base-exp t3_gated_top25_md30 \
        --name "$NAME" --lambda-max "$LAM" --warmup "$WARMUP" \
        --one-sided --apply-to all --use-taylor-slope \
        --seed "$SEED" --num-workers 4
fi
echo "done: $NAME"