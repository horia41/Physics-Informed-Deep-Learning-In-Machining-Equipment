#!/bin/bash
# ============================================================================
# MATWI Stage 3 — use_taylor_slope=TRUE at LOWER lambda (Snellius)
# ============================================================================
# Question: was lambda=0.5 the reason the Taylor-slope variant degraded (esp.
# vision sym-T -> 62.86 overall)? Re-run the two -T winners at small lambda.
#   both winners x lambda {0.01,0.02,0.1,0.2} x seeds {42,43,44} = 24 tasks
#   34 epochs, warmup=2, OUTPUT into confirm/ (names tagged _tslope) so they
#   sit beside the existing ctrl / ceil / ceil-T(0.5) / sym / sym-T(0.5) rows.
#
#   index: cfg=i/12 (0=fusion ceil-T, 1=vision sym-T)
#          rem=i%12  lam=rem/3 (->{0.01,0.02,0.1,0.2})  seed=rem%3
#
#   sbatch /scratch-shared/hionescu/project_pinn/pinn/run_taylor_tslope_lowlam.sh
# ============================================================================
#
#SBATCH --job-name=tslope_lowlam
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-23
#SBATCH --output=/scratch-shared/hionescu/project_pinn/pinn/runs/confirm/logs/tslolam_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/pinn/runs/confirm/logs/tslolam_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/pinn
DATA=$ROOT/dataset/matwi
CONST=$CODE/taylor_constants.json
WARMUP=2; EP=34
if [ ! -f "$CONST" ]; then echo "ERROR: $CONST not found."; exit 1; fi

LAMBDAS=(0.01 0.02 0.1 0.2)
SEEDS=(42 43 44)

i=$SLURM_ARRAY_TASK_ID
cfg=$(( i / 12 )); rem=$(( i % 12 )); l=$(( rem / 3 )); s=$(( rem % 3 ))
LAM=${LAMBDAS[$l]}; SEED=${SEEDS[$s]}

echo "=== task $i | use_taylor_slope=TRUE | cfg=$cfg lambda=$LAM epochs=$EP seed=$SEED | node=$(hostname) ==="
cd "$CODE"

if [ "$cfg" -eq 0 ]; then
    NAME="t3gated_taylor_ceil_l${LAM}_tslope_e${EP}_s${SEED}"
    echo "  -> $NAME"
    python train_vision_sensor_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/confirm/stage3_fusion" --constants "$CONST" \
        --base-exp t3_gated_top25_md30 \
        --name "$NAME" --lambda-max "$LAM" --warmup "$WARMUP" \
        --one-sided --apply-to all --use-taylor-slope \
        --epochs "$EP" --seed "$SEED" --num-workers 4
else
    NAME="vis_taylor_sym_l${LAM}_tslope_e${EP}_s${SEED}"
    echo "  -> $NAME"
    python train_vision_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/confirm/stage3_vision" --constants "$CONST" \
        --name "$NAME" --lambda-max "$LAM" --warmup "$WARMUP" \
        --apply-to all --use-taylor-slope \
        --backbone efficientnetv2_s --set-range 1-13 --data-loss mse \
        --epochs "$EP" --lr 3e-4 --batch-size 32 --seed "$SEED" --num-workers 4
fi
echo "done: $NAME"