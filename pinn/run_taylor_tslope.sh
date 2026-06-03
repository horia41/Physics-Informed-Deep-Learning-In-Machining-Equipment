#!/bin/bash
# ============================================================================
# MATWI Stage 3 — use_taylor_slope=True check (Snellius)
# ============================================================================
# Drives the physics loss from the FITTED Taylor constants (slope = W_f/T_taylor,
# T_taylor=(C/Vc)^(1/n)) instead of each set's observed mid-life slope. Single
# comparison cell against the confirmation runs you already have:
#   - the two winners only, λ=0.5, 34 epochs, seeds {42,43,44}
#   - written into the SAME confirm/ dirs, names tagged _tslope, so the existing
#     controls + observed-slope versions sit beside them for a 3-way compare.
# Controls are NOT re-run (use_taylor_slope is a no-op at λ=0).
#
#   index : family variant   -> name
#     0-2 : fusion ceil(λ0.5) t3gated_taylor_ceil_l0.5_tslope_e34_s{42,43,44}
#     3-5 : vision sym (λ0.5) vis_taylor_sym_l0.5_tslope_e34_s{42,43,44}
#
#   mkdir -p /scratch-shared/hionescu/project_pinn/pinn/runs/confirm/logs
#   sbatch /scratch-shared/hionescu/project_pinn/pinn/run_taylor_tslope.sh
# ============================================================================
#
#SBATCH --job-name=taylor_tslope
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch-shared/hionescu/project_pinn/pinn/runs/confirm/logs/tslope_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/pinn/runs/confirm/logs/tslope_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/pinn
DATA=$ROOT/dataset/matwi
CONST=$CODE/taylor_constants.json
WARMUP=2; EP=34
if [ ! -f "$CONST" ]; then echo "ERROR: $CONST not found."; exit 1; fi

SEEDS=(42 43 44)
i=$SLURM_ARRAY_TASK_ID
c=$(( i / 3 )); s=$(( i % 3 ))
SEED=${SEEDS[$s]}

echo "=== task $i | use_taylor_slope=TRUE | epochs=$EP seed=$SEED | node=$(hostname) ==="
cd "$CODE"

if [ "$c" -eq 0 ]; then
    NAME="t3gated_taylor_ceil_l0.5_tslope_e${EP}_s${SEED}"
    echo "  -> $NAME"
    python train_vision_sensor_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/confirm/stage3_fusion" --constants "$CONST" \
        --base-exp t3_gated_top25_md30 \
        --name "$NAME" --lambda-max 0.5 --warmup "$WARMUP" \
        --one-sided --apply-to all --use-taylor-slope \
        --epochs "$EP" --seed "$SEED" --num-workers 4
else
    NAME="vis_taylor_sym_l0.5_tslope_e${EP}_s${SEED}"
    echo "  -> $NAME"
    python train_vision_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/confirm/stage3_vision" --constants "$CONST" \
        --name "$NAME" --lambda-max 0.5 --warmup "$WARMUP" \
        --apply-to all --use-taylor-slope \
        --backbone efficientnetv2_s --set-range 1-13 --data-loss mse \
        --epochs "$EP" --lr 3e-4 --batch-size 32 --seed "$SEED" --num-workers 4
fi
echo "done: $NAME"