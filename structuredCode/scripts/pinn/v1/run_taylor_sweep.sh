#!/bin/bash
# ============================================================================
# MATWI Stage 3 — Taylor lambda SWEEP (Snellius)
# ============================================================================
# Both families x both penalty shapes x lambda {0.1, 0.2, 0.5}, warmup=2.
# 12 array tasks (one GPU each). Outputs land in the EXISTING family dirs so
# the aggregator deltas them against the already-present lambda=0 controls;
# the lambda is encoded in each run's NAME (and echoed in its log).
#
#   index : family variant lambda            -> run name
#   ---------------------------------------------------------------
#     0-2 : vision sym  {0.1,0.2,0.5}         vis_taylor_sym_l*
#     3-5 : vision ceil {0.1,0.2,0.5}         vis_taylor_ceil_l*
#     6-8 : fusion sym  {0.1,0.2,0.5}         t3gated_taylor_sym_l*
#    9-11 : fusion ceil {0.1,0.2,0.5}         t3gated_taylor_ceil_l*
#
# Before submitting (sweep logs live in their own folder):
#   mkdir -p /scratch-shared/hionescu/project_pinn/pinn/runs/stage3_sweep/logs
#   sbatch /scratch-shared/hionescu/project_pinn/pinn/run_taylor_sweep.sh
# After: re-run the aggregator on the two family dirs — it now shows the full
# lambda ladder (0, 0.05, 0.1, 0.2, 0.5) with deltas vs control.
# ============================================================================
#
#SBATCH --job-name=taylor_sweep
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --output=/scratch-shared/hionescu/project_pinn/pinn/runs/stage3_sweep/logs/sweep_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/pinn/runs/stage3_sweep/logs/sweep_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/pinn
DATA=$ROOT/dataset/matwi
CONST=$CODE/taylor_constants.json
WARMUP=2

if [ ! -f "$CONST" ]; then echo "ERROR: $CONST not found (run fit_taylor.py first)."; exit 1; fi

SPECS=(
  "vision sym 0.1"  "vision sym 0.2"  "vision sym 0.5"
  "vision ceil 0.1" "vision ceil 0.2" "vision ceil 0.5"
  "fusion sym 0.1"  "fusion sym 0.2"  "fusion sym 0.5"
  "fusion ceil 0.1" "fusion ceil 0.2" "fusion ceil 0.5"
)
read -r FAM VAR LAM <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"

if [ "$VAR" = "ceil" ]; then PHYS="--one-sided --apply-to all"; else PHYS="--apply-to all"; fi

echo "=== task $SLURM_ARRAY_TASK_ID | family=$FAM variant=$VAR lambda=$LAM warmup=$WARMUP | node=$(hostname) ==="
cd "$CODE"

if [ "$FAM" = "vision" ]; then
    NAME="vis_taylor_${VAR}_l${LAM}"
    echo "  -> $NAME"
    python train_vision_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/stage3_vision" --constants "$CONST" \
        --name "$NAME" --lambda-max "$LAM" --warmup "$WARMUP" $PHYS \
        --backbone efficientnetv2_s --set-range 1-13 --data-loss mse \
        --epochs 17 --lr 3e-4 --batch-size 32 --seed 42 --num-workers 4
else
    NAME="t3gated_taylor_${VAR}_l${LAM}"
    echo "  -> $NAME"
    python train_vision_sensor_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/stage3_fusion" --constants "$CONST" \
        --base-exp t3_gated_top25_md30 \
        --name "$NAME" --lambda-max "$LAM" --warmup "$WARMUP" $PHYS \
        --seed 42 --num-workers 4
fi
echo "done: $NAME"