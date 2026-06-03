#!/bin/bash
# ============================================================================
# MATWI Stage 3 — CONFIRMATION run (Snellius)
# ============================================================================
# The two sweep winners + their matched controls, each at {17, 34} epochs and
# seeds {42, 43, 44}.  4 configs × 2 epoch counts × 3 seeds = 24 array tasks.
#
#   config (i//6) : family role  lambda  -> name base
#   ---------------------------------------------------------------
#     0 : vision ctrl 0.0          vision_only_ctrl
#     1 : vision sym  0.5          vis_taylor_sym_l0.5
#     2 : fusion ctrl 0.0          t3gated_ctrl
#     3 : fusion ceil 0.5          t3gated_taylor_ceil_l0.5      <-- main winner
#   epochs (rem//3): 0->17  1->34      seed (rem%3): 0->42 1->43 2->44
#   full name: <base>_e<EP>_s<SEED>
#
# Outputs go to a SEPARATE confirm/ tree (controls ride along, so seed/epoch-
# matched deltas are computable). Submit:
#   mkdir -p /scratch-shared/hionescu/project_pinn/pinn/runs/confirm/logs
#   sbatch /scratch-shared/hionescu/project_pinn/pinn/run_taylor_confirm.sh
# ============================================================================
#
#SBATCH --job-name=taylor_confirm
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-23
#SBATCH --output=/scratch-shared/hionescu/project_pinn/pinn/runs/confirm/logs/cf_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/pinn/runs/confirm/logs/cf_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/pinn
DATA=$ROOT/dataset/matwi
CONST=$CODE/taylor_constants.json
WARMUP=2
if [ ! -f "$CONST" ]; then echo "ERROR: $CONST not found."; exit 1; fi

CFG=( "vision ctrl 0.0" "vision sym 0.5" "fusion ctrl 0.0" "fusion ceil 0.5" )
EPOCHS_OPTS=(17 34)
SEEDS=(42 43 44)

i=$SLURM_ARRAY_TASK_ID
c=$(( i / 6 )); rem=$(( i % 6 )); e=$(( rem / 3 )); s=$(( rem % 3 ))
read -r FAM ROLE LAM <<< "${CFG[$c]}"
EP=${EPOCHS_OPTS[$e]}; SEED=${SEEDS[$s]}

# role -> physics flags + name base
if   [ "$ROLE" = ceil ]; then PHYS="--one-sided --apply-to all"; LTAG="_l${LAM}"
elif [ "$ROLE" = sym  ]; then PHYS="--apply-to all";             LTAG="_l${LAM}"
else                          PHYS="";                           LTAG=""; LAM="0.0"; fi

echo "=== task $i | family=$FAM role=$ROLE lambda=$LAM epochs=$EP seed=$SEED warmup=$WARMUP | node=$(hostname) ==="
cd "$CODE"

if [ "$FAM" = vision ]; then
    [ "$ROLE" = ctrl ] && BASE="vision_only_ctrl" || BASE="vis_taylor_${ROLE}"
    NAME="${BASE}${LTAG}_e${EP}_s${SEED}"
    echo "  -> $NAME"
    python train_vision_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/confirm/stage3_vision" --constants "$CONST" \
        --name "$NAME" --lambda-max "$LAM" --warmup "$WARMUP" $PHYS \
        --backbone efficientnetv2_s --set-range 1-13 --data-loss mse \
        --epochs "$EP" --lr 3e-4 --batch-size 32 --seed "$SEED" --num-workers 4
else
    [ "$ROLE" = ctrl ] && BASE="t3gated_ctrl" || BASE="t3gated_taylor_${ROLE}"
    NAME="${BASE}${LTAG}_e${EP}_s${SEED}"
    echo "  -> $NAME"
    python train_vision_sensor_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/confirm/stage3_fusion" --constants "$CONST" \
        --base-exp t3_gated_top25_md30 \
        --name "$NAME" --lambda-max "$LAM" --warmup "$WARMUP" $PHYS \
        --epochs "$EP" --seed "$SEED" --num-workers 4
fi
echo "done: $NAME"