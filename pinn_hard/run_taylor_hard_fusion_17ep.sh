#!/bin/bash
# ============================================================================
# MATWI Stage 3 — HARD Taylor constraint on the FUSION model (17 epochs)
# ============================================================================
# g_theta = t3_gated_top25_md30 (gated fusion, image + top25 sensor features,
# modality dropout 0.3). VB_pred = VB_taylor(s,t) + C*tanh(g_theta(image,sensor)),
# so the bounded correction is driven by BOTH modalities. Loss = data MSE on the
# constrained output (no lambda). Uses the clean (non-air-cut) DatasetClass, so
# results are directly comparable to the pinnV2 SOFT fusion runs (ceil-T l0.5).
# Self-contained: all fusion files + taylor_hard.py live in this folder.
#
#   cells (i // 3):
#     0 ctrl (unconstrained fusion, reproduces t3gated_ctrl)
#     1 hard C=100 um
#     2 hard C=150 um
#     3 hard C=200 um
#   seed (i % 3): 0->42 1->43 2->44     4 cells x 3 seeds = 12 tasks (0-11)
#
#   mkdir -p /scratch-shared/hionescu/project_pinn/pinn_hard/runs/hard_fusion/logs
#   sbatch /scratch-shared/hionescu/project_pinn/pinn_hard/run_taylor_hard_fusion_17ep.sh
# ============================================================================
#
#SBATCH --job-name=hard_fusion
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=/scratch-shared/hionescu/project_pinn/pinn_hard/runs/hard_fusion/logs/hf_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/pinn_hard/runs/hard_fusion/logs/hf_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/pinn_hard
DATA=$ROOT/dataset/matwi
CONST=$CODE/taylor_constants.json
EP=17
if [ ! -f "$CONST" ]; then echo "ERROR: $CONST not found. Run fit_taylor.py first."; exit 1; fi

CELLS=(
  "ctrl 0"
  "hard 100"
  "hard 150"
  "hard 200"
)
SEEDS=(42 43 44)

i=$SLURM_ARRAY_TASK_ID
cell=$(( i / 3 )); s=$(( i % 3 ))
read -r ROLE C <<< "${CELLS[$cell]}"
SEED=${SEEDS[$s]}
cd "$CODE"

if [ "$ROLE" = ctrl ]; then
    NAME="t3gated_ctrl_e${EP}_s${SEED}"; HARD=""
else
    NAME="t3gated_hard_C${C}_e${EP}_s${SEED}"; HARD="--hard --C-um ${C}"
fi

echo "=== $i | $NAME ==="
srun python train_vision_sensor_hard.py \
    --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
    --output-dir "$CODE/runs/hard_fusion/stage3_fusion" --constants "$CONST" \
    --base-exp t3_gated_top25_md30 \
    --name "$NAME" $HARD \
    --epochs "$EP" --seed "$SEED" --num-workers 4
echo "done: $NAME"
