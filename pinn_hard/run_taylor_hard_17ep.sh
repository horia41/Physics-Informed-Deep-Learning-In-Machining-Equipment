#!/bin/bash
# ============================================================================
# MATWI Stage 3 — HARD Taylor constraint on the VISION model (17 epochs)
# ============================================================================
# Hard constraint (taylor_hard.py): VB_pred = VB_taylor(s,t) + C*tanh(g_theta(x)),
# so predictions are structurally confined to the band [VB_taylor +/- C] (RQ3).
# Loss is plain data MSE on the constrained output (physics is in the
# architecture, no lambda). Self-contained except the Stage-1 vision base, which
# is loaded from ../vision-only/train_vision.py (present in the repo), exactly
# like pinnV2/run_taylorV2_17ep.sh's vision half.
#
#   cells (i // 3):
#     0 ctrl (unconstrained, reproduces vision_only_ctrl)
#     1 hard C=100 um
#     2 hard C=150 um   (>= wear_cap-failure=150 -> can reach the 450um cap)
#     3 hard C=200 um
#   seed (i % 3): 0->42 1->43 2->44     4 cells x 3 seeds = 12 tasks (0-11)
#
# Compare the hard_* runs vs the cell-0 ctrl (in-folder A/B) and vs the SOFT
# result (pinnV2 vision sym-T l0.5) to answer RQ3: hard vs soft constraint.
#
# RUN ONCE FIRST if you do not already have constants (login node, ~1s):
#   cd /scratch-shared/hionescu/project_pinn/pinn_hard
#   python fit_taylor.py --labels-csv ../dataset/matwi/labels.csv \
#       --sets-csv ../dataset/matwi/sets.csv --out ./taylor_constants.json
#   # (a copy of pinnV2/taylor_constants.json is already included and identical)
#   mkdir -p /scratch-shared/hionescu/project_pinn/pinn_hard/runs/hard/logs
#   sbatch /scratch-shared/hionescu/project_pinn/pinn_hard/run_taylor_hard_17ep.sh
# ============================================================================
#
#SBATCH --job-name=taylor_hard
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-11
#SBATCH --output=/scratch-shared/hionescu/project_pinn/pinn_hard/runs/hard/logs/hd_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/pinn_hard/runs/hard/logs/hd_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/pinn_hard
DATA=$ROOT/dataset/matwi
CONST=$CODE/taylor_constants.json
EP=17
if [ ! -f "$CONST" ]; then echo "ERROR: $CONST not found. Run fit_taylor.py first (see header)."; exit 1; fi

# ROLE: ctrl (no constraint) or hard with band half-width C (um).
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
    NAME="vision_only_ctrl_e${EP}_s${SEED}"; HARD=""
else
    NAME="hard_C${C}_e${EP}_s${SEED}"; HARD="--hard --C-um ${C}"
fi

echo "=== $i | $NAME ==="
srun python train_vision_hard.py \
    --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
    --output-dir "$CODE/runs/hard/stage3_vision" --constants "$CONST" \
    --name "$NAME" $HARD \
    --backbone efficientnetv2_s --set-range 1-13 --data-loss mse \
    --epochs "$EP" --lr 3e-4 --batch-size 32 --seed "$SEED" --num-workers 4
echo "done: $NAME"
