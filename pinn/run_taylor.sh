#!/bin/bash
# ============================================================================
# MATWI Stage 3 — Taylor physics on the VISION-ONLY model (Snellius)
# ============================================================================
# Single seed, 3-way controlled comparison (array 0-2, one GPU each):
#   0  vision_only_ctrl       lambda=0.0   (control, no physics)
#   1  vis_taylor_sym         lambda=0.05  symmetric, all materials
#   2  vis_taylor_ceil_rvs    lambda=0.05  one-sided ceiling, RVS only
#
# RUN ONCE ON THE LOGIN NODE FIRST (CPU, ~1 s):
#   source /scratch-shared/hionescu/project_pinn/my_env/bin/activate
#   python /scratch-shared/hionescu/project_pinn/pinn/fit_taylor.py \
#       --labels-csv /scratch-shared/hionescu/project_pinn/dataset/matwi/labels.csv \
#       --sets-csv   /scratch-shared/hionescu/project_pinn/dataset/matwi/sets.csv \
#       --out        /scratch-shared/hionescu/project_pinn/pinn/taylor_constants.json
#   mkdir -p /scratch-shared/hionescu/project_pinn/pinn/runs/stage3_vision/logs
#   sbatch /scratch-shared/hionescu/project_pinn/pinn/run_taylor.sh
# ============================================================================
#
#SBATCH --job-name=taylor_vis
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch-shared/hionescu/project_pinn/pinn/runs/stage3_vision/logs/vis_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/pinn/runs/stage3_vision/logs/vis_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/pinn
DATA=$ROOT/dataset/matwi
OUT=$CODE/runs/stage3_vision
CONST=$CODE/taylor_constants.json
mkdir -p "$OUT"

if [ ! -f "$CONST" ]; then
    echo "ERROR: $CONST not found. Run fit_taylor.py on the login node first (see header)."; exit 1
fi

NAMES=(  vision_only_ctrl  vis_taylor_sym    vis_taylor_ceil_rvs )
LAMBDA=( 0.0               0.05              0.05 )
EXTRA=(  ""                "--apply-to all"  "--one-sided --apply-to rvs" )

i=$SLURM_ARRAY_TASK_ID
echo "=== task $i -> ${NAMES[$i]} on $(hostname) ==="
cd "$CODE"
srun python train_vision_taylor.py \
    --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
    --output-dir "$OUT" --constants "$CONST" \
    --name "${NAMES[$i]}" --lambda-max "${LAMBDA[$i]}" ${EXTRA[$i]} \
    --backbone efficientnetv2_s --set-range 1-13 --data-loss mse \
    --epochs 17 --lr 3e-4 --batch-size 32 --seed 42 --num-workers 4
echo "done: ${NAMES[$i]}"
