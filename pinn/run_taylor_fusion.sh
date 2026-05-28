#!/bin/bash
# ============================================================================
# MATWI Stage 3 — Taylor physics on the BEST FUSION model (Snellius)
# base = t3_gated_top25_md30 (gated fusion, top25, modality dropout 0.3)
# ============================================================================
# Single seed, 3-way controlled comparison (array 0-2, one GPU each):
#   0  t3gated_ctrl            lambda=0.0   (fusion control, reproduces ~21.4 µm)
#   1  t3gated_taylor_sym      lambda=0.05  symmetric, all materials
#   2  t3gated_taylor_ceil_rvs lambda=0.05  one-sided ceiling, RVS only
#
# Needs taylor_constants.json (fit once on the login node — see run_taylor.sh).
#   mkdir -p /scratch-shared/hionescu/project_pinn/pinn/runs/stage3_fusion/logs
#   sbatch /scratch-shared/hionescu/project_pinn/pinn/run_taylor_fusion.sh
# Submit in parallel with run_taylor.sh.
# ============================================================================
#
#SBATCH --job-name=taylor_fus
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch-shared/hionescu/project_pinn/pinn/runs/stage3_fusion/logs/fus_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/pinn/runs/stage3_fusion/logs/fus_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/pinn
DATA=$ROOT/dataset/matwi
OUT=$CODE/runs/stage3_fusion
CONST=$CODE/taylor_constants.json
mkdir -p "$OUT"

if [ ! -f "$CONST" ]; then
    echo "ERROR: $CONST not found. Run fit_taylor.py on the login node first."; exit 1
fi

NAMES=(  t3gated_ctrl  t3gated_taylor_sym  t3gated_taylor_ceil_rvs )
LAMBDA=( 0.0           0.05                0.05 )
EXTRA=(  ""            "--apply-to all"    "--one-sided --apply-to rvs" )

i=$SLURM_ARRAY_TASK_ID
echo "=== task $i -> ${NAMES[$i]} on $(hostname) ==="
cd "$CODE"
srun python train_vision_sensor_taylor.py \
    --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
    --output-dir "$OUT" --constants "$CONST" \
    --base-exp t3_gated_top25_md30 \
    --name "${NAMES[$i]}" --lambda-max "${LAMBDA[$i]}" ${EXTRA[$i]} \
    --seed 42 --num-workers 4
echo "done: ${NAMES[$i]}"
