#!/bin/bash
#SBATCH --job-name=v2_impr
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch-shared/hionescu/project_pinn/vision-sensor/improve_attempt/logs/v2_impr_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/vision-sensor/improve_attempt/logs/v2_impr_%A_%a.err

# ══════════════════════════════════════════════════════════════════════════════
# Task 3 — improving vision+sensor fusion WITHOUT Taylor's equation
# ══════════════════════════════════════════════════════════════════════════════
# 2×2 controlled grid: modality dropout {0.0, 0.3} × fusion {intermediate, gated}
# All on top25 features. Compare against vision_only_647 reference = 22.4 µm.
#
# Before submitting (logs dir must exist, and timm weights must be cached):
#   mkdir -p /scratch-shared/hionescu/project_pinn/runs/task3_fusion/logs
#   sbatch run_task3_fusion.sh
#
# After all 4 jobs finish, aggregate:
#   python gather_results_sensor.py \
#     --output-dir /scratch-shared/hionescu/project_pinn/runs/task3_fusion
# ══════════════════════════════════════════════════════════════════════════════

CODE_DIR="/scratch-shared/hionescu/project_pinn/vision-sensor/improve_attempt"
DATA_DIR="/scratch-shared/hionescu/project_pinn/dataset/matwi"
OUTPUT_DIR="/scratch-shared/hionescu/project_pinn/vision-sensor/improve_attempt"

module load 2023 Python/3.11.3-GCCcore-12.3.0

source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

# ── Experiment mapping ────────────────────────────────────────────────────────
EXPERIMENTS=(
    "t3_intermediate_top25"        # 0  intermediate, no dropout (task-3 anchor)
    "t3_intermediate_top25_md30"   # 1  intermediate, 30% modality dropout
    "t3_gated_top25"               # 2  gated fusion, no dropout
    "t3_gated_top25_md30"          # 3  gated fusion, 30% modality dropout
)

EXP_NAME="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "════════════════════════════════════════════════════════════════"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Array index : ${SLURM_ARRAY_TASK_ID}"
echo "  Experiment  : ${EXP_NAME}"
echo "  Node        : $(hostname)"
echo "  GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Started     : $(date)"
echo "════════════════════════════════════════════════════════════════"

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/${EXP_NAME}"

# ── Run experiment ────────────────────────────────────────────────────────────
cd "${CODE_DIR}"

python3 train_vision_sensorV2.py \
    --data-dir   "${DATA_DIR}" \
    --labels-csv "${DATA_DIR}/labels.csv" \
    --sets-csv   "${DATA_DIR}/sets.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --only       "${EXP_NAME}" \
    --num-workers 4 \
    --seed 42

echo ""
echo "  Finished : $(date)"
echo "════════════════════════════════════════════════════════════════"
