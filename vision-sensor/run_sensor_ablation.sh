#!/bin/bash
#SBATCH --job-name=sensor_fusion
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-12
#SBATCH --output=/scratch-shared/your_username/name_of_project_folder/runs/sensor_ablation/logs/sensor_%A_%a.out
#SBATCH --error=/scratch-shared/your_username/name_of_project_folder/runs/sensor_ablation/logs/sensor_%A_%a.err


CODE_DIR="/scratch-shared/your_username/name_of_project_folder/vision-sensor"
DATA_DIR="/scratch-shared/your_username/name_of_project_folder/dataset/matwi"
OUTPUT_DIR="/scratch-shared/your_username/name_of_project_folder/runs/sensor_ablation"

module load 2023 Python/3.11.3-GCCcore-12.3.0 # here again whatever version of python you loaded into your environment, I loaded the 2023 3.11.3 version

source /scratch-shared/your_username/name_of_project_folder/name_of_environment_folder/bin/activate

# ── Experiment mapping ────────────────────────────────────────────────────────
EXPERIMENTS=(
    "vision_only_647"            # 0
    "early_raw25"                # 1
    "early_top25"                # 2
    "early_all40"                # 3
    "intermediate_raw25"         # 4
    "intermediate_top25"         # 5
    "intermediate_all40"         # 6
    "late_raw25"                 # 7
    "late_top25"                 # 8
    "late_all40"                 # 9
    "intermediate_top25_gated"   # 10  air-cut ablation
    "early_top25_gated"          # 11  air-cut ablation
    "intermediate_all40_gated"   # 12  air-cut ablation
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

python3 train_vision_sensor.py \
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
