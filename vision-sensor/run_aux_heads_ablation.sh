#!/bin/bash
#SBATCH --job-name=aux_heads_fusion
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/scratch-shared/your_username/name_of_project_folder/runs/aux_heads_ablation/logs/aux_%A_%a.out
#SBATCH --error=/scratch-shared/your_username/name_of_project_folder/runs/aux_heads_ablation/logs/aux_%A_%a.err

# Base paths on cluster
CODE_DIR="/scratch-shared/your_username/name_of_project_folder/vision-sensor"
DATA_DIR="/scratch-shared/your_username/name_of_project_folder/dataset/matwi"
OUTPUT_DIR="/scratch-shared/your_username/name_of_project_folder/runs/aux_heads_ablation"

module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/your_username/name_of_project_folder/name_of_environment_folder/bin/activate

# Task-3 recommended starting point: intermediate + top25.
BASE_EXP="intermediate_top25"

# Alpha sweep, beta fixed.
ALPHAS=(0.1 0.3 0.5)
BETA=0.3

ALPHA="${ALPHAS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="${BASE_EXP}_aux_a${ALPHA}_b${BETA}"

mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/${RUN_NAME}"

cd "${CODE_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Array index : ${SLURM_ARRAY_TASK_ID}"
echo "  Base exp    : ${BASE_EXP}"
echo "  Alpha       : ${ALPHA}"
echo "  Beta        : ${BETA}"
echo "  Run name    : ${RUN_NAME}"
echo "  Node        : $(hostname)"
echo "  Started     : $(date)"
echo "════════════════════════════════════════════════════════════════"

python3 train_vision_sensor.py \
    --data-dir "${DATA_DIR}" \
    --labels-csv "${DATA_DIR}/labels.csv" \
    --sets-csv "${DATA_DIR}/sets.csv" \
    --output-dir "${OUTPUT_DIR}" \
    --only "${BASE_EXP}" \
    --num-workers 4 \
    --seed 42 \
    --enable-sensor-aux-head \
    --enable-vision-aux-head \
    --alpha-sensor-aux "${ALPHA}" \
    --beta-vision-aux "${BETA}"

echo ""
echo "  Finished : $(date)"
echo "════════════════════════════════════════════════════════════════"
