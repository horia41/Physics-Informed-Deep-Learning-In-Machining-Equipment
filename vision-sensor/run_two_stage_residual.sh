#!/bin/bash
#SBATCH --job-name=two_stage_residual
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch-shared/your_username/name_of_project_folder/runs/two_stage_residual/logs/two_stage_%j.out
#SBATCH --error=/scratch-shared/your_username/name_of_project_folder/runs/two_stage_residual/logs/two_stage_%j.err

# -----------------------------------------------------------------------------
# One-call runner for two-stage residual fusion:
#   Stage A (vision-only) + Stage B (sensor residual) + final eval
# -----------------------------------------------------------------------------

CODE_DIR="/scratch-shared/your_username/name_of_project_folder/vision-sensor"
DATA_DIR="/scratch-shared/your_username/name_of_project_folder/dataset/matwi"
OUTPUT_DIR="/scratch-shared/your_username/name_of_project_folder/runs/two_stage_residual"

module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/your_username/name_of_project_folder/name_of_environment_folder/bin/activate

mkdir -p "${OUTPUT_DIR}/logs"

# -----------------------------
# Configurable run parameters
# -----------------------------
RUN_NAME="two_stage_residual_top25_sensors_only"
FEATURE_SET="top25"             # raw25 | top25 | all40
RESIDUAL_INPUT="sensors_only"   # sensors_only | hybrid
MODALITY_DROPOUT_P="0.0"        # e.g. 0.0 / 0.1 / 0.2

STAGE_A_EPOCHS="17"
STAGE_B_EPOCHS="17"
BATCH_SIZE="32"
LR_STAGE_A="3e-4"
LR_STAGE_B="3e-4"
SEED="42"

# Optional reuse of a pre-trained Stage A checkpoint:
# STAGE_A_CKPT="/scratch-shared/your_username/name_of_project_folder/runs/some_run/stage_a/best_model.pt"
# SKIP_STAGE_A="--skip-stage-a-train"
STAGE_A_CKPT=""
SKIP_STAGE_A=""

cd "${CODE_DIR}"

echo "======================================================================="
echo "  Job ID      : ${SLURM_JOB_ID}"
echo "  Node        : $(hostname)"
echo "  Started     : $(date)"
echo "  Run name    : ${RUN_NAME}"
echo "  Feature set : ${FEATURE_SET}"
echo "  Residual in : ${RESIDUAL_INPUT}"
echo "  Dropout p   : ${MODALITY_DROPOUT_P}"
echo "======================================================================="

CMD=(
  python3 train_two_stage_residual.py
  --data-dir "${DATA_DIR}"
  --labels-csv "${DATA_DIR}/labels.csv"
  --sets-csv "${DATA_DIR}/sets.csv"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --feature-set "${FEATURE_SET}"
  --residual-input "${RESIDUAL_INPUT}"
  --modality-dropout-p "${MODALITY_DROPOUT_P}"
  --stage-a-epochs "${STAGE_A_EPOCHS}"
  --stage-b-epochs "${STAGE_B_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr-stage-a "${LR_STAGE_A}"
  --lr-stage-b "${LR_STAGE_B}"
  --num-workers 4
  --seed "${SEED}"
)

if [[ -n "${STAGE_A_CKPT}" ]]; then
  CMD+=(--stage-a-checkpoint "${STAGE_A_CKPT}")
fi

if [[ -n "${SKIP_STAGE_A}" ]]; then
  CMD+=(${SKIP_STAGE_A})
fi

printf 'Running command:\n%s\n\n' "${CMD[*]}"
"${CMD[@]}"

echo ""
echo "  Finished : $(date)"
echo "======================================================================="
