#!/bin/bash
#SBATCH --job-name=matwi_vision
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-8
#SBATCH --output=/scratch-shared/your_username/name_of_project_folder/runs/vision_ablation/logs/vision_%A_%a.out
#SBATCH --error=/scratch-shared/your_username/name_of_project_folder/runs/vision_ablation/logs/vision_%A_%a.err

# ══════════════════════════════════════════════════════════════════════════════
# MATWI Vision Baseline Ablation — Snellius SLURM Array Job
# ══════════════════════════════════════════════════════════════════════════════
#
# Runs 9 experiments as a SLURM array (one GPU per experiment, in parallel):
#   0: resnet50_imagenet_L1
#   1: resnet50_imagenet_MSE
#   2: resnet50_dataset_L1
#   3: resnet50_dataset_MSE
#   4: efficientnetv2_imagenet_L1
#   5: efficientnetv2_imagenet_MSE
#   6: efficientnetv2_dataset_L1
#   7: efficientnetv2_dataset_MSE
#   8: efficientnetv2_imagenet_L1_mlp_sched  (best practices reference)
#
# Usage:
#   mkdir -p logs
#   sbatch run_vision_ablation.sh
#
# After all jobs finish, gather results:
#   python train_vision.py \
#     --data-dir $DATA_DIR --labels-csv $DATA_DIR/labels.csv \
#     --sets-csv $DATA_DIR/sets.csv --output-dir $OUTPUT_DIR \
#     --list-experiments
#
# Or re-run the comparison summary only (no training):
#   # Results are already in $OUTPUT_DIR/<experiment_name>/results.json
# ══════════════════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────────────────
CODE_DIR="/scratch-shared/your_username/name_of_project_folder/vision-only"
DATA_DIR="/scratch-shared/your_username/name_of_project_folder/dataset/matwi"
OUTPUT_DIR="/scratch-shared/your_username/name_of_project_folder/runs/vision_ablation"

# ── Environment setup ─────────────────────────────────────────────────────────

module load 2023 Python/3.11.3-GCCcore-12.3.0 # here again whatever version of python you loaded into your environment, I loaded the 2023 3.11.3 version

source /scratch-shared/your_username/name_of_project_folder/name_of_environment_folder/bin/activate

# ── Experiment mapping ────────────────────────────────────────────────────────
EXPERIMENTS=(
    "resnet50_imagenet_L1"
    "resnet50_imagenet_MSE"
    "resnet50_dataset_L1"
    "resnet50_dataset_MSE"
    "efficientnetv2_imagenet_L1"
    "efficientnetv2_imagenet_MSE"
    "efficientnetv2_dataset_L1"
    "efficientnetv2_dataset_MSE"
    "efficientnetv2_imagenet_L1_mlp_sched"
)

EXPERIMENT_NAME="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

echo "========================================"
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Array task     : $SLURM_ARRAY_TASK_ID"
echo "  Experiment     : $EXPERIMENT_NAME"
echo "  Node           : $SLURMD_NODENAME"
echo "  GPU            : $CUDA_VISIBLE_DEVICES"
echo "  Date           : $(date)"
echo "========================================"

# ── Create output dirs ────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"

# ── Run ───────────────────────────────────────────────────────────────────────
cd "${CODE_DIR}"

python3 train_vision.py \
    --data-dir    "$DATA_DIR" \
    --labels-csv  "$DATA_DIR/labels.csv" \
    --sets-csv    "$DATA_DIR/sets.csv" \
    --output-dir  "$OUTPUT_DIR" \
    --only        "$EXPERIMENT_NAME" \
    --num-workers 4 \
    --seed        42 \
    --device      auto \
    --log-every   10

echo ""
echo "Finished: $EXPERIMENT_NAME at $(date)"
echo "Results:  $OUTPUT_DIR/$EXPERIMENT_NAME/results.json"