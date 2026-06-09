#!/bin/bash
# ============================================================================
# MATWI — Vision baselines, 3 seeds, BOTH populations (Snellius)
# ============================================================================
# Re-runs the two headline vision-only configs over seeds {42,43,44} on BOTH
# the full 664 set and the 647 multimodal subset (--require-sensors), so they
# slot directly into the same mean±std comparison as the Stage-3 / fusion runs.
#
#   config (i//6) :
#     0 : efficientnetv2_dataset_MSE   (simple head, dataset norm, MSE, lr 3e-4)
#     1 : resnet50_imagenet_L1         (simple head, imagenet norm, L1,  lr 3e-4)
#   population ((i/3)%2) : 0 -> 664 (full)   1 -> 647 (--require-sensors)
#   seed (i%3)           : 0 -> 42   1 -> 43   2 -> 44
#   2 configs × 2 pops × 3 seeds = 12 array tasks (0-11)
#
# Epochs (17), batch size, wear cap etc. are baked into the ExperimentConfig.
# Outputs: runs/baselines_3seed/vision_<POP>/s<SEED>/<config>/results.json
#
# Submit:
#   mkdir -p /scratch-shared/hionescu/project_pinn/runs/baselines_3seed/logs
#   sbatch /scratch-shared/hionescu/project_pinn/vision-only/run_vision_baselines_3seed.sh
# ============================================================================
#SBATCH --job-name=vis_base3
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-11
#SBATCH --output=/scratch-shared/hionescu/project_pinn/runs/baselines_3seed/logs/vis_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/runs/baselines_3seed/logs/vis_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/vision-only
DATA=$ROOT/dataset/matwi
OUTROOT=$ROOT/runs/baselines_3seed

CONFIGS=( "efficientnetv2_dataset_MSE" "resnet50_imagenet_L1" )
SEEDS=(42 43 44)

i=$SLURM_ARRAY_TASK_ID
c=$(( i / 6 ))          # 0..1  config
p=$(( (i / 3) % 2 ))   # 0=664, 1=647
s=$(( i % 3 ))         # 0..2  seed index
CFG=${CONFIGS[$c]}
SEED=${SEEDS[$s]}

if [ "$p" -eq 1 ]; then POP=647; REQ="--require-sensors"; else POP=664; REQ=""; fi

OUTDIR=$OUTROOT/vision_${POP}/s${SEED}
mkdir -p "$OUTDIR" "$OUTROOT/logs"

echo "=== task $i | config=$CFG | pop=$POP | seed=$SEED | node=$(hostname) | $(date) ==="
cd "$CODE"

python3 train_vision.py \
    --data-dir   "$DATA" \
    --labels-csv "$DATA/labels.csv" \
    --sets-csv   "$DATA/sets.csv" \
    --output-dir "$OUTDIR" \
    --only       "$CFG" $REQ \
    --num-workers 4 \
    --seed        "$SEED" \
    --device      auto

echo "Finished task $i at $(date)"
