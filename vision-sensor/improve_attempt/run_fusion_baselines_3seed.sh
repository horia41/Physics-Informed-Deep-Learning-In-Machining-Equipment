#!/bin/bash
# ============================================================================
# MATWI — Fusion baselines, 3 seeds (Snellius)
# ============================================================================
# Re-runs the two fusion configs of interest over seeds {42,43,44} so they get
# mean±std comparable to the vision baselines and the Stage-3 Taylor runs.
# Both run on the 647-sample multimodal subset (the only one with sensors).
#
#   config (i//3) :
#     0 : t3_intermediate_top25     (intermediate fusion, top25, no mod-dropout)
#     1 : t3_gated_top25_md30       (gated fusion, top25, modality dropout 0.3)
#   seed (i%3)    : 0 -> 42  1 -> 43  2 -> 44
#   2 configs × 3 seeds = 6 array tasks (0-5)
#
# Epochs (17), lr 3e-4, MSE, dataset norm etc. are baked into the config.
# Outputs: runs/baselines_3seed/fusion/s<SEED>/<config>/results.json
#
# Submit:
#   mkdir -p /scratch-shared/hionescu/project_pinn/runs/baselines_3seed/logs
#   sbatch /scratch-shared/hionescu/project_pinn/vision-sensor/improve_attempt/run_fusion_baselines_3seed.sh
# ============================================================================
#SBATCH --job-name=fus_base3
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch-shared/hionescu/project_pinn/runs/baselines_3seed/logs/fus_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/runs/baselines_3seed/logs/fus_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/vision-sensor/improve_attempt
DATA=$ROOT/dataset/matwi
OUTROOT=$ROOT/runs/baselines_3seed

CONFIGS=( "t3_intermediate_top25" "t3_gated_top25_md30" )
SEEDS=(42 43 44)

i=$SLURM_ARRAY_TASK_ID
c=$(( i / 3 ))   # 0..1  config
s=$(( i % 3 ))   # 0..2  seed index
CFG=${CONFIGS[$c]}
SEED=${SEEDS[$s]}

OUTDIR=$OUTROOT/fusion/s${SEED}
mkdir -p "$OUTDIR" "$OUTROOT/logs"

echo "=== task $i | config=$CFG | seed=$SEED | node=$(hostname) | $(date) ==="
cd "$CODE"

python3 train_vision_sensorV2.py \
    --data-dir   "$DATA" \
    --labels-csv "$DATA/labels.csv" \
    --sets-csv   "$DATA/sets.csv" \
    --output-dir "$OUTDIR" \
    --only       "$CFG" \
    --num-workers 4 \
    --seed        "$SEED" \
    --device      auto

echo "Finished task $i at $(date)"
