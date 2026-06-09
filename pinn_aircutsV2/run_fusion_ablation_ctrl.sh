#!/bin/bash
# ============================================================================
# MATWI Stage 3 — AIR-CUT × FUSION-MODE ablation (control only, 17 epochs)
# ============================================================================
# Question this answers: does wavelet air-cut removal help a NON-gated fusion
# model, and was the learned gate + modality-dropout masking the air-cut effect
# in t3_gated_top25_md30? Also cleanly separates the detector effect from the
# absolute->relative band-energy change (both arms use this folder's relative
# energy, so gate-vs-no-gate is a pure detector A/B).
#
# 3 x 2 design (3 fusion models x {ungated, gated}), control (lambda 0) only,
# 3 seeds = 18 tasks (0-17):
#   cell (i // 3):
#     0  gated_md30     ungated   (t3_gated_top25_md30,        no --gate-aircuts)
#     1  gated_md30     gated     (t3_gated_top25_md30,        --gate-aircuts)
#     2  interm_md30    ungated   (t3_intermediate_top25_md30, no --gate-aircuts)
#     3  interm_md30    gated     (t3_intermediate_top25_md30, --gate-aircuts)
#     4  interm_simple  ungated   (intermediate_top25,         no --gate-aircuts)
#     5  interm_simple  gated     (intermediate_top25,         --gate-aircuts)
#   seed (i % 3): 0->42 1->43 2->44
#
# The three fusion rows form a progression that isolates each mechanism:
#   gated_md30    = learned gate + modality-dropout 0.3
#   interm_md30   = plain intermediate concat + modality-dropout 0.3  (drop the gate)
#   interm_simple = plain intermediate concat, NO dropout            (the original
#                   Stage-2 vision-sensor model; == t3_intermediate_top25)
# So gated_md30 vs interm_md30 isolates the GATE; interm_md30 vs interm_simple
# isolates modality DROPOUT; and within each row the ungated-vs-gated column is a
# clean wavelet-detector A/B (both share this folder's relative band energy).
#
# COMPUTE-LEAN OPTIONS:
#   sbatch --array=12-17 run_fusion_ablation_ctrl.sh   # ONLY the simple model (cells 4,5)
#   sbatch --array=6-17  run_fusion_ablation_ctrl.sh   # both intermediate models, skip gated
# (cells 0,1 reproduce the gated model; skip them if you trust the prior 26.70).
#
# RUN ONCE FIRST (login node) — same constants as the main run; needed even for
# control because train_vision_sensor_taylor.py loads them (physics is just off):
#   cd /scratch-shared/hionescu/project_pinn/pinn_aircutsV2
#   python fit_taylor.py \
#       --labels-csv /scratch-shared/hionescu/project_pinn/dataset/matwi/labels.csv \
#       --sets-csv   /scratch-shared/hionescu/project_pinn/dataset/matwi/sets.csv \
#       --out        ./taylor_constants.json
#   mkdir -p /scratch-shared/hionescu/project_pinn/pinn_aircutsV2/runs/fusion_ablation/logs
#   sbatch /scratch-shared/hionescu/project_pinn/pinn_aircutsV2/run_fusion_ablation_ctrl.sh
#
# AGGREGATE (each cell is its own dir/"family", so they don't collapse):
#   cd /scratch-shared/hionescu/project_pinn/pinn_aircutsV2
#   python aggregate_confirm.py \
#       runs/fusion_ablation/gated_md30_ungated \
#       runs/fusion_ablation/gated_md30_gated \
#       runs/fusion_ablation/interm_md30_ungated \
#       runs/fusion_ablation/interm_md30_gated \
#       runs/fusion_ablation/interm_simple_ungated \
#       runs/fusion_ablation/interm_simple_gated
#   # compare the six 'overall' means (the Δ column is self-vs-self here = 0).
# ============================================================================
#
#SBATCH --job-name=fusion_ablation
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-17
#SBATCH --output=/scratch-shared/hionescu/project_pinn/pinn_aircutsV2/runs/fusion_ablation/logs/fa_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/pinn_aircutsV2/runs/fusion_ablation/logs/fa_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/pinn_aircutsV2
DATA=$ROOT/dataset/matwi
CONST=$CODE/taylor_constants.json
EP=17
if [ ! -f "$CONST" ]; then echo "ERROR: $CONST not found. Run fit_taylor.py first (see header)."; exit 1; fi

# Each cell: TAG  BASE_EXP  GATE(on/off). All control (lambda 0).
CELLS=(
  "gated_md30     t3_gated_top25_md30          off"
  "gated_md30     t3_gated_top25_md30          on"
  "interm_md30    t3_intermediate_top25_md30   off"
  "interm_md30    t3_intermediate_top25_md30   on"
  "interm_simple  intermediate_top25           off"
  "interm_simple  intermediate_top25           on"
)
SEEDS=(42 43 44)

i=$SLURM_ARRAY_TASK_ID
cell=$(( i / 3 )); s=$(( i % 3 ))
read -r TAG BASE_EXP GATE <<< "${CELLS[$cell]}"
SEED=${SEEDS[$s]}
cd "$CODE"

if [ "$GATE" = on ]; then
    GATESTATE=gated;   GATEFLAG="--gate-aircuts"
else
    GATESTATE=ungated; GATEFLAG=""
fi

OUTDIR="$CODE/runs/fusion_ablation/${TAG}_${GATESTATE}"
NAME="${TAG}_${GATESTATE}_ctrl_e${EP}_s${SEED}"

echo "=== $i | $NAME | base=$BASE_EXP gate=$GATE ==="
srun python train_vision_sensor_taylor.py \
    --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
    --output-dir "$OUTDIR" --constants "$CONST" \
    --base-exp "$BASE_EXP" \
    --name "$NAME" \
    --lambda-max 0.0 \
    $GATEFLAG \
    --epochs "$EP" --seed "$SEED" --num-workers 4
echo "done: $NAME"
