#!/bin/bash
# ============================================================================
# MATWI Stage 3 — CONFIRMATION run (Snellius)
# ============================================================================
# The two sweep winners + their matched controls, each at {17, 34} epochs and
# seeds {42, 43, 44}.  4 configs × 2 epoch counts × 3 seeds = 24 array tasks.
#
#   config (i//6) : family role  lambda  -> name base
#   ---------------------------------------------------------------
#     0 : vision ctrl 0.0          vision_only_ctrl
#     1 : vision sym  0.5          vis_taylor_sym_l0.5
#     2 : fusion ctrl 0.0          t3gated_ctrl
#     3 : fusion ceil 0.5          t3gated_taylor_ceil_l0.5      <-- main winner
#   epochs (rem//3): 0->17  1->34      seed (rem%3): 0->42 1->43 2->44
#   full name: <base>_e<EP>_s<SEED>
#
# Outputs go to a SEPARATE confirm/ tree (controls ride along, so seed/epoch-
# matched deltas are computable). Submit:
#   mkdir -p /scratch-shared/hionescu/project_pinn/pinn/runs/confirm/logs
#   sbatch /scratch-shared/hionescu/project_pinn/pinn/run_taylor_confirm.sh
# ============================================================================
#
#SBATCH --job-name=taylor_confirm
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-23
#SBATCH --output=/scratch-shared/hionescu/project_pinn/pinn/runs/confirm/logs/cf_%A_%a.out
#SBATCH --error=/scratch-shared/hionescu/project_pinn/pinn/runs/confirm/logs/cf_%A_%a.err

set -euo pipefail
module load 2023 Python/3.11.3-GCCcore-12.3.0
source /scratch-shared/hionescu/project_pinn/my_env/bin/activate

ROOT=/scratch-shared/hionescu/project_pinn
CODE=$ROOT/pinn
DATA=$ROOT/dataset/matwi
CONST=$CODE/taylor_constants.json
WARMUP=2
if [ ! -f "$CONST" ]; then echo "ERROR: $CONST not found."; exit 1; fi

CFG=( "vision ctrl 0.0" "vision sym 0.5" "fusion ctrl 0.0" "fusion ceil 0.5" )
EPOCHS_OPTS=(17 34)
SEEDS=(42 43 44)

i=$SLURM_ARRAY_TASK_ID
c=$(( i / 6 )); rem=$(( i % 6 )); e=$(( rem / 3 )); s=$(( rem % 3 ))
read -r FAM ROLE LAM <<< "${CFG[$c]}"
EP=${EPOCHS_OPTS[$e]}; SEED=${SEEDS[$s]}

# role -> physics flags + name base
if   [ "$ROLE" = ceil ]; then PHYS="--one-sided --apply-to all"; LTAG="_l${LAM}"
elif [ "$ROLE" = sym  ]; then PHYS="--apply-to all";             LTAG="_l${LAM}"
else                          PHYS="";                           LTAG=""; LAM="0.0"; fi

echo "=== task $i | family=$FAM role=$ROLE lambda=$LAM epochs=$EP seed=$SEED warmup=$WARMUP | node=$(hostname) ==="
cd "$CODE"

if [ "$FAM" = vision ]; then
    [ "$ROLE" = ctrl ] && BASE="vision_only_ctrl" || BASE="vis_taylor_${ROLE}"
    NAME="${BASE}${LTAG}_e${EP}_s${SEED}"
    echo "  -> $NAME"
    python train_vision_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/confirm/stage3_vision" --constants "$CONST" \
        --name "$NAME" --lambda-max "$LAM" --warmup "$WARMUP" $PHYS \
        --backbone efficientnetv2_s --set-range 1-13 --data-loss mse \
        --epochs "$EP" --lr 3e-4 --batch-size 32 --seed "$SEED" --num-workers 4
else
    [ "$ROLE" = ctrl ] && BASE="t3gated_ctrl" || BASE="t3gated_taylor_${ROLE}"
    NAME="${BASE}${LTAG}_e${EP}_s${SEED}"
    echo "  -> $NAME"
    python train_vision_sensor_taylor.py \
        --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
        --output-dir "$CODE/runs/confirm/stage3_fusion" --constants "$CONST" \
        --base-exp t3_gated_top25_md30 \
        --name "$NAME" --lambda-max "$LAM" --warmup "$WARMUP" $PHYS \
        --epochs "$EP" --seed "$SEED" --num-workers 4
fi
echo "done: $NAME"
[hionescu@int6 project_pinn]$ ls -lh pinn
total 100K
-rw-r----- 1 hionescu hionescu 4.2K May 28 17:29 aggregate_confirm.py
-rw-r----- 1 hionescu hionescu 3.4K May 28 15:10 aggregate_stage3.py
-rw-r----- 1 hionescu hionescu  13K May 28 13:21 fit_taylor.py
drwxr-x--- 2 hionescu hionescu 4.0K May 28 13:38 __pycache__
drwxr-x--- 6 hionescu hionescu 4.0K May 28 16:10 runs
-rw-r----- 1 hionescu hionescu 3.7K May 28 16:12 run_taylor_confirm.sh
-rw-r----- 1 hionescu hionescu 2.4K May 28 14:06 run_taylor_fusion.sh
-rw-r----- 1 hionescu hionescu 3.5K May 28 15:25 run_taylor_sweep.sh
-rw-r----- 1 hionescu hionescu 3.0K May 28 18:57 run_taylor_tslope_lowlam.sh
-rw-r----- 1 hionescu hionescu 3.1K May 28 17:30 run_taylor_tslope.sh
-rw-r----- 1 hionescu hionescu 4.2K May 28 13:27 taylor_constants.json
-rw-r----- 1 hionescu hionescu 6.5K May 28 13:22 taylor_physics_loss.py
-rw-r----- 1 hionescu hionescu  13K May 28 16:11 train_vision_sensor_taylor.py
-rw-r----- 1 hionescu hionescu  14K May 28 13:22 train_vision_taylor.py
[hionescu@int6 project_pinn]$ cat pinn/run_taylor_fusion.sh
#!/bin/bash
# ============================================================================
# MATWI Stage 3 — Taylor physics on the BEST FUSION model (Snellius)
# base = t3_gated_top25_md30 (gated fusion, top25, modality dropout 0.3)
# ============================================================================
# Single seed, 3-way controlled comparison (array 0-2, one GPU each):
#   0  t3gated_ctrl            lambda=0.0   (fusion control, reproduces ~21.4 µm)
#   1  t3gated_taylor_sym      lambda=0.05  symmetric, all materials
#   2  t3gated_taylor_ceil lambda=0.05  one-sided ceiling (CK45 train data; RVS is val/test only)
#
# Needs taylor_constants.json (fit once on the login node — see run_taylor.sh).
#   mkdir -p /scratch-shared/hionescu/project_pinn/pinn/runs/stage3_fusion/logs
#   sbatch /scratch-shared/hionescu/project_pinn/pinn/run_taylor_fusion.sh
# Submit in parallel with run_taylor.sh.
# ============================================================================
#
#SBATCH --job-name=taylor_fus
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
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

NAMES=(  t3gated_ctrl  t3gated_taylor_sym  t3gated_taylor_ceil )
LAMBDA=( 0.0           0.05                0.05 )
EXTRA=(  ""            "--apply-to all"    "--one-sided --apply-to all" )

i=$SLURM_ARRAY_TASK_ID
echo "=== task $i -> ${NAMES[$i]} on $(hostname) ==="
cd "$CODE"
python train_vision_sensor_taylor.py \
    --data-dir "$DATA" --labels-csv "$DATA/labels.csv" --sets-csv "$DATA/sets.csv" \
    --output-dir "$OUT" --constants "$CONST" \
    --base-exp t3_gated_top25_md30 \
    --name "${NAMES[$i]}" --lambda-max "${LAMBDA[$i]}" ${EXTRA[$i]} \
    --seed 42 --num-workers 4
echo "done: ${NAMES[$i]}"