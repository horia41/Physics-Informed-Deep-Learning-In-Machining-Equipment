#!/bin/bash
#SBATCH --job-name=sensor_check
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch-shared/your_username/name_of_project_folder/runs/sensor_check/sensor_check_%j.out
#SBATCH --error=/scratch-shared/your_username/name_of_project_folder/runs/sensor_check/sensor_check_%j.err

CODE_DIR="/scratch-shared/your_username/name_of_project_folder/vision-sensor"
DATA_DIR="/scratch-shared/your_username/name_of_project_folder/dataset/matwi"
OUTPUT_DIR="/scratch-shared/your_username/name_of_project_folder/runs/sensor_check"

module load 2023 Python/3.11.3-GCCcore-12.3.0 # here again whatever version of python you loaded into your environment, I loaded the 2023 3.11.3 version

source /scratch-shared/your_username/name_of_project_folder/name_of_environment_folder/bin/activate

mkdir -p "${OUTPUT_DIR}"

cd "${CODE_DIR}"

python3 sensor_check.py \
    --data-dir    "${DATA_DIR}" \
    --labels-csv  "${DATA_DIR}/labels.csv" \
    --sets-csv    "${DATA_DIR}/sets.csv" \
    --set-range   1-13

echo ""
echo "Done at $(date)"