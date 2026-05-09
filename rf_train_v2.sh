#!/bin/bash
#SBATCH --job-name=rf_train_v2
#SBATCH --account=DELL-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32          # RF uses n_jobs=-1; give it plenty of cores
#SBATCH --mem=128G                  # texture features on large images are RAM-heavy
#SBATCH --time=12:00:00
#SBATCH --output=/rds/user/zx335/hpc-work/RandomForest/Test1/v2_logs/random_forest_classification_v2_%j.out

module purge
module load rhel8/default-amp
module load python/3.11.0-icl

source /rds/user/zx335/hpc-work/RandomForest/Test1/rf_venv/bin/activate

mkdir -p /rds/user/zx335/hpc-work/RandomForest/Test1/v2_logs

python /rds/user/zx335/hpc-work/RandomForest/Test1/random_forest_classification_v2.py
