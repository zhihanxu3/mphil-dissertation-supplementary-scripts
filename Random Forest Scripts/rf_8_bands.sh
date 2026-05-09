#!/bin/bash
# ============================================================
# Random Forest Classification (8 bands) — HPC SLURM Batch Script
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
# Description:
#   Submits random_forest_classification_8_bands.py as a CPU batch
#   job on the HPC cluster. Allocates 32 cores and 128 GB RAM
#   to accommodate the parallel Random Forest training and
#   the memory demands of texture feature computation on large
#   PlanetScope imagery.
#   Submit via: sbatch rf_8_bands.sh
#
# Important — before submitting:
#   Open random_forest_classification_8_bands.py and confirm that
#   MODE is set correctly at the top of the file:
#     MODE = "train"  to train and save a new model
#     MODE = "apply"  to load a saved model and run prediction
#
# Input:  Determined by paths set inside the Python script
# Output: Determined by paths set inside the Python script
# ============================================================

#SBATCH --job-name=rf_8_bands
#SBATCH --account=ACCOUNT_CODE          # the HPC project account code
#SBATCH --partition=CPU_PARTITION       # e.g. icelake, cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32             # RF uses n_jobs=-1; give it plenty of cores
#SBATCH --mem=128G                     # texture features on large images are RAM-heavy
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/your/logs/rf_8_bands_%j.out  # %j is replaced by job ID automatically

module purge
module load rhel8/default-amp           # adjust to the HPC's available modules
module load python/3.11.0-icl

source /path/to/your/rf_venv/bin/activate

mkdir -p /path/to/your/logs            # creates log folder if it does not exist

python /path/to/your/random_forest_classification_8_bands.py
