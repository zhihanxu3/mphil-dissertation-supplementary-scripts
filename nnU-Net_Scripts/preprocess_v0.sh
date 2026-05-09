#!/bin/bash
# ============================================================
# nnU-Net Preprocessing — HPC SLURM Batch Script
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
# Description:
#   Runs nnU-Net v2 dataset planning and preprocessing on an
#   HPC cluster using the default model planner for 2D configuration.
#   Designed for Cambridge University HPC (icelake partition)
#   but adaptable to other SLURM-based systems by adjusting
#   the directives and module names below.
#   Must be submitted via: sbatch preprocess.sh in a terminal
#
# Input:  Raw dataset folder under nnUNet_raw (Dataset 501)
# Output: Preprocessed plans saved to nnUNet_preprocessed
# ============================================================

# ------ SLURM job configuration ----------------------------
#SBATCH --job-name=nnunet_pre
#SBATCH --account=ACCOUNT_CODE         # the HPC project account code
#SBATCH --partition=PARTITION_TYPE      # e.g. icelake, gpu, cpu, CPU is sufficient for this task
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16             
#SBATCH --mem=64G                      # adjust based on dataset size
#SBATCH --time=04:00:00                
#SBATCH --output=/path/to/logs/pre_%j.out  # %j is replaced by job ID automatically

# ------ Environment setup ----------------------------------
# Adjust module names to match the HPC's available modules
module purge
module load rhel8/default-amp           # base environment
module load python/3.11.0-icl
module load gdal/3.5.1-gcc-8.5.0

# Activate your nnU-Net virtual environment
source /path/to/your/nnunet_env/bin/activate

# ------ nnU-Net environment variables ----------------------
# These three variables must point to the nnU-Net working directories
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# ------ Run preprocessing ----------------------------------
# -d 501        : dataset ID (change to match the dataset number)
# --verify_dataset_integrity: checks the imagesTr/labelsTr files to ensure they are correctly paired before preprocessing begins

# 4. Run Preprocessing
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
