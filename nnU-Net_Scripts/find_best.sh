#!/bin/bash
# ============================================================
# nnU-Net Find Best Configuration — HPC SLURM Batch Script
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
# Description:
#   Evaluates all trained folds and configurations for Dataset
#   501 and identifies the best-performing model or ensemble
#   combination based on cross-validation metrics. Run this
#   after all five training folds are complete. The result
#   determines which configuration is used for final inference.
#   Submit via: sbatch find_best.sh in a terminal
#
# Input:  All trained fold results under nnUNet_results
# Output: Best configuration written to nnUNet_results as
#         inference_information.json
# ============================================================

# ------ SLURM job configuration ----------------------------
#SBATCH --job-name=nnunet_best
#SBATCH --account=ACCOUNT_CODE          # the HPC project account code
#SBATCH --partition=CPU_PARTITION  
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/logs/best_%j.out  # %j is replaced by job ID automatically

# ------ Environment setup ----------------------------------
module purge
module load rhel8/default-amp           # adjust to the HPC's available modules

source /path/to/your/nnunet_env/bin/activate

# ------ nnU-Net environment variables ----------------------
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# ------ Find best configuration ----------------------------
# 501  : dataset ID (change to match the dataset number)
# -c 2d: evaluate the 2D configuration
nnUNetv2_find_best_configuration 501 -c 2d
