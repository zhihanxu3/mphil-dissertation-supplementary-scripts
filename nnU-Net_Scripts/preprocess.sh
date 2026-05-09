#!/bin/bash
# ============================================================
# nnU-Net Preprocessing — HPC SLURM Batch Script
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
# Description:
#   Runs nnU-Net v2 dataset planning and preprocessing on an
#   HPC cluster using the Residual Encoder planner (ResEncL)
#   for 2D configuration. Designed for Cambridge University HPC 
#   (icelake partition) but adaptable to other SLURM-based systems
#   by adjusting the directives and module names below.
#   Must be submitted via: sbatch preprocess.sh in a terminal
#
# Input:  Raw dataset folder under nnUNet_raw (Dataset 501)
# Output: Preprocessed plans saved to nnUNet_preprocessed
# ============================================================

# ------ SLURM job configuration ----------------------------
#SBATCH --job-name=nnunet_pre
#SBATCH --account=YOUR_ACCOUNT          # the HPC project account code
#SBATCH --partition=YOUR_PARTITION      # e.g. icelake, gpu, cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16             # preprocessing is CPU-bound; 16 recommended
#SBATCH --mem=64G                      # adjust based on dataset size
#SBATCH --time=04:00:00                # increase for larger datasets
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
# -pl ResEncL   : uses Residual Encoder planner (larger capacity than default)
# -c 2d         : 2D configuration (use 3d_fullres for volumetric data)
nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncL -c 2d
