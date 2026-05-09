#!/bin/bash
# ============================================================
# nnU-Net Training Resume — HPC SLURM Batch Script
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
# Description:
#   Resumes an interrupted nnU-Net v2 training job from the
#   latest saved checkpoint for a specified fold. Use this
#   when a training job hits the walltime limit or is killed
#   before completion. Change the fold index (0–4) in the
#   nnUNetv2_train command to match whichever fold needs
#   resuming. All other settings mirror the original
#   train_fold scripts.
#   Submit via: sbatch resume_fold.sh in a terminal
#
# Input:  Latest checkpoint under nnUNet_results for the fold
# Output: Continued training appended to the same fold folder
# ============================================================

# ------ SLURM job configuration ----------------------------
#SBATCH --job-name=nnunet_resume
#SBATCH --account=ACCOUNT_CODE          # the HPC project account code
#SBATCH --partition=GPU_PARTITION  # e.g. ampere, gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00                
#SBATCH --output=/path/to/logs/resume_f_%j.out  # %j is replaced by job ID automatically

# ------ Environment setup ----------------------------------
module purge
module load rhel8/default-amp           # adjust to the HPC's available modules

source /path/to/your/nnunet_env/bin/activate

# ------ nnU-Net environment variables ----------------------
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# ------ Resume training ------------------------------------
# Change the fold index (here: 0) to whichever fold need to be resumed
# --npz : retain softmax probability maps for ensembling
# --c   : resume from the latest checkpoint instead of starting fresh
nnUNetv2_train 501 2d 0 --npz --c
