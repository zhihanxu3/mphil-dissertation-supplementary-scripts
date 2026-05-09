#!/bin/bash
# ============================================================
# nnU-Net Training — Fold 4 — 5-Fold Cross-Validation — HPC SLURM Batch Script
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
# Description:
#   Trains nnU-Net v2 on fold 4 of a 5-fold cross-validation (folds 0–4)
#   split for the dataset 501, 2D configuration. Five separate
#   scripts (train_fold_0.sh to train_fold_4.sh) are provided
#   because each fold is submitted as an independent GPU job,
#   allowing all five to run in parallel on the HPC cluster
#   rather than sequentially in a single job. The full model
#   ensemble uses all five trained folds at inference time.
#   Submit via: sbatch train_fold_4.sh in a terminal
#
# Input:  Preprocessed data under nnUNet_preprocessed
# Output: Trained fold 4 model saved to nnUNet_results
# ============================================================

# ------ SLURM job configuration ----------------------------
#SBATCH --job-name=nnunet_f4
#SBATCH --account=YOUR_ACCOUNT          # the HPC project account code
#SBATCH --partition=YOUR_GPU_PARTITION  # e.g. ampere, gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                   # single GPU per fold
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G                      # adjust based on dataset size
#SBATCH --time=12:00:00                
#SBATCH --output=/path/to/logs/train_f4_%j.out  # %j is replaced by job ID automatically

# ------ Environment setup ----------------------------------
module purge
module load rhel8/default-amp           # adjust to the HPC's available modules

source /path/to/your/nnunet_env/bin/activate

# ------ nnU-Net environment variables ----------------------
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# ------ Train fold 0 ---------------------------------------
# 501  : dataset ID (change to match the dataset number)
# 2d   : 2D configuration
# 4    : fold index (0–4 across the five scripts)
# --npz : saves softmax probability maps needed for ensembling
nnUNetv2_train 501 2d 4 --npz
