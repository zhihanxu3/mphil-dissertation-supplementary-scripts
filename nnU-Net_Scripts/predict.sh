#!/bin/bash
# ============================================================
# nnU-Net Inference and Post-processing — HPC SLURM Batch Script
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
# Description:
#   Runs nnU-Net v2 inference on new images using the full
#   5-fold ensemble (folds 0-4), then applies default model
#   post-processing using the parameters determined by find_best.sh.
#   The two steps are run sequentially in a single job:
#   prediction first, then post-processing on the raw output.
#   Submit via: sbatch predict.sh in a terminal
#
# Input:  PlanetScope tiles to predict in predict_input/
# Output: Raw predictions in predict_output/
#         Post-processed predictions in predict_output_pp/
# ============================================================

# ------ SLURM job configuration ----------------------------
#SBATCH --job-name=nnunet_predict
#SBATCH --account=ACCOUNT_CODE          # HPC project account code
#SBATCH --partition=GPU_PARTITION       # e.g. ampere, gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32             
#SBATCH --mem=64G
#SBATCH --time=06:00:00                
#SBATCH --output=/path/to/logs/predict_%j.out  # %j is replaced by job ID automatically

# ------ Environment setup ----------------------------------
module purge
module load rhel8/default-amp           # adjust to the HPC's available modules

source /path/to/your/nnunet_env/bin/activate

# ------ nnU-Net environment variables ----------------------
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# ------ Output directories ---------------------------------
PREDICT_INPUT="/path/to/predict_input"         # folder containing images to predict
PREDICT_OUTPUT="/path/to/predict_output"       # raw ensemble predictions saved here
PREDICT_OUTPUT_PP="/path/to/predict_output_pp" # post-processed predictions saved here
RESULTS_DIR="$nnUNet_results/Dataset501_Planet/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4"

mkdir -p "$PREDICT_OUTPUT"
mkdir -p "$PREDICT_OUTPUT_PP"

# ------ GPU settings ---------------------------------------
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# ------ Step 1: Run inference ------------------------------
# -d  : dataset name
# -i  : input folder
# -o  : output folder for raw predictions
# -f  : use all five folds as an ensemble (0 1 2 3 4)
# -tr : trainer class
# -c  : 2D configuration
# -p  : plans identifier
nnUNetv2_predict \
    -d Dataset501_Planet \
    -i "$PREDICT_INPUT" \
    -o "$PREDICT_OUTPUT" \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainer \
    -c 2d \
    -p nnUNetPlans

# ------ Step 2: Apply post-processing ----------------------
# -i           : raw predictions from Step 1
# -o           : folder for final post-processed output
# -pp_pkl_file : post-processing parameters from find_best.sh
# -np          : number of parallel processes
# -plans_json  : plans file from cross-validation results
nnUNetv2_apply_postprocessing \
    -i "$PREDICT_OUTPUT" \
    -o "$PREDICT_OUTPUT_PP" \
    -pp_pkl_file "$RESULTS_DIR/postprocessing.pkl" \
    -np 8 \
    -plans_json "$RESULTS_DIR/plans.json"
