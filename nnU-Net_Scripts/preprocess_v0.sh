#!/bin/bash
#SBATCH --job-name=nnunet_pre
#SBATCH --account=DELL-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/rds/user/zx335/hpc-work/nnunet_project/logs/pre_%j.out

# 1. Environment Setup
module purge
module load rhel8/default-amp


# 2. Activate Environment
source /rds/user/zx335/hpc-work/nnunet_project/env_gpu/bin/activate

# 3. Set Paths (Crucial for nnU-Net)
export nnUNet_raw="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_raw"
export nnUNet_preprocessed="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_preprocessed"
export nnUNet_results="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_results"

# 4. Run Preprocessing
# Note: -d 501 assumes your folder is named "Dataset501_YourName"
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity