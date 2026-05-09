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
module load python/3.11.0-icl
module load gdal/3.5.1-gcc-8.5.0

# 2. Activate Environment
source /rds/user/zx335/hpc-work/nnunet_project/env_gpu/bin/activate

# 3. Set Paths
export nnUNet_raw="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_raw"
export nnUNet_preprocessed="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_preprocessed"
export nnUNet_results="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_results"

# 4. Run Preprocessing using the Residual Encoder Planner
nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncL -c 2d