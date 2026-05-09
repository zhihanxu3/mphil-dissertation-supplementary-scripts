#!/bin/bash
#SBATCH --job-name=nnunet_resume_f${FOLD}
#SBATCH --account=DELL-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/rds/user/zx335/hpc-work/nnunet_project/logs/resume_f0_%j.out

module purge
module load rhel8/default-amp


source /rds/user/zx335/hpc-work/nnunet_project/env_gpu/bin/activate

export nnUNet_raw="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_raw"
export nnUNet_preprocessed="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_preprocessed"
export nnUNet_results="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_results"

nnUNetv2_train 501 2d 0 --npz --c   # --c resumes from the latest checkpoint