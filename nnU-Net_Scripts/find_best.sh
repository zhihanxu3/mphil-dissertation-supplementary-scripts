#!/bin/bash
#SBATCH --job-name=nnunet_best
#SBATCH --account=DELL-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/rds/user/zx335/hpc-work/nnunet_project/logs/best_%j.out

module purge
module load rhel8/default-amp

source /rds/user/zx335/hpc-work/nnunet_project/env_gpu/bin/activate

export nnUNet_raw="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_raw"
export nnUNet_preprocessed="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_preprocessed"
export nnUNet_results="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_results"

nnUNetv2_find_best_configuration 501 -c 2d