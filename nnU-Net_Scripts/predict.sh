#!/bin/bash
#SBATCH --job-name=nnunet_predict
#SBATCH --account=DELL-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/rds/user/zx335/hpc-work/nnunet_project/logs/predict_%j.out

module purge
module load rhel8/default-amp

source /rds/user/zx335/hpc-work/nnunet_project/env_gpu/bin/activate

export nnUNet_raw="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_raw"
export nnUNet_preprocessed="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_preprocessed"
export nnUNet_results="/rds/user/zx335/hpc-work/nnunet_project/nnUNet_results"

mkdir -p /rds/user/zx335/hpc-work/nnunet_project/predict_output
mkdir -p /rds/user/zx335/hpc-work/nnunet_project/predict_output_pp

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

nnUNetv2_predict -d Dataset501_Planet -i /rds/user/zx335/hpc-work/nnunet_project/predict_input -o /rds/user/zx335/hpc-work/nnunet_project/predict_output -f 0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans

nnUNetv2_apply_postprocessing -i /rds/user/zx335/hpc-work/nnunet_project/predict_output -o /rds/user/zx335/hpc-work/nnunet_project/predict_output_pp -pp_pkl_file /rds/user/zx335/hpc-work/nnunet_project/nnUNet_results/Dataset501_Planet/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /rds/user/zx335/hpc-work/nnunet_project/nnUNet_results/Dataset501_Planet/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json