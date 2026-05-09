#!/bin/bash
#SBATCH --job-name=rf_apply_v2
#SBATCH --account=DELL-SL3-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=09:00:00             # apply over many large tiles takes longer
#SBATCH --output=/rds/user/zx335/hpc-work/RandomForest/Test1/v2_logs/random_forest_classification_v2_%j.out

module purge
module load rhel8/default-amp
module load python/3.11.0-icl

source /rds/user/zx335/hpc-work/RandomForest/Test1/rf_venv/bin/activate

mkdir -p /rds/user/zx335/hpc-work/RandomForest/Test1/v2_logs

# Switch MODE to "apply" at runtime without editing the script
# (overrides the MODE variable at the top of random_forest_classification_v2.py)
sed 's/^MODE = "train"/MODE = "apply"/' \
    /rds/user/zx335/hpc-work/RandomForest/Test1/random_forest_classification_v2.py \
    > /tmp/random_forest_classification_v2_apply.py

python /tmp/random_forest_classification_v2_apply.py
