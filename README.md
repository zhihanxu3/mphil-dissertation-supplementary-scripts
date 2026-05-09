# mphil-dissertation-supplementary-scripts
Supplementary Python scripts and result data for my MPhil in Polar Studies dissertation on supraglacial channel mapping


## nnU-Net Supraglacial Meltwater Feature Mapping — Workflow Guide

This section describes how to use the nnU-Net scripts in this repository to train and apply a deep learning segmentation model for supraglacial meltwater feature mapping from PlanetScope imagery. Two model variants are covered: a full 8-band model using PlanetScope SuperDove imagery, and a 4-band model for compatibility with 2019 (pre-2021) PlanetScope Dove imagery.

Please refer to Isensee et al. (2021) for detailed description of nnU-Net algorithms, architecture, and expected directory structure for data organisation

### Part 1 — 8-Band SuperDove Model (Dataset501)

#### Step 1 — Split training image bands
Run 'split_bands_training_imagery.py' on your imagesTr folder. This splits each
8-band PlanetScope GeoTIFF into eight single-band files named
`[tile_id]_0000.tif` through `[tile_id]_0007.tif` as required
by nnU-Net, and removes the original 8-band files afterwards.

#### Step 2 — Prepare label masks and dataset.json
Run 'prepare_nnunet_training_masks.py'. This remaps your RF annotation
class IDs to the nnU-Net 3-class label scheme (background, lake,
channel), resolves any shape mismatches between masks and images,
saves corrected labels to labelsTr/, and generates dataset.json
describing the 8-band channel configuration.

#### Step 3 — Preprocess
Submit 'preprocess.sh' to the HPC for preprocess remotely:
sbatch preprocess.sh

This runs nnU-Net's planning and preprocessing pipeline using the
Residual Encoder planner (ResEncL) for the 2D configuration.

#### Step 4 — Train all five folds
Submit one job per fold. All five can be submitted at once and
will run in parallel on the GPU cluster:
sbatch train_fold_0.sh
sbatch train_fold_1.sh
sbatch train_fold_2.sh
sbatch train_fold_3.sh
sbatch train_fold_4.sh

The full model requires all five folds to be complete. If any job 
hits the walltime limit before finishing, resume it with:
sbatch resume_fold.sh

#### Step 5 — Find best configuration
Once all five folds are complete, submit 'find_best.sh':
sbatch find_best.sh

This evaluates all fold results and identifies the best-performing
model or ensemble combination. The output is saved as
`inference_information.json` under nnUNet_results and is required
before running prediction.

#### Step 6 — Split prediction image bands
Run 'split_bands_prediction_imagery.py' on your imagesPr folder to split prediction 
images into individual band files as required.

#### Step 7 — Predict and post-process
Submit 'predict.sh:
sbatch predict.sh

This runs inference using the full 5-fold ensemble, writing raw
predictions to 'predict_output/', then applies post-processing
parameters from Step 5 and writes final outputs to
'predict_output_pp/'.


### Part 2 — 4-Band Dove Model (Dataset502)

This model is trained on only four bands (Blue,
Green I, Red, NIR) to enable prediction on 2019 PlanetScope
Dove imagery.

#### Step 1 — Extract 4 bands from training images
Run 'extract_4bands_for_training_images.py'. This extracts the
four relevant bands from the existing 8-band SuperDove training
tiles and saves them as band-split files into the Dataset502
imagesTr folder, ready for nnU-Net directly.

#### Step 2 — Prepare label masks and dataset.json
Re-run 'prepare_nnunet_dataset.py' with paths updated. 
The label masks are identical to the 8-band model.
But output paths need to be changed.

#### Steps 3 to 7 — Preprocess, train, find best, and predict
The SLURM scripts for these steps (preprocess, train folds 0-4,
resume, find best, predict) are not separately provided for the
4-band model here, as they are nearly identical to their 8-band
counterparts. The changes required are mainly related to updating 
dataset ID, folder paths to point to Dataset502 directories,
and 'predict_input/' to contain the 4-band split prediction tiles. 

All other SLURM directives, module loads, environment variables,
and nnU-Net commands need no change.
