# ============================================================
# nnU-Net training masks preparation — Labels and dataset.json
# Standalone Python Script
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
#
# Description:
#   Prepares the labelled mask component of the nnU-Net
#   training dataset (Dataset501_Planet). For each training
#   tile, this script remaps the original RF annotation class
#   IDs to the three-class nnU-Net label scheme (background,
#   lake, channel), handles any pixel-level shape mismatches
#   between mask and image, and saves the corrected labels to
#   the nnUNet_raw/labelsTr folder. It also generates the
#   required dataset.json describing the 8-band channel
#   configuration and label mapping.

#   Run this script before submitting preprocess.sh 
#   and after splitting training images into individual band tiles
#
# Label remapping (RF annotation → nnU-Net):
#   1 (Snow / clean ice) → 0  background
#   3 (Supraglacial lake) → 1  lake
#   4 (Supraglacial channel) → 2  channel
#   all others             → 3  ignore
#
# Inputs:
#   - SRC_MASKS : folder of RF annotation mask GeoTIFFs
#   - IMG_TR    : imagesTr folder already created, with
#                 band-split training image tiles
#
# Outputs:
#   - labelsTr/*.tif     : remapped label GeoTIFFs
#   - dataset.json       : nnU-Net dataset descriptor
# ============================================================

import os, json, re, tifffile
import numpy as np
import imageio.v3 as iio
from pathlib import Path
import rasterio
from rasterio.errors import RasterioIOError

# ==========================================================
# USER INPUT
# ==========================================================

SRC_MASKS = Path("/path/to/your/nnUNet_masks")                      # source RF annotation masks
OUT_ROOT  = Path("/path/to/your/nnUNet_raw/Dataset501_Planet")      # nnU-Net raw dataset root
LBL_TR    = OUT_ROOT / "labelsTr"

# Regex to extract tile ID from filename (e.g. 2_0006 or 16A_0000)
CASE_RE = re.compile(r"(\d+[A-Z]?_\d+)")

def remap_mask(arr):
    arr = np.asarray(arr)
    out = np.full(arr.shape, 3, dtype=np.uint8) # Default everything to Ignore (3)
    out[arr == 1] = 0  # Background (Ice/Snow)
    out[arr == 3] = 1  # Lake
    out[arr == 4] = 2  # Channel
    return out

def main():
    LBL_TR.mkdir(parents=True, exist_ok=True)
    IMG_TR = OUT_ROOT / "imagesTr"
    
    mask_files = sorted([p for p in SRC_MASKS.iterdir() if p.suffix.lower() in ['.tif', '.tiff']])
    
    if not mask_files:
        print(f"Error: No masks found in {SRC_MASKS}")
        return

    tokens = []
    for m in mask_files:
        # Extract ID (e.g., 2_0006 or 16A_0000)
        match = CASE_RE.search(m.stem)
        token = match.group(1) if match else m.stem
        
        # 1. Reference the split image to get the target shape (Height, Width)
        # We check band _0000 specifically
        img_ref_path = IMG_TR / f"{token}_0000.tif"
        
        if not img_ref_path.exists():
            print(f"Skipping {token}: Corresponding image not found in {IMG_TR}")
            continue

        try:
            with rasterio.open(img_ref_path) as src:
                target_shape = src.shape  # (Height, Width)
            
            # 2. Read and Remap the mask
            mask_arr = iio.imread(m)
            remapped = remap_mask(mask_arr) # Uses your 0, 1, 2, 3 logic
            
            # 3. Handle Shape Mismatch
            if remapped.shape != target_shape:
                print(f"Fixing mismatch for {token}: Mask {remapped.shape} -> Image {target_shape}")
                
                # Create a new blank canvas filled with 'Ignore' (3)
                final_mask = np.full(target_shape, 3, dtype=np.uint8)
                
                # Calculate valid overlap to copy
                h_copy = min(remapped.shape[0], target_shape[0])
                w_copy = min(remapped.shape[1], target_shape[1])
                
                # Paste original mask onto the canvas
                final_mask[:h_copy, :w_copy] = remapped[:h_copy, :w_copy]
                remapped = final_mask
            
            # 4. Save the corrected mask
            tifffile.imwrite(LBL_TR / f"{token}.tif", remapped, compression='deflate')
            tokens.append(token)
            print(f"Processed: {token}.tif")

        except Exception as e:
            print(f"Failed to process {token}: {e}")

    # 5. Write the 8-channel dataset.json
    dataset_json = {
        "channel_names": {
            "0": "CoastalBlue", "1": "Blue", "2": "GreenI", "3": "GreenII",
            "4": "Yellow", "5": "Red", "6": "RedEdge", "7": "NIR"
        },
        "labels": {
            "background": 0, 
            "lake": 1, 
            "channel": 2, 
            "ignore": 3
        },
        "numTraining": len(tokens),
        "file_ending": ".tif"
    }

    with open(OUT_ROOT / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"\nSuccess! Fixed and processed {len(tokens)} masks.")
    print(f"You can now resubmit: sbatch preprocess.sh")

if __name__ == "__main__":
    main()
