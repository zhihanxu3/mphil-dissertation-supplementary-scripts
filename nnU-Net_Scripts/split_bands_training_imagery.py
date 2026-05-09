# ============================================================
# nnU-Net Image Band Splitting
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
#
# Description:
#   Splits 8-band PlanetScope GeoTIFFs into individual
#   single-band files as required by nnU-Net's imagesTr/
#   imagesPr folder convention, where each channel must be
#   stored as a separate file suffixed _0000 to _0007.
#   Already-split single-band files are skipped automatically.
#   The original 8-band file is deleted after splitting.
#   Run this before prepare_nnunet_dataset.py and
#   preprocess.sh.
#
# Inputs:
#   - input_dir : folder containing 8-band PlanetScope
#                 GeoTIFFs (imagesTr/ or imagesPr/)
#
# Outputs:
#   - Per-band GeoTIFFs in the same folder, named
#     [tile_id]_0000.tif through [tile_id]_0007.tif
#   - Original 8-band files removed after splitting
# ============================================================

import os, glob, rasterio
from tqdm import tqdm

# ==========================================================
# USER INPUT
# ==========================================================

# Folder containing 8-band PlanetScope tiles to split
# (typically imagesTr/ for training or imagesPr/ for prediction)
input_dir = "/path/to/your/nnUNet_raw/Dataset501_Planet/imagesPr"

def split_multiband_to_nnunet():
    # 1. Grab every .tif in the folder
    all_tifs = glob.glob(os.path.join(input_dir, "*.tif"))
    
    print(f"Scanning {len(all_tifs)} files for 8-band images...")

    for img_path in tqdm(all_tifs):
        try:
            with rasterio.open(img_path) as src:
                # If it's already a single band (split), skip it immediately
                if src.count == 1:
                    continue
                
                # If it's your 8-band PlanetScope image
                if src.count == 8:
                    base_name = os.path.basename(img_path).replace(".tif", "")
                    
                    # Prepare metadata once
                    meta = src.meta.copy()
                    meta.update({"count": 1, "compress": "lzw", "predictor": 2})
                    
                    # Split bands 1 through 8
                    for b in range(1, 9):
                        out_name = f"{base_name}_{b-1:04d}.tif"
                        out_path = os.path.join(input_dir, out_name)
                        
                        with rasterio.open(out_path, 'w', **meta) as dst:
                            dst.write(src.read(b), 1)
                    
                    # Close the source before deleting
                    src_to_delete = img_path
            
            # Delete the 8-band file only after the 'with' block is closed
            os.remove(src_to_delete)
            
        except Exception as e:
            # This handles cases where a file might be corrupted or not a GeoTIFF
            continue

if __name__ == "__main__":
    split_multiband_to_nnunet()
