# ============================================================
# 4-Band Image Extraction for 2019 PlanetScope Imagery Prediction Task
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
#
# Description:
#   Extracts four spectral bands (Blue, Green, Red, NIR)
#   from 2025 8-band PlanetScope SuperDove training images and
#   saves each as a separate single-band file, producing a
#   4-band dataset (Dataset502_Planet4band) compatible with
#   the older 4-band PlanetScope Dove sensor. This allows
#   a parallel model to be trained and applied to 2019
#   imagery that lacks the four additional SuperDove bands.
#   Run before the corresponding preprocess.sh for
#   Dataset502.
#
# Bands extracted from 8-band SuperDove (1-indexed):
#   Band 2 = Blue
#   Band 4 = Green
#   Band 6 = Red
#   Band 8 = NIR
#
# Inputs:
#   - INPUT_DIR  : imagesTr/ folder of 8-band SuperDove tiles
#                  (already band-split by split_bands.py)
#
# Outputs:
#   - Per-band GeoTIFFs written to OUTPUT_DIR, named
#     [tile_id]_0000.tif through [tile_id]_0003.tif
# ============================================================

import os
import glob
import numpy as np
import rasterio

# ==========================================================
# USER INPUT
# ==========================================================

INPUT_DIR  = "/path/to/your/Dataset501_Planet/imagesTr"           # source 8-band split tiles
OUTPUT_DIR = "/path/to/your/Dataset502_Planet4band/imagesTr"      # 4-band output saved here
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bands to extract from 8-band SuperDove (1-indexed for rasterio)
BANDS_TO_EXTRACT = [2, 4, 6, 8]

tif_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
print("Images found: %d" % len(tif_files))

for img_path in tif_files:
    fname = os.path.basename(img_path)
    case  = os.path.splitext(fname)[0]  

    with rasterio.open(img_path) as src:
        if src.count != 8:
            print("  [SKIP] %s has %d bands, expected 8" % (fname, src.count))
            continue

        profile = src.profile.copy()
        profile.update(count=1, compress="lzw")

        for new_idx, band_num in enumerate(BANDS_TO_EXTRACT):
            out_name = "%s_%04d.tif" % (case, new_idx)
            out_path = os.path.join(OUTPUT_DIR, out_name)
            data = src.read(band_num)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)

    print("Done: %s  ->  4 band files written" % case)

print("\nAll done. Output in: %s" % OUTPUT_DIR)
