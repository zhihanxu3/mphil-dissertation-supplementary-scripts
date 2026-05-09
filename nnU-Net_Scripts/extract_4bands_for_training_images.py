# extract_4bands_for_training_images.py
import os
import glob
import numpy as np
import rasterio

INPUT_DIR  = "/rds/user/zx335/hpc-work/nnunet_project/nnUNet_raw/Dataset501_Planet/imagesTr"
OUTPUT_DIR = "/rds/user/zx335/hpc-work/nnunet_project/nnUNet_raw/Dataset502_Planet4band/imagesTr"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bands to extract from 8-band SuperDove (1-indexed for rasterio)
# Band 2 = Blue, Band 3 = Green I, Band 6 = Red, Band 8 = NIR
BANDS_TO_EXTRACT = [2, 3, 6, 8]

tif_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
print("Images found: %d" % len(tif_files))

for img_path in tif_files:
    fname = os.path.basename(img_path)
    case  = os.path.splitext(fname)[0]   # e.g. planet_tile_11B_0000

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