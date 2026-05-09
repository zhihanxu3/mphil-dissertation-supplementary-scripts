import os, glob, rasterio
from tqdm import tqdm

input_dir  = '/rds/user/zx335/hpc-work/nnunet_project/nnUNet_raw/Dataset501_Planet/imagesPr'
output_dir = '/rds/user/zx335/hpc-work/nnunet_project/predict_input'

os.makedirs(output_dir, exist_ok=True)

all_tifs = glob.glob(os.path.join(input_dir, '*.tif'))
print(f'Scanning {len(all_tifs)} files...')

for img_path in all_tifs:
    try:
        with rasterio.open(img_path) as src:
            if src.count != 8:
                print(f'Skipping {os.path.basename(img_path)} - not 8 bands')
                continue
            base_name = os.path.basename(img_path).replace('.tif', '')
            meta = src.meta.copy()
            meta.update({'count': 1, 'compress': 'lzw', 'predictor': 2})
            for b in range(1, 9):
                out_name = f'{base_name}_{b-1:04d}.tif'
                out_path = os.path.join(output_dir, out_name)
                with rasterio.open(out_path, 'w', **meta) as dst:
                    dst.write(src.read(b), 1)
            print(f'Split: {base_name}')
    except Exception as e:
        print(f'Error on {os.path.basename(img_path)}: {e}')
        continue

print('Done. Band-split files written to:', output_dir)