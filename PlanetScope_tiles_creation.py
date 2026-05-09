# ============================================================
# PlanetScope Large Image Tiling Script
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
# Description:
#   Splits a large PlanetScope GeoTIFF into fixed-size tiles
#   using GDAL. Each tile preserves the correct geotransform,
#   projection, and all spectral bands. Tiles are more manageable 
#   for training data annotation and classification.
# Inputs:  Single large PlanetScope GeoTIFF
# Outputs: Series of GeoTIFF tiles named
#          planet_tile_[AOI]_[NNNN].tif saved to output_folder
# ============================================================

import os
from osgeo import gdal

# ==========================================================
# USER INPUT
# ==========================================================

input_raster  = r"C:\path\to\your\input_image.tif"    # large PlanetScope GeoTIFF
output_folder = r"C:\path\to\your\output_tiles_folder" # folder to save tiles (created if absent)

tile_size = 4096  # tile width and height in pixels
AOI = 2           # AOI identifier used in output filenames

# ==========================================================
# LOAD RASTER
# ==========================================================

os.makedirs(output_folder, exist_ok=True)

ds = gdal.Open(input_raster)
if ds is None:
    raise RuntimeError("Could not open input raster — check input_raster path.")

x_size      = ds.RasterXSize
y_size      = ds.RasterYSize
band_count  = ds.RasterCount
geotransform = ds.GetGeoTransform()
projection  = ds.GetProjection()
driver      = gdal.GetDriverByName("GTiff")

# ==========================================================
# TILE AND SAVE
# ==========================================================

tile_id = 0

for y in range(0, y_size, tile_size):
    for x in range(0, x_size, tile_size):

        # Clamp tile dimensions at image edges
        x_win = min(tile_size, x_size - x)
        y_win = min(tile_size, y_size - y)

        tile_path = os.path.join(output_folder, f"planet_tile_{AOI}_{tile_id:04d}.tif")

        out_ds = driver.Create(
            tile_path, x_win, y_win, band_count,
            ds.GetRasterBand(1).DataType,
            options=["COMPRESS=LZW", "TILED=YES"]
        )

        # Shift the geotransform origin to the top-left corner of this tile
        new_gt = (
            geotransform[0] + x * geotransform[1],
            geotransform[1], 0.0,
            geotransform[3] + y * geotransform[5],
            0.0, geotransform[5]
        )
        out_ds.SetGeoTransform(new_gt)
        out_ds.SetProjection(projection)

        for band in range(1, band_count + 1):
            data = ds.GetRasterBand(band).ReadAsArray(x, y, x_win, y_win)
            out_ds.GetRasterBand(band).WriteArray(data)

        out_ds.FlushCache()
        out_ds = None
        tile_id += 1

print(f"Done. {tile_id} tiles saved to: {output_folder}")
