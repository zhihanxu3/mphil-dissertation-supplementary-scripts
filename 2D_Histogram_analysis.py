# ============================================================
# 2D Spectral Histogram Analysis
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
#
# Description:
#   Generates a 2D histogram of pixel reflectance values for
#   two user-selected PlanetScope bands (e.g. Blue vs Red,
#   Blue vs NIR, Green vs NIR) for manually annotated training
#   pixels. Each annotation class is plotted as a 2D spectral
#   density distribution on a log colour scale, allowing visual
#   inspection of class separability in 2D spectral space
#   before random forest classification. The class scheme,
#   names, and colours are defined centrally to stay consistent
#   with the 1D histogram script. Sampling is capped to balance
#   class sizes and avoid overrepresentation of larger classes.
#   Run inside the QGIS Python Console.
#
# Inputs:
#   - A PlanetScope multispectral raster layer loaded in QGIS
#     (referenced by layer name)
#   - A vector annotation layer loaded in QGIS (GeoPackage or
#     similar) with a field storing integer class labels
#
# Outputs:
#   - An interactive 2D histogram plot (matplotlib) displayed
#     in the QGIS session, showing per-class spectral density
#     on a log colour scale
# ============================================================

import numpy as np
import random
from osgeo import gdal
from qgis.core import QgsProject
import matplotlib.pyplot as plt

# ==========================================================
# USER INPUT
# ==========================================================

# Layer names as shown in the QGIS Layers panel
raster_layer_name = "planet_tile_0001"      # change to my raster layer name
vector_layer_name = "planet_mask_annotations"  # change to my annotation layer name

# PlanetScope band indices to compare (1-based GDAL indexing)
# e.g. Blue vs Red, Blue vs NIR (band 2 vs 8), Green vs NIR (band 4 vs 8)
band_x = 2  # Blue
band_y = 5  # Red

# Maximum samples drawn per class; also capped relative to smallest class
target_samples = 20000
ignore_label = 255  # nodata / unannotated pixel value

# ==========================================================
# FIXED CLASS SCHEME
# ==========================================================

# All expected class IDs in my annotation scheme
ALL_CLASSES = [0,1,2,3,4,5,6]

# Human-readable names for each class ID
CLASS_NAMES = {
    0: "Dirty ice / debris-influenced ice",
    1: "Snow / clean ice",
    2: "Blue ice",
    3: "Supraglacial lake",
    4: "Supraglacial river / channel",
    5: "Slush",
    6: "Cloud / cloud shadow"
}

# Consistent colours assigned to each class for all plots
CLASS_COLORS = {
    0: "#8B5A2B",   # brownish — dirty ice
    1: "#E69F00",   # light orange — snow / clean ice
    2: "#8EC9FF",   # light blue — blue ice
    3: "#08306B",   # dark blue — supraglacial lake
    4: "#CC79A7",   # purple — supraglacial river
    5: "#D55E00",   # vermillion — slush
    6: "#000000"    # black — cloud / cloud shadow
}

# ==========================================================
# LOAD LAYERS
# ==========================================================

project = QgsProject.instance()

raster_layer = project.mapLayersByName(raster_layer_name)[0]
vector_layer = project.mapLayersByName(vector_layer_name)[0]

# Open the raster via GDAL to access raw band arrays
raster_path = raster_layer.source()
ds = gdal.Open(raster_path)

cols = ds.RasterXSize
rows = ds.RasterYSize

# Read the two chosen bands into arrays
band_x_data = ds.GetRasterBand(band_x).ReadAsArray().astype(float)
band_y_data = ds.GetRasterBand(band_y).ReadAsArray().astype(float)

# ==========================================================
# CREATE CLASS RASTER FROM VECTOR LAYER
# ==========================================================

# Rasterise the vector annotation layer into an in-memory
# class raster that matches the image grid exactly.
# 255 is used as the nodata / unannotated pixel value.
mem_driver = gdal.GetDriverByName("MEM")
mask_ds = mem_driver.Create("", cols, rows, 1, gdal.GDT_Byte)
mask_ds.SetGeoTransform(ds.GetGeoTransform())
mask_ds.SetProjection(ds.GetProjection())

mask_band = mask_ds.GetRasterBand(1)
mask_band.Fill(ignore_label)           # fill with nodata before burning
mask_band.SetNoDataValue(ignore_label)

# Access the underlying OGR layer from the QGIS vector layer
from osgeo import ogr
vector_path = vector_layer.source().split("|")[0]  # strip QGIS layer options if present
vector_ds = ogr.Open(vector_path)
ogr_layer = vector_ds.GetLayer()

# Read the class field name from the first attribute field
layer_defn = ogr_layer.GetLayerDefn()
class_field = layer_defn.GetFieldDefn(0).GetName()

# Burn class ID values from the vector attribute into the raster
gdal.RasterizeLayer(
    mask_ds,
    [1],
    ogr_layer,
    options=[f"ATTRIBUTE={class_field}"]
)

class_raster = mask_band.ReadAsArray()

# ==========================================================
# HANDLE NODATA
# ==========================================================

# Build a boolean mask of all annotated (valid) pixels
valid_mask = (class_raster != ignore_label)

# Identify which classes are actually present in this tile
present_classes = np.unique(class_raster[valid_mask])
present_classes = [c for c in present_classes if c in ALL_CLASSES]

print("Present classes:", present_classes)

# ==========================================================
# COLLECT PIXELS BY CLASS
# ==========================================================

# Use numpy boolean indexing to extract pixel pairs per class —
# faster than looping over individual features or coordinates
class_pixels = {}

for cls in present_classes:
    mask = (class_raster == cls) & valid_mask

    bx = band_x_data[mask]
    by = band_y_data[mask]

    # Remove pixel pairs where either band value is NaN
    valid = (~np.isnan(bx)) & (~np.isnan(by))
    bx = bx[valid]
    by = by[valid]

    if len(bx) == 0:
        continue

    # Store as list of (x, y) pairs for sampling
    class_pixels[cls] = list(zip(bx, by))

# ==========================================================
# APPLY SAMPLING RULE
# ==========================================================

# Cap each class at target_samples but also at 2x the smallest
# class size, so no single class dominates the plot
class_sizes = {k: len(v) for k, v in class_pixels.items()}
min_size = min(class_sizes.values())

final_samples = {}

for cls, pixels in class_pixels.items():
    max_allowed = min(target_samples, 2 * min_size, len(pixels))
    final_samples[cls] = random.sample(pixels, max_allowed)

# ==========================================================
# PLOT 2D HISTOGRAMS
# ==========================================================

plt.figure(figsize=(10, 8))

for cls, samples in final_samples.items():
    data = np.array(samples)
    plt.hist2d(
        data[:, 0],
        data[:, 1],
        bins=200,
        cmap="viridis",
        alpha=0.6,
        norm=plt.matplotlib.colors.LogNorm()  # log scale to reveal low-density structure
    )

plt.xlabel(f"Band {band_x} Surface Reflectance")
plt.ylabel(f"Band {band_y} Surface Reflectance")
plt.title(f"2D Spectral Density: Band {band_x} vs Band {band_y} by Class")
plt.colorbar(label="Pixel density (log scale)")
plt.tight_layout()
plt.show()
