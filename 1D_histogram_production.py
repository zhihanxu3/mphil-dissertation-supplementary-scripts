# ============================================================
# 1D Spectral Histogram Analysis
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
#
# Description:
#   Produces per-band 1D spectral histograms and a 2D band
#   scatter plot for manually annotated training pixels on
#   PlanetScope imagery. Each annotation class is plotted as
#   a probability density curve, allowing visual inspection
#   of spectral separability across all bands before random
#   forest classification. The script accepts either a
#   pre-existing raster training mask or a vector annotation
#   layer (GeoPackage); the vector path rasterises the layer
#   on the fly into an in-memory class raster so that both
#   inputs follow exactly the same downstream logic.
#   Run in QGIS python module.
#
# Inputs:
#   - image_path        : PlanetScope multispectral GeoTIFF
#   - mask_raster_path  : Pre-rasterised training mask (.tif),
#                         used when use_vector = False
#   - vector_mask_path  : Vector annotation layer (.gpkg),
#                         used when use_vector = True
#
# Outputs:
#   - One PNG histogram per band saved to output_dir,
#     named [tile_id]_Band_[N]_Histogram.png
#   - One 2D scatter plot saved to output_dir,
#     named [tile_id]_Band[X]_vs_Band[Y]_2D.png
# ============================================================

import os
import numpy as np
from osgeo import gdal, ogr
import matplotlib.pyplot as plt

# ==========================================================
# USER INPUT
# ==========================================================

image_path = r"enter image path here"
mask_raster_path = r"enter mask raster path here"
vector_mask_path = r"enter mask vector path here"

output_dir = r"enter oytput folder path here"
os.makedirs(output_dir, exist_ok=True)

tile_id = "10_0004"
use_vector = True  # set True to rasterise from vector, False to load pre-made raster mask

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
# LOAD IMAGE
# ==========================================================

img_ds = gdal.Open(image_path)
cols = img_ds.RasterXSize
rows = img_ds.RasterYSize
band_count = img_ds.RasterCount

# Read all bands at once into a 3D array [band, row, col]
bands = np.array([
    img_ds.GetRasterBand(i+1).ReadAsArray().astype(float)
    for i in range(band_count)
])

# Retain geotransform and projection for mask alignment
geotransform = img_ds.GetGeoTransform()
projection = img_ds.GetProjection()

# ==========================================================
# LOAD OR CREATE CLASS RASTER
# ==========================================================

if use_vector:
    # Rasterise the vector annotation layer into an in-memory
    # class raster that matches the image grid exactly.
    # 255 is used as the nodata / unannotated pixel value.
    mem_driver = gdal.GetDriverByName("MEM")
    mask_ds = mem_driver.Create("", cols, rows, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(geotransform)
    mask_ds.SetProjection(projection)

    mask_band = mask_ds.GetRasterBand(1)
    mask_band.Fill(255)           # fill with nodata
    mask_band.SetNoDataValue(255)

    vector_ds = ogr.Open(vector_mask_path)
    layer = vector_ds.GetLayer()

    # Read the class field name from the first attribute field
    layer_defn = layer.GetLayerDefn()
    class_field = layer_defn.GetFieldDefn(0).GetName()

    # Burn class ID values from the vector attribute into the raster
    gdal.RasterizeLayer(
        mask_ds,
        [1],
        layer,
        options=[f"ATTRIBUTE={class_field}"]
    )

    class_raster = mask_band.ReadAsArray()
    nodata_value = 255

else:
    # Load a pre-rasterised training mask directly from disk
    mask_ds = gdal.Open(mask_raster_path)
    mask_band = mask_ds.GetRasterBand(1)
    nodata_value = mask_band.GetNoDataValue()
    class_raster = mask_band.ReadAsArray()

# ==========================================================
# HANDLE NODATA
# ==========================================================

# Build a boolean mask of all annotated (valid) pixels
valid_mask = (class_raster != nodata_value)

# Identify which classes are actually present in this tile
present_classes = np.unique(class_raster[valid_mask])
present_classes = [c for c in present_classes if c in ALL_CLASSES]

print("Present classes:", present_classes)

# ==========================================================
# 1D HISTOGRAMS — ONE PER BAND
# ==========================================================

for b in range(band_count):

    plt.figure(figsize=(8,6))

    for cls in ALL_CLASSES:

        # Select pixels belonging to this class that are also valid
        mask = (class_raster == cls) & valid_mask
        if not np.any(mask):
            continue

        pixels = bands[b][mask]
        pixels = pixels[~np.isnan(pixels)]  # drop any residual NaN values
        if pixels.size == 0:
            continue

        plt.hist(
            pixels,
            bins=60,
            density=True,       # normalise to probability density so classes are comparable
            alpha=0.5,
            color=CLASS_COLORS[cls],
            label=f"{cls} – {CLASS_NAMES[cls]}",
            histtype="stepfilled"
        )

    plt.xlabel(f"Band {b+1} Surface Reflectance")
    plt.ylabel("Probability Density")
    plt.title(f"Spectral Histogram – Band {b+1}")
    plt.legend(frameon=False, fontsize=8)
    plt.grid(False)

    out_path = os.path.join(output_dir, f"{tile_id}_Band_{b+1}_Histogram.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# ==========================================================
# 2D SCATTER — BAND 6 vs BAND 8
# ==========================================================

# Inspect separability between two bands in 2D spectral space.
# Change x_band / y_band to explore other band combinations.
x_band = 6   # e.g. Red Edge
y_band = 8   # e.g. NIR

plt.figure(figsize=(8,7))

for cls in ALL_CLASSES:

    mask = (class_raster == cls) & valid_mask
    if not np.any(mask):
        continue

    x_vals = bands[x_band-1][mask]
    y_vals = bands[y_band-1][mask]

    # Remove pixels where either band is NaN
    valid = (~np.isnan(x_vals)) & (~np.isnan(y_vals))
    x_vals = x_vals[valid]
    y_vals = y_vals[valid]

    if len(x_vals) == 0:
        continue

    # Cap at 10,000 pixels per class to keep the plot readable
    if len(x_vals) > 10000:
        idx = np.random.choice(len(x_vals), 10000, replace=False)
        x_vals = x_vals[idx]
        y_vals = y_vals[idx]

    plt.scatter(
        x_vals,
        y_vals,
        s=5,
        alpha=0.05,             # low alpha so dense clusters remain visible
        color=CLASS_COLORS[cls],
        label=f"{cls} – {CLASS_NAMES[cls]}"
    )

plt.xlabel(f"Band {x_band} Surface Reflectance")
plt.ylabel(f"Band {y_band} Surface Reflectance")
plt.title(f"Spectral Separability: Band {x_band} vs Band {y_band}")

leg = plt.legend(frameon=False, fontsize=11, markerscale=5)
for lh in leg.legendHandles:
    lh.set_alpha(1)  # restore full opacity in the legend markers only
plt.grid(False)

out_path_2d = os.path.join(output_dir, f"{tile_id}_Band{x_band}_vs_Band{y_band}_2D.png")
plt.savefig(out_path_2d, dpi=300, bbox_inches="tight")
plt.close()

print("Done. All histogram outputs saved to:", output_dir)
