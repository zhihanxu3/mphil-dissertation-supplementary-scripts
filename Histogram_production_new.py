import os
import numpy as np
from osgeo import gdal, ogr
import matplotlib.pyplot as plt

# ==========================================================
# USER INPUT
# ==========================================================

image_path = r"D:\MPhil_Project_Preparation\PlanetScope_Data\SchemeA_multi_training\Training_Images\planet_tile_10_0004.tif"
mask_raster_path = r"D:\MPhil_Project_Preparation\PlanetScope_Data\SchemeA_multi_training\Training_Masks\10_0004_mask_raster.tif"
vector_mask_path = r"D:\MPhil_Project_Preparation\PlanetScope_Data\planet_tile_10_0004\planet_tile_10_0004_mask_gpack.gpkg"

output_dir = r"D:\MPhil_Project_Preparation\PlanetScope_Data\Histograms\planet_tile_10"
os.makedirs(output_dir, exist_ok=True)

tile_id = "10_0004"
use_vector = True

# ==========================================================
# FIXED CLASS SCHEME (UPDATED COLOURS)
# ==========================================================

ALL_CLASSES = [0,1,2,3,4,5,6]

CLASS_NAMES = {
    0: "Dirty ice / debris-influenced ice",
    1: "Snow / clean ice",
    2: "Blue ice",
    3: "Supraglacial lake",
    4: "Supraglacial river / channel",
    5: "Slush",
    6: "Cloud / cloud shadow"
}

CLASS_COLORS = {
    0: "#8B5A2B",   # brownish dirty ice
    1: "#E69F00",   # snow (light orange)
    2: "#8EC9FF",   # light blue (blue ice)  <-- UPDATED
    3: "#08306B",   # dark blue (lake)       <-- UPDATED
    4: "#CC79A7",   # purple (river)
    5: "#D55E00",   # vermillion (slush)
    6: "#000000"    # black (cloud/shadow)
}

# ==========================================================
# LOAD IMAGE
# ==========================================================

img_ds = gdal.Open(image_path)
cols = img_ds.RasterXSize
rows = img_ds.RasterYSize
band_count = img_ds.RasterCount

bands = np.array([
    img_ds.GetRasterBand(i+1).ReadAsArray().astype(float)
    for i in range(band_count)
])

geotransform = img_ds.GetGeoTransform()
projection = img_ds.GetProjection()

# ==========================================================
# LOAD OR CREATE CLASS RASTER
# ==========================================================

if use_vector:

    mem_driver = gdal.GetDriverByName("MEM")
    mask_ds = mem_driver.Create("", cols, rows, 1, gdal.GDT_Byte)
    mask_ds.SetGeoTransform(geotransform)
    mask_ds.SetProjection(projection)

    mask_band = mask_ds.GetRasterBand(1)
    mask_band.Fill(255)
    mask_band.SetNoDataValue(255)

    vector_ds = ogr.Open(vector_mask_path)
    layer = vector_ds.GetLayer()

    layer_defn = layer.GetLayerDefn()
    class_field = layer_defn.GetFieldDefn(0).GetName()

    gdal.RasterizeLayer(
        mask_ds,
        [1],
        layer,
        options=[f"ATTRIBUTE={class_field}"]
    )

    class_raster = mask_band.ReadAsArray()
    nodata_value = 255

else:
    mask_ds = gdal.Open(mask_raster_path)
    mask_band = mask_ds.GetRasterBand(1)
    nodata_value = mask_band.GetNoDataValue()
    class_raster = mask_band.ReadAsArray()

# ==========================================================
# HANDLE NODATA
# ==========================================================

valid_mask = (class_raster != nodata_value)
present_classes = np.unique(class_raster[valid_mask])
present_classes = [c for c in present_classes if c in ALL_CLASSES]

print("Present classes:", present_classes)

# ==========================================================
# HISTOGRAMS (UNCHANGED)
# ==========================================================

for b in range(band_count):

    plt.figure(figsize=(8,6))

    for cls in ALL_CLASSES:

        mask = (class_raster == cls) & valid_mask
        if not np.any(mask):
            continue

        pixels = bands[b][mask]
        pixels = pixels[~np.isnan(pixels)]
        if pixels.size == 0:
            continue

        plt.hist(
            pixels,
            bins=60,
            density=True,
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

    out_path = os.path.join(output_dir,
                            f"{tile_id}_Band_{b+1}_Histogram.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# ==========================================================
# 2D SCATTER: BAND 1 vs BAND 5 (UPDATED)
# ==========================================================

x_band = 6   # Band 1
y_band = 8   # Band 5

plt.figure(figsize=(8,7))

for cls in ALL_CLASSES:

    mask = (class_raster == cls) & valid_mask
    if not np.any(mask):
        continue

    x_vals = bands[x_band-1][mask]
    y_vals = bands[y_band-1][mask]

    valid = (~np.isnan(x_vals)) & (~np.isnan(y_vals))

    x_vals = x_vals[valid]
    y_vals = y_vals[valid]

    if len(x_vals) == 0:
        continue

    # ===============================
    # SAMPLE MAX 20,000 PIXELS
    # ===============================
    if len(x_vals) > 10000:
        idx = np.random.choice(len(x_vals), 10000, replace=False)
        x_vals = x_vals[idx]
        y_vals = y_vals[idx]

    plt.scatter(
        x_vals,
        y_vals,
        s=5,
        alpha=0.05,
        color=CLASS_COLORS[cls],
        label=f"{cls} – {CLASS_NAMES[cls]}"
    )

plt.xlabel("Band 6 Surface Reflectance")
plt.ylabel("Band 8 Surface Reflectance")
plt.title("Spectral Separability: Band 6 vs Band 8")
leg = plt.legend(
    frameon=False,
    fontsize=11,
    markerscale=5
)

for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.grid(False)

out_path_2d = os.path.join(
    output_dir,
    f"{tile_id}_Band6_vs_Band8_2D.png"
)

plt.savefig(out_path_2d, dpi=300, bbox_inches="tight")
plt.close()

print("✔ Processing complete.")