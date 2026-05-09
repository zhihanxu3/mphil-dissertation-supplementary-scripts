import numpy as np
import random
from osgeo import gdal
from qgis.core import QgsProject
import matplotlib.pyplot as plt

# -----------------------------
# USER SETTINGS
# -----------------------------

# Raster layer name (as shown in QGIS)
raster_layer_name = "planet_tile_0001"  # change to my layer name

# Vector mask layer name
vector_layer_name = "planet_mask_annotations"

# Field storing class labels
class_field = "class_id"

# PlanetScope band indices (1-based GDAL indexing)
band_x = 2  # Blue
band_y = 5  # Red
# can also try 
# Blue vs NIR (band 2 vs band 8) or
# Green vs NIR(band 4 vs band 8)


target_samples = 20000
ignore_label = 255

# -----------------------------
# LOAD LAYERS
# -----------------------------

project = QgsProject.instance()

raster_layer = project.mapLayersByName(raster_layer_name)[0]
vector_layer = project.mapLayersByName(vector_layer_name)[0]

raster_path = raster_layer.source()
ds = gdal.Open(raster_path)

band_x_data = ds.GetRasterBand(band_x).ReadAsArray()
band_y_data = ds.GetRasterBand(band_y).ReadAsArray()

# -----------------------------
# COLLECT PIXELS BY CLASS
# -----------------------------

class_pixels = {}

for feature in vector_layer.getFeatures():
    cls = feature[class_field]
    if cls == ignore_label:
        continue

    geom = feature.geometry()
    if geom is None:
        continue

    # Rasterize single feature to mask
    mem_drv = gdal.GetDriverByName("MEM")
    mask_ds = mem_drv.Create(
        "", ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte
    )
    mask_ds.SetGeoTransform(ds.GetGeoTransform())
    mask_ds.SetProjection(ds.GetProjection())

    gdal.RasterizeLayer(
        mask_ds,
        [1],
        vector_layer,
        burn_values=[1],
        where=f"{class_field} = {cls}"
    )

    mask = mask_ds.GetRasterBand(1).ReadAsArray()

    ys, xs = np.where(mask == 1)
    for y, x in zip(ys, xs):
        bx = band_x_data[y, x]
        by = band_y_data[y, x]

        if np.isnan(bx) or np.isnan(by):
            continue

        class_pixels.setdefault(cls, []).append((bx, by))

# -----------------------------
# APPLY SAMPLING RULE
# -----------------------------

class_sizes = {k: len(v) for k, v in class_pixels.items()}
min_size = min(class_sizes.values())

final_samples = {}

for cls, pixels in class_pixels.items():
    max_allowed = min(target_samples, 2 * min_size, len(pixels))
    final_samples[cls] = random.sample(pixels, max_allowed)

# -----------------------------
# PLOT 2D HISTOGRAMS
# -----------------------------

plt.figure(figsize=(10, 8))

for cls, samples in final_samples.items():
    data = np.array(samples)
    plt.hist2d(
        data[:, 0],
        data[:, 1],
        bins=200,
        cmap="viridis",
        alpha=0.6,
        norm=plt.matplotlib.colors.LogNorm()
    )

plt.xlabel("Blue band reflectance")
plt.ylabel("Red band reflectance")
plt.title("2D spectral density (Blue vs Red) by class")
plt.colorbar(label="Pixel density (log scale)")
plt.tight_layout()
plt.show()
