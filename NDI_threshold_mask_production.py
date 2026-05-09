# ============================================================
# NDI Threshold-Based Binary Mask Production
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
# Description:
#   Computes any two-band normalised difference index (NDI)
#   and produces a single-class binary raster mask by applying
#   a user-defined threshold. Designed for quick mask
#   generation (mainly for NDWIice, NDWI) before manual
#   annotation or random forest classification.
# Inputs:  PlanetScope multispectral GeoTIFF
# Outputs: Single-band GeoTIFF mask (value = class_id where
#          NDI > threshold, 0 elsewhere)
# ============================================================

from qgis.core import QgsRasterLayer
from qgis.analysis import QgsRasterCalculator, QgsRasterCalculatorEntry

# ==========================================================
# USER INPUT
# ==========================================================

input_raster_path  = r"C:\path\to\your\input_image.tif"   # PlanetScope GeoTIFF
output_raster_path = r"C:\path\to\your\output_mask.tif"   # where to save the mask

# NDI band selection (1-based band numbers in your raster)
# e.g. NDWIice: band_a = Green (4), band_b = Red (6)
#      NDWI:    band_a = Green (4), band_b = NIR  (8)
band_a_number = 4   # numerator band   (e.g. Green)
band_b_number = 6   # denominator band (e.g. Red)
band_a_label  = "band_a"
band_b_label  = "band_b"

threshold = 0.10    # pixels with NDI > threshold are assigned class_id
class_id  = 3       # class label that the output mask will have that can facilitate later machine learning training data annotation

# ==========================================================
# LOAD RASTER
# ==========================================================

raster = QgsRasterLayer(input_raster_path, "input_raster")
if not raster.isValid():
    raise Exception("Raster failed to load — check input_raster_path.")

# ==========================================================
# SET UP CALCULATOR ENTRIES
# ==========================================================

def make_entry(label, raster, band_number):
    e = QgsRasterCalculatorEntry()
    e.ref = f"{label}@1"
    e.raster = raster
    e.bandNumber = band_number
    return e

entries = [
    make_entry(band_a_label, raster, band_a_number),
    make_entry(band_b_label, raster, band_b_number)
]

# NDI = (band_a - band_b) / (band_a + band_b); assign class_id where NDI > threshold
expression = (
    f"(({band_a_label}@1 - {band_b_label}@1) / "
    f"({band_a_label}@1 + {band_b_label}@1) > {threshold}) * {class_id}"
)

# ==========================================================
# RUN AND SAVE
# ==========================================================

calc = QgsRasterCalculator(
    expression, output_raster_path, "GTiff",
    raster.extent(), raster.width(), raster.height(), entries
)

if calc.processCalculation() != 0:
    raise Exception("Raster calculation failed.")

print(f"Mask saved to: {output_raster_path}")
