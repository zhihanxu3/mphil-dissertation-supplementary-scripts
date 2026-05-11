"""
================================================================================
Supraglacial Meltwater Area, Volume and Drainage Density Statistics

Author : Zhihan Xu
Date   : May 2026
================================================================================
Computes per-tile, per-AOI, and per-region statistics of supraglacial meltwater
features (lakes, channels, slush) from PlanetScope classification rasters
produced by:

    Random Forest (6-class):  3 = lake, 4 = channel, 5 = slush
    nnU-Net (3-class):        2 = lake, 3 = channel  (no slush)

PlanetScope pixel size is 3 m, so each pixel covers 9 m^2.  Areas are reported
in km^2 (per class, plus totals).  For supraglacial channels, drainage density 
is also computed (Dd, channel length per unit catchment area, km/km^2) by 
skeletonising the channel mask to get the channel length along its flowline.

Volumes are estimated by pairing each classification raster with its source
PlanetScope SuperDove imagery and applying the empirical depth formula of
Williamson et al. (2018) -- the same formula used by Zhang et al. (2023):

        d_pixel = 0.2764 * R_red^(-0.8952)        (Red reflectance in [0, 1])

PlanetScope has been radiometrically harmonised to Sentinel-2 so this formula
transfers directly.  Volumes are reported per class and as two combined
totals: (1) Lake + Channel (RF only; slush may not fit well in this empirical 
law) (2) Lake + Channel + Slush.

Regional grouping:
    Northwest Greenland (NW):       AOIs 1 - 7
    Central-West Greenland (CW):    AOIs 8 - 14
    Southwest Greenland (SW):       AOIs 15 - 22

Inputs:
1. Classification rasters (EPSG:3413, 3 m pixels, 4096 x 4096 typical)
    RF and DL result dataset filepath structures in our study are: 
       <BASE>/RF_6classes/Planet_<aoi>/planet_tile_<aoi>_<tile>.tif
       <BASE>/DL_3classes/Planet_<aoi>/planet_tile_<aoi>_<tile>.tif

    The script is written based on this structure. If the organisation 
    is different, change relevant codes in Section 1  'Detect tile filename' 
    block, Section 4 'def parse_tile_filename(name: str) -> 
    Optional[Tuple[str, str]]' block, and Section 8 'def build_per_tile_df' block.

2. Corresponding Source PlanetScope SuperDove imagery (8-band SR, EPSG:3413, 3 m)
       <IMAGERY_DIR>/Planet_AOI_<aoi>_sub_images/planet_tile_<aoi>_<tile>.tif
   Used only to read Band 6 (Red, 650-680 nm) for the depth retrieval.
   Each classification tile must have a geometrically matching imagery tile
   (same shape, CRS, and affine transform).

   The script is written based on this structure. If the organisation 
   is different, change relevant codes inchange Section 5 'def 
   find_imagery(orig_aoi: str, tile_num: str) -> Optional[Path]' 
   function block

Outputs (under <BASE>/Statistics_Results/):
    /tables/csv      per-tile, per-AOI, per-region tables (CSV)
    /tables/excel    same content as Excel workbooks
    /tables/word     same content as Word docx files (paste-ready for SI)
    /figures/        publication-quality PNG + PDF figures
    /run.log         run log

Run in a normal Python environment (e.g., VS Code)

Requirements:
    pip install numpy pandas rasterio scikit-image matplotlib seaborn ^
                openpyxl python-docx tqdm

================================================================================
"""

from __future__ import annotations

import logging
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from skimage.morphology import skeletonize
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CONFIGURATION  -- edit paths here only
# ============================================================================

# Root folder holding the two classifier output folders
BASE_DIR    = Path(r"C:\path\to\your\base_folder")
RF_DIR      = BASE_DIR / "RF_6classes"      # 6-class Random Forest results
DL_DIR      = BASE_DIR / "DL_3classes"      # 3-class deep-learning results

# Folder holding original PlanetScope imagery (needed for depth/volume calc).
# Each AOI lives in a sub-folder named  Planet_AOI_<orig_aoi>_sub_images/
IMAGERY_DIR = Path(r"C:\path\to\your\PlanetScope_imagery_folder")

# Output folder
OUT_DIR     = BASE_DIR / "Statistics_Results"
CSV_DIR     = OUT_DIR / "tables" / "csv"
XLSX_DIR    = OUT_DIR / "tables" / "excel"
DOCX_DIR    = OUT_DIR / "tables" / "word"
FIG_DIR     = OUT_DIR / "figures"
LOG_PATH    = OUT_DIR / "run.log"

# PlanetScope SuperDove band assignment (1-indexed for rasterio)
PS_RED_BAND = 6        # band 6 = Red (650-680 nm)

# Pixel size in metres
PIXEL_SIZE  = 3.0
PIXEL_AREA  = PIXEL_SIZE ** 2          # 9 m^2 per pixel

# Williamson et al. (2018) and Zhang et al. (2023) empirical depth coefficients
DEPTH_A     = 0.2764
DEPTH_B     = -0.8952

# Physically reasonable bounds for the depth retrieval.
#   R_MIN : reflectance below this is masked (Because R has a negative power, 
#           depth grows toward infinity as R approaches zero, thus there 
#           should be a threshold limiting the minimum possible value of depth)
#           When R = 0.005, derived depth is 31.73 m
R_MIN = 0.005

# RF / DL class definitions
RF_WATER_CLASSES = {
    3: ("Supraglacial Lakes",    "#1F4E79"),
    4: ("Supraglacial Channels", "#2E9CCA"),
    5: ("Slush",                 "#B5C7D3"),
}
DL_WATER_CLASSES = {
    2: ("Supraglacial Lakes",    "#1F4E79"),
    3: ("Supraglacial Channels", "#2E9CCA"),
}

# AOI re-naming (original -> formal name used in paper)
AOI_RENAME: Dict[str, str] = {
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
    "8": "10", "9": "11", "10": "12",
    "11A": "13A", "11B": "13B",
    "12": "15", "13": "17",
    "14": "14",
    "15": "19",
    "16A": "20A", "16B": "20B",
    "17": "18", "18": "21", "19": "22",
}

REGION_NAMES = {
    "NW": "Northwest Greenland",
    "CW": "Central-West Greenland",
    "SW": "Southwest Greenland",
}

# Detect tile filename: planet_tile_<aoi>_<tile>.tif or planet_tile_<aoi>_<tile>.tif
TILE_RE = re.compile(
    r"^planet_tile_(?P<aoi>\d+[A-Z]?)_(?P<tile>\d+)\.tif$",
    re.IGNORECASE,
)


# ============================================================================
# 2. PUBLICATION-QUALITY MATPLOTLIB STYLE  (matches my evaluation-metrics script)
# ============================================================================

def configure_plot_style() -> None:
    mpl.rcParams.update({
        "figure.dpi":         110,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "axes.labelsize":     11,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.9,
        "axes.edgecolor":     "#333333",
        "axes.titlepad":      10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "xtick.color":        "#333333",
        "ytick.color":        "#333333",
        "xtick.major.width":  0.9,
        "ytick.major.width":  0.9,
        "legend.fontsize":    9,
        "legend.frameon":     False,
        "lines.linewidth":    1.4,
        "patch.linewidth":    0.6,
    })


# ============================================================================
# 3. LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    for d in (OUT_DIR, CSV_DIR, XLSX_DIR, DOCX_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("water_stats")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setFormatter(fmt); log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); log.addHandler(sh)
    return log

LOG: logging.Logger  # populated in main()


# ============================================================================
# 4. HELPERS  --  filename parsing, AOI / region remapping, sort key
# ============================================================================

def parse_tile_filename(name: str) -> Optional[Tuple[str, str]]:
    """Return (orig_aoi, tile_number) from a filename like
    'planet_tile_11A_0003.tif'.  Returns None if no match."""
    m = TILE_RE.match(name)
    if not m:
        return None
    return m.group("aoi"), m.group("tile")


def to_formal_aoi(orig_aoi: str) -> str:
    """Map original AOI label (e.g. '11A') to the formal name used in the paper."""
    if orig_aoi not in AOI_RENAME:
        raise KeyError(f"No formal name defined for original AOI '{orig_aoi}'")
    return AOI_RENAME[orig_aoi]


def get_region(formal_aoi: str) -> str:
    """Return 'NW' / 'CW' / 'SW' based on the formal AOI numeric prefix."""
    n = int(re.match(r"(\d+)", formal_aoi).group(1))
    if 1  <= n <= 7:  return "NW"
    if 8  <= n <= 14: return "CW"
    if 15 <= n <= 22: return "SW"
    raise ValueError(f"Formal AOI '{formal_aoi}' is outside expected range")


def aoi_sort_key(formal_aoi: str) -> Tuple[int, str]:
    """Sort '12' < '13A' < '13B' < '15' correctly."""
    m = re.match(r"(\d+)([A-Z]?)", formal_aoi)
    return (int(m.group(1)), m.group(2) or "")


# ============================================================================
# 5. RASTER I/O + CONSISTENCY CHECK
# ============================================================================

def find_imagery(orig_aoi: str, tile_num: str) -> Optional[Path]:
    """Locate the source PlanetScope tile that corresponds to a classification."""
    folder = IMAGERY_DIR / f"Planet_AOI_{orig_aoi}_sub_images"
    cand   = folder / f"planet_tile_{orig_aoi}_{tile_num}.tif"
    return cand if cand.exists() else None


def consistency_check(cls_path: Path, img_path: Path) -> bool:
    """Verify that the classification raster and the PS imagery share the same
    shape, CRS, and affine transform.  I print a warning rather than crashing
    so the run can continue on the next tile."""
    with rasterio.open(cls_path) as c, rasterio.open(img_path) as i:
        same_shape = c.shape == i.shape
        same_crs   = c.crs   == i.crs
        # Transforms may differ in floating-point precision; compare to ~1e-6 m
        same_trf   = all(abs(a - b) < 1e-6
                         for a, b in zip(c.transform[:6], i.transform[:6]))
        ok = same_shape and same_crs and same_trf
        if not ok:
            LOG.warning(
                "Inconsistent geometry for %s vs %s: shape=%s crs=%s transform=%s",
                cls_path.name, img_path.name,
                same_shape, same_crs, same_trf,
            )
        return ok


def read_class_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def read_red_reflectance(path: Path) -> np.ndarray:
    """Read the Red band and convert to reflectance in [0, 1].
    PlanetScope SR products are typically scaled by 10000."""
    with rasterio.open(path) as src:
        red = src.read(PS_RED_BAND).astype(np.float32)
    # Auto-detect scaling: if values are clearly > 1.5, assume scale = 10000
    if np.nanmax(red) > 1.5:
        red = red / 10000.0
    return red


# ============================================================================
# 6. PHYSICAL QUANTITIES  --  channel length and water depth
# ============================================================================

def channel_length_km(channel_mask: np.ndarray, pixel_size: float = PIXEL_SIZE) -> float:
    """Total channel centreline length via skeletonisation.
    Each skeleton pixel contributes one pixel-width of length (simple count
    approximation, the standard quick method in supraglacial hydrology)."""
    if channel_mask.sum() == 0:
        return 0.0
    skel = skeletonize(channel_mask.astype(bool))
    return float(skel.sum()) * pixel_size / 1000.0   # metres -> km


def water_depth(red: np.ndarray, r_min: float = R_MIN) -> np.ndarray:
    """Apply Zhang et al. (2023) empirical depth model.
    Pixels with R <= r_min or non-finite are set to NaN to 
    avoid unrealistic depth values."""
    d = np.full_like(red, np.nan, dtype=np.float32)
    valid = (red > r_min) & np.isfinite(red)
    d[valid] = DEPTH_A * np.power(red[valid], DEPTH_B)
    return d


# ============================================================================
# 7. PER-TILE STATISTICS
# ============================================================================

@dataclass
class TileStats:
    """One row in the per-tile table.  All quantities for a single tile."""
    classifier: str
    orig_aoi:   str
    formal_aoi: str
    region:     str
    tile_id:    str
    tile_area_km2: float
    # area metrics, per class id
    area_km2:        Dict[int, float]
    area_frac_pct:   Dict[int, float]
    # channel-only metrics
    channel_length_km: float
    drainage_density_per_km: float
    # volume metrics, per class id
    volume_m3:        Dict[int, float]
    vol_density_m3_km2: Dict[int, float]
    # combined totals
    total_water_km2:    float
    total_water_pct:    float
    lc_volume_m3:       float
    all_volume_m3:      float
    lc_vol_density:     float
    all_vol_density:    float
    # diagnostics
    has_imagery: bool


def compute_tile_stats(
    cls_path: Path,
    classifier: str,
    classes_dict: Dict[int, Tuple[str, str]],
) -> TileStats:
    """Compute every statistic for one classification raster."""

    parsed = parse_tile_filename(cls_path.name)
    if parsed is None:
        raise ValueError(f"Cannot parse tile name: {cls_path.name}")
    orig_aoi, tile_num = parsed
    formal_aoi  = to_formal_aoi(orig_aoi)
    region      = get_region(formal_aoi)

    cls_arr = read_class_raster(cls_path)
    n_pix   = cls_arr.size
    tile_area_km2 = n_pix * PIXEL_AREA / 1e6     # km^2

    # --- area per class ---------------------------------------------------
    area_km2:      Dict[int, float] = {}
    area_frac_pct: Dict[int, float] = {}
    for cid in classes_dict:
        count = int((cls_arr == cid).sum())
        a_km2 = count * PIXEL_AREA / 1e6
        area_km2[cid]      = a_km2
        area_frac_pct[cid] = 100.0 * a_km2 / tile_area_km2

    # --- channel length & drainage density -------------------------------
    channel_cid = list(classes_dict.keys())[1]   # 4 for RF, 3 for DL
    ch_mask = (cls_arr == channel_cid)
    ch_len_km = channel_length_km(ch_mask)
    dd_per_km = ch_len_km / tile_area_km2 if tile_area_km2 > 0 else 0.0

    # --- volume calculation ----------------------------------------------
    img_path = find_imagery(orig_aoi, tile_num)
    volume_m3:        Dict[int, float] = {c: 0.0 for c in classes_dict}
    vol_density_m3_km2: Dict[int, float] = {c: 0.0 for c in classes_dict}
    has_imagery = img_path is not None

    if has_imagery:
        # Geometry sanity-check before sampling reflectance
        consistency_check(cls_path, img_path)
        red = read_red_reflectance(img_path)
        depth = water_depth(red)               # NaN where invalid
        for cid in classes_dict:
            mask = (cls_arr == cid) & np.isfinite(depth)
            if mask.any():
                v = float(np.nansum(depth[mask]) * PIXEL_AREA)   # m^3
            else:
                v = 0.0
            volume_m3[cid]        = v
            vol_density_m3_km2[cid] = v / tile_area_km2 if tile_area_km2 > 0 else 0.0
    else:
        LOG.warning("No PS imagery for %s (orig AOI %s tile %s) -- volumes set to NaN",
                    cls_path.name, orig_aoi, tile_num)
        for cid in classes_dict:
            volume_m3[cid]          = np.nan
            vol_density_m3_km2[cid] = np.nan

    # --- combined totals -------------------------------------------------
    lake_cid, channel_cid = list(classes_dict.keys())[0], list(classes_dict.keys())[1]
    has_slush = len(classes_dict) >= 3
    slush_cid = list(classes_dict.keys())[2] if has_slush else None

    total_water_km2 = sum(area_km2.values())
    total_water_pct = 100.0 * total_water_km2 / tile_area_km2

    lc_vol     = volume_m3[lake_cid] + volume_m3[channel_cid]
    all_vol    = lc_vol + (volume_m3[slush_cid] if has_slush else 0.0)
    lc_vd      = lc_vol  / tile_area_km2 if tile_area_km2 > 0 else 0.0
    all_vd     = all_vol / tile_area_km2 if tile_area_km2 > 0 else 0.0

    tile_id = f"{formal_aoi}_{tile_num}"

    return TileStats(
        classifier=classifier, orig_aoi=orig_aoi, formal_aoi=formal_aoi,
        region=region, tile_id=tile_id, tile_area_km2=tile_area_km2,
        area_km2=area_km2, area_frac_pct=area_frac_pct,
        channel_length_km=ch_len_km, drainage_density_per_km=dd_per_km,
        volume_m3=volume_m3, vol_density_m3_km2=vol_density_m3_km2,
        total_water_km2=total_water_km2, total_water_pct=total_water_pct,
        lc_volume_m3=lc_vol, all_volume_m3=all_vol,
        lc_vol_density=lc_vd, all_vol_density=all_vd,
        has_imagery=has_imagery,
    )


# ============================================================================
# 8. ASSEMBLE PER-TILE / PER-AOI / PER-REGION DATAFRAMES
# ============================================================================

# Short column names used in tables; classifier-specific values inserted via .format
COL_AREA   = "{cls}_area_km2"
COL_FRAC   = "{cls}_frac_pct"
COL_VOL    = "{cls}_volume_m3"
COL_VDENS  = "{cls}_voldens_m3_per_km2"

# Short labels for class IDs to keep column names clean
SHORT = {2: "Lake", 3: "Lake_RF",    # ambiguous: dispatched per-classifier below
         4: "Channel", 5: "Slush"}


def class_label(cid: int, classes_dict: Dict[int, Tuple[str, str]]) -> str:
    """Produce a short, unambiguous column-label component for a class id."""
    name = classes_dict[cid][0]
    if "Lake"    in name: return "Lake"
    if "Channel" in name: return "Channel"
    if "Slush"   in name: return "Slush"
    return f"Class{cid}"


def stats_to_row(ts: TileStats, classes_dict: Dict[int, Tuple[str, str]]) -> dict:
    """Flatten a TileStats into a dict suitable for a DataFrame row."""
    row = {
        "AOI":    ts.formal_aoi,
        "Tile":   ts.tile_id,
        "Region": ts.region,
        "tile_area_km2": round(ts.tile_area_km2, 4),
    }
    for cid, (name, _) in classes_dict.items():
        lab = class_label(cid, classes_dict)
        row[f"{lab}_area_km2"]   = round(ts.area_km2[cid], 6)
        row[f"{lab}_frac_pct"]   = round(ts.area_frac_pct[cid], 4)
    row["TotalWater_area_km2"] = round(ts.total_water_km2, 6)
    row["TotalWater_frac_pct"] = round(ts.total_water_pct, 4)
    row["Channel_length_km"]   = round(ts.channel_length_km, 4)
    row["Drainage_density_per_km"] = round(ts.drainage_density_per_km, 5)
    for cid in classes_dict:
        lab = class_label(cid, classes_dict)
        row[f"{lab}_volume_m3"]        = (round(ts.volume_m3[cid], 1)
                                          if np.isfinite(ts.volume_m3[cid]) else np.nan)
        row[f"{lab}_voldens_m3_per_km2"] = (round(ts.vol_density_m3_km2[cid], 1)
                                          if np.isfinite(ts.vol_density_m3_km2[cid]) else np.nan)
    row["LakeChannel_volume_m3"]   = (round(ts.lc_volume_m3, 1)
                                       if np.isfinite(ts.lc_volume_m3) else np.nan)
    row["LakeChannel_voldens_m3_per_km2"] = (round(ts.lc_vol_density, 1)
                                       if np.isfinite(ts.lc_vol_density) else np.nan)
    if len(classes_dict) >= 3:
        row["AllWater_volume_m3"]   = (round(ts.all_volume_m3, 1)
                                       if np.isfinite(ts.all_volume_m3) else np.nan)
        row["AllWater_voldens_m3_per_km2"] = (round(ts.all_vol_density, 1)
                                       if np.isfinite(ts.all_vol_density) else np.nan)
    return row


def build_per_tile_df(
    classifier_dir: Path,
    classifier: str,
    classes_dict: Dict[int, Tuple[str, str]],
) -> pd.DataFrame:
    """Scan one classifier folder, compute stats for every tile, return a DataFrame.
    The folder layout expected:   <classifier_dir>/Planet_<orig_aoi>/planet_tile_<aoi>_<tile>.tif"""

    rows: List[dict] = []
    aoi_folders = sorted([p for p in classifier_dir.iterdir() if p.is_dir()])
    LOG.info("Found %d AOI folders under %s", len(aoi_folders), classifier_dir)

    for aoi_folder in aoi_folders:
        tile_files = sorted(aoi_folder.glob("planet_tile_*.tif"))
        if not tile_files:
            LOG.warning("No prediction tiles in %s", aoi_folder)
            continue
        for tf in tqdm(tile_files, desc=f"{classifier} | {aoi_folder.name}", leave=False):
            try:
                ts = compute_tile_stats(tf, classifier, classes_dict)
                rows.append(stats_to_row(ts, classes_dict))
            except Exception as e:
                LOG.exception("Failed on %s: %s", tf, e)

    df = pd.DataFrame(rows)
    # Order rows by (AOI numeric, AOI letter, tile)
    if not df.empty:
        df["_sort"] = df["AOI"].map(aoi_sort_key)
        df = df.sort_values(by=["_sort", "Tile"]).drop(columns="_sort").reset_index(drop=True)
    return df


def aggregate_per_aoi(per_tile: pd.DataFrame,
                      classes_dict: Dict[int, Tuple[str, str]]) -> pd.DataFrame:
    """Aggregate per-tile results to AOI level.
    Densities and fractions are AVERAGED across tiles (sums would be misleading
    because tiles only sample part of each AOI).  Total areas / lengths are
    given as summed values for information only."""

    if per_tile.empty:
        return pd.DataFrame()

    # Columns to average (per-area densities and fractions)
    mean_cols = [c for c in per_tile.columns
                 if c.endswith("_frac_pct")
                 or c.endswith("_voldens_m3_per_km2")
                 or c == "Drainage_density_per_km"]
    # Columns to sum (raw totals; informational)
    sum_cols = [c for c in per_tile.columns
                if c.endswith("_area_km2")
                or c.endswith("_volume_m3")
                or c == "Channel_length_km"
                or c == "tile_area_km2"]

    grouped = per_tile.groupby("AOI", sort=False)
    agg = {c: "mean" for c in mean_cols}
    agg.update({c: "sum" for c in sum_cols})
    agg["Region"] = "first"
    out = grouped.agg(agg).reset_index()
    out["n_tiles"] = grouped.size().values

    # Rename: averaged columns get an explicit "Mean_" prefix; totals "Total_"
    rename_map = {}
    for c in mean_cols:    rename_map[c] = f"Mean_{c}"
    for c in sum_cols:     rename_map[c] = f"Total_{c}"
    out = out.rename(columns=rename_map)

    # Re-order: identity columns first, then totals, then means
    front = ["AOI", "Region", "n_tiles"]
    rest  = [c for c in out.columns if c not in front]
    out   = out[front + rest]

    out["_sort"] = out["AOI"].map(aoi_sort_key)
    out = out.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    # Round for legibility
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(4)
    return out


def aggregate_per_region(per_aoi: pd.DataFrame) -> pd.DataFrame:
    """Region averages.  I take the mean of per-AOI means (not pooling tiles),
    so each AOI carries equal weight regardless of how many tiles I happened
    to process there."""

    if per_aoi.empty:
        return pd.DataFrame()

    mean_cols = [c for c in per_aoi.columns if c.startswith("Mean_")]
    grouped = per_aoi.groupby("Region", sort=False)
    out = grouped[mean_cols].mean().reset_index()
    out["n_AOIs"] = grouped.size().values
    out["Region_name"] = out["Region"].map(REGION_NAMES)

    front = ["Region", "Region_name", "n_AOIs"]
    rest  = [c for c in out.columns if c not in front]
    out   = out[front + rest]

    # Standard order: NW, CW, SW
    order = {"NW": 0, "CW": 1, "SW": 2}
    out["_o"] = out["Region"].map(order)
    out = out.sort_values("_o").drop(columns="_o").reset_index(drop=True)

    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(4)
    return out


# ============================================================================
# 9. TABLE OUTPUT  --  CSV, Excel, Word
# ============================================================================

def save_csv(df: pd.DataFrame, name: str) -> None:
    p = CSV_DIR / f"{name}.csv"
    df.to_csv(p, index=False)
    LOG.info("Wrote %s", p)


def save_excel(dfs: Dict[str, pd.DataFrame], name: str) -> None:
    """All DataFrames go into one workbook, one per sheet."""
    p = XLSX_DIR / f"{name}.xlsx"
    with pd.ExcelWriter(p, engine="openpyxl") as xw:
        for sheet, df in dfs.items():
            df.to_excel(xw, sheet_name=sheet[:31], index=False)
    LOG.info("Wrote %s", p)


def save_word(df: pd.DataFrame, name: str, caption: str) -> None:
    """Produce a paste-ready supplementary table in Word.  Landscape, Arial 8pt,
    bold dark-blue header row, alternating light-blue row shading. """
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.section import WD_ORIENT
    from docx.enum.table import WD_ALIGN_VERTICAL
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    doc = Document()
    # Landscape A4
    sec = doc.sections[0]
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    sec.left_margin = sec.right_margin = Cm(1.2)
    sec.top_margin  = sec.bottom_margin = Cm(1.4)

    # Title + caption
    title = doc.add_paragraph()
    run = title.add_run(name.replace("_", " "))
    run.font.name = "Arial"; run.font.size = Pt(11); run.bold = True
    cap = doc.add_paragraph()
    crun = cap.add_run(caption)
    crun.font.name = "Arial"; crun.font.size = Pt(9); crun.italic = True

    # Build the table
    n_rows, n_cols = df.shape
    tbl = doc.add_table(rows=n_rows + 1, cols=n_cols)
    tbl.style = "Light Grid Accent 1"
    tbl.autofit = True

    # Header row
    for j, col in enumerate(df.columns):
        cell = tbl.rows[0].cells[j]
        cell.text = ""
        para = cell.paragraphs[0]
        r = para.add_run(str(col))
        r.font.name = "Arial"; r.font.size = Pt(7); r.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        # Dark navy fill
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), "1F4E79")
        cell._tc.get_or_add_tcPr().append(shd)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Data rows
    for i in range(n_rows):
        for j in range(n_cols):
            cell = tbl.rows[i + 1].cells[j]
            v = df.iat[i, j]
            s = "" if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else str(v))
            cell.text = ""
            r = cell.paragraphs[0].add_run(s)
            r.font.name = "Arial"; r.font.size = Pt(7)
            if i % 2 == 1:
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"), "EBF3FB")
                cell._tc.get_or_add_tcPr().append(shd)

    p = DOCX_DIR / f"{name}.docx"
    doc.save(p)
    LOG.info("Wrote %s", p)


# ============================================================================
# 10. FIGURES
# ============================================================================

def _aoi_xticks(ax, aois: List[str]) -> None:
    ax.set_xticks(np.arange(len(aois)))
    ax.set_xticklabels(aois, rotation=0, ha="center")
    ax.set_xlabel("AOI")


def fig_area_per_aoi(per_aoi_rf: pd.DataFrame, per_aoi_dl: pd.DataFrame) -> None:
    """Figure: total water area summed across processed tiles per AOI, stacked
    by class.  Two panels (RF top, DL bottom).  Areas are sums (km^2) for tiles 
    within each AOI and shown in the figure as 'cumulative sampled area'."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)

    for ax, df, classes, title in [
        (axes[0], per_aoi_rf, RF_WATER_CLASSES, "Random Forest (6-class)"),
        (axes[1], per_aoi_dl, DL_WATER_CLASSES, "Deep learning nnU-Net (3-class)"),
    ]:
        if df.empty:
            ax.set_axis_off(); continue
        aois = df["AOI"].tolist()
        x = np.arange(len(aois))
        bottom = np.zeros(len(aois))
        for cid, (name, color) in classes.items():
            lab = class_label(cid, classes)
            col = f"Total_{lab}_area_km2"
            y = df[col].fillna(0).values
            ax.bar(x, y, bottom=bottom, color=color, edgecolor="white",
                   linewidth=0.5, label=name)
            bottom = bottom + y
        ax.set_title(title)
        ax.set_ylabel(r"Cumulative sampled area (km$^2$)")
        _aoi_xticks(ax, aois)
        ax.legend(loc="upper right", ncols=len(classes))
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"fig01_area_per_aoi.{ext}")
    plt.close()


def fig_density_per_aoi(per_aoi_rf: pd.DataFrame, per_aoi_dl: pd.DataFrame) -> None:
    """Figure: per-AOI mean water-area fraction (% of tile area), per class."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, df, classes, title in [
        (axes[0], per_aoi_rf, RF_WATER_CLASSES, "Random Forest (6-class)"),
        (axes[1], per_aoi_dl, DL_WATER_CLASSES, "Deep learning nnU-Net (3-class)"),
    ]:
        if df.empty:
            ax.set_axis_off(); continue
        aois = df["AOI"].tolist()
        x = np.arange(len(aois))
        width = 0.8 / max(len(classes), 1)
        for k, (cid, (name, color)) in enumerate(classes.items()):
            lab = class_label(cid, classes)
            col = f"Mean_{lab}_frac_pct"
            y = df[col].fillna(0).values
            ax.bar(x + (k - (len(classes) - 1) / 2) * width, y, width=width,
                   color=color, edgecolor="white", linewidth=0.4, label=name)
        ax.set_title(title)
        ax.set_ylabel("Mean water area fraction (%)")
        _aoi_xticks(ax, aois)
        ax.legend(loc="upper right", ncols=len(classes))
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"fig02_fraction_per_aoi.{ext}")
    plt.close()


def fig_volume_density_per_aoi(per_aoi_rf: pd.DataFrame, per_aoi_dl: pd.DataFrame) -> None:
    """Figure: mean volume density (m^3 / km^2) per class per AOI."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, df, classes, title in [
        (axes[0], per_aoi_rf, RF_WATER_CLASSES, "Random Forest (6-class)"),
        (axes[1], per_aoi_dl, DL_WATER_CLASSES, "Deep learning nnU-Net (3-class)"),
    ]:
        if df.empty:
            ax.set_axis_off(); continue
        aois = df["AOI"].tolist()
        x = np.arange(len(aois))
        width = 0.8 / max(len(classes), 1)
        for k, (cid, (name, color)) in enumerate(classes.items()):
            lab = class_label(cid, classes)
            col = f"Mean_{lab}_voldens_m3_per_km2"
            y = df[col].fillna(0).values
            ax.bar(x + (k - (len(classes) - 1) / 2) * width, y, width=width,
                   color=color, edgecolor="white", linewidth=0.4, label=name)
        ax.set_title(title)
        ax.set_ylabel(r"Mean volume density (m$^3$ km$^{-2}$)")
        _aoi_xticks(ax, aois)
        ax.legend(loc="upper right", ncols=len(classes))
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"fig03_volume_density_per_aoi.{ext}")
    plt.close()


def fig_drainage_density(per_aoi_rf: pd.DataFrame, per_aoi_dl: pd.DataFrame) -> None:
    """Figure: mean drainage density per AOI (channels only)."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    bw = 0.4

    def plot_one(df, offset, color, label):
        if df.empty: return
        aois = df["AOI"].tolist()
        x = np.arange(len(aois))
        y = df["Mean_Drainage_density_per_km"].fillna(0).values
        ax.bar(x + offset, y, bw, color=color, edgecolor="white",
               linewidth=0.4, label=label)
        return aois

    aois = plot_one(per_aoi_rf, -bw / 2, "#2E9CCA", "Random Forest")
    a2   = plot_one(per_aoi_dl, +bw / 2, "#1F4E79", "Deep learning nnU-Net")
    aois = aois or a2 or []
    ax.set_ylabel(r"Drainage density (km km$^{-2}$)")
    ax.set_title("Mean drainage density per AOI -- supraglacial channels")
    _aoi_xticks(ax, aois)
    ax.legend(loc="upper right")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"fig04_drainage_density.{ext}")
    plt.close()


def fig_region_summary(per_region_rf: pd.DataFrame,
                       per_region_dl: pd.DataFrame) -> None:
    """Two-panel figure: regional mean fraction (top) and volume density (bottom),
    with grouped bars for each region."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    def grouped(ax, df, classes, value_prefix, ylabel):
        if df.empty:
            ax.set_axis_off(); return
        regions = df["Region"].tolist()
        x = np.arange(len(regions))
        width = 0.8 / max(len(classes), 1)
        for k, (cid, (name, color)) in enumerate(classes.items()):
            lab = class_label(cid, classes)
            col = f"Mean_{lab}_{value_prefix}"
            y = df[col].fillna(0).values
            ax.bar(x + (k - (len(classes) - 1) / 2) * width, y, width=width,
                   color=color, edgecolor="white", linewidth=0.4, label=name)
        ax.set_xticks(x); ax.set_xticklabels([REGION_NAMES[r] for r in regions])
        ax.set_ylabel(ylabel)

    # Use RF for top panel as it has all three classes; DL drawn faintly behind
    grouped(axes[0], per_region_rf, RF_WATER_CLASSES, "frac_pct",
            "Mean water area fraction (%)")
    axes[0].set_title("Area fraction by region (Random Forest)")
    axes[0].legend(loc="upper right")

    grouped(axes[1], per_region_rf, RF_WATER_CLASSES, "voldens_m3_per_km2",
            r"Mean volume density (m$^3$ km$^{-2}$)")
    axes[1].set_title("Volume density by region (Random Forest)")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"fig05_region_summary.{ext}")
    plt.close()


# ============================================================================
# 11. MAIN
# ============================================================================

def main() -> None:
    global LOG
    configure_plot_style()
    LOG = setup_logging()
    LOG.info("=== Supraglacial meltwater area / volume statistics ===")
    LOG.info("Base directory : %s", BASE_DIR)
    LOG.info("Imagery folder : %s", IMAGERY_DIR)
    LOG.info("Pixel size     : %s m   (pixel area = %s m^2)", PIXEL_SIZE, PIXEL_AREA)

    # ---- Random Forest ----
    LOG.info("Processing Random Forest (6-class) tiles ...")
    per_tile_rf  = build_per_tile_df(RF_DIR, "RF", RF_WATER_CLASSES)
    per_aoi_rf   = aggregate_per_aoi(per_tile_rf, RF_WATER_CLASSES)
    per_region_rf = aggregate_per_region(per_aoi_rf)

    # ---- Deep learning ----
    LOG.info("Processing nnU-Net (3-class) tiles ...")
    per_tile_dl  = build_per_tile_df(DL_DIR, "DL", DL_WATER_CLASSES)
    per_aoi_dl   = aggregate_per_aoi(per_tile_dl, DL_WATER_CLASSES)
    per_region_dl = aggregate_per_region(per_aoi_dl)

    # ---- Save tables (CSV / Excel / Word) ---------------------------------
    LOG.info("Writing tables ...")
    save_csv(per_tile_rf,    "per_tile_RF")
    save_csv(per_tile_dl,    "per_tile_DL")
    save_csv(per_aoi_rf,     "per_aoi_RF")
    save_csv(per_aoi_dl,     "per_aoi_DL")
    save_csv(per_region_rf,  "per_region_RF")
    save_csv(per_region_dl,  "per_region_DL")

    save_excel({
        "per_tile_RF":  per_tile_rf,
        "per_tile_DL":  per_tile_dl,
        "per_aoi_RF":   per_aoi_rf,
        "per_aoi_DL":   per_aoi_dl,
        "per_region_RF": per_region_rf,
        "per_region_DL": per_region_dl,
    }, "water_statistics")

    save_word(per_tile_rf, "Table_S_per_tile_RF",
              "Per-tile water-feature statistics from the 6-class Random Forest "
              "classification.  Areas in km^2, area fractions in %, channel "
              "length in km, drainage density in km/km^2, volumes in m^3, "
              "volume densities in m^3/km^2.")
    save_word(per_tile_dl, "Table_S_per_tile_DL",
              "Per-tile water-feature statistics from the 3-class nnU-Net "
              "(deep-learning) classification.")
    save_word(per_aoi_rf, "Table_S_per_aoi_RF",
              "Per-AOI summary (Random Forest).  Total_* columns are sums across "
              "processed tiles; Mean_* columns are means across tiles within an "
              "AOI.  Densities and fractions are reported as means because "
              "processed tiles do not cover the full AOI extent.")
    save_word(per_aoi_dl, "Table_S_per_aoi_DL",
              "Per-AOI summary (nnU-Net).")
    save_word(per_region_rf, "Table_S_per_region_RF",
              "Drainage-basin (region) summary, Random Forest.  Each value is "
              "the mean of per-AOI means within the region (equal weight per AOI).")
    save_word(per_region_dl, "Table_S_per_region_DL",
              "Drainage-basin (region) summary, nnU-Net.")

    # ---- Figures ----------------------------------------------------------
    LOG.info("Drawing figures ...")
    fig_area_per_aoi(per_aoi_rf, per_aoi_dl)
    fig_density_per_aoi(per_aoi_rf, per_aoi_dl)
    fig_volume_density_per_aoi(per_aoi_rf, per_aoi_dl)
    fig_drainage_density(per_aoi_rf, per_aoi_dl)
    fig_region_summary(per_region_rf, per_region_dl)

    LOG.info("All outputs written to %s", OUT_DIR)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
