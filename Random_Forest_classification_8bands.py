# ============================================================
# Random Forest Classification — Supraglacial Feature Mapping
# Author: Zhihan Xu
# Date: May 2026
# ============================================================
#
# Description:
#   Trains and applies a Random Forest classifier to map
#   supraglacial features from PlanetScope SuperDove 8-band
#   surface reflectance imagery. Features are engineered from
#   raw band values, spectral indices (NDWIice, NDWI, NDGI,
#   NDVI, etc.), and local texture statistics. Training pixels
#   are drawn from manually labelled RF masks and supplemented
#   by nnU-Net DL masks for lake and channel classes.
#   Post-processing applies a probability threshold, skeleton
#   enforcement, and an elongation filter to refine channel
#   predictions across the full image.
#
# Classification scheme:
#   0 = Dirty ice / debris-influenced ice
#   1 = Snow / clean ice
#   2 = Blue ice
#   3 = Supraglacial lake
#   4 = Supraglacial river / channel
#   5 = Slush

#PlanetScope SuperDove 8-band order (0-indexed in Python arrays)

#  Index 0  = Coastal Blue  (431–452 nm)
#  Index 1  = Blue          (465–515 nm)
#  Index 2  = Green I       (513–549 nm)
#  Index 3  = Green II      (547–583 nm)
#  Index 4  = Yellow        (600–620 nm)
#  Index 5  = Red           (650–680 nm)
#  Index 6  = Red Edge      (697–713 nm)
#  Index 7  = NIR           (845–885 nm)

# Modes:
#   MODE = "train" — collects training pixels from RF and DL
#                    masks, trains the model, saves to disk
#   MODE = "apply" — loads saved model, predicts all tiles in
#                    APPLY_IMAGE_DIR, runs post-processing
#
# Inputs (train):
#   - RF training GeoTIFFs and corresponding class masks
#   - DL training masks (nnU-Net outputs) for lake/channel

# Inputs (apply):
#   - PlanetScope GeoTIFF tiles in APPLY_IMAGE_DIR
#   - Trained model .pkl file at MODEL_PATH
#
# Outputs:
#   - Predictions_raw/  : per-tile raw classification GeoTIFFs
#   - Predictions_final/: post-processed classification GeoTIFFs
#   - RF_Model_6classes_8bands.pkl : saved trained model
# ============================================================



import os
import re
import gc
import glob
import warnings

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import generic_filter, label as ndimage_label
from skimage.morphology import skeletonize, binary_dilation, disk
from skimage.measure import regionprops, label as sk_label
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# =====================  USER SETTINGS  =======================================
# =============================================================================

MODE = "apply"          # "train"  or  "apply"

# ------ Paths ----------------------------------------------------------------

BASE = "/rds/user/path/to/your/base_folder"   

# RF-purpose training data (original manually labelled rasters)
RF_TRAIN_MASK_DIR  = os.path.join(BASE, "Training_Masks")
RF_TRAIN_IMAGE_DIR = os.path.join(BASE, "Training_Images")

# DL-purpose training masks (nnU-Net masks — reused here for channel/lake pixels)
DL_TRAIN_MASK_DIR  = "/rds/user/path/to/your/DL/masks"
DL_TRAIN_IMAGE_DIR = os.path.join(BASE, "Training_Images")   # same image folder

# Images to apply the model to
APPLY_IMAGE_DIR = os.path.join(BASE, "Apply_Images")

# Output directories
OUTPUT_DIR     = os.path.join(BASE, "Output")
PRED_DIR_RAW   = os.path.join(OUTPUT_DIR, "Predictions_raw")    # before post-process
PRED_DIR_FINAL = os.path.join(OUTPUT_DIR, "Predictions_final")  # after post-process
MODEL_PATH     = os.path.join(OUTPUT_DIR, "RF_Model_6classes_8bands.pkl")

for d in [OUTPUT_DIR, PRED_DIR_RAW, PRED_DIR_FINAL]:
    os.makedirs(d, exist_ok=True)

# ------ Class and sampling settings ------------------------------------------

# Classes present in RF-purpose masks
VALID_RF_CLASSES = {0, 1, 2, 3, 4, 5}

# DL mask label values (original nnU-Net mask values, before any remapping)
DL_CHANNEL_CLASS = 4    # channel label in DL masks
DL_LAKE_CLASS    = 3    # lake    label in DL masks
DL_NODATA_VALUES = {0, 255}  # nodata/ignore in DL masks

# Prediction settings
CHANNEL_CLASS     = 4   # channel label in output predictions
SAFE_NODATA       = 255

# Pixel cap per class per RF training mask
MAX_PIXELS_CLASS_0125 = 45000   # debris ice, snow/ice, blue ice, slush
MAX_PIXELS_CLASS_34   = 25000   # lake and channel
#lower cap because pixels for lake an channel will also be got from DL training masks

# Scale factor: Planet SR is stored as DN × 10000
SCALE_FACTOR = 1 / 10000.0

# Tile size for apply-mode prediction
TILE_SIZE = 1024

# Post-processing parameters
CHANNEL_PROB_THRESHOLD = 0.70   # minimum probability to accept channel label
                                 # tune between 0.55 (lenient) and 0.70 (strict)
MAX_HALF_WIDTH         = 3      # skeleton re-expansion in pixels
                                 # 1 → max 3 px wide, 2 → max 5 px wide
MIN_CHANNEL_AREA       = 20     # minimum channel blob size (pixels)
MIN_ECCENTRICITY       = 0.85   # blobs less elongated than this are removed


# =============================================================================
# =====================  FEATURE ENGINEERING  =================================
# =============================================================================

def compute_features(ps):
    """
    Compute spectral indices and texture features from a 8-band PS array.

    Parameters
    ----------
    ps : np.ndarray, shape (H, W, 8), float32, already scaled to [0, 1]

    Returns
    -------
    features : np.ndarray, shape (H, W, N_FEATURES)

    Band assignments (0-indexed):
        0 = Coastal Blue  (CB)   431–452 nm
        1 = Blue          (B)    465–515 nm
        2 = Green I       (G1)   513–549 nm
        3 = Green II      (G2)   547–583 nm
        4 = Yellow        (Y)    600–620 nm
        5 = Red           (R)    650–680 nm
        6 = Red Edge      (RE)   697–713 nm
        7 = NIR           (N)    845–885 nm
    """
    eps = 1e-6

    CB = ps[:, :, 0]   # Coastal Blue
    B  = ps[:, :, 1]   # Blue
    G1 = ps[:, :, 2]   # Green I
    G2 = ps[:, :, 3]   # Green II
    Y  = ps[:, :, 4]   # Yellow
    R  = ps[:, :, 5]   # Red
    RE = ps[:, :, 6]   # Red Edge
    N  = ps[:, :, 7]   # NIR

    # ------------------------------------------------------------------
    # Corrected spectral indices for supraglacial water on ice/snow
    # ------------------------------------------------------------------

    # NDWI1 (McFeeters 1996): open water detection using Green I and NIR
    # High for snow/ice, lower for liquid water.
    NDWI1 = (G1 - N) / (G1 + N + eps)

    # NDWIice: adapted for ice/liquid discrimination; uses Blue vs Red (Yang and Smith, 2013)
    # Liquid supraglacial water has lower NIR reflectance than frozen surface
    NDWIice = (B - R) / (B + R + eps)

    # NDGI (Normalised Difference Greenness Index):
    # channels and lakes appear slightly greener than surrounding white ice
    NDGI = (G2 - R) / (G2 + R + eps)

    # NDVI: vegetation index — low for all surface types here,
    # but still provides contrast between liquid water (very low) and ice
    NDVI = (N - R) / (N + R + eps)

    # NDWI (McFeeters 1996): uses Green II and NIR
    # High for snow/ice, lower for liquid water.
    NDWI = (G2 - N) / (G2 + N + eps)

    # NDWI_CB: Coastal Blue captures very short-wave scattering
    # which is enhanced over liquid water compared to snow/ice
    NDWI_CB = (CB - N) / (CB + N + eps)

    # Blue-NIR ratio: liquid water has relatively higher blue reflectance
    # than NIR compared to snow/ice (which is high in both)
    B_NIR = (B - N) / (B + N + eps)

    # Yellow band helps separate slush (brownish/yellowish) from clean ice
    # and from channels (which are darker uniformly)
    Y_R = Y / (R + eps)

    # Red Edge vs NIR: liquid water strongly absorbs in red-edge;
    # ratio drops clearly for open water vs ice
    RE_N = RE / (N + eps)

    # ------------------------------------------------------------------
    # Brightness and colour ratios
    # ------------------------------------------------------------------

    # Overall visible brightness: channels and lakes are darker than white ice
    brightness = (B + G1 + G2 + R) / 4.0

    # Blue fraction of visible brightness:
    # liquid water scatters blue more relative to red than white ice does
    blue_fraction = B / (brightness + eps)

    # Coastal blue fraction: similar principle, uses shortest wavelength
    CB_fraction = CB / (brightness + eps)

    # R/G ratio: very high for debris ice (brownish), low for water
    R_G_ratio = R / (G1 + eps)

    # B/R ratio: high for clear water, lower for turbid or ice surfaces
    B_R_ratio = B / (R + eps)

    # ------------------------------------------------------------------
    # Texture features (local statistics in sliding windows)
    # ------------------------------------------------------------------
    # Channels create sharp dark-bright boundaries against surrounding ice.

    # Standard deviation (measures local variability / edges)
    NIR_std_3   = generic_filter(N,  np.std, size=3)
    Green_std_3 = generic_filter(G1, np.std, size=3)
    NIR_std_7   = generic_filter(N,  np.std, size=7)
    Blue_std_5  = generic_filter(B,  np.std, size=5)
    CB_std_5    = generic_filter(CB, np.std, size=5)

    # Local range (max - min): captures sharp channel edges efficiently
    NIR_range_5  = generic_filter(N,  lambda x: x.max() - x.min(), size=5)
    Blue_range_5 = generic_filter(B,  lambda x: x.max() - x.min(), size=5)

    # ------------------------------------------------------------------
    # Stack all features
    # ------------------------------------------------------------------
    features = np.dstack([
        # Raw bands (8)
        CB, B, G1, G2, Y, R, RE, N,
        # Normalised indices (7)
        NDWI1, NDWIice, NDGI, NDVI, NDWI, NDWI_CB, B_NIR,
        # Additional spectral ratios (4)
        Y_R, RE_N, R_G_ratio, B_R_ratio,
        # Brightness and colour fractions (3)
        brightness, blue_fraction, CB_fraction,
        # Texture — std (5)
        NIR_std_3, Green_std_3, NIR_std_7, Blue_std_5, CB_std_5,
        # Texture — range (2)
        NIR_range_5, Blue_range_5,
    ])
    # Total: 8 + 7 + 4 + 3 + 5 + 2 = 29 features

    return features.astype(np.float32)


# =============================================================================
# =====================  POST-PROCESSING  =====================================
# =============================================================================

def enforce_channel_width(pred, max_half_width=2):
    """
    Collapse predicted channel blobs to their skeleton centreline,
    then re-expand by at most max_half_width pixels.

    This prevents artificially wide channel predictions caused by
    mixed-pixel spectral contamination at channel boundaries.

    max_half_width=1 → channels at most 3 pixels wide
    max_half_width=2 → channels at most 5 pixels wide (recommended start)
    """
    channel_mask = (pred == CHANNEL_CLASS)
    if not np.any(channel_mask):
        return pred

    # Skeletonize: reduce to 1-pixel-wide centreline
    skeleton = skeletonize(channel_mask)

    # Re-expand centreline by controlled amount
    selem = disk(max_half_width)
    controlled = binary_dilation(skeleton, selem)

    # Only keep pixels that were both channel and within the controlled buffer
    final_channel = controlled & channel_mask

    pred_out = pred.copy()
    removed  = channel_mask & ~final_channel
    pred_out[removed]       = 1              # reassign stripped boundary to ice
    pred_out[final_channel] = CHANNEL_CLASS

    return pred_out


def filter_channel_by_shape(pred, min_area=20, min_eccentricity=0.85):
    """
    Remove channel predictions that are either:
      (a) too small (< min_area pixels), OR
      (b) too circular (eccentricity < min_eccentricity)

    Real channels and streams are highly elongated (eccentricity ≈ 1.0).
    A perfect circle has eccentricity = 0.
    A very thin line has eccentricity → 1.

    min_eccentricity=0.85 removes blobs less elongated than roughly a 2:1
    aspect-ratio ellipse. Curved channels retain high eccentricity.
    """
    channel_mask = (pred == CHANNEL_CLASS)
    if not np.any(channel_mask):
        return pred

    labeled = sk_label(channel_mask)
    props   = regionprops(labeled)

    pred_out = pred.copy()
    for region in props:
        remove = (region.area < min_area) or (region.eccentricity < min_eccentricity)
        if remove:
            rows, cols = region.coords[:, 0], region.coords[:, 1]
            pred_out[rows, cols] = 1   # reassign to snow/ice

    return pred_out


def postprocess_full_image(raw_path, out_path,
                            max_half_width=MAX_HALF_WIDTH,
                            min_eccentricity=MIN_ECCENTRICITY,
                            min_area=MIN_CHANNEL_AREA):
    """
    Load a full raw prediction raster, apply morphological post-processing
    (skeleton enforcement then elongation filter), save result.

    Must be run on the FULL image (not tile-by-tile) so that channel
    features crossing tile boundaries are not incorrectly severed.
    """
    print("  Post-processing:", os.path.basename(raw_path), flush=True)

    with rasterio.open(raw_path) as src:
        profile = src.profile.copy()
        pred    = src.read(1)

    print("  → Enforcing channel width (skeleton + re-expand)...", flush=True)
    pred = enforce_channel_width(pred, max_half_width=max_half_width)

    print("  → Filtering by shape (area + eccentricity)...", flush=True)
    pred = filter_channel_by_shape(pred, min_area=min_area,
                                   min_eccentricity=min_eccentricity)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(pred, 1)

    print("  Post-processing complete → saved:", os.path.basename(out_path),
          flush=True)


# =============================================================================
# =====================  UTILITY FUNCTIONS  ===================================
# =============================================================================

def load_image_as_float(path):
    """Read a multi-band GeoTIFF, return (H, W, bands) float32 scaled to [0,1]."""
    with rasterio.open(path) as src:
        arr = src.read()  # (bands, H, W)
    arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    arr *= SCALE_FACTOR
    return arr


def find_matching_image(mask_filename, image_dir, stem_prefix=""):
    """
    Attempt to find the corresponding image for a mask by fuzzy stem matching.

    For RF masks like   "2_0004_mask_raster.tif"
      → image stem is  "2_0004"
    For DL masks like   "planet_tile_2_0006_mask_raster.tif"
      → image stem is  "planet_tile_2_0006"

    The function strips everything from "_mask_raster" onward from the mask
    stem and then looks for a .tif in image_dir whose stem contains that token.
    """
    mask_stem = os.path.splitext(mask_filename)[0]

    # Remove the "_mask_raster" suffix to recover the tile identifier
    core = re.sub(r'_mask_raster$', '', mask_stem, flags=re.IGNORECASE)

    # Search image directory for a file whose stem matches the core token
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(".tif"):
            continue
        img_stem = os.path.splitext(fname)[0]
        # Direct stem match
        if img_stem == core:
            return os.path.join(image_dir, fname)
        # Core is a suffix of the image stem (handles "planet_tile_" prefix cases)
        if img_stem.endswith(core):
            return os.path.join(image_dir, fname)
        # Core contains the image stem (handles when the core has extra prefix)
        if core.endswith(img_stem):
            return os.path.join(image_dir, fname)

    return None


def sample_pixels(flat_feats, flat_labels, target_class, max_count):
    """
    From flat (N_pixels, N_features) and flat (N_pixels,) label arrays,
    extract up to max_count pixels for target_class with random sampling.
    """
    idx = np.where(flat_labels == target_class)[0]
    if len(idx) == 0:
        return None, None
    if len(idx) > max_count:
        idx = np.random.choice(idx, max_count, replace=False)
    return flat_feats[idx], flat_labels[idx]


# =============================================================================
# =====================  TRAINING MODE  =======================================
# =============================================================================

def train_mode():
    """
    Collect training pixels from two sources:

    Source A — RF-purpose masks  (/Training_Masks)
        6-class scheme: 0=debris ice, 1=snow/ice, 2=blue ice,
                        3=lake, 4=channel, 5=slush
        Sampling caps:
            classes 0,1,2,5 → max MAX_PIXELS_CLASS_0125 per mask
            classes 3,4     → max MAX_PIXELS_CLASS_34   per mask

    Source B — DL-purpose masks  (nnUNet_masks)
        Original labels: 3=lake, 4=channel, 0/255=nodata/ignore
        Only channel (class 4) and an equal random sample of lake (class 3)
        pixels are extracted, then relabelled to RF class scheme
        (channel→4, lake→3).  This supplements narrow-channel training pixels.
    """
    np.random.seed(42)

    all_feats  = []
    all_labels = []

    # ------------------------------------------------------------------ #
    # Source A: RF-purpose training masks                                 #
    # ------------------------------------------------------------------ #
    rf_mask_files = sorted(glob.glob(os.path.join(RF_TRAIN_MASK_DIR, "*.tif")))
    print(f"Found {len(rf_mask_files)} RF-purpose training masks.", flush=True)

    for mask_path in rf_mask_files:
        mask_fname = os.path.basename(mask_path)
        img_path   = find_matching_image(mask_fname, RF_TRAIN_IMAGE_DIR)

        if img_path is None:
            print(f"  [SKIP] No matching image for: {mask_fname}", flush=True)
            continue

        print(f"  [A] Mask: {mask_fname}  →  Image: {os.path.basename(img_path)}",
              flush=True)

        img = load_image_as_float(img_path)

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        if img.shape[:2] != mask.shape:
            print(f"      Shape mismatch — img {img.shape[:2]} vs mask "
                  f"{mask.shape}. Skipping.", flush=True)
            continue

        feats = compute_features(img)
        H, W, D = feats.shape

        flat_feats  = feats.reshape(-1, D)
        flat_labels = mask.reshape(-1).astype(np.int32)

        for cls in VALID_RF_CLASSES:
            cap = (MAX_PIXELS_CLASS_34
                   if cls in (3, 4) else MAX_PIXELS_CLASS_0125)
            f, l = sample_pixels(flat_feats, flat_labels, cls, cap)
            if f is not None:
                all_feats.append(f)
                all_labels.append(l)

        del img, mask, feats, flat_feats, flat_labels
        gc.collect()

    # ------------------------------------------------------------------ #
    # Source B: DL-purpose training masks                                 #
    # ------------------------------------------------------------------ #
    dl_mask_files = sorted(glob.glob(os.path.join(DL_TRAIN_MASK_DIR, "*.tif")))
    print(f"\nFound {len(dl_mask_files)} DL-purpose training masks.", flush=True)

    for mask_path in dl_mask_files:
        mask_fname = os.path.basename(mask_path)
        img_path   = find_matching_image(mask_fname, DL_TRAIN_IMAGE_DIR)

        if img_path is None:
            print(f"  [SKIP] No matching image for: {mask_fname}", flush=True)
            continue

        print(f"  [B] Mask: {mask_fname}  →  Image: {os.path.basename(img_path)}",
              flush=True)

        img = load_image_as_float(img_path)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int32)

        if img.shape[:2] != mask.shape:
            print(f"      Shape mismatch — img {img.shape[:2]} vs mask "
                  f"{mask.shape}. Skipping.", flush=True)
            continue

        feats = compute_features(img)
        H, W, D = feats.shape

        flat_feats  = feats.reshape(-1, D)
        flat_labels = mask.reshape(-1)

        # --- Extract ALL channel pixels (class 4) ---
        ch_idx = np.where(flat_labels == DL_CHANNEL_CLASS)[0]
        n_channel = len(ch_idx)

        if n_channel == 0:
            print("      No channel pixels found in this DL mask. Skipping.",
                  flush=True)
            del img, mask, feats, flat_feats, flat_labels
            gc.collect()
            continue

        # --- Extract an equal-sized random sample of lake pixels (class 3) ---
        lk_idx = np.where(flat_labels == DL_LAKE_CLASS)[0]
        if len(lk_idx) > n_channel:
            lk_idx = np.random.choice(lk_idx, n_channel, replace=False)

        combined_idx = np.concatenate([ch_idx, lk_idx])
        dl_feats     = flat_feats[combined_idx]
        dl_labels_raw = flat_labels[combined_idx]

        # Remap to RF class scheme (DL labels 3 and 4 already match RF labels)
        dl_labels = dl_labels_raw.copy()

        print(f"      DL pixels added: {n_channel} channel, "
              f"{len(lk_idx)} lake.", flush=True)

        all_feats.append(dl_feats)
        all_labels.append(dl_labels)

        del img, mask, feats, flat_feats, flat_labels, dl_feats, dl_labels
        gc.collect()

    # ------------------------------------------------------------------ #
    # Train the Random Forest                                             #
    # ------------------------------------------------------------------ #
    if not all_feats:
        print("ERROR: No training pixels collected. Check paths and filenames.",
              flush=True)
        return

    X = np.vstack(all_feats).astype(np.float32)
    y = np.concatenate(all_labels).astype(np.int32)

    # Remove any nodata pixels
    valid = (y != 255)
    X = X[valid]
    y = y[valid]

    # Replace any NaN from texture computations at image edges
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    print(f"\nTotal training pixels: {len(y)}", flush=True)
    for cls in sorted(np.unique(y)):
        print(f"  Class {cls}: {np.sum(y == cls):,} pixels", flush=True)

    print("\nTraining Random Forest...", flush=True)

    # Class weights: upweight lake and channel to compensate for class imbalance
    # and the greater difficulty of channel discrimination, and to force model to
    # pursue more accurate channel extraction
    class_weights = {0: 1, 1: 1, 2: 1, 3: 3, 4: 6, 5: 2}

    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight=class_weights,
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    clf.fit(X, y)

    # Save model and class list
    bundle = {"model": clf, "classes": list(clf.classes_)}
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}", flush=True)

    # Print feature importances for diagnostics
    feat_names = [
        "CB", "B", "G1", "G2", "Y", "R", "RE", "N",
        "NDWI1", "NDWIice", "NDGI", "NDVI", "NDWI", "NDWI_CB", "B_NIR",
        "Y_R", "RE_N", "R_G_ratio", "B_R_ratio",
        "brightness", "blue_fraction", "CB_fraction",
        "NIR_std3", "Green_std3", "NIR_std7", "Blue_std5", "CB_std5",
        "NIR_range5", "Blue_range5",
    ]
    importances = clf.feature_importances_
    ranked = sorted(zip(feat_names, importances), key=lambda x: -x[1])
    print("\nTop 15 feature importances:")
    for name, imp in ranked[:15]:
        print(f"  {name:15s}  {imp:.4f}")


# =============================================================================
# =====================  APPLY MODE  ==========================================
# =============================================================================

def apply_mode():
    """
    Apply the trained model to all .tif images in APPLY_IMAGE_DIR.

    Two-pass approach:
      Pass 1 — tile-by-tile prediction with probability-threshold channel
               acceptance (Strategies 6). Writes raw predictions to PRED_DIR_RAW.
      Pass 2 — full-image post-processing using skeleton enforcement and
               elongation filter (Strategies 4+5). Writes final outputs to
               PRED_DIR_FINAL.  This MUST be done on the full image, not
               per-tile, to avoid severing channels at tile boundaries.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}", flush=True)
        print("Run with MODE='train' first.", flush=True)
        return

    bundle      = joblib.load(MODEL_PATH)
    clf         = bundle["model"]
    classes     = np.array(bundle["classes"])
    channel_idx = int(np.where(classes == CHANNEL_CLASS)[0][0])

    apply_images = sorted(glob.glob(os.path.join(APPLY_IMAGE_DIR, "*.tif")))
    print(f"Images to predict: {len(apply_images)}", flush=True)

    for img_path in apply_images:
        img_name = os.path.basename(img_path)
        stem     = os.path.splitext(img_name)[0]
        print(f"\n{'='*60}", flush=True)
        print(f"Processing: {img_name}", flush=True)

        raw_path   = os.path.join(PRED_DIR_RAW,   f"{stem}_raw.tif")
        final_path = os.path.join(PRED_DIR_FINAL, f"{stem}_final.tif")

        with rasterio.open(img_path) as src:
            profile = src.profile.copy()
            height  = src.height
            width   = src.width

        out_profile = profile.copy()
        out_profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=SAFE_NODATA,
            compress="lzw",
            tiled=True,
            bigtiff="yes",
        )

        # ---------------------------------------------------------------- #
        # Pass 1: tile-by-tile prediction                                  #
        # ---------------------------------------------------------------- #
        print("Pass 1: tiled prediction...", flush=True)

        with rasterio.open(img_path) as src:
            with rasterio.open(raw_path, "w", **out_profile) as dst:

                for row in range(0, height, TILE_SIZE):
                    for col in range(0, width, TILE_SIZE):

                        win_h  = min(TILE_SIZE, height - row)
                        win_w  = min(TILE_SIZE, width  - col)
                        window = Window(col, row, win_w, win_h)

                        img_data = src.read(window=window)

                        # Transpose to (H, W, bands) and scale
                        img_tile = np.transpose(
                            img_data, (1, 2, 0)
                        ).astype(np.float32) * SCALE_FACTOR

                        feats = compute_features(img_tile)
                        h, w, d = feats.shape

                        # Valid pixels: not all-zero across bands
                        valid_px = ~np.all(img_tile == 0, axis=2)
                        flat_feats  = feats.reshape(-1, d)
                        flat_valid  = valid_px.reshape(-1)

                        pred_flat = np.full(
                            flat_feats.shape[0], SAFE_NODATA, dtype=np.uint8
                        )

                        if np.any(flat_valid):
                            valid_feats = np.nan_to_num(
                                flat_feats[flat_valid], nan=0.0
                            )

                            # --- Strategy 6: probability threshold ---
                            proba      = clf.predict_proba(valid_feats)
                            pred_valid = classes[np.argmax(proba, axis=1)]

                            # Only accept channel if probability ≥ threshold
                            is_channel    = (pred_valid == CHANNEL_CLASS)
                            below_thr     = (
                                proba[:, channel_idx] < CHANNEL_PROB_THRESHOLD
                            )
                            needs_reassign = is_channel & below_thr

                            if np.any(needs_reassign):
                                p_no_ch = proba[needs_reassign].copy()
                                p_no_ch[:, channel_idx] = 0.0
                                pred_valid[needs_reassign] = classes[
                                    np.argmax(p_no_ch, axis=1)
                                ]

                            pred_flat[flat_valid] = pred_valid.astype(np.uint8)

                        preds = pred_flat.reshape(h, w)
                        dst.write(preds, 1, window=window)

                        print(
                            f"  Tile row={row:5d} col={col:5d} done",
                            flush=True,
                        )

                        del img_data, img_tile, feats, flat_feats, pred_flat
                        gc.collect()

        print(f"Raw prediction saved: {raw_path}", flush=True)

        # ---------------------------------------------------------------- #
        # Pass 2: full-image morphological post-processing                 #
        # ---------------------------------------------------------------- #
        postprocess_full_image(
            raw_path, final_path,
            max_half_width=MAX_HALF_WIDTH,
            min_eccentricity=MIN_ECCENTRICITY,
            min_area=MIN_CHANNEL_AREA,
        )

    print("\nAll images processed.", flush=True)


# =============================================================================
# =====================  ENTRY POINT  =========================================
# =============================================================================

if __name__ == "__main__":
    print(f"MODE: {MODE}", flush=True)
    print(f"Model path: {MODEL_PATH}", flush=True)
    if MODE == "train":
        train_mode()
    elif MODE == "apply":
        apply_mode()
    else:
        print(f"ERROR: Unknown MODE '{MODE}'. Use 'train' or 'apply'.", flush=True)
