# ============================================================
# Random Forest Classification — Supraglacial Feature Mapping
#Using 4 bands (Blue, Green, Red, NIR) extracted from 8-band PlanetScope SuperDove training imagery
# Author: Zhihan Xu
# Date: May 2026
# ============================================================


# Purpose: train a model compatible with 2019 PlanetScope 4-band imagery
# (which has no Coastal Blue, Green II, Yellow, or Red Edge bands).
# Model architecture follows exactly same as the 8 band classification RF model

# Classification scheme
# ---------------------
#  0 = Debris-influenced ice
#  1 = Snow / clean ice
#  2 = Blue ice
#  3 = Supraglacial lake
#  4 = Supraglacial channel
#  5 = Slush

# Band extraction from 8-band SuperDove training images (0-indexed)
# -----------------------------------------------------------------
# SuperDove index 1 -> Blue   (465-515 nm)
# SuperDove index 2 -> Green  (513-549 nm)
# SuperDove index 5 -> Red    (650-680 nm)
# SuperDove index 7 -> NIR    (845-885 nm)

# 2019 PlanetScope 4-band apply images band order (0-indexed)
# ------------------------------------------------------------
# Index 0 -> Blue
# Index 1 -> Green
# Index 2 -> Red
# Index 3 -> NIR

# Modes
# -----
# MODE = "train"  ->  collect 4-band pixels from RF and DL masks, train model
# MODE = "apply"  ->  load model, predict on 4-band images, post-process

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
#   - RF_Model_6classes_4bands.pkl : saved trained model


import os
import re
import gc
import glob
import warnings

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import generic_filter
from skimage.morphology import skeletonize, binary_dilation, disk
from skimage.measure import regionprops, label as sk_label
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# USER SETTINGS
# =============================================================================

MODE = "apply"          # "train" or "apply"

BASE = "/rds/user/path/to/your/base_folder"

# Training data paths 
RF_TRAIN_MASK_DIR  = os.path.join(BASE, "Training_Masks")
RF_TRAIN_IMAGE_DIR = os.path.join(BASE, "Training_Images")

DL_TRAIN_MASK_DIR  = "/rds/user/path/to/your/DL/masks"
DL_TRAIN_IMAGE_DIR = os.path.join(BASE, "Training_Images")

# Apply images: 2019 PlanetScope 4-band imagery
APPLY_IMAGE_DIR = os.path.join(BASE, "Apply_Images_2019")

# Outputs — separate from v2 to avoid confusion
OUTPUT_DIR     = os.path.join(BASE, "Output_4bands")
PRED_DIR_RAW   = os.path.join(OUTPUT_DIR, "Predictions_raw")
PRED_DIR_FINAL = os.path.join(OUTPUT_DIR, "Predictions_final")
MODEL_PATH     = os.path.join(OUTPUT_DIR, "RF_Model_6classes_4bands.pkl")

for d in [OUTPUT_DIR, PRED_DIR_RAW, PRED_DIR_FINAL]:
    os.makedirs(d, exist_ok=True)

# Class settings
VALID_RF_CLASSES = {0, 1, 2, 3, 4, 5}
DL_CHANNEL_CLASS = 4
DL_LAKE_CLASS    = 3
DL_NODATA_VALUES = {0, 255}
CHANNEL_CLASS    = 4
LAKE_CLASS       = 3
BACKGROUND_CLASS = 1    # snow/ice — default reassignment for removed objects
SAFE_NODATA      = 255

# Pixel sampling caps per class per training mask
MAX_PIXELS_CLASS_0125 = 45000
MAX_PIXELS_CLASS_34   = 25000

SCALE_FACTOR = 1 / 10000.0
TILE_SIZE    = 1024

# Band indices to extract from 8-band SuperDove training images
# These correspond to Blue, Green I, Red, NIR
TRAIN_BAND_INDICES = [1, 2, 5, 7]   # 0-indexed

# Post-processing parameters
CHANNEL_PROB_THRESHOLD = 0.70
MAX_HALF_WIDTH         = 3
MIN_CHANNEL_AREA       = 20
MIN_ECCENTRICITY       = 0.85
MIN_LAKE_AREA          = 150


# =============================================================================
# FEATURE ENGINEERING — 4-band version
# =============================================================================

def compute_features_4band(img):
    """
    Compute spectral indices and texture features from a 4-band array.

    Input shape: (H, W, 4), float32, scaled to [0, 1]

    Band order (consistent for both training extraction and apply):
        Index 0 -> Blue   (B)
        Index 1 -> Green  (G)
        Index 2 -> Red    (R)
        Index 3 -> NIR    (N)

    Returns array of shape (H, W, N_FEATURES).
    Total features: 4 raw + 7 indices + 3 ratios + 2 brightness + 5 texture = 21
    """
    eps = 1e-6

    B = img[:, :, 0]   # Blue
    G = img[:, :, 1]   # Green
    R = img[:, :, 2]   # Red
    N = img[:, :, 3]   # NIR

    # ------------------------------------------------------------------
    # Spectral indices — only those computable from B, G, R, NIR
    # ------------------------------------------------------------------

    # NDWI (McFeeters): positive for open water, negative for ice/snow
    NDWI = (G - N) / (G + N + eps)

    # NDVI: low for all glacier surfaces, distinguishes water (very low)
    NDVI = (N - R) / (N + R + eps)

    # NDGI (Normalised Difference Greenness):
    # channels and lakes appear slightly greener than white ice
    NDGI = (G - R) / (G + R + eps)

    # Blue-NIR normalised difference:
    # liquid water has relatively more blue and less NIR than snow/ice
    B_NIR = (B - N) / (B + N + eps)

    # NDWIice: NIR vs Red — liquid water suppresses NIR more than frozen ice
    NDWIice = (N - R) / (N + R + eps)

    # Blue-Red normalised: water appears bluer relative to red than ice
    B_R_norm = (B - R) / (B + R + eps)

    # ------------------------------------------------------------------
    # Band ratios
    # ------------------------------------------------------------------

    # R/G: high for debris ice (brownish), low for water
    R_G_ratio = R / (G + eps)

    # B/R: high for clear water, lower for ice
    B_R_ratio = B / (R + eps)

    # G/R: subtle greenness indicator
    G_R_ratio = G / (R + eps)

    # ------------------------------------------------------------------
    # Brightness and colour fractions
    # ------------------------------------------------------------------

    # Overall visible brightness: channels/lakes are darker than white ice
    brightness = (B + G + R) / 3.0

    # Blue fraction: liquid water is relatively more blue than ice
    blue_fraction = B / (brightness + eps)

    # ------------------------------------------------------------------
    # Texture features
    # ------------------------------------------------------------------

    # Standard deviation in sliding windows — channels create local contrast
    NIR_std3   = generic_filter(N, np.std, size=3)
    Green_std3 = generic_filter(G, np.std, size=3)
    NIR_std7   = generic_filter(N, np.std, size=7)
    Blue_std5  = generic_filter(B, np.std, size=5)

    # Local range (max-min): captures sharp channel-ice boundaries
    NIR_range5 = generic_filter(N, lambda x: x.max() - x.min(), size=5)

    # ------------------------------------------------------------------
    # Stack all features
    # ------------------------------------------------------------------
    features = np.dstack([
        # Raw bands (4)
        B, G, R, N,
        # Spectral indices (6)
        NDWI, NDVI, NDGI, B_NIR, NDWIice, B_R_norm,
        # Band ratios (3)
        R_G_ratio, B_R_ratio, G_R_ratio,
        # Brightness and fractions (2)
        brightness, blue_fraction,
        # Texture (5)
        NIR_std3, Green_std3, NIR_std7, Blue_std5, NIR_range5,
    ])
    # Total: 4 + 6 + 3 + 2 + 5 = 20 features

    return features.astype(np.float32)


# =============================================================================
# POSTPROCESSING
# =============================================================================

def enforce_channel_width(pred):
    """Collapse channel predictions to skeleton, re-expand by MAX_HALF_WIDTH."""
    mask = (pred == CHANNEL_CLASS)
    if not np.any(mask):
        return pred
    skeleton = skeletonize(mask)
    expanded = binary_dilation(skeleton, disk(MAX_HALF_WIDTH))
    final    = expanded & mask
    out = pred.copy()
    out[mask & ~final] = BACKGROUND_CLASS
    out[final]         = CHANNEL_CLASS
    return out


def filter_by_size_and_shape(pred):
    """
    Remove lake blobs smaller than MIN_LAKE_AREA.
    Remove channel blobs smaller than MIN_CHANNEL_AREA or too circular.
    """
    out = pred.copy()
    for target_class, min_area, check_shape in [
        (LAKE_CLASS,    MIN_LAKE_AREA,    False),
        (CHANNEL_CLASS, MIN_CHANNEL_AREA, True),
    ]:
        mask = (pred == target_class)
        if not np.any(mask):
            continue
        labeled = sk_label(mask)
        for region in regionprops(labeled):
            remove = region.area < min_area
            if check_shape and not remove:
                remove = region.eccentricity < MIN_ECCENTRICITY
            if remove:
                r, c = region.coords[:, 0], region.coords[:, 1]
                out[r, c] = BACKGROUND_CLASS
    return out


def postprocess_full_image(raw_path, out_path):
    print("  Post-processing:", os.path.basename(raw_path), flush=True)
    with rasterio.open(raw_path) as src:
        profile = src.profile.copy()
        pred    = src.read(1)
    pred = enforce_channel_width(pred)
    pred = filter_by_size_and_shape(pred)
    profile.update(dtype=rasterio.uint8, count=1,
                   compress="lzw", nodata=SAFE_NODATA)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(pred.astype(np.uint8), 1)
    print("  Saved:", os.path.basename(out_path), flush=True)


# =============================================================================
# UTILITIES
# =============================================================================

def find_matching_image(mask_filename, image_dir):
    """Match mask to image by stripping _mask_raster suffix."""
    stem = os.path.splitext(mask_filename)[0]
    core = re.sub(r'_mask_raster$', '', stem, flags=re.IGNORECASE)
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(".tif"):
            continue
        img_stem = os.path.splitext(fname)[0]
        if img_stem == core:
            return os.path.join(image_dir, fname)
        if img_stem.endswith(core) or core.endswith(img_stem):
            return os.path.join(image_dir, fname)
    return None


def load_4bands_from_image(path, band_indices=None):
    """
    Load specific band indices from a multi-band GeoTIFF.
    If band_indices is None, assumes image already has 4 bands (apply mode).
    Returns (H, W, 4) float32 scaled to [0, 1].
    """
    with rasterio.open(path) as src:
        n_bands = src.count
        if band_indices is not None:
            # Extract specific bands from 8-band training image
            bands = np.stack(
                [src.read(i + 1) for i in band_indices], axis=-1
            ).astype(np.float32)
        else:
            # Read all bands — for 4-band apply images
            if n_bands != 4:
                raise ValueError(
                    f"{os.path.basename(path)} has {n_bands} bands, expected 4."
                )
            arr = src.read()   # (4, H, W)
            bands = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    bands *= SCALE_FACTOR
    return bands


def sample_pixels(flat_feats, flat_labels, target_class, max_count):
    idx = np.where(flat_labels == target_class)[0]
    if len(idx) == 0:
        return None, None
    if len(idx) > max_count:
        idx = np.random.choice(idx, max_count, replace=False)
    return flat_feats[idx], flat_labels[idx]


# =============================================================================
# TRAINING MODE
# =============================================================================

def train_mode():
    np.random.seed(42)
    all_feats  = []
    all_labels = []

    # --- Source A: RF-purpose masks (extract 4 bands from 8-band images) ---
    rf_mask_files = sorted(glob.glob(
        os.path.join(RF_TRAIN_MASK_DIR, "*.tif")))
    print("RF masks found: %d" % len(rf_mask_files), flush=True)

    for mask_path in rf_mask_files:
        mask_fname = os.path.basename(mask_path)
        img_path   = find_matching_image(mask_fname, RF_TRAIN_IMAGE_DIR)
        if img_path is None:
            print("  [SKIP] No image for: %s" % mask_fname, flush=True)
            continue
        print("  [A] %s" % mask_fname, flush=True)

        try:
            img = load_4bands_from_image(img_path, TRAIN_BAND_INDICES)
        except Exception as e:
            print("      Load error: %s" % str(e), flush=True)
            continue

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        if img.shape[:2] != mask.shape:
            print("      Shape mismatch. Skipping.", flush=True)
            continue

        feats = compute_features_4band(img)
        flat_feats  = feats.reshape(-1, feats.shape[2])
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

    # --- Source B: DL-purpose masks (channel and lake pixels only) ---
    dl_mask_files = sorted(glob.glob(
        os.path.join(DL_TRAIN_MASK_DIR, "*.tif")))
    print("DL masks found: %d" % len(dl_mask_files), flush=True)

    for mask_path in dl_mask_files:
        mask_fname = os.path.basename(mask_path)
        img_path   = find_matching_image(mask_fname, DL_TRAIN_IMAGE_DIR)
        if img_path is None:
            print("  [SKIP] No image for: %s" % mask_fname, flush=True)
            continue
        print("  [B] %s" % mask_fname, flush=True)

        try:
            img = load_4bands_from_image(img_path, TRAIN_BAND_INDICES)
        except Exception as e:
            print("      Load error: %s" % str(e), flush=True)
            continue

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int32)

        if img.shape[:2] != mask.shape:
            print("      Shape mismatch. Skipping.", flush=True)
            continue

        feats = compute_features_4band(img)
        flat_feats  = feats.reshape(-1, feats.shape[2])
        flat_labels = mask.reshape(-1)

        ch_idx    = np.where(flat_labels == DL_CHANNEL_CLASS)[0]
        n_channel = len(ch_idx)
        if n_channel == 0:
            print("      No channel pixels. Skipping.", flush=True)
            del img, mask, feats, flat_feats, flat_labels
            gc.collect()
            continue

        lk_idx = np.where(flat_labels == DL_LAKE_CLASS)[0]
        if len(lk_idx) > n_channel:
            lk_idx = np.random.choice(lk_idx, n_channel, replace=False)

        idx      = np.concatenate([ch_idx, lk_idx])
        dl_feats = flat_feats[idx]
        dl_lbls  = flat_labels[idx].copy()

        print("      Channel: %d  Lake: %d" % (n_channel, len(lk_idx)),
              flush=True)
        all_feats.append(dl_feats)
        all_labels.append(dl_lbls)

        del img, mask, feats, flat_feats, flat_labels, dl_feats, dl_lbls
        gc.collect()

    if not all_feats:
        print("ERROR: No training pixels collected.", flush=True)
        return

    X = np.vstack(all_feats).astype(np.float32)
    y = np.concatenate(all_labels).astype(np.int32)

    valid = (y != 255)
    X = X[valid]
    y = y[valid]
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    print("\nTotal training pixels: %d" % len(y), flush=True)
    for cls in sorted(np.unique(y)):
        print("  Class %d: %d pixels" % (cls, int(np.sum(y == cls))),
              flush=True)

    print("\nTraining Random Forest...", flush=True)
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

    bundle = {"model": clf, "classes": list(clf.classes_),
              "n_features": X.shape[1], "band_mode": "4band"}
    joblib.dump(bundle, MODEL_PATH)
    print("Model saved: %s" % MODEL_PATH, flush=True)

    feat_names = [
        "B", "G", "R", "N",
        "NDWI", "NDVI", "NDGI", "NDSI", "B_NIR", "NDWIice", "B_R_norm",
        "R_G_ratio", "B_R_ratio", "G_R_ratio",
        "brightness", "blue_fraction",
        "NIR_std3", "Green_std3", "NIR_std7", "Blue_std5", "NIR_range5",
    ]
    ranked = sorted(zip(feat_names, clf.feature_importances_),
                    key=lambda x: -x[1])
    print("\nTop 15 feature importances:")
    for name, imp in ranked[:15]:
        print("  %-15s  %.4f" % (name, imp))


# =============================================================================
# APPLY MODE
# =============================================================================

def apply_mode():
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model not found at %s" % MODEL_PATH, flush=True)
        return

    bundle      = joblib.load(MODEL_PATH)
    clf         = bundle["model"]
    classes     = np.array(bundle["classes"])
    channel_idx = int(np.where(classes == CHANNEL_CLASS)[0][0])

    apply_images = sorted(glob.glob(
        os.path.join(APPLY_IMAGE_DIR, "*.tif")))
    print("Images to predict: %d" % len(apply_images), flush=True)

    for img_path in apply_images:
        img_name = os.path.basename(img_path)
        stem     = os.path.splitext(img_name)[0]
        print("\n" + "=" * 60, flush=True)
        print("Processing: %s" % img_name, flush=True)

        raw_path   = os.path.join(PRED_DIR_RAW,   stem + "_raw.tif")
        final_path = os.path.join(PRED_DIR_FINAL, stem + "_final.tif")

        with rasterio.open(img_path) as src:
            profile = src.profile.copy()
            height  = src.height
            width   = src.width
            n_bands = src.count

        # Validate band count of apply image
        if n_bands != 4:
            print("  [SKIP] Expected 4 bands, got %d: %s" % (
                n_bands, img_name), flush=True)
            continue

        out_profile = profile.copy()
        out_profile.update(
            count=1, dtype=rasterio.uint8, nodata=SAFE_NODATA,
            compress="lzw", tiled=True, bigtiff="yes",
        )

        # Pass 1: tiled prediction
        print("Pass 1: prediction...", flush=True)
        with rasterio.open(img_path) as src:
            with rasterio.open(raw_path, "w", **out_profile) as dst:
                for row in range(0, height, TILE_SIZE):
                    for col in range(0, width, TILE_SIZE):
                        win_h  = min(TILE_SIZE, height - row)
                        win_w  = min(TILE_SIZE, width  - col)
                        window = Window(col, row, win_w, win_h)

                        img_data = src.read(window=window)   # (4, H, W)
                        img_tile = np.transpose(
                            img_data, (1, 2, 0)
                        ).astype(np.float32) * SCALE_FACTOR

                        feats = compute_features_4band(img_tile)
                        h, w, d = feats.shape

                        valid_px   = ~np.all(img_tile == 0, axis=2)
                        flat_feats = feats.reshape(-1, d)
                        flat_valid = valid_px.reshape(-1)

                        pred_flat = np.full(
                            flat_feats.shape[0], SAFE_NODATA, dtype=np.uint8)

                        if np.any(flat_valid):
                            vf    = np.nan_to_num(
                                flat_feats[flat_valid], nan=0.0)
                            proba = clf.predict_proba(vf)
                            pv    = classes[np.argmax(proba, axis=1)]

                            # Raise acceptance threshold for channel
                            is_ch  = (pv == CHANNEL_CLASS)
                            low_p  = proba[:, channel_idx] < CHANNEL_PROB_THRESHOLD
                            reassign = is_ch & low_p
                            if np.any(reassign):
                                p2 = proba[reassign].copy()
                                p2[:, channel_idx] = 0.0
                                pv[reassign] = classes[np.argmax(p2, axis=1)]

                            pred_flat[flat_valid] = pv.astype(np.uint8)

                        preds = pred_flat.reshape(h, w)
                        dst.write(preds, 1, window=window)
                        print("  Tile row=%d col=%d done" % (row, col),
                              flush=True)

                        del img_data, img_tile, feats, flat_feats, pred_flat
                        gc.collect()

        print("Raw saved: %s" % raw_path, flush=True)

        # Pass 2: full-image morphological post-processing
        postprocess_full_image(raw_path, final_path)

    print("\nAll images processed.", flush=True)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("MODE: %s" % MODE, flush=True)
    print("Model: %s" % MODEL_PATH, flush=True)
    if MODE == "train":
        train_mode()
    elif MODE == "apply":
        apply_mode()
    else:
        print("ERROR: Unknown MODE. Use train or apply.", flush=True)
