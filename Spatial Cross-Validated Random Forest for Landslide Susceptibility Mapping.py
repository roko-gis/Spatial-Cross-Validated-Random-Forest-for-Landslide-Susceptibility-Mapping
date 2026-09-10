# -*- coding: utf-8 -*-
"""
Spatial Cross-Validated Random Forest for Landslide Susceptibility Mapping
Version 1.0.0
"""

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterNumber,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterString,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessing,
    QgsRasterLayer,
    QgsProject,
    QgsProcessingException,
    QgsPalettedRasterRenderer,
    QgsColorRampShader,
    QgsRasterShader,
    QgsSingleBandPseudoColorRenderer,
    Qgis
)
from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtGui import QColor

import time
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from osgeo import gdal
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    matthews_corrcoef, confusion_matrix
)
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib

# Optional SHAP - imported only if needed
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


@dataclass
class LSMConfig:
    """Configuration parameters for the landslide susceptibility model."""
    seed: int = 42
    version: str = "1.0.0"
    chunk_size: int = 50000
    n_splits: int = 5
    max_samples: int = 50000
    n_estimators: int = 250
    max_depth: int = 10
    min_samples_leaf: int = 4
    smooth_sigma: float = 1.0
    nodata_val: float = -9999.0
    categorical_keywords: List[str] = field(default_factory=lambda: [
        'landcover', 'land_cover', 'lc', 'landuse', 'land_use',
        'lulc', 'corine', 'clc', 'geology', 'lithology',
        'soil_type', 'soil_class', 'ip_landform', 'landform'
    ])

cfg = LSMConfig()

# Susceptibility class names and colors (for symbology)
LSM_CLASSES = [
    (1, "Very Low", QColor(0, 128, 0)),
    (2, "Low", QColor(173, 255, 47)),
    (3, "Moderate", QColor(255, 255, 0)),
    (4, "High", QColor(255, 165, 0)),
    (5, "Very High", QColor(255, 0, 0))
]


# ============================================================
# Utility functions
# ============================================================

def check_raster_alignment(base_ds, pred_ds, name, feedback):
    """Check if predictor raster aligns with base raster; report issues."""
    base_gt = base_ds.GetGeoTransform()
    pred_gt = pred_ds.GetGeoTransform()
    problems = []
    if base_ds.GetProjection() != pred_ds.GetProjection():
        problems.append("CRS mismatch")
    if abs(base_gt[0] - pred_gt[0]) > 1e-6 or abs(base_gt[3] - pred_gt[3]) > 1e-6:
        problems.append("Origin mismatch")
    if abs(base_gt[1] - pred_gt[1]) > 1e-6 or abs(base_gt[5] - pred_gt[5]) > 1e-6:
        problems.append("Pixel size mismatch")
    if base_ds.RasterXSize != pred_ds.RasterXSize or base_ds.RasterYSize != pred_ds.RasterYSize:
        problems.append(f"Dimensions mismatch ({pred_ds.RasterXSize}x{pred_ds.RasterYSize} "
                        f"vs {base_ds.RasterXSize}x{base_ds.RasterYSize})")
    if problems:
        feedback.pushWarning(f"Predictor '{name}' is NOT perfectly aligned:")
        for p in problems:
            feedback.pushWarning(f"  - {p}")
        feedback.pushWarning("  → Will attempt resampling")
        return False
    return True


def is_categorical_raster(src_ds, name):
    """
    Heuristic to detect categorical raster based on name and integer values.
    Reads only a sample of the raster to save memory/time.
    """
    name_lower = name.lower()
    if not any(kw in name_lower for kw in cfg.categorical_keywords):
        return False

    # Read a sample (first 10000 cells)
    band = src_ds.GetRasterBand(1)
    xsize = min(src_ds.RasterXSize, 200)
    ysize = min(src_ds.RasterYSize, 50)
    sample = band.ReadAsArray(0, 0, xsize, ysize).astype(np.float32)
    sample = sample[~np.isnan(sample)]
    sample = sample[sample != cfg.nodata_val]
    if len(sample) == 0:
        return False
    # Check if all sampled values are close to integers
    return np.all(np.abs(sample - np.round(sample)) < 0.001)


def align_raster_to_base(src_path: str, base_ds: gdal.Dataset, feedback,
                         is_categorical: bool = False) -> str:
    """
    Reproject/resample a raster to match base raster.
    Returns path to temporary warped raster.
    """
    tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
    tmp_path = tmp.name
    tmp.close()

    base_gt = base_ds.GetGeoTransform()
    base_proj = base_ds.GetProjection()
    base_cols = base_ds.RasterXSize
    base_rows = base_ds.RasterYSize

    if is_categorical:
        resample_alg = gdal.GRA_NearestNeighbour
    else:
        resample_alg = gdal.GRA_Bilinear

    warp_options = gdal.WarpOptions(
        format='GTiff',
        dstSRS=base_proj,
        outputBounds=[base_gt[0], base_gt[3] + base_gt[5] * base_rows,
                      base_gt[0] + base_gt[1] * base_cols, base_gt[3]],
        xRes=abs(base_gt[1]),
        yRes=abs(base_gt[5]),
        resampleAlg=resample_alg,
        dstNodata=cfg.nodata_val
    )
    gdal.Warp(tmp_path, src_path, options=warp_options)
    return tmp_path


def prepare_predictors(feature_layers, base_ds, rows, cols, feedback,
                       use_resampling=True):
    """
    Load and prepare predictor rasters: align, handle NoData,
    and optionally one-hot encode categorical layers.
    """
    features_list = []
    feature_names = []
    temp_files = []  # track temporary files for cleanup

    for layer in feature_layers:
        name = layer.name()
        src_path = layer.source()
        src_ds = gdal.Open(src_path)
        if src_ds is None:
            feedback.pushWarning(f"Cannot open predictor: {name} – skipped")
            continue

        try:
            # Check alignment
            aligned = check_raster_alignment(base_ds, src_ds, name, feedback)
            if not aligned and use_resampling:
                # Determine categorical before resampling using sample
                is_cat = is_categorical_raster(src_ds, name)
                feedback.pushInfo(f"  Resampling '{name}' with " +
                                  ("nearest neighbour" if is_cat else "bilinear") + "...")
                new_path = align_raster_to_base(src_path, base_ds, feedback, is_categorical=is_cat)
                temp_files.append(new_path)
                src_ds = None
                src_ds = gdal.Open(new_path)
                if src_ds is None:
                    feedback.pushWarning(f"  Failed to open resampled raster for {name}, skipping.")
                    continue
            elif not aligned and not use_resampling:
                # Legacy crop/pad
                arr = src_ds.ReadAsArray().astype(np.float32)
                if arr.shape != (rows, cols):
                    temp = np.full((rows, cols), np.nan, dtype=np.float32)
                    h = min(arr.shape[0], rows)
                    w = min(arr.shape[1], cols)
                    temp[:h, :w] = arr[:h, :w]
                    arr = temp
                    feedback.pushWarning(f"  '{name}' was cropped/padded to {rows}x{cols}")
                arr[arr == cfg.nodata_val] = np.nan
                features_list.append(arr)
                feature_names.append(name)
                src_ds = None
                continue

            # Read array (if we got here, src_ds is aligned)
            arr = src_ds.ReadAsArray().astype(np.float32)
            arr[arr == cfg.nodata_val] = np.nan
            features_list.append(arr)
            feature_names.append(name)
        except Exception as e:
            feedback.pushWarning(f"Error processing predictor '{name}': {str(e)}")
        finally:
            if src_ds is not None:
                src_ds = None

    # Clean up temporary resampled files
    for tmp in temp_files:
        try:
            os.remove(tmp)
        except OSError:
            pass

    feedback.pushInfo("Encoding categorical layers...")
    new_features, new_names = [], []
    for name, arr in zip(feature_names, features_list):
        name_lower = name.lower()
        arr_clean = arr[~np.isnan(arr)]
        is_categorical = any(kw in name_lower for kw in cfg.categorical_keywords)
        is_integer_like = len(arr_clean) > 0 and np.all(np.abs(arr_clean - np.round(arr_clean)) < 0.001)
        if is_categorical and is_integer_like:
            unique_vals = np.unique(np.round(arr_clean))
            if 2 <= len(unique_vals) <= 50:
                feedback.pushInfo(f"  {name}: {len(unique_vals)} classes → One-Hot")
                for val in unique_vals:
                    new_features.append((np.round(arr) == val).astype(np.float32))
                    new_names.append(f"{name}_Class{int(val)}")
            else:
                new_features.append(arr)
                new_names.append(name)
        else:
            new_features.append(arr)
            new_names.append(name)
    feedback.pushInfo(f"Final number of predictors: {len(new_names)}")
    return new_features, new_names


def extract_inventory_samples(vector_layer, gt, rows, cols, valid_mask, feedback):
    """Extract positive sample indices from point landslide inventory."""
    positions = set()
    for feat in vector_layer.getFeatures():
        try:
            geom = feat.geometry()
            if geom is None or geom.isEmpty():
                continue
            # Only accept point geometries
            if geom.type() != 0:  # 0 = Point
                continue
            if geom.isMultipart():
                pts = geom.asMultiPoint()
                if not pts:
                    continue
                # Add all points from multipoint
                for pt in pts:
                    x = int((pt.x() - gt[0]) / gt[1])
                    y = int((pt.y() - gt[3]) / gt[5])
                    if 0 <= x < cols and 0 <= y < rows:
                        positions.add(y * cols + x)
            else:
                pt = geom.asPoint()
                x = int((pt.x() - gt[0]) / gt[1])
                y = int((pt.y() - gt[3]) / gt[5])
                if 0 <= x < cols and 0 <= y < rows:
                    positions.add(y * cols + x)
        except (AttributeError, ValueError, TypeError):
            continue
    pos = np.array(list(positions), dtype=np.int64)
    pos = pos[valid_mask.ravel()[pos]]
    feedback.pushInfo(f"Positive landslide samples: {len(pos)} unique cells")
    return pos


def generate_background_samples(pos, valid_mask, rows, cols, min_dist_px, max_samples,
                                bg_ratio=2.0, buffer_px=None, feedback=None):
    """Generate background/pseudo-absence samples far enough from positive ones."""
    land_mask = np.zeros(rows * cols, dtype=np.uint8)
    land_mask[pos] = 1
    if buffer_px is not None:
        buffer_dist = buffer_px
    else:
        buffer_dist = min_dist_px
    dist_map = distance_transform_edt(1 - land_mask.reshape(rows, cols))
    bg_candidates = np.where(valid_mask.ravel() & (dist_map.ravel() > buffer_dist))[0]
    bg_pool = np.setdiff1d(bg_candidates, pos)
    n_pos = len(pos)
    n_bg = min(int(n_pos * bg_ratio), len(bg_pool), max_samples)
    if n_bg < 10:
        raise QgsProcessingException("Too few background candidates after applying minimum distance.")
    bg = np.random.choice(bg_pool, n_bg, replace=False)
    if feedback:
        feedback.pushInfo(f"Background samples: {n_bg} selected from {len(bg_pool)} candidates (ratio {bg_ratio})")
    diagnostics = {
        "positive_samples": n_pos,
        "background_candidates": int(len(bg_pool)),
        "background_samples": n_bg,
        "background_positive_ratio": float(n_bg / n_pos) if n_pos > 0 else 0.0,
        "min_distance_px": float(buffer_dist)
    }
    return bg, diagnostics


def create_spatial_groups(yy, xx, block_pixels):
    """Assign a unique group ID based on spatial block."""
    return ((yy // block_pixels) * 100000 + (xx // block_pixels)).astype(np.int64)


def calculate_morans_i(residuals, coords, n_neighbors=8, n_permutations=99):
    """Calculate Moran's I and permutation-based two-sided p-value."""
    n = len(residuals)
    if n < 20:
        return 0.0, 1.0
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=min(n_neighbors + 1, n))
    row_ind, col_ind, data = [], [], []
    for i in range(n):
        for j in idx[i, 1:]:
            if j < n:
                row_ind.extend([i, j])
                col_ind.extend([j, i])
                data.extend([1.0, 1.0])
    W = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1e-6
    W_std = W.multiply(1.0 / row_sums[:, np.newaxis])
    z = residuals - np.mean(residuals)
    I = (n * z @ (W_std @ z)) / (np.sum(W_std) * np.sum(z ** 2))

    I_perm = np.zeros(n_permutations)
    for p in range(n_permutations):
        z_perm = np.random.permutation(z)
        I_perm[p] = (n * z_perm @ (W_std @ z_perm)) / (np.sum(W_std) * np.sum(z_perm ** 2))

    # Two-sided p-value
    abs_observed = abs(I)
    abs_perm = np.abs(I_perm)
    count_extreme = np.sum(abs_perm >= abs_observed)
    p_value = (count_extreme + 1) / (n_permutations + 1)
    return float(I), float(p_value)


def build_model(groups=None):
    """
    Construct the Random Forest pipeline with calibration.
    If groups is provided, use spatial group k-fold for calibration.
    """
    rf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        random_state=cfg.seed,
        class_weight='balanced_subsample',
        n_jobs=-1
    )
    if groups is not None:
        # Use spatial group k-fold with 3 splits (or fewer if groups are too few)
        n_unique = len(np.unique(groups))
        if n_unique >= 3:
            cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=cfg.seed)
        elif n_unique == 2:
            cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=cfg.seed)
        else:
            cv = 3  # fallback to non-spatial
        calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=cv)
    else:
        calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=3)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", calibrated)
    ])


def run_spatial_cv(X, y, groups, coords, feedback, use_spatial_calibration=True):
    """
    Perform spatial cross-validation and return metrics.
    Returns fold_metrics, all_y_true, all_y_pred, test_groups.
    """
    feedback.pushInfo(f"Running Spatial Cross-Validation ({cfg.n_splits} folds)...")
    cv = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    fold_metrics = []
    all_y_true, all_y_pred = [], []
    all_test_groups = []
    for fold_idx, (tr, te) in enumerate(cv.split(X, y, groups), 1):
        if len(np.unique(y[tr])) < 2 or len(y[tr]) < 30:
            continue

        # Prepare model with spatial calibration if requested
        if use_spatial_calibration:
            try:
                model = build_model(groups=groups[tr])
                model.fit(X[tr], y[tr], model__groups=groups[tr])
            except Exception:
                # Fallback to non-spatial calibration if spatial fails
                feedback.pushInfo(f"  Fold {fold_idx}: spatial calibration failed, using default calibration.")
                model = build_model(groups=None)
                model.fit(X[tr], y[tr])
        else:
            model = build_model(groups=None)
            model.fit(X[tr], y[tr])

        p = model.predict_proba(X[te])[:, 1]
        if np.any(np.isnan(p)):
            continue
        auc = roc_auc_score(y[te], p)
        pr_auc = average_precision_score(y[te], p)
        brier = brier_score_loss(y[te], p)
        y_pred = (p >= 0.5).astype(int)
        cm = confusion_matrix(y[te], y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        mcc = matthews_corrcoef(y[te], y_pred)
        moran_i, moran_p = calculate_morans_i(y[te] - p, coords[te])
        metrics = {
            'fold': fold_idx, 'auc': auc, 'pr_auc': pr_auc, 'brier': brier,
            'sensitivity': sens, 'specificity': spec, 'mcc': mcc,
            'moran_i': moran_i, 'moran_p': moran_p
        }
        fold_metrics.append(metrics)
        all_y_true.extend(y[te].tolist())
        all_y_pred.extend(p.tolist())
        all_test_groups.extend(groups[te].tolist())
        feedback.pushInfo(
            f"Fold {fold_idx}: AUC={auc:.3f}  PR-AUC={pr_auc:.3f}  Brier={brier:.3f}  "
            f"MCC={mcc:.3f}  Moran’s I={moran_i:.3f} (p={moran_p:.3f})"
        )
    return fold_metrics, np.array(all_y_true), np.array(all_y_pred), np.array(all_test_groups)


def block_bootstrap_auc(y_true, y_pred, groups, n_boot=200, seed=42):
    """
    Compute bootstrap 95% CI for AUC using spatial blocks as units.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_blocks = len(unique_groups)
    aucs = []
    for _ in range(n_boot):
        # Sample blocks with replacement
        sampled_blocks = rng.choice(unique_groups, size=n_blocks, replace=True)
        # Collect observations belonging to sampled blocks
        y_true_boot = []
        y_pred_boot = []
        for block in sampled_blocks:
            mask = groups == block
            y_true_boot.extend(y_true[mask].tolist())
            y_pred_boot.extend(y_pred[mask].tolist())
        if len(y_true_boot) < 2 or len(np.unique(y_true_boot)) < 2:
            continue
        try:
            auc_boot = roc_auc_score(y_true_boot, y_pred_boot)
            aucs.append(auc_boot)
        except ValueError:
            pass
    if len(aucs) == 0:
        return [np.nan, np.nan]
    return np.nanpercentile(aucs, [2.5, 97.5])


def save_raster(arr, path, gt, proj, valid_mask, is_categorical=False):
    """Save a raster with appropriate metadata."""
    path = Path(path)
    if is_categorical:
        mem = gdal.GetDriverByName('MEM').Create('', arr.shape[1], arr.shape[0], 1, gdal.GDT_Byte)
        mem.SetGeoTransform(gt)
        mem.SetProjection(proj)
        band = mem.GetRasterBand(1)
        band.WriteArray(np.where(valid_mask, arr, 0).astype(np.uint8))
        band.SetNoDataValue(0)
        ct = gdal.ColorTable()
        for val, _, color in LSM_CLASSES:
            ct.SetColorEntry(val, (color.red(), color.green(), color.blue(), 255))
        band.SetRasterColorTable(ct)
        gdal.GetDriverByName('GTiff').CreateCopy(str(path), mem, 0, ['COMPRESS=DEFLATE', 'TILED=YES'])
        mem = None
    else:
        ds = gdal.GetDriverByName('GTiff').Create(str(path), arr.shape[1], arr.shape[0], 1,
                                                  gdal.GDT_Float32, ["COMPRESS=DEFLATE", "TILED=YES"])
        ds.SetGeoTransform(gt)
        ds.SetProjection(proj)
        band = ds.GetRasterBand(1)
        band.WriteArray(np.where(valid_mask, arr, cfg.nodata_val).astype(np.float32))
        band.SetNoDataValue(cfg.nodata_val)
        ds = None
    return path


def apply_lsm_symbology(layer):
    """Apply paletted renderer with class names to the susceptibility classes raster."""
    classes = [QgsPalettedRasterRenderer.Class(val, color, name) for val, name, color in LSM_CLASSES]
    renderer = QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes)
    layer.setRenderer(renderer)
    layer.triggerRepaint()


def apply_continuous_symbology(layer, color_stops, min_val=0.0, max_val=1.0):
    """
    Apply a continuous pseudo-color renderer with given color stops.
    color_stops: list of (value, QColor) tuples.
    """
    shader = QgsColorRampShader(min_val, max_val)
    shader.setColorRampType(QgsColorRampShader.Interpolated)
    shader.setColorRampItemList([QgsColorRampShader.ColorRampItem(val, color) for val, color in color_stops])
    raster_shader = QgsRasterShader()
    raster_shader.setRasterShaderFunction(shader)
    renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, raster_shader)
    layer.setRenderer(renderer)
    layer.triggerRepaint()


# ============================================================
# Helper functions for improvements
# ============================================================

def jenks_breaks_true(values, n_classes=5, max_samples_for_breaks=10000):
    """
    Jenks natural breaks optimization for 1D array (Fisher-Jenks algorithm).
    If input is too large, a random subset is used to compute breaks.
    Returns n_classes-1 break values.
    """
    if len(values) > max_samples_for_breaks:
        # Randomly sample to reduce computational cost
        idx = np.random.choice(len(values), max_samples_for_breaks, replace=False)
        values = values[idx]
    values = np.sort(values)
    n = len(values)
    if n <= n_classes:
        return np.linspace(values[0], values[-1], n_classes + 1)[1:-1]

    # Build variance matrix (SSD) for all contiguous intervals
    # Use float32 to save memory
    mat_ssd = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sum_vals = 0.0
        sum_sq = 0.0
        for j in range(i, n):
            sum_vals += values[j]
            sum_sq += values[j] ** 2
            k = j - i + 1
            mean = sum_vals / k
            mat_ssd[i, j] = sum_sq - k * mean * mean

    # Dynamic programming to minimize total SSD
    dp = np.full((n_classes, n), np.inf, dtype=np.float32)
    bounds = np.zeros((n_classes, n), dtype=int)

    # First class (k=0)
    dp[0, :] = mat_ssd[0, :]

    for k in range(1, n_classes):
        for i in range(k, n):
            for j in range(k-1, i):
                val = dp[k-1, j] + mat_ssd[j+1, i]
                if val < dp[k, i]:
                    dp[k, i] = val
                    bounds[k, i] = j

    # Backtrack to find break indices
    breaks = []
    idx = n - 1
    for k in range(n_classes-1, 0, -1):
        j = bounds[k, idx]
        breaks.append(values[j+1])
        idx = j
    breaks = breaks[::-1]
    return np.array(breaks)


def compute_shap_values(model, X_sample, feature_names, max_samples=500):
    """Compute SHAP values for a subset and return mean absolute SHAP."""
    if not SHAP_AVAILABLE:
        return None
    if len(X_sample) > max_samples:
        idx = np.random.choice(len(X_sample), max_samples, replace=False)
        X_sample = X_sample[idx]

    # Aggregate SHAP over all calibrated classifiers
    try:
        calibrated = model.named_steps['model']
        if hasattr(calibrated, 'calibrated_classifiers_') and calibrated.calibrated_classifiers_:
            # CalibratedClassifierCV
            shap_values_sum = None
            n_estimators_used = 0
            for clf in calibrated.calibrated_classifiers_:
                rf = clf.estimator
                explainer = shap.TreeExplainer(rf)
                sv = explainer.shap_values(X_sample)
                if isinstance(sv, list):
                    sv = sv[1]  # class 1
                if shap_values_sum is None:
                    shap_values_sum = np.zeros_like(sv)
                shap_values_sum += sv
                n_estimators_used += 1
            if n_estimators_used > 0:
                shap_values = shap_values_sum / n_estimators_used
            else:
                return None
        else:
            # Fallback: use first classifier
            rf_model = calibrated.calibrated_classifiers_[0].estimator
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
    except Exception:
        # If SHAP fails, return None
        return None

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    return dict(zip(feature_names, mean_abs_shap.tolist()))


# ============================================================
# Main algorithm class
# ============================================================

class SpatialRFLandslideAlgorithm(QgsProcessingAlgorithm):
    """QGIS processing algorithm for spatial cross-validated RF landslide susceptibility."""

    INPUT_VECTOR = 'INPUT_VECTOR'
    INPUT_BASE = 'INPUT_BASE'
    INPUT_PREDICTORS = 'INPUT_PREDICTORS'
    MIN_DISTANCE = 'MIN_DISTANCE'
    BLOCK_SIZE = 'BLOCK_SIZE'
    OUTPUT_DIR = 'OUTPUT_DIR'
    OUTPUT_PREFIX = 'OUTPUT_PREFIX'

    N_ESTIMATORS = 'N_ESTIMATORS'
    MAX_DEPTH = 'MAX_DEPTH'
    MIN_SAMPLES_LEAF = 'MIN_SAMPLES_LEAF'
    SMOOTH_SIGMA = 'SMOOTH_SIGMA'
    CV_SPLITS = 'CV_SPLITS'

    BG_RATIO = 'BG_RATIO'
    BUFFER_DIST = 'BUFFER_DIST'
    TEST_SIZE = 'TEST_SIZE'
    COMPUTE_SHAP = 'COMPUTE_SHAP'
    CLASS_METHOD = 'CLASS_METHOD'
    MODEL_INPUT = 'MODEL_INPUT'
    IMPUTE_NODATA = 'IMPUTE_NODATA'
    VIF_THRESHOLD = 'VIF_THRESHOLD'
    RESAMPLE_PREDICTORS = 'RESAMPLE_PREDICTORS'
    SPATIAL_CALIBRATION = 'SPATIAL_CALIBRATION'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return SpatialRFLandslideAlgorithm()

    def name(self):
        return 'spatial_rf_landslide_susceptibility'

    def displayName(self):
        return self.tr(f'Spatial CV Random Forest – Landslide Susceptibility (v{cfg.version})')

    def group(self):
        return self.tr('Landslide Susceptibility')

    def groupId(self):
        return 'landslide_susceptibility'

    def shortHelpString(self):
        return self.tr(
            "<h3>Spatial CV Random Forest for Landslide Susceptibility</h3>"
            "<p>This algorithm performs spatial cross-validated Random Forest classification "
            "for landslide susceptibility mapping. It includes:</p>"
            "<ul>"
            "<li>Spatial block cross-validation to assess model performance</li>"
            "<li>Automatic background/pseudo-absence sampling</li>"
            "<li>Feature importance and multicollinearity diagnostics</li>"
            "<li>Probability, smoothed susceptibility, ambiguity, and class maps</li>"
            "<li>Optional hold-out test set for independent evaluation</li>"
            "<li>Optional SHAP values for model interpretation</li>"
            "<li>Optional spatially aware probability calibration</li>"
            "</ul>"
            "<h4>Parameters</h4>"
            "<ul>"
            "<li><b>Landslide inventory</b>: point locations of landslides</li>"
            "<li><b>Base raster</b>: defines extent, resolution, and valid area mask</li>"
            "<li><b>Predictor rasters</b>: environmental/terrain variables</li>"
            "<li><b>Minimum distance</b>: buffer distance (in meters) for background sampling</li>"
            "<li><b>Spatial block size</b>: size (in meters) of spatial blocks for cross-validation</li>"
            "<li><b>Output prefix</b>: prefix for all output files</li>"
            "<li><b>Advanced parameters</b>: tune Random Forest, add hold-out, SHAP, etc.</li>"
            "<li><b>Spatially aware calibration</b>: uses spatial groups for internal calibration cross-validation (slower but more conservative)</li>"
            "</ul>"
        )

    def initAlgorithm(self, config=None):
        """Define input parameters."""
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.INPUT_VECTOR,
            self.tr('Landslide inventory (point locations)'),
            [QgsProcessing.TypeVectorPoint]
        ))

        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT_BASE,
            self.tr('Base / Reference raster (defines extent and valid area)')
        ))

        self.addParameter(QgsProcessingParameterMultipleLayers(
            self.INPUT_PREDICTORS,
            self.tr('Predictor rasters'),
            QgsProcessing.TypeRaster
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_DISTANCE,
            self.tr('Minimum distance from landslides (meters)'),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=75.0,
            minValue=0.0
        ))

        self.addParameter(QgsProcessingParameterNumber(
            self.BLOCK_SIZE,
            self.tr('Spatial block size (meters)'),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=1000,
            minValue=100
        ))

        self.addParameter(QgsProcessingParameterString(
            self.OUTPUT_PREFIX,
            self.tr('Output file prefix'),
            defaultValue='LSM'
        ))

        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUTPUT_DIR,
            self.tr('Output folder')
        ))

        # Advanced parameters
        param = QgsProcessingParameterNumber(
            self.N_ESTIMATORS,
            self.tr('Number of trees (Random Forest)'),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=250,
            minValue=10,
            maxValue=2000
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        param = QgsProcessingParameterNumber(
            self.MAX_DEPTH,
            self.tr('Maximum tree depth'),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=10,
            minValue=1,
            maxValue=100
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        param = QgsProcessingParameterNumber(
            self.MIN_SAMPLES_LEAF,
            self.tr('Minimum samples per leaf'),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=4,
            minValue=1,
            maxValue=100
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        param = QgsProcessingParameterNumber(
            self.SMOOTH_SIGMA,
            self.tr('Smoothing sigma (Gaussian)'),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=1.0,
            minValue=0.0,
            maxValue=10.0
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        param = QgsProcessingParameterNumber(
            self.CV_SPLITS,
            self.tr('Number of spatial cross-validation folds'),
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=5,
            minValue=2,
            maxValue=10
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        param = QgsProcessingParameterNumber(
            self.BG_RATIO,
            self.tr('Background/positive ratio'),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=2.0,
            minValue=1.0,
            maxValue=10.0
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        param = QgsProcessingParameterNumber(
            self.BUFFER_DIST,
            self.tr('Additional buffer for background sampling (meters, 0 = same as min distance)'),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0,
            minValue=0.0
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        param = QgsProcessingParameterNumber(
            self.TEST_SIZE,
            self.tr('Test set fraction (0 = no hold-out)'),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.2,
            minValue=0.0,
            maxValue=0.5
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        param = QgsProcessingParameterBoolean(
            self.COMPUTE_SHAP,
            self.tr('Compute SHAP values (may be slow)'),
            defaultValue=False
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        param = QgsProcessingParameterEnum(
            self.CLASS_METHOD,
            self.tr('Classification method for susceptibility classes'),
            options=['Percentile (20/40/60/80)', 'Fixed (0.2,0.4,0.6,0.8)', 'Jenks natural breaks'],
            defaultValue=0
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        param = QgsProcessingParameterFile(
            self.MODEL_INPUT,
            self.tr('Optional pre-trained model (.pkl)'),
            optional=True,
            extension='pkl'
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        param = QgsProcessingParameterBoolean(
            self.IMPUTE_NODATA,
            self.tr('Impute missing predictor values'),
            defaultValue=True
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        param = QgsProcessingParameterNumber(
            self.VIF_THRESHOLD,
            self.tr('VIF threshold for predictor removal (0 = no removal)'),
            type=QgsProcessingParameterNumber.Double,
            defaultValue=10.0,
            minValue=0.0,
            maxValue=100.0
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        param = QgsProcessingParameterBoolean(
            self.RESAMPLE_PREDICTORS,
            self.tr('Resample predictors to base raster (recommended)'),
            defaultValue=True
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

        # New parameter: spatial calibration
        param = QgsProcessingParameterBoolean(
            self.SPATIAL_CALIBRATION,
            self.tr('Use spatially aware probability calibration (slower)'),
            defaultValue=False
        )
        param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
        self.addParameter(param)

    # Helper methods
    def _load_raster_data(self, base_raster_path):
        ds = gdal.Open(base_raster_path)
        if ds is None:
            raise QgsProcessingException("Cannot open base raster")
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        rows, cols = ds.RasterYSize, ds.RasterXSize
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue() or cfg.nodata_val
        base_data = band.ReadAsArray().astype(np.float32)
        valid_mask = (base_data != nodata) & (~np.isnan(base_data))
        pixel_size = abs(gt[1])
        return ds, gt, proj, rows, cols, valid_mask, pixel_size

    def _compute_diagnostics_imputed(self, X_imputed, y, feature_names, coords):
        """Compute MI, VIF, Moran's I on imputed data."""
        mi_scores = mutual_info_classif(X_imputed, y, random_state=cfg.seed)
        mi_order = np.argsort(mi_scores)[::-1]
        mi_percent = mi_scores / mi_scores.sum() * 100 if mi_scores.sum() > 0 else np.zeros_like(mi_scores)

        vif_values = []
        for i in range(X_imputed.shape[1]):
            try:
                reg = LinearRegression().fit(np.delete(X_imputed, i, axis=1), X_imputed[:, i])
                r2 = max(0.0, min(reg.score(np.delete(X_imputed, i, axis=1), X_imputed[:, i]), 0.9999))
                vif_values.append(1.0 / (1.0 - r2) if r2 < 1 else np.inf)
            except (ValueError, np.linalg.LinAlgError):
                vif_values.append(np.nan)

        predictor_moran = []
        for i in range(X_imputed.shape[1]):
            mi, p = calculate_morans_i(X_imputed[:, i], coords)
            predictor_moran.append((mi, p))

        return mi_scores, mi_order, mi_percent, vif_values, predictor_moran

    def _apply_vif_removal(self, X_imputed, feature_names, vif_values, vif_threshold):
        """Iteratively remove predictors with VIF above threshold."""
        if vif_threshold <= 0:
            return X_imputed, feature_names, []
        X_work = X_imputed.copy()
        names = feature_names.copy()
        vifs = vif_values.copy()
        removed = []
        while len(names) > 1 and max(vifs) > vif_threshold:
            idx = np.argmax(vifs)
            removed.append(names[idx])
            X_work = np.delete(X_work, idx, axis=1)
            del names[idx]
            del vifs[idx]
            # Recompute VIF for remaining
            new_vifs = []
            for i in range(X_work.shape[1]):
                try:
                    reg = LinearRegression().fit(np.delete(X_work, i, axis=1), X_work[:, i])
                    r2 = max(0.0, min(reg.score(np.delete(X_work, i, axis=1), X_work[:, i]), 0.9999))
                    new_vifs.append(1.0 / (1.0 - r2) if r2 < 1 else np.inf)
                except:
                    new_vifs.append(np.nan)
            vifs = new_vifs
        return X_work, names, removed

    def _generate_susceptibility_maps_normalized_smoothing(self, model, features_list, feature_names,
                                                           valid_mask, rows, cols, feedback,
                                                           class_method='percentile'):
        """Generate maps with normalized Gaussian smoothing."""
        flat_prob = np.full(rows * cols, np.nan, dtype=np.float32)
        valid_cells = np.where(valid_mask.ravel())[0]
        flat_features = [f.ravel() for f in features_list]
        total = len(valid_cells)
        for i in range(0, total, cfg.chunk_size):
            if feedback.isCanceled():
                break
            batch_idx = valid_cells[i:i + cfg.chunk_size]
            batch = np.column_stack([f[batch_idx] for f in flat_features])
            flat_prob[batch_idx] = model.predict_proba(batch)[:, 1]
            progress = int(100 * min(i + len(batch_idx), total) / total)
            feedback.setProgress(progress)

        prob_raw = flat_prob.reshape(rows, cols)

        # Normalized Gaussian smoothing to avoid NoData as zero
        valid_float = valid_mask.astype(np.float32)
        # Fill NaNs with 0 for smoothing
        prob_filled = np.where(valid_mask, np.nan_to_num(prob_raw, nan=0.0), 0.0)
        numerator = gaussian_filter(prob_filled, sigma=cfg.smooth_sigma)
        denominator = gaussian_filter(valid_float, sigma=cfg.smooth_sigma)
        denominator_safe = np.where(denominator > 0, denominator, 1.0)
        prob_smooth = numerator / denominator_safe
        prob_smooth[~valid_mask] = np.nan

        ambiguity = 1.0 - 2.0 * np.abs(prob_smooth - 0.5)
        ambiguity[~valid_mask] = np.nan

        valid_vals = prob_smooth[valid_mask]
        susceptibility = np.zeros((rows, cols), dtype=np.uint8)
        if len(valid_vals) > 0:
            if class_method == 0:
                thresholds = np.nanpercentile(valid_vals, [20, 40, 60, 80])
            elif class_method == 1:
                thresholds = np.array([0.2, 0.4, 0.6, 0.8])
            elif class_method == 2:
                thresholds = jenks_breaks_true(valid_vals, n_classes=5)
            else:
                thresholds = np.nanpercentile(valid_vals, [20, 40, 60, 80])

            flat = susceptibility.ravel()
            vidx = np.where(valid_mask.ravel())[0]
            flat[vidx] = 3
            flat[vidx[valid_vals <= thresholds[0]]] = 1
            flat[vidx[(valid_vals > thresholds[0]) & (valid_vals <= thresholds[1])]] = 2
            flat[vidx[(valid_vals > thresholds[2]) & (valid_vals <= thresholds[3])]] = 4
            flat[vidx[valid_vals > thresholds[3]]] = 5
            susceptibility = flat.reshape(rows, cols)

        return prob_raw, prob_smooth, ambiguity, susceptibility

    def _write_report(self, csv_path, fold_metrics, auc_ci, feature_names_all,
                      mi_order, mi_percent, vif_values, predictor_moran,
                      used_in_model, removal_reason,
                      n_samples, n_pos, n_bg, n_groups, pixel_size,
                      min_distance_m, min_dist_px, test_metrics=None,
                      shap_values=None):
        """Write full CSV summary report with used_in_model flag."""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)

            w.writerow(["DATA SUMMARY"])
            w.writerow(["Variable", "Value"])
            w.writerow(["Version", cfg.version])
            w.writerow(["Title", "Spatial Cross-Validated Random Forest for Landslide Susceptibility Mapping"])
            w.writerow(["Output folder", str(csv_path.parent)])
            w.writerow(["Predictors (original)", len(feature_names_all)])
            w.writerow(["Predictors used in model", int(np.sum(used_in_model))])
            w.writerow(["Samples", n_samples])
            w.writerow(["Positive landslide samples", n_pos])
            w.writerow(["Background samples", n_bg])
            w.writerow(["Spatial blocks", n_groups])
            w.writerow(["Pixel size (m)", f"{pixel_size:.2f}"])
            w.writerow(["Min distance (m)", f"{min_distance_m:.1f}"])
            w.writerow(["Min distance (px)", f"{min_dist_px:.2f}"])
            w.writerow([])

            w.writerow(["OVERALL PERFORMANCE (CV)"])
            w.writerow(["Metric", "Mean", "Std", "Min", "Max"])
            if fold_metrics:
                for key, name in [('auc', 'AUC'), ('pr_auc', 'PR-AUC'), ('brier', 'Brier score'),
                                  ('sensitivity', 'Sensitivity'), ('specificity', 'Specificity'),
                                  ('mcc', 'MCC'), ('moran_i', "Moran's I"), ('moran_p', "Moran's I p-value")]:
                    vals = [m[key] for m in fold_metrics]
                    w.writerow([name, f"{np.mean(vals):.4f}", f"{np.std(vals):.4f}",
                                f"{np.min(vals):.4f}", f"{np.max(vals):.4f}"])
            w.writerow([])
            w.writerow(["AUC 95% CI (block bootstrap)", f"{auc_ci[0]:.4f}", f"{auc_ci[1]:.4f}"])
            w.writerow([])

            w.writerow(["SPATIAL CV RESULTS"])
            w.writerow(["Fold", "AUC", "PR-AUC", "Brier", "Sensitivity", "Specificity", "MCC", "Moran's I", "p-value"])
            for m in fold_metrics:
                w.writerow([
                    m['fold'], f"{m['auc']:.4f}", f"{m['pr_auc']:.4f}", f"{m['brier']:.4f}",
                    f"{m['sensitivity']:.4f}", f"{m['specificity']:.4f}", f"{m['mcc']:.4f}",
                    f"{m['moran_i']:.4f}", f"{m['moran_p']:.4f}"
                ])
            w.writerow([])

            if test_metrics:
                w.writerow(["HELD-OUT TEST SET PERFORMANCE"])
                w.writerow(["Metric", "Value"])
                for key, val in test_metrics.items():
                    w.writerow([key, f"{val:.4f}"])
                w.writerow([])

            w.writerow(["FEATURE DIAGNOSTICS"])
            w.writerow(["Rank", "Feature", "MI (%)", "VIF", "Moran's I", "Moran's I p-value",
                        "Used in model", "Removal reason"])
            for rank, i in enumerate(mi_order, 1):
                moran_i, moran_p = predictor_moran[i]
                used = "YES" if used_in_model[i] else "NO"
                reason = removal_reason.get(feature_names_all[i], "")
                w.writerow([rank, feature_names_all[i], f"{mi_percent[i]:.2f}",
                            f"{vif_values[i]:.2f}" if not (np.isinf(vif_values[i]) or np.isnan(vif_values[i])) else "inf",
                            f"{moran_i:.4f}", f"{moran_p:.4f}",
                            used, reason])
            w.writerow([])

            if shap_values:
                w.writerow(["SHAP IMPORTANCE (mean |SHAP|)"])
                w.writerow(["Feature", "Mean |SHAP|"])
                for feat, val in sorted(shap_values.items(), key=lambda x: x[1], reverse=True):
                    w.writerow([feat, f"{val:.6f}"])
                w.writerow([])

    def processAlgorithm(self, parameters, context, feedback):
        vector_layer = self.parameterAsVectorLayer(parameters, self.INPUT_VECTOR, context)
        base_raster = self.parameterAsRasterLayer(parameters, self.INPUT_BASE, context)
        feature_layers = self.parameterAsLayerList(parameters, self.INPUT_PREDICTORS, context)
        min_distance_m = self.parameterAsDouble(parameters, self.MIN_DISTANCE, context)
        spatial_block = self.parameterAsInt(parameters, self.BLOCK_SIZE, context)
        output_dir = Path(self.parameterAsString(parameters, self.OUTPUT_DIR, context))
        output_prefix = self.parameterAsString(parameters, self.OUTPUT_PREFIX, context)

        cfg.n_estimators = self.parameterAsInt(parameters, self.N_ESTIMATORS, context)
        cfg.max_depth = self.parameterAsInt(parameters, self.MAX_DEPTH, context)
        cfg.min_samples_leaf = self.parameterAsInt(parameters, self.MIN_SAMPLES_LEAF, context)
        cfg.smooth_sigma = self.parameterAsDouble(parameters, self.SMOOTH_SIGMA, context)
        cfg.n_splits = self.parameterAsInt(parameters, self.CV_SPLITS, context)

        bg_ratio = self.parameterAsDouble(parameters, self.BG_RATIO, context)
        extra_buffer = self.parameterAsDouble(parameters, self.BUFFER_DIST, context)
        test_size = self.parameterAsDouble(parameters, self.TEST_SIZE, context)
        compute_shap = self.parameterAsBool(parameters, self.COMPUTE_SHAP, context)
        class_method = self.parameterAsEnum(parameters, self.CLASS_METHOD, context)
        model_input = self.parameterAsString(parameters, self.MODEL_INPUT, context)
        impute_nodata = self.parameterAsBool(parameters, self.IMPUTE_NODATA, context)
        vif_threshold = self.parameterAsDouble(parameters, self.VIF_THRESHOLD, context)
        resample_predictors = self.parameterAsBool(parameters, self.RESAMPLE_PREDICTORS, context)
        spatial_calibration = self.parameterAsBool(parameters, self.SPATIAL_CALIBRATION, context)

        if not vector_layer or not base_raster or len(feature_layers) == 0:
            raise QgsProcessingException("Please select inventory, base raster and at least one predictor.")

        output_dir.mkdir(parents=True, exist_ok=True)
        TS = int(time.time())

        feedback.pushInfo("Loading base raster...")
        ds, gt, proj, rows, cols, valid_mask, pixel_size = self._load_raster_data(base_raster.source())
        feedback.pushInfo(f"Dimensions: {rows}×{cols} | Valid cells: {np.sum(valid_mask):,} | Pixel size: {pixel_size:.2f}")

        min_dist_px = max(1.0, min_distance_m / pixel_size)
        if extra_buffer > 0:
            buffer_px = min_dist_px + max(1.0, extra_buffer / pixel_size)
        else:
            buffer_px = min_dist_px
        feedback.pushInfo(f"Minimum distance: {min_distance_m:.1f} m → {min_dist_px:.2f} pixels")
        if extra_buffer > 0:
            feedback.pushInfo(f"Additional buffer: {extra_buffer:.1f} m → total exclusion buffer {buffer_px:.2f} pixels")

        feedback.pushInfo("Preparing predictor rasters...")
        features_list, feature_names = prepare_predictors(
            feature_layers, ds, rows, cols, feedback,
            use_resampling=resample_predictors
        )
        ds = None
        if len(features_list) == 0:
            raise QgsProcessingException("No valid predictor rasters loaded!")

        feedback.pushInfo("Sampling landslide and background locations...")
        pos = extract_inventory_samples(vector_layer, gt, rows, cols, valid_mask, feedback)
        bg, bg_diag = generate_background_samples(
            pos, valid_mask, rows, cols, min_dist_px, cfg.max_samples,
            bg_ratio=bg_ratio, buffer_px=buffer_px, feedback=feedback
        )

        idx = np.concatenate([pos, bg])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(bg))])
        yy, xx = np.unravel_index(idx, (rows, cols))
        X_raw = np.column_stack([f.ravel()[idx] for f in features_list])
        coords = np.column_stack((xx, yy))

        # Handle missing values if impute_nodata is False
        if not impute_nodata:
            valid = ~np.any(np.isnan(X_raw), axis=1)
            X_raw = X_raw[valid]
            y = y[valid]
            yy = yy[valid]
            xx = xx[valid]
            coords = coords[valid]

        n_samples = len(X_raw)
        n_pos = int(np.sum(y == 1))
        n_bg = int(np.sum(y == 0))
        feedback.pushInfo(f"Final samples: {n_samples} ({n_pos} landslides / {n_bg} background)")

        block_pixels = max(int(spatial_block / pixel_size), 1)
        groups = create_spatial_groups(yy, xx, block_pixels)
        n_groups = len(np.unique(groups))
        feedback.pushInfo(f"Spatial blocks: {n_groups}")
        if n_groups < cfg.n_splits:
            raise QgsProcessingException(f"Only {n_groups} spatial groups – need at least {cfg.n_splits}")

        # Prepare imputed diagnostics data
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X_raw)

        feedback.pushInfo("Computing feature diagnostics (MI, VIF, Moran's I) on imputed data...")
        mi_scores, mi_order, mi_percent, vif_values, predictor_moran = self._compute_diagnostics_imputed(
            X_imputed, y, feature_names, coords
        )

        # VIF removal
        removed_predictors = []
        X_model_imputed = X_imputed
        feature_names_model = feature_names.copy()
        used_in_model = np.ones(len(feature_names), dtype=bool)
        removal_reason = {}
        if vif_threshold > 0:
            feedback.pushInfo(f"Applying VIF removal (threshold = {vif_threshold})...")
            X_model_imputed, feature_names_model, removed_predictors = self._apply_vif_removal(
                X_imputed, feature_names_model, vif_values, vif_threshold
            )
            if removed_predictors:
                feedback.pushInfo(f"Removed predictors: {', '.join(removed_predictors)}")
                for r in removed_predictors:
                    idx_rem = feature_names.index(r)
                    used_in_model[idx_rem] = False
                    removal_reason[r] = f"VIF > {vif_threshold}"
                selected_indices = [i for i, used in enumerate(used_in_model) if used]
                features_list_model = [features_list[i] for i in selected_indices]
            else:
                features_list_model = features_list
        else:
            features_list_model = features_list

        # Prepare raw data subset for model (with NaN, pipeline will impute)
        X_raw_model = X_raw.copy()
        if not np.all(used_in_model):
            X_raw_model = X_raw_model[:, used_in_model]

        # Hold-out split
        test_metrics = None
        X_train_raw, y_train, groups_train, coords_train = X_raw_model, y, groups, coords
        X_test_raw, y_test, groups_test, coords_test = None, None, None, None
        if test_size > 0 and len(X_train_raw) > 50:
            feedback.pushInfo(f"Splitting data into train/test with test size = {test_size:.2f}...")
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=cfg.seed)
            train_idx, test_idx = next(gss.split(X_train_raw, y, groups))
            X_train_split = X_train_raw[train_idx]
            y_train_split = y[train_idx]
            groups_train_split = groups[train_idx]
            coords_train_split = coords[train_idx]
            X_test_split = X_train_raw[test_idx]
            y_test_split = y[test_idx]
            groups_test_split = groups[test_idx]
            coords_test_split = coords[test_idx]

            X_train_raw, y_train, groups_train, coords_train = X_train_split, y_train_split, groups_train_split, coords_train_split
            X_test_raw, y_test, groups_test, coords_test = X_test_split, y_test_split, groups_test_split, coords_test_split

            feedback.pushInfo(f"Train: {len(X_train_raw)} samples, Test: {len(X_test_raw)} samples")

        # Load pre-trained model if provided
        loaded_model = None
        if model_input and Path(model_input).exists():
            feedback.pushInfo(f"Loading pre-trained model from {model_input}...")
            try:
                loaded_model = joblib.load(model_input)
                # Check if model has feature names
                if hasattr(loaded_model, 'feature_names_in_'):
                    expected_features = list(loaded_model.feature_names_in_)
                    if expected_features != feature_names_model:
                        feedback.pushWarning("Loaded model feature names do not match current predictors. Results may be incorrect.")
            except Exception as e:
                feedback.pushWarning(f"Failed to load model: {str(e)}")
                loaded_model = None

        if loaded_model is not None:
            model = loaded_model
            fold_metrics = []
            all_y_true, all_y_pred = np.array([]), np.array([])
            all_test_groups = np.array([])
            auc_ci = [np.nan, np.nan]
            feedback.pushInfo("Using pre-trained model; skipping cross-validation and training.")
        else:
            # Spatial CV on training set
            if len(np.unique(y_train)) >= 2 and len(y_train) >= 30:
                fold_metrics, all_y_true, all_y_pred, all_test_groups = run_spatial_cv(
                    X_train_raw, y_train, groups_train, coords_train, feedback,
                    use_spatial_calibration=spatial_calibration
                )
                if len(all_y_true) > 0:
                    # Use block bootstrap if groups are available and lengths match
                    if len(all_test_groups) == len(all_y_true):
                        feedback.pushInfo("Computing block bootstrap 95% CI for AUC...")
                        auc_ci = block_bootstrap_auc(all_y_true, all_y_pred, all_test_groups,
                                                     n_boot=200, seed=cfg.seed)
                    else:
                        # Fallback to observation bootstrap if lengths mismatch
                        feedback.pushInfo("Group lengths mismatch; using observation bootstrap...")
                        auc_bootstrap = []
                        for _ in range(200):
                            idx_bs = np.random.choice(len(all_y_true), min(len(all_y_true), 500), replace=True)
                            try:
                                auc_bootstrap.append(roc_auc_score(all_y_true[idx_bs], all_y_pred[idx_bs]))
                            except ValueError:
                                pass
                        auc_ci = np.nanpercentile(auc_bootstrap, [2.5, 97.5]) if auc_bootstrap else [np.nan, np.nan]
                    feedback.pushInfo(f"AUC 95% CI: [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
                else:
                    auc_ci = [np.nan, np.nan]
            else:
                fold_metrics = []
                all_y_true, all_y_pred = np.array([]), np.array([])
                all_test_groups = np.array([])
                auc_ci = [np.nan, np.nan]
                feedback.pushWarning("Not enough training samples for CV.")

            # Train final model on full training set (or all data if no test split)
            feedback.pushInfo("Training final model...")
            if spatial_calibration:
                model = build_model(groups=groups_train)
                try:
                    model.fit(X_train_raw, y_train, model__groups=groups_train)
                except Exception:
                    feedback.pushInfo("Spatial calibration failed during final model fitting; using default calibration.")
                    model = build_model(groups=None)
                    model.fit(X_train_raw, y_train)
            else:
                model = build_model(groups=None)
                model.fit(X_train_raw, y_train)

        # Evaluate on test set if available
        if X_test_raw is not None and len(X_test_raw) > 0:
            feedback.pushInfo("Evaluating on held-out test set...")
            p_test = model.predict_proba(X_test_raw)[:, 1]
            y_pred_test = (p_test >= 0.5).astype(int)
            auc_test = roc_auc_score(y_test, p_test)
            pr_auc_test = average_precision_score(y_test, p_test)
            brier_test = brier_score_loss(y_test, p_test)
            mcc_test = matthews_corrcoef(y_test, y_pred_test)
            cm = confusion_matrix(y_test, y_pred_test)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
            sens_test = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec_test = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            test_metrics = {
                'AUC': auc_test,
                'PR-AUC': pr_auc_test,
                'Brier': brier_test,
                'Sensitivity': sens_test,
                'Specificity': spec_test,
                'MCC': mcc_test
            }
            feedback.pushInfo(f"Test AUC = {auc_test:.3f}")

        # Generate maps using the final model and the selected features
        feedback.pushInfo("Generating susceptibility maps...")
        prob_raw, prob_smooth, ambiguity, susceptibility = self._generate_susceptibility_maps_normalized_smoothing(
            model, features_list_model, feature_names_model, valid_mask, rows, cols,
            feedback, class_method=class_method
        )

        # Save outputs
        feedback.pushInfo("Saving outputs...")
        output_paths = {
            'raw': save_raster(prob_raw, output_dir / f"{output_prefix}_Probability_Raw_v{cfg.version}_{TS}.tif", gt, proj, valid_mask),
            'smooth': save_raster(prob_smooth, output_dir / f"{output_prefix}_Susceptibility_Smoothed_v{cfg.version}_{TS}.tif", gt, proj, valid_mask),
            'amb': save_raster(ambiguity, output_dir / f"{output_prefix}_Probability_Ambiguity_v{cfg.version}_{TS}.tif", gt, proj, valid_mask),
            'classes': save_raster(susceptibility, output_dir / f"{output_prefix}_Susceptibility_Classes_v{cfg.version}_{TS}.tif", gt, proj, valid_mask, True)
        }

        model_path = output_dir / f"{output_prefix}_model_v{cfg.version}_{TS}.pkl"
        if loaded_model is None:
            joblib.dump(model, model_path)
        else:
            import shutil
            shutil.copyfile(model_input, model_path)

        # Metadata with versions
        import sklearn
        import scipy
        metadata = {
            'version': cfg.version,
            'timestamp': TS,
            'crs': proj,
            'extent': [gt[0], gt[3], gt[1], gt[5]],
            'predictors': feature_names_model,
            'params': {
                'n_estimators': cfg.n_estimators,
                'max_depth': cfg.max_depth,
                'min_samples_leaf': cfg.min_samples_leaf,
                'smooth_sigma': cfg.smooth_sigma,
                'n_splits': cfg.n_splits,
                'background_ratio': bg_ratio,
                'test_size': test_size,
                'class_method': class_method,
                'vif_threshold': vif_threshold,
                'impute_nodata': impute_nodata,
                'resample_predictors': resample_predictors,
                'spatial_calibration': spatial_calibration,
                'calibration_method': 'isotonic' if not spatial_calibration else 'isotonic_spatial_group_cv'
            },
            'environment': {
                'python_version': sys.version,
                'qgis_version': Qgis.QGIS_VERSION,
                'gdal_version': gdal.VersionInfo('RELEASE_NAME'),
                'numpy_version': np.__version__,
                'scipy_version': scipy.__version__,
                'sklearn_version': sklearn.__version__,
                'shap_available': SHAP_AVAILABLE
            },
            'random_seed': cfg.seed,
            'spatial_block_m': spatial_block,
            'min_distance_m': min_distance_m,
            'additional_buffer_m': extra_buffer
        }
        with open(output_dir / f"{output_prefix}_model_metadata_v{cfg.version}_{TS}.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # SHAP (if requested)
        shap_values = None
        if compute_shap and loaded_model is None:
            if SHAP_AVAILABLE:
                feedback.pushInfo("Computing SHAP values...")
                max_shap_samples = 500
                if len(X_train_raw) > max_shap_samples:
                    shap_idx = np.random.choice(len(X_train_raw), max_shap_samples, replace=False)
                    X_shap = X_train_raw[shap_idx]
                else:
                    X_shap = X_train_raw
                shap_values = compute_shap_values(model, X_shap, feature_names_model)
                if shap_values:
                    feedback.pushInfo("SHAP values computed.")
            else:
                feedback.pushWarning("SHAP not installed; skipping.")

        # Symbology
        prob_colors = [
            (0.0, QColor(0, 0, 255)),
            (0.25, QColor(0, 255, 255)),
            (0.5, QColor(0, 255, 0)),
            (0.75, QColor(255, 255, 0)),
            (1.0, QColor(255, 0, 0))
        ]
        amb_colors = [
            (0.0, QColor(0, 128, 0)),
            (0.5, QColor(255, 255, 0)),
            (1.0, QColor(255, 0, 0))
        ]

        for key, p in output_paths.items():
            if p.exists():
                layer = QgsRasterLayer(str(p), f"{output_prefix} {key} v{cfg.version}")
                if layer.isValid():
                    if key == 'classes':
                        apply_lsm_symbology(layer)
                    elif key in ('raw', 'smooth'):
                        apply_continuous_symbology(layer, prob_colors, 0.0, 1.0)
                    elif key == 'amb':
                        apply_continuous_symbology(layer, amb_colors, 0.0, 1.0)
                    QgsProject.instance().addMapLayer(layer)

        # Write CSV report
        csv_path = output_dir / f"{output_prefix}_summary_v{cfg.version}_{TS}.csv"
        self._write_report(
            csv_path, fold_metrics, auc_ci, feature_names,
            mi_order, mi_percent, vif_values, predictor_moran,
            used_in_model, removal_reason,
            n_samples, n_pos, n_bg, n_groups, pixel_size,
            min_distance_m, min_dist_px, test_metrics=test_metrics,
            shap_values=shap_values
        )

        feedback.pushInfo(f"Full report saved: {csv_path}")
        if fold_metrics:
            feedback.pushInfo("=" * 50)
            feedback.pushInfo(f"Mean CV AUC: {np.mean([m['auc'] for m in fold_metrics]):.3f} ± {np.std([m['auc'] for m in fold_metrics]):.3f}")
            if test_metrics:
                feedback.pushInfo(f"Test AUC: {test_metrics['AUC']:.3f}")
            feedback.pushInfo(f"Output folder: {output_dir}")
            feedback.pushInfo("=" * 50)

        return {self.OUTPUT_DIR: str(output_dir)}