#!/usr/bin/env python3

import os, re, glob, argparse, warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.spatial import procrustes

AP_LIST = ["AP1","AP2","AP3","AP4","AP5","AP6","AP7","AP8"]
VERBOSE = True
log_file = None

def _log(msg):
    s = str(msg)
    if VERBOSE:
        print(s, flush=True)
    global log_file
    if log_file is not None:
        log_file.write(s +"\n")
        log_file.flush()

def procrustes_alignment(ref_points: np.ndarray, target_points: np.ndarray, ignore_scale: bool=False):
    """Perform Procrustes alignment between reference and target points"""
    # Input validation
    if len(ref_points) == 0 or len(target_points) == 0:
        raise ValueError("Input point arrays cannot be empty")
    if ref_points.shape[1] != target_points.shape[1]:
        raise ValueError("Reference and target points must have same dimensions")

    try:
        mtx1, mtx2, disparity = procrustes(ref_points, target_points)
        if ignore_scale:
            scale = np.linalg.norm(mtx1) / (np.linalg.norm(mtx2) + 1e-12)
            mtx2 *= scale  # FIXED: was 'mtx *= scale'
        return mtx1, mtx2, float(disparity)
    except np.linalg.LinAlgError as e:
        _log(f"[WARNING] Procrustes alignment failed: {e}")
        return ref_points, target_points, float('inf')

def iterative_procrustes_alignment(list_of_shapes, n_iter: int=10, tolerance: float=1e-6):
    """Perform iterative Procrustes alignment with improved initialization and convergence"""
    if len(list_of_shapes) == 0:
        return None, []
    if len(list_of_shapes) == 1:
        return list_of_shapes[0], list_of_shapes

    # IMPROVED: Better initialization - use shape closest to centroid
    centroids = [np.mean(shape, axis=0) for shape in list_of_shapes]
    mean_centroid = np.mean(centroids, axis=0)
    centroid_distances = [np.linalg.norm(c - mean_centroid) for c in centroids]
    best_ref_idx = np.argmin(centroid_distances)
    ref = list_of_shapes[best_ref_idx].copy()
    _log(f"[PROC] Using shape {best_ref_idx} as initial reference (closest to centroid)")

    aligned_shapes = list_of_shapes[:]
    prev_mean_disp = None
    no_improvement_count = 0

    for it in range(n_iter):
        mean_disp_vals = []
        new_aligned = []

        for shape in aligned_shapes:
            _, mtx_shape, disp = procrustes_alignment(ref, shape, ignore_scale=False)
            mean_disp_vals.append(disp)
            new_aligned.append(mtx_shape)

        ref_new = np.mean(np.stack(new_aligned, axis=0), axis=0)
        ref_shift = float(np.linalg.norm(ref_new - ref))

        mean_disp = float(np.mean(mean_disp_vals)) if mean_disp_vals else float("nan")
        min_disp = float(np.min(mean_disp_vals)) if mean_disp_vals else float("nan")
        max_disp = float(np.max(mean_disp_vals)) if mean_disp_vals else float("nan")

        # IMPROVED: Better convergence criteria
        delta_disp = (prev_mean_disp - mean_disp) if (prev_mean_disp is not None) else float("nan")
        _log(f"[PROC] iter={it+1} ref_shift={ref_shift: .6g} "
             f"disp_mean={mean_disp: .6g} disp_min={min_disp: .6g} disp_max={max_disp: .6g} "
             f"improvement={delta_disp:.6g}"
        )

        # Check for convergence based on both reference shift and disparity improvement
        converged = ref_shift < tolerance
        if prev_mean_disp is not None:
            disp_improvement = abs(delta_disp) < tolerance
            if disp_improvement:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            # Early stopping if no significant improvement for 3 iterations
            if no_improvement_count >= 3 or converged:
                _log(f"[PROC] Converged after {it+1} iterations")
                break

        prev_mean_disp = mean_disp
        ref = ref_new
        aligned_shapes = new_aligned

    return ref, aligned_shapes

def preserve_and_adjust_points(point_2d: np.ndarray) -> np.ndarray:
    adjusted = point_2d.copy()
    seen = {}
    for i, (x,y) in enumerate(point_2d):
        key = (round(float(x), 5), round(float(y), 5))
        cnt = seen.get(key, 0)
        if cnt > 0:
            adjusted[i] += np.random.normal(0.0, 1e-5, size=2)
        seen[key] = cnt + 1
    return adjusted

def _load_particles_xy(path: str) -> np.ndarray:
    """Load particles file with x,y coordinates (ignoring z)"""

    pts = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:

            t = line.strip()
            if not t:
                continue
            parts = t.split()
            if len(parts) < 3:
                continue
            try:
                x, y = float(parts[0]), float(parts[1])

            except Exception:
                continue
            pts.append((x, y))
    if not pts:
        raise ValueError(f"no valid 'x y z' points in {path}")
    return np.asarray(pts, dtype=float)

def _valid_path(val) -> bool:
    if not isinstance(val, str) or len(val) == 0 or val == "NULL":
        return False

    # Fix Windows path separators for Linux
    normalized_path = val.replace('\\', '/')
    return os.path.exists(normalized_path)


def preprocess_with_procrustes(metadata_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    meta = pd.read_csv(metadata_csv)

    # Normalize meta columns to match the user's key names
    required = {"patient_id","group_sex","group_age","group_direction", "ST_number"}
    if not required.issubset(set(meta.columns)):
        raise ValueError(f"metadata_csv missing required columns: {required - set(meta.columns)}")
    meta_cols = list(meta.columns)


    #global
    global_shapes = []
    global_parts_meta = []
    global_splits = []

    for idx, row in meta.iterrows():
        ap_seq = []
        split_info = []
        for ap in AP_LIST:
            col = f"landmarks_file_{ap}"
            if col in meta.columns:
                p=row[col]
                if _valid_path(p):
                    try:
                        # Fix Windows path separators for Linux
                        normalized_p = p.replace('\\', '/')
                        pts_ap = _load_particles_xy(normalized_p)
                        ap_seq.append(pts_ap)
                        split_info.append((ap, pts_ap.shape[0]))
                    except Exception as e:
                        warnings.warn(f"Skip {ap} for row {idx}: {e}")
        if not ap_seq:
            continue
        concat = np.vstack(ap_seq)
        concat = preserve_and_adjust_points(concat)
        global_shapes.append(concat)
        global_parts_meta.append({col: row[col] for col in meta_cols})
        global_splits.append(split_info)

    _, aligned_global_shapes = iterative_procrustes_alignment(global_shapes, n_iter=10)

    if aligned_global_shapes:
        final_ref = np.mean(np.stack(aligned_global_shapes, axis=0), axis=0)
        disps = []
        for shp in global_shapes:
            _, _, d = procrustes_alignment(final_ref, shp, ignore_scale=False)
            disps.append(float(d))
        _log(f"[PROC][GLOBAL] final_disp_mean={np.mean(disps):.6g} "
             f"min/max={np.min(disps):.6g}/{np.max(disps):.6g} n={len(disps)}")
    rows_global = []
    for meta_payload, split_info, aligned in zip(global_parts_meta, global_splits, aligned_global_shapes):
        start = 0
        for ap, n in split_info:
            block = aligned[start:start+n]
            for j, (x,y) in enumerate(block):
                rows_global.append({
                    **meta_payload,
                    "ap": ap,
                    "landmark_index": int(j),
                    "x": float(x),
                    "y": float(y),
                    "z": 0.0,
                })
            start += n

    #bonewise
    rows_bone = []
    for ap in AP_LIST:
        col = f"landmarks_file_{ap}"
        if col not in meta.columns:
            continue
        shapes_ap = []
        metas_ap = []
        for idx, row in meta.iterrows():
            p = row[col] if col in row else None
            if _valid_path(p):
                try:
                    # Fix Windows path separators for Linux
                    normalized_p = p.replace('\\', '/')
                    pts = _load_particles_xy(normalized_p)
                    shapes_ap.append(pts)
                    metas_ap.append({c: row[c] for c in meta_cols})
                except Exception as e:
                    warnings.warn(f"Skip {ap} for row {idx}: {e}")
        if not shapes_ap:
            continue
        _, aligned_shapes_ap = iterative_procrustes_alignment(shapes_ap, n_iter=10)

        if aligned_shapes_ap:
            final_ref_b = np.mean(np.stack(aligned_shapes_ap, axis=0), axis=0)
            disps_b = []
            for shp in shapes_ap:
                _, _, d = procrustes_alignment(final_ref_b, shp, ignore_scale=False)
                disps_b.append(float(d))
            _log(f"[PROC][BONE:{ap}] final_disp_mean={np.mean(disps_b):.6g} "
                f"min/max={np.min(disps_b):.6g}/{np.max(disps_b):.6g} n={len(disps_b)}")

        for meta_payload, aligned in zip(metas_ap, aligned_shapes_ap):

            for j, (x, y) in enumerate(aligned):
                rows_bone.append({
                    **meta_payload,
                    "ap": ap,
                    "landmark_index": int(j),
                    "x": float(x),
                    "y": float(y),
                    "z": 0.0
                })


    df_g = pd.DataFrame(rows_global)
    df_b = pd.DataFrame(rows_bone)
    out_g = os.path.join(output_dir, "aligned_global.feather")
    out_b = os.path.join(output_dir, "aligned_bonewise.feather")

    # Try feather; fallback to CSV if pyarrow missing
    try:
        df_g.reset_index(drop=True).to_feather(out_g)
        _log(f"[SAVED] Global alignment: {out_g}")
    except Exception as e:
        out_g_csv = out_g.replace(".feather",".csv")
        df_g.to_csv(out_g_csv, index=False)
        _log(f"[SAVED] Global alignment (CSV fallback): {out_g_csv}")

    try:
        df_b.reset_index(drop=True).to_feather(out_b)
        _log(f"[SAVED] Bonewise alignment: {out_b}")
    except Exception as e:
        out_b_csv = out_b.replace(".feather",".csv")
        df_b.to_csv(out_b_csv, index=False)
        _log(f"[SAVED] Bonewise alignment (CSV fallback): {out_b_csv}")

    return out_g, out_b

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess foot disease data with Procrustes alignment')
    parser.add_argument('--metadata_csv', type=str,
                       default="test_data/output/vtp_aligned_output_None.csv",
                       help='Path to metadata CSV file')
    parser.add_argument('--output_dir', type=str,
                       default="test_data/output/procrustes_results",
                       help='Output directory for results')

    args = parser.parse_args()

    metadata_csv = args.metadata_csv
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_path = os.path.join(output_dir, "processing_log.txt")
    log_file = open(log_path, 'w', encoding='utf-8')
    _log(f"[LOG] Writing to {log_path}")
    _log(f"[INPUT] Metadata CSV: {metadata_csv}")
    _log(f"[OUTPUT] Results directory: {output_dir}")

    try:
        out_global, out_bonewise = preprocess_with_procrustes(metadata_csv, output_dir)
        _log("[SUCCESS] Preprocessing completed successfully!")
        print(f"Results saved:")
        print(f"  Global alignment: {out_global}")
        print(f"  Bonewise alignment: {out_bonewise}")
        print(f"  Log file: {log_path}")
    except Exception as e:
        _log(f"[ERROR] Processing failed: {e}")
        raise
    finally:
        try:
            log_file.close()
        except Exception:
            pass