
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
        log_file.write(s +"\\n")
        log_file.flush()

def procrustes_alignment(ref_points: np.ndarray, target_points: np.ndarray, ignore_scale: bool=False):
    
    mtx1, mtx2, disparity = procrustes(ref_points, target_points)
    if ignore_scale:
        scale = np.linalg.norm(mtx1) / (np.linalg.norm(mtx2) + 1e-12)
        mtx *= scale
    return mtx1, mtx2, float(disparity)

def iterative_procrustes_alingnment(list_of_shapes, n_iter: int=10):
    if len(list_of_shapes) == 0:
        return None, []
    if len(list_of_shapes) == 1:
        return list_of_shapes[0], list_of_shapes
    ref = list_of_shapes[0].copy()
    
    aligned_shapes = list_of_shapes[:]
    prev_distance = np.inf

    prev_mean_disp = None

    for it in range(n_iter):
        mean_disp_vals = []
        new_aligned = []
        for shape in aligned_shapes:
            _, mtx_shape, disp = procrustes_alignment(ref, shape, ignore_scale=False)
            mean_disp_vals.append(disp)
            new_aligned.append(mtx_shape)
        ref_new = np.mean(np.stack(new_aligned, axis=0), axis=0)
        distance = float(np.linalg.norm(ref_new - ref))

        mean_disp = float(np.mean(mean_disp_vals)) if mean_disp_vals else float("nan")
        min_disp = float(np.min(mean_disp_vals)) if mean_disp_vals else float("nan")
        max_disp = float(np.max(mean_disp_vals)) if mean_disp_vals else float("nan")
        delta_disp = (prev_mean_disp - mean_disp) if (prev_mean_disp is not None) else float("nan")
        _log(f"[PROC] iter={it+1} ref_shift={distance: .6g} "
             f"disp_mean={mean_disp: .6g} disp_min={min_disp: .6g} disp_max={max_disp: .6g}"
             f"improvement={delta_disp:.6g}"
        )
        if distance > prev_distance:
            break
        prev_distance = distance
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
    return isinstance(val, str) and len(val) > 0 and val != "NULL" and os.path.exists(val)


def preprocess_with_procrustes(metadata_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    meta = pd.read_csv(metadata_csv)
    # normalize meta columns to match the user's key names
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
                        pts_ap = _load_particles_xy(p)
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

    _, aligned_global_shapes = iterative_procrustes_alingnment(global_shapes, n_iter=10)

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
                    pts = _load_particles_xy(p)
                    shapes_ap.append(pts)
                    metas_ap.append({c: row[c] for c in meta_cols})
                except Exception as e:
                    warnings.warn(f"Skip {ap} for now {idx}: {e}")
        if not shapes_ap:
            continue
        _, aligned_shapes_ap = iterative_procrustes_alingnment(shapes_ap, n_iter=10)

        if aligned_shapes_ap:
            final_ref_b = np.mean(np.stack(aligned_shapes_ap, axis=0), axis=0)
            disps_b = []
            for shp in shapes_ap:
                _, _, d = procrustes_alignment(final_ref_b, shp, ignore_scale=False)
                disps_b.append(float(d))
            _log(f"[PROC][BONE:{ap}] final_disp_mean={np.mean(disps_b):.6g} "
                f"min/max={np.min(disps_b):.6g}/{np.max(disps_b):.6g} n={len(disps)}")
        
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
    except Exception:
        df_g.to_csv(out_g.replace(".feather",".csv"), index=False)
    try:
        df_b.reset_index(drop=True).to_feather(out_b)
    except Exception:
        df_b.to_csv(out_b.replace(".feather",".csv"), index=False)
    print("[Saved]", out_g, "and", out_b)

if __name__ == "__main__":
    metadata_csv = "/home/ubuntu/analysis/2025_08_10_codes/파일매핑테이블생성/X1_aligned_output_None.csv"
    output_dir = "/home/ubuntu/analysis/2025_08_10_codes/최종분석/results_procrustes"
    os.makedirs(output_dir, exist_ok=True)
    
    
    log_path = os.path.join(output_dir, "processing_log.txt")
    log_file = open(log_path, 'w', encoding='utf-8')
    _log(f"[LOG] writing to [log_path]")

    try:
        preprocess_with_procrustes(metadata_csv, output_dir)
    finally:
        try:
            log_file.close()
        except Exception:
            pass
