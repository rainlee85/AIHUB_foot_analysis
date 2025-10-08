#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified VTP Alignment Pipeline (Code-config only)
Configuration is controlled by the CONFIG block below (no CLI arguments).

Pipeline:
  1) L/R 변환: Left 데이터를 선택 축(기본 X)으로 반사
     - 단, 'no_mirror_states'에 속한 라벨(예: reflected)인 항목은 미러링 금지
  2) 중심 정렬: AP7 중심을 원점(0,0,0)으로 이동 (그룹 공통 translation)
  3) 회전 정렬: AP7→AP4 방향벡터를 글로벌 +Y축과 정렬 (그룹 공통 rotation)

Excel 라벨링(선택):
  - 컬럼(대소문자 무시): subject, direction, viewpoint, study_number, must be
  - exclude_states: 완전히 제외할 상태값 집합 (예: {'removed'})
  - no_mirror_states: 좌우반전 금지할 상태값 집합 (예: {'reflected'})

그룹키: (patient_name, study_num, series_num, viewpoint)
파일명 파싱: utils.parse_filename 를 사용(기존 환경과 동일).
"""

# =========================
# ======= CONFIG ==========
# =========================
CONFIG = {
    # 경로
    "input_dir": "./test_data/vtp_files",             # 입력 루트 폴더
    "output_dir": "./test_data/vtp_aligned",           # 출력 폴더
    "outliers_xlsx": None,  # 라벨 엑셀 (없으면 None)

    # 동작 옵션
    "mirror_axis": "X",           # 'X' | 'Y' | 'Z'
    "skip_rotation": False,       # True 이면 AP7→AP4 회전 정렬 생략
    "strict_ap": True,           # True 이면 AP7과 AP4가 없으면 그룹 스킵

    # 라벨 처리
    "exclude_states": {"removed"},     # 이 상태 라벨은 완전 제외
    "no_mirror_states": {"reflected"}, # 이 상태 라벨은 L이라도 미러링 금지
}

import os
import logging
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R



logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='./alignment.log',
                    filemode='w'
                    )
filename_pattern = re.compile(
    r'^'
    f'(.*?)_'
    f'(.*?)_'
    r'(ST\d+)_'
    r'(SE\d+)_'
    r'(IM\d+)_'
    r'(T\d|X\d)_'
    r'(.*?)_'
    r'(R|L)_'
    r'(.*?)_'
    r'LabelPolyLine\.vtp$'
)

def parse_filename(fname: str):
    match = filename_pattern.match(fname)
    if not match:
        return None
    
    patient_name = match.group(1)
    device_name = match.group(2)
    study_num = match.group(3)
    series_num = match.group(4)
    image_num = match.group(5)
    viewpoint = match.group(6)
    dummy_field = match.group(7)
    direction = match.group(8)
    bone_name = match.group(9)

    return {
        "patient_name": patient_name,
        "device_name": device_name,
        "study_num": study_num,
        "series_num": series_num,
        "image_num": image_num,
        "viewpoint": viewpoint,
        "dummy_field": dummy_field,
        "direction": direction,
        "bone_name": bone_name,
        "orginal_filename": fname
    }
# Loaded from Excel (if provided)
OUTLIERS = set()   # keys to exclude entirely
STATE_MAP = {}     # key -> state (lowercased)
# key format: (subject, direction, study_number, viewpoint)


def mirror_points(points: np.ndarray, axis: str = "X") -> np.ndarray:
    pts = points.copy()
    axis = axis.upper()
    if axis == "X":
        pts[:, 0] = -pts[:, 0]
    elif axis == "Y":
        pts[:, 1] = -pts[:, 1]
    elif axis == "Z":
        pts[:, 2] = -pts[:, 2]
    else:
        raise ValueError(f"Unknown mirror axis: {axis}")
    return pts


def calc_center(mesh: pv.PolyData) -> np.ndarray:
    return np.array(mesh.center, dtype=np.float64)


def compute_rotation_from_u_to_v(u: np.ndarray, v: np.ndarray) -> Optional[R]:
    """Return Rotation that maps unit vector u -> unit vector v. If invalid, return None."""
    if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
        return None
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if np.isclose(dot, 1.0, atol=1e-8):
        return R.identity()
    if np.isclose(dot, -1.0, atol=1e-8):
        a = np.array([1.0, 0.0, 0.0])
        if abs(u[0]) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        axis = np.cross(u, a)
        axis /= np.linalg.norm(axis)
        return R.from_rotvec(axis * np.pi)
    axis = np.cross(u, v)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return R.identity()
    axis /= norm
    angle = np.arccos(dot)
    return R.from_rotvec(axis * angle)


# ===== Excel loaders =====
def load_outliers_from_xlsx(xlsx_path: str, exclude_states: set) -> set:
    """Return a set of keys to exclude entirely based on 'must be' states."""
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_subject = pick("subject", "patient_name", "patient")
    c_direction = pick("direction", "dir")
    c_viewpoint = pick("viewpoint", "vp", "view")
    c_study = pick("study_number", "study_num", "study", "st")
    c_state = pick("must be", "state", "label")

    out = set()
    for _, row in df.iterrows():
        state = str(row[c_state]).strip().lower()
        if state in exclude_states:
            subj = str(row[c_subject]).strip()
            dire = str(row[c_direction]).strip()
            vp = str(row[c_viewpoint]).strip()
            st = str(row[c_study]).strip()
            out.add((subj, dire, st, vp))
    return out


def load_state_map_from_xlsx(xlsx_path: str) -> dict:
    """Returns a dict mapping (subject, direction, study_number, viewpoint) -> 'state' (lowercased)."""
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_subject = pick("subject", "patient_name", "patient")
    c_direction = pick("direction", "dir")
    c_viewpoint = pick("viewpoint", "vp", "view")
    c_study = pick("study_number", "study_num", "study", "st")
    c_state = pick("must be", "state", "label")

    m = {}
    for _, row in df.iterrows():
        subj = str(row[c_subject]).strip()
        dire = str(row[c_direction]).strip()
        vp = str(row[c_viewpoint]).strip()
        st = str(row[c_study]).strip()
        state = str(row[c_state]).strip().lower()
        m[(subj, dire, st, vp)] = state
    return m



def process_group(file_infos: List[Tuple[str, dict]], mirror_axis: str, skip_rotation: bool,
                  strict_ap: bool, output_dir: str, no_mirror_states: set) -> None:
    loaded = []
    ap7_mesh_map = {"L": None, "R": None}
    ap4_mesh_map = {"L": None, "R": None}

    for fpath, info in file_infos:
        mesh = pv.read(fpath)
        key_for_state = (info.get("patient_name"), info.get("direction"), info.get("study_num"), info.get("viewpoint"))
        state = STATE_MAP.get(key_for_state)
        should_skip_mirror = state in no_mirror_states if state is not None else False

        if info.get("direction") == "L" and not should_skip_mirror:
            mesh.points = mirror_points(mesh.points, axis=mirror_axis)

        loaded.append((mesh, info))

    by_side = {"L": [], "R": []}
    for mesh, info in loaded:
        side = info.get("direction", "R")
        if side not in ("L", "R"):
            side = 'R'
        by_side[side].append((mesh, info))

        bone_name = info.get("bone_name", "")
        fname = info.get("original_filename", "")
        if ap7_mesh_map[side] is None and (bone_name == "AP7" or ("AP7" in fname)):
            ap7_mesh_map[side] = mesh
        if ap4_mesh_map[side] is None and (bone_name == "AP4" or ("AP4" in fname)):
            ap4_mesh_map[side] = mesh

    def center_and_rotate_one_side(side: str):
        group = by_side[side]
        if not group:
            return
        ap7_mesh = ap7_mesh_map[side]
        ap4_mesh = ap4_mesh_map[side]
        if ap7_mesh is None:
            logging.warning("AP7 not found in group -> cannot center on AP7. Skipping this group.")
            if strict_ap:
                return
            else:
                return

        c_ap7 = calc_center(ap7_mesh)
        c_ap4 = calc_center(ap4_mesh)
        logging.info(f"[{side}] original AP7 center (X,Y): ({c_ap7[0]:.2f}, {c_ap7[1]}:.2f)")
        logging.info(f"[{side}] original AP4 center (X,Y): ({c_ap4[0]:.2f}, {c_ap4[1]}:.2f)")
        tvec = -c_ap7
        for mesh, _ in loaded:
            mesh.translate(tvec, inplace=True)
        
        if not skip_rotation and ap4_mesh is not None:
            ap7_c = calc_center(ap7_mesh)
            ap4_c = calc_center(ap4_mesh)
            logging.info("---Starting Z-axis rotation based on AP4 ---")
            logging.info(f"[{side}] AP7 center (X,Y): ({ap7_c[0]:.2f}, {ap7_c[1]}:.2f)")
            logging.info(f"[{side}] AP4 center (X,Y): ({ap4_c[0]:.2f}, {ap4_c[1]}:.2f)")
            vec = ap4_c[:2] - ap7_c[:2]
            if np.linalg.norm(vec) < 1e-6:
                logging.warning(f"AP4 and AP7 have identical ")
            else:
            
                current_angle = np.arctan2(vec[1], vec[0])
                target_angle = np.pi / 2
                angle_diff = target_angle - current_angle

                logging.info(f"[{side}] Current angle (deg): {np.rad2deg(current_angle):.2f}")
                logging.info(f"[{side}] Taget angle (deg): {np.rad2deg(target_angle):.2f}")
                logging.info(f"[{side}] --> Rotation needed (deg): {np.rad2deg(angle_diff):.2f}")
                c, s =  np.cos(angle_diff), np.sin(angle_diff)
                R2 = np.array([[c, -s, 0],
                            [s, c,  0],
                            [0, 0, 1]])
                
                for mesh, _ in loaded:
                    mesh.points = mesh.points @ R2.T
                logging.info(f"[{side}] Rotation applied successfully.")
        elif not skip_rotation:
            logging.warning("AP4 not found; skipping rotation.")
    center_and_rotate_one_side("L")
    center_and_rotate_one_side("R")

    os.makedirs(output_dir, exist_ok=True)
    for mesh, info in loaded:
        out_name = info.get("original_filename") or os.path.basename(info.get("filepath", "unknown.vtp"))
        out_path = os.path.join(output_dir, out_name)
        mesh.save(out_path)
        logging.info(f"Saved: {out_path}")


def run():
    input_dir = CONFIG["input_dir"]
    output_dir = CONFIG["output_dir"]
    mirror_axis = CONFIG["mirror_axis"]
    skip_rotation = CONFIG["skip_rotation"]
    strict_ap = CONFIG["strict_ap"]
    exclude_states = set(s.lower() for s in CONFIG["exclude_states"])
    no_mirror_states = set(s.lower() for s in CONFIG["no_mirror_states"])

    xlsx = CONFIG.get("outliers_xlsx")
    if xlsx and os.path.isfile(xlsx):
        OUTLIERS.update(load_outliers_from_xlsx(xlsx, exclude_states))
        STATE_MAP.update(load_state_map_from_xlsx(xlsx))
        logging.info(f"Loaded {len(OUTLIERS)} excluded keys and {len(STATE_MAP)} labeled keys from {xlsx}")
    else:
        logging.info("No outliers Excel or file not found.")

    groups: Dict[Tuple[str, str, str, str], List[Tuple[str, dict]]] = {}
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(".vtp"):
                continue
            info = parse_filename(fname)
            if not info:
                logging.warning(f"Skip (pattern mismatch): {fname}")
                continue
            key4 = (info.get("patient_name"), info.get("direction"), info.get("study_num"), info.get("viewpoint"))
            if key4 in OUTLIERS:
                logging.info(f"Excluded by Excel: {fname}")
                continue
            group_key = (info.get("patient_name"), info.get("study_num"), info.get("series_num"), info.get("viewpoint"), info.get("image_num"))
            fpath = os.path.join(root, fname)
            info["original_filename"] = fname
            info["filepath"] = fpath
            groups.setdefault(group_key, []).append((fpath, info))

    if not groups:
        logging.warning("No groups found.")
        return

    for gkey, file_infos in groups.items():
        logging.info(f"Processing group: {gkey} with {len(file_infos)} files")
        process_group(file_infos, mirror_axis, skip_rotation, strict_ap, output_dir, no_mirror_states)


if __name__ == "__main__":
    run()
