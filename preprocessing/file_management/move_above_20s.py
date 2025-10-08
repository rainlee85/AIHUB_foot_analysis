import os
import re
import json
import shutil
import logging


# --- 설정: 경로/상수 변수화 ---
DEFAULT_VTP_FOLDER = "/home/ubuntu/data/vtp/X1"
DEFAULT_CLINICAL_FOLDER = "/home/ubuntu/analysis/clinical_info"
DEFAULT_OUTPUT_FOLDER = "/home/ubuntu/data/vtp_above_20s/X1"

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def get_patient_age(clinical_folder, patient_name):
    """임상정보 폴더에서 환자 나이 추출"""
    clinical_file = os.path.join(clinical_folder, f"{patient_name}_ClinicalInfo.json")
    if not os.path.exists(clinical_file):
        return None
    try:
        with open(clinical_file, 'r', encoding='utf-8') as f:
            clinical_data = json.load(f)
            age = clinical_data.get("Age", None)
        return float(age)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logging.error(f"fail for {patient_name}: {e}")
        return None

def copy_vtp_files_above_20(vtp_folder=DEFAULT_VTP_FOLDER, clinical_folder=DEFAULT_CLINICAL_FOLDER, output_folder=DEFAULT_OUTPUT_FOLDER):
    """20세 이상 환자의 VTP 파일만 복사"""
    os.makedirs(output_folder, exist_ok=True)
    for root, dirs, files in os.walk(vtp_folder):
        for fname in files:
            if not fname.lower().endswith(".vtp"):
                continue
            info = parse_filename(fname)
            if not info:
                continue
            patient_name = info['patient_name']
            age = get_patient_age(clinical_folder, patient_name)
            if age is not None and isinstance(age, float) and age >= 20:
                src_path = os.path.join(root, fname)
                dest_path = os.path.join(output_folder, fname)
                try:
                    shutil.copy2(src_path, dest_path)
                    logging.info(f"Copied: {src_path} -> {dest_path}")
                except Exception as e:
                    logging.error(f"Failed to copy {src_path}: {e}")

if __name__ == "__main__":
    copy_vtp_files_above_20()