import os
import shutil

start = "/home/ubuntu/download/data/raw_result"
target = "/home/ubuntu/download/data/vtp"

folder_mapping = {
    'T1': 'T1_TG',
    'X1': 'X1_FSAP',
    'X2': 'X2_FS',
    'X3': 'X3_FLO',
    'X4': 'X4_HAV',
    'X5': 'X5_AWBAP',
    'X6': 'X6_AWBL',
    'X7': 'X7_KWBAP',
    'X8': 'X8_KWBL'
}

def copy_files(dir, filename):
    if 'CR' in filename:
        for pattern, folder_name in folder_mapping.items():
            if pattern in filename:
                target_path = os.path.join(target, folder_name, filename)
                shutil.copy(os.path.join(dir, filename), target_path)
                print(f"Copied {filename} to {target}")
                break

for root, dirs, files in os.walk(start):
    for file in files:
        copy_files(root, file)