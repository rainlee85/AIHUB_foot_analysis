import os
import shutil

folder_path = "/home/ubuntu/analysis/vtp_original"

file_names = os.listdir(folder_path)

for file_name in file_names:
    label_info = file_name.split("_")[-1].split(".")[0]
    target_folder = os.path.join(folder_path, label_info)
    os.makedirs(target_folder, exist_ok=True)

    source_path = os.path.join(folder_path, file_name)
    target_path = os.path.join(target_folder, file_name)
    shutil.move(source_path, target_path)