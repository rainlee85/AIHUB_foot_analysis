import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import math
import numpy as np
import vtk

def load_vtp_file(file_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    point_array = np.zeros((num_points, 3))
    for i in range(num_points):
        point_array[i] = points.GetPoint(i)
    return point_array[:, :2]


def parse_vtp_filename(filename):
    filename = filename.replace('.vtp','')


    if "_R_" in filename:
        direction = 'R'
        parts = filename.split("_R_")
    elif "_L_" in filename:
        direction = 'L'
        parts = filename.split("_L_")
    else:
        raise ValueError("File name does not contain direction")
    
    st_number = int([part for part in parts[0].split('_') if part.startswith('ST')][0][2:])
    patient_id = parts[0].split('_')[0]
    
    return patient_id, direction, st_number


def render_and_save_vtp_group(vtp_files, output_path):
    fig, ax = plt.subplots(figsize =(6,6))
    ax.set_facecolor('black')

    for file_path in vtp_files:
        points = load_vtp_file(file_path)
        if points.size == 0:
            continue

        ax.plot(points[:, 0], points[:, 1], color='white', linewidth=1)
        ax.plot([points[-1, 0], points[0,0]], [points[-1,1], points[0,1]], color='white', linewidth=1)
        ax.scatter(points[:, 0], points[:, 1], color='white', s=5)

    ax.axis('equal')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"Saved image to {output_path}")

def create_mosaic_images(image_folder, output_folder, n_cols=8, n_rows=5):
    max_image_per_page = n_cols * n_rows
    images = []
    labels = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".png"):
            img_path = os.path.join(image_folder, filename)
            with Image.open(img_path) as img:
                images.append(img.copy())
            patient_id, direction, st_number = filename.replace('.png','').split('_')
            labels.append(f"{patient_id} - {direction} -{st_number}")
    
    if not images:
        print("no images")
        return
    
    total_pages = math.ceil(len(images) / max_image_per_page)

    for page in range(total_pages):
        start_idx = page * max_image_per_page
        end_idx = min((page + 1) * max_image_per_page, len(images))
        page_images = images[start_idx:end_idx]
        page_labels = labels[start_idx:end_idx]

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20,12))
        axs = axs.flatten()

        for i, (img, label) in enumerate(zip(page_images, page_labels)):
            axs[i].imshow(img)
            axs[i].axis('off')
            axs[i].text(0.5, -0.1, label, size=8, ha='center', transform=axs[i].transAxes)

        for i in range(len(page_images), len(axs)):
            axs[i].axis('off')

        plt.tight_layout()
        output_path = os.path.join(output_folder, f"mosaic_page_{page+1}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()


def process_vtp_files(input_folder, temp_image_folder, final_output_folder, n_cols=8, n_rows=5):
    if not os.path.exists(temp_image_folder):
        os.makedirs(temp_image_folder)
    if not os.path.exists(final_output_folder):
        os.makedirs(final_output_folder)

    grouped_files = defaultdict(list)

    for filename in os.listdir(input_folder):
        if filename.endswith(".vtp"):
            try:
                patient_id, direction, st_number = parse_vtp_filename(filename)
                group_key = (patient_id, direction, st_number)
                grouped_files[group_key].append(os.path.join(input_folder, filename))
            except ValueError as e:
                print(f"Skipping file {filename}: {e}")

    for (patient_id, direction, st_number), files in grouped_files.items():
        output_image_path = os.path.join(temp_image_folder, f"{patient_id}_{direction}_ST{st_number}.png")
        print(f"Processing group: Patient ID: {patient_id}, Direction: {direction}, ST Number: {st_number}")
        render_and_save_vtp_group(files, output_image_path)
    
    create_mosaic_images(temp_image_folder, final_output_folder, n_cols, n_rows)

input_folder = r"J:\analyze_aihub_foot_disease_data\test_data\vtp_files"
temp_image_folder = r"J:\analyze_aihub_foot_disease_data\test_data\visual\vtp_aligned\temp"
final_output_folder = r"J:\analyze_aihub_foot_disease_data\test_data\visual\vtp_aligned\mosaic"

process_vtp_files(input_folder, temp_image_folder, final_output_folder, n_cols=8, n_rows=5)



        



