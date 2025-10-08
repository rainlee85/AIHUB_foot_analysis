import numpy as np
import matplotlib.pyplot as plt
from pycpd import RigidRegistration, AffineRegistration, DeformableRegistration
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import vtk
import os
from concurrent.futures import ThreadPoolExecutor


# 라벨별로 처리한 VTP 파일 경로를 지정하고 불러오기
folder_path = "./test_data/vtp_aligned"  # 실제 경로로 변경
output_folder = "./test_data/particles_files"  # 저장할 폴더 경로
os.makedirs(output_folder, exist_ok=True) 


# VTP 파일을 불러와 포인트 데이터를 추출하는 함수
def load_vtp_file(file_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    points = np.array([polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())])
    return points


# 라벨을 파일명에서 추출하는 함수 
def extract_label_from_filename(filename):
    parts = filename.split("_")
    if len(parts) > 5:
        label = "_".join(parts[5:]).split(".")[0]
        label = label.replace("_LabelPolyLine","")
        label = label.replace("_L_", "_").replace("_R_", "_")
        label = label.replace("_FSLL_", "_").replace("_FSLR_", "_")
        return label
    return "Unknown"

def resample_points(points, num_points):
    differences = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(differences ** 2, axis=1))
    total_length = np.insert(np.cumsum(segment_lengths), 0, 0)
    uniform_length = np.linspace(0, total_length[-1], num_points)
    uniform_x = np.interp(uniform_length, total_length, points[:, 0])
    uniform_y = np.interp(uniform_length, total_length, points[:, 1])
    return np.column_stack((uniform_x, uniform_y))

def load_vtp_files_by_label(folder_path):
    labeled_points = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.vtp'):
            label = extract_label_from_filename(filename)
            file_path = os.path.join(folder_path, filename)
            points = load_vtp_file(file_path)

            if label not in labeled_points:
                 labeled_points[label] = []
            labeled_points[label].append((points, filename))
    return labeled_points


# 점 리샘플링 함수
def compute_average_length(points_list):
    total_length = 0.0
    for points in points_list:
        differences = np.diff(points, axis=0)
        segment_lengths = np.sqrt(np.sum(differences ** 2, axis=1))
        total_length += np.sum(segment_lengths)
    return total_length / len(points_list) 


# 정렬 및 매칭 함수 정의 (변형 없이 대응 관계만 맞추기)
def align_points_and_find_indices(target, source):
    
    target_centroid = np.mean(target, axis=0)
    source_centroid = np.mean(source, axis=0)

    # target_centerd = target - target_centroid
    target_centerd = target
    # source_centered = source - source_centroid
    source_centered = source

    try:
        rigid_reg = RigidRegistration(**{'X': target_centerd, 'Y': source_centered, 'max_iterations': 200})
        transformed_source_rigid = rigid_reg.register()[0].reshape(len(source), 2)
    except np.linalg.LinAlgError as e:
        return source, np.arange(len(source))
    
    try:
        affine_reg = AffineRegistration(**{'X': target_centerd, 'Y': transformed_source_rigid, 'max_iterations': 200})
        transformed_source_affine = affine_reg.register()[0].reshape(len(source), 2)
    except np.linalg.LinAlgError as e:
        tree = cKDTree(transformed_source_rigid)
        _, indices = tree.query(target, k=1)
        indices = indices.flatten()
        reordered_source = source[indices]  # 변형되지 않은 원본 데이터를 재정렬
        return reordered_source, indices
    
    try:    
        deform_reg = DeformableRegistration(X=target_centerd, Y=transformed_source_affine, alpha=0.00001, beta=1000)
        # deform_reg = DeformableRegistration(**{'X': target_centerd, 'Y': transformed_source_affine})
        transformed_source_Deform = deform_reg.register()[0].reshape(len(source), 2)
    except np.linalg.LinAlgError as e:
        tree = cKDTree(transformed_source_affine)
        _, indices = tree.query(target, k=1)
        indices = indices.flatten()
        reordered_source = source[indices]  # 변형되지 않은 원본 데이터를 재정렬
        return reordered_source, indices

    tree = cKDTree(transformed_source_Deform)
    _, indices = tree.query(target, k=1)
    indices = indices.flatten()
    reordered_source = source[indices]  # 변형되지 않은 원본 데이터를 재정렬
    return reordered_source, indices

def procrustes_alignment(X, Y):
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    Xc = X - X_mean
    Yc = Y- Y_mean

    U, _, Vt = np.linalg.svd(Yc.T @ Xc)
    R = U @ Vt

    Y_aligned = (Yc @ R) + X_mean
    return Y_aligned

# 평균 데이터 계산 및 업데이트
def calculate_mean_shape(points_list):
    return np.mean(points_list, axis=0)


# GPA 수행 함수 정의
def perform_gpa(points_list, tolerance=1e-6):
    n_shapes = len(points_list)
    mean_shape = np.mean(points_list, axis=0)
    prev_error = np.inf
    iteration = 0
    max_iterations = 100
    aligned_shapes = []

    while iteration < max_iterations:
        aligned_shapes.clear()
        for shape in points_list:
            mu1 = np.mean(mean_shape, axis=0)
            mu2 = np.mean(shape, axis=0)
            shape1_cent = mean_shape - mu1
            shape2_cent = shape - mu2
            U, _, Vt = np.linalg.svd(np.dot(shape1_cent.T, shape2_cent))
            R = np.dot(Vt.T, U.T)
            aligned_shape = np.dot(shape - mu2, R) + mu1
            aligned_shapes.append(aligned_shape)

        new_mean_shape = np.mean(aligned_shapes, axis=0)
        error = np.linalg.norm(new_mean_shape - mean_shape)

        if error < tolerance or error >= prev_error:
            break

        mean_shape = new_mean_shape
        prev_error = error
        iteration += 1

    return mean_shape, aligned_shapes


# .particles 파일로 포인트 데이터 저장 함수
def save_points_as_particles(resampled_points_list, reordered_indices_list, output_folder, original_filenames):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    saved_files = []
    for resampled_points, indices, original_filename in zip(resampled_points_list, reordered_indices_list,
                                                            original_filenames):
        base_filename = os.path.splitext(original_filename)[0]
        output_path = os.path.join(output_folder, f"{base_filename}.particles")
        saved_files.append(output_path)  # 저장된 파일 경로를 리스트에 추가
        with open(output_path, 'w') as file:
            reordered_points = resampled_points[indices]
            for point in reordered_points:
                if len(point) == 3:
                    file.write(f"{point[0]} {point[1]} {point[2]}\n")
                elif len(point) == 2:
                    file.write(f"{point[0]} {point[1]} 0.0\n")  # z 값이 없는 경우 0.0으로 채움
            file.write("\n")
        print(f"Particles file saved at: {output_path}")

    return saved_files  # 저장된 파일 목록을 반환


# 저장된 .particles 파일 불러오기
def load_particles_file(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():  # 빈 줄이 아닌 경우
                points.append([float(val) for val in line.strip().split()])
    return np.array(points)


# PCA 모델 생성 함수 정의
def create_statistical_shape_model(points_list):
    data_matrix = np.array([shape.flatten() for shape in points_list])
    pca = PCA(n_components=min(len(points_list), data_matrix.shape[1]))
    pca.fit(data_matrix)
    return pca


# PCA 모드 시각화 함수 정의
def plot_pca_modes(pca_model, mean_shape):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].plot(mean_shape[:, 0], mean_shape[:, 1], 'o', label='Mean Shape')
    axes[0].set_title("Mean Shape")
    axes[0].axis('equal')
    for i, ax in enumerate(axes[1:]):
        mode_variation = pca_model.components_[i].reshape(mean_shape.shape)
        ax.plot(mean_shape[:, 0] + 100 * mode_variation[:, 0], mean_shape[:, 1] + 100 * mode_variation[:, 1], 'o',
                label='Positive')
        ax.plot(mean_shape[:, 0] - 100 * mode_variation[:, 0], mean_shape[:, 1] - 100 * mode_variation[:, 1], 'o',
                label='Negative')
        ax.set_title(f"PCA Mode {i + 1}")
        ax.legend()
        ax.axis('equal')
    plt.show()

    # PCA 결과를 콘솔에 출력
    explained_variance = pca_model.explained_variance_ratio_
    print("Explained Variance Ratios:", explained_variance)



labeled_points = load_vtp_files_by_label(folder_path)


label_avg_length = {}
for label, points_info_list in labeled_points.items():
    original_points_list = [points for points, _ in points_info_list]
    avg_length = compute_average_length(original_points_list)
    label_avg_length[label] = avg_length

base_label = min(label_avg_length, key=label_avg_length.get)
base_length = label_avg_length[base_label]
base_num_points = 50
label_optimal_num_points = {}


label_optimal_num_points = {}
label_info_path = os.path.join(output_folder, "label_info.txt")
with open(label_info_path, 'w') as f:
    for label, avg_length in label_avg_length.items():
        ratio = avg_length / base_length
        num_points = int(round(base_num_points * ratio))
        num_points = max(30, min(num_points, 150))
        label_optimal_num_points[label] = num_points
        f.write(f"Label {label}: Avg length={avg_length:.2f}, Num points={num_points}\n")


# 라벨별로 프로세스 수행
log_file_path = os.path.join(output_folder, "processing_log.txt")
with open(log_file_path, 'w') as log_file:
    labeled_points = load_vtp_files_by_label(folder_path)

    for label, points_info_list in labeled_points.items():
        print(f"Processing label: {label}")
        log_file.write(f"Processing label: {label}\n")
        # 리샘플링 수행
        optimal_num_points = label_optimal_num_points[label]
        resampled_points_list = [resample_points(points, optimal_num_points) for points, _ in points_info_list]
        original_filenames = [filename for _, filename in points_info_list]

        # 첫 번째 데이터를 기준으로 나머지 데이터를 정렬 및 대응 관계 맞추기 (기준: 평균 데이터)
        
        mean_shape = resampled_points_list[2]  # 첫 번째 데이터로 초기 기준 설정
        
        tolerance = 1e-2  # 수렴 기준
        max_iterations = 100 # 최대 반복 횟수
        early_stop_patience = 5
        iteration = 0
        no_improvement_counter = 0
        prev_mean_shape = None
        prev_mean_diff = None
        warning_counter = 0

        while iteration < max_iterations:
            # with ThreadPoolExecutor(max_workers=6) as executor:
            #     results = list(executor.map(
            #         lambda points: (lambda reordered, idxs: (reordered, idxs, procrustes_alignment(mean_shape, reordered)))(
            #         *align_points_and_find_indices(mean_shape, points)
            #         ),
            #         resampled_points_list
            #     ))

            transformed_and_reordered_points_list = []
            reordered_indices_list = []
            aligned_point_list = []
        
            # transformed_and_reordered_points_list, reordered_indices_list, aligned_point_list = zip(*results)



            # 모든 데이터에 대해 기준(mean_shape)에 맞춰 대응 관계 정렬 수행
            for points in resampled_points_list:
                reordered_source, indices = align_points_and_find_indices(mean_shape, points)
                transformed_and_reordered_points_list.append(reordered_source)  # 원본 데이터를 변형하지 않고 대응 순서만 정렬
                reordered_indices_list.append(indices)
                
                aligned_source = procrustes_alignment(mean_shape, reordered_source)
                aligned_point_list.append(aligned_source)

            # 새로운 평균 형태 계산
            
            new_mean_shape = calculate_mean_shape(aligned_point_list)

            # 수렴 확인
            if prev_mean_shape is not None:
                mean_diff = np.linalg.norm(new_mean_shape - prev_mean_shape)
                print(f"Iteration {iteration}, mean difference: {mean_diff:.8f}")
                log_file.write(f"Iteration {iteration}, mean difference: {mean_diff:.8f}\n")
                if prev_mean_diff is not None:
                    if mean_diff > prev_mean_diff:
                        print(f"Warning: mean_diff increased at iteration {iteration}: {mean_diff: .8f}> {prev_mean_diff: .8f}")
                        mean_shape = prev_mean_shape
                        print("Too many consecutive mean_diff increase. Stopping.")
                        break
                    else:
                        prev_mean_diff = mean_diff 
                        
                else:
                    prev_mean_diff = mean_diff 
                
                if mean_diff < tolerance:
                    no_improvement_counter +=1
                    if no_improvement_counter >= early_stop_patience:
                        print(f"Converged after {iteration} iterations")
                        break
                else:
                    no_improvement_counter = 0
                
                

            # 평균 데이터를 업데이트
            mean_shape = new_mean_shape
            prev_mean_shape = new_mean_shape
            iteration += 1

        # 리샘플링된 포인트 데이터를 .particles 파일로 내보내기
        if iteration == max_iterations:
            print(f"Reached max iterations ({max_iterations}) without full convergence.")
        saved_files = save_points_as_particles(resampled_points_list, reordered_indices_list, output_folder,
                                            original_filenames)

        # # 저장된 .particles 파일 불러오기
        # loaded_points_list = [load_particles_file(file_path) for file_path in saved_files]

        # # GPA 수행 및 평균 형태와 정렬된 형태들 반환
        # mean_shape, aligned_shapes = perform_gpa(loaded_points_list)

        # # PCA 모델 생성
        # pca_model = create_statistical_shape_model(aligned_shapes)

        # # PCA 모드 시각화
        # plot_pca_modes(pca_model, mean_shape)

        # # 원본 데이터의 분포를 확인 (PCA 결과와 비교하기 위해)
        # for i, points in enumerate(aligned_shapes):
        #     plt.plot(points[:, 0], points[:, 1], 'o-', label=f"Data {i + 1}")
        # plt.title(f"Aligned Shapes for Label: {label}")
        # plt.axis('equal')
        # plt.legend()
        # plt.show()
