#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic VTP test files for foot disease analysis
Creates realistic foot landmark data for testing the analysis pipeline
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import random
import json

# Anatomical parts with their expected landmark counts (from CLAUDE.md)
AP_LANDMARK_COUNTS = {
    "AP1": 50,   # distal 1st phalange
    "AP2": 70,   # proximal 1st phalange
    "AP3": 112,  # 2nd metatarsal
    "AP4": 123,  # 1st metatarsal
    "AP5": 114,  # 5th metatarsal
    "AP6": 78,   # navicular
    "AP7": 109,  # talus
    "AP8": 126   # calcaneus
}

# Disease and category mappings from the reference code
DIAGINOSIS_MAPPING = {
    '정상': 'normal',
    '하지부동': 'leg_length_discrepancy',
    'kneevarus': 'knee_varus',
    '퇴행성관절염' : 'degenerative_arthritis',
    'kneevalgus': 'knee_valgus',
    'heelvalgus': 'heel_valgus',
    '요족' : 'high_arch',
    '부주상골증후군': 'accessory_navicular_syndrome',
    '평발': 'flat_foot',
    'heelvarus': 'heel_varus',
    '족부관절염': 'foot_arthritis',
    '무지외반증': 'hallux_valgus',
    '발뒤꿈치통증증후군': 'heel_pain_syndrome',
    '족근골유합': 'tarsal_coalition',
    '외상후관절염' : 'post_traumatic_arthritis',
    '내족지보행': 'in_toeing',
    '외족지보행': 'out_toeing',
    '류마티스 관절염': 'rhematoid_arthritis'
}

CATEGORY_MAPPING = {
    '정상': 'normal',
    '족부질환': 'foot_disease',
    '족관절관절염': 'ankle_arthritis',
    '보행장애': 'gait_disorder'
}

CATEGORY_TO_DISEASE_MAPPING = {
    'normal': ['normal'],
    'foot_disease': ['foot_arthritis','hallux_valgus', 'tarsal_coalition', 'heel_pain_syndrome','accessory_navicular_syndrome'],
    'ankle_arthritis': ['degenerative_arthritis', 'rhematoid_arthritis', 'post_traumatic_arthritis'],
    'gait_disorder': ['flat_foot','high_arch', 'knee_valgus', 'heel_valgus','knee_varus','heel_varus','in_toeing','out_toeing', 'leg_length_discrepancy']
}

# All available diseases and categories
ALL_DISEASES = list(DIAGINOSIS_MAPPING.values())
ALL_CATEGORIES = list(CATEGORY_MAPPING.values())

# Demographics for stratification testing
SEXES = ['M', 'F']
AGE_GROUPS = ['20s', '30s', '40s', '50s', '60s+']

def generate_foot_landmarks(ap_name, n_points, direction="R", base_shape="foot", noise_level=0.1):
    """Generate realistic 3D foot landmark coordinates for a specific anatomical part

    Args:
        ap_name: Anatomical part name (AP1-AP8)
        n_points: Number of landmark points to generate
        direction: 'L' for left foot, 'R' for right foot
        base_shape: Shape type for generation
        noise_level: Amount of random variation to add
    """

    # Base foot dimensions (approximate, in mm)
    foot_length = 250
    foot_width = 100
    foot_height = 50

    # Define rough anatomical positions for each AP
    ap_positions = {
        "AP1": (0.85, 0.15, 0),    # distal toe
        "AP2": (0.75, 0.15, 0),    # proximal toe
        "AP3": (0.65, 0.3, 0),    # 2nd metatarsal
        "AP4": (0.65, 0.15, 0),   # 1st metatarsal
        "AP5": (0.65, 0.85, 0),   # 5th metatarsal
        "AP6": (0.45, 0.25, 0),    # navicular (midfoot)
        "AP7": (0.35, 0.4, 0),     # talus (ankle)
        "AP8": (0.15, 0.5, 0)      # calcaneus (heel)
    }

    if ap_name not in ap_positions:
        # Default position if AP not found
        center_x, center_y, center_z = 0.5, 0.5, 0
    else:
        center_x, center_y, center_z = ap_positions[ap_name]

    # Scale by foot dimensions
    center = np.array([
        center_x * foot_length,
        center_y * foot_width,
        center_z * foot_height
    ])

    # Generate points around the anatomical center
    if base_shape == "foot":
        # Create elliptical distribution for foot-like shape
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)

        # Vary radii for each AP
        ap_radii = {
            "AP1": (8, 6, 3),    # small, toe
            "AP2": (10, 8, 4),   # slightly larger
            "AP3": (12, 8, 5),   # metatarsal
            "AP4": (15, 10, 6),  # largest metatarsal
            "AP5": (12, 8, 5),   # 5th metatarsal
            "AP6": (18, 12, 8),  # navicular (broader)
            "AP7": (20, 15, 12), # talus (ankle, largest)
            "AP8": (25, 20, 10)  # calcaneus (heel, long)
        }

        radii = ap_radii.get(ap_name, (15, 10, 6))

        # Create elliptical point cloud
        points = np.zeros((n_points, 3))
        for i, angle in enumerate(angles):
            # Add some randomness to radius
            r_var = 1 + noise_level * (np.random.random() - 0.5)

            # Elliptical coordinates
            x = radii[0] * r_var * np.cos(angle)
            y = radii[1] * r_var * np.sin(angle)
            z = radii[2] * r_var * (np.random.random() - 0.5)

            points[i] = center + np.array([x, y, z])

        # Add additional random variation
        points += noise_level * np.random.randn(n_points, 3) * 2

    else:
        # Simple random distribution around center
        scale = 20  # mm
        points = center + scale * np.random.randn(n_points, 3)

    # Mirror left foot along the Y-axis (medial-lateral axis)
    # This ensures left and right feet are anatomically correct mirror images
    if direction == "L":
        # Mirror by negating the X coordinate (width/medial-lateral axis)
        points[:, 0] = -points[:, 0]

    return points

def create_vtp_content(points):
    """Create VTP file content with landmark points"""
    n_points = len(points)

    # VTP header
    content = ['<?xml version="1.0"?>']
    content.append('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">')
    content.append('  <PolyData>')
    content.append(f'    <Piece NumberOfPoints="{n_points}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="0">')

    # Points section
    content.append('      <Points>')
    content.append('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">')

    for point in points:
        content.append(f'          {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}')

    content.append('        </DataArray>')
    content.append('      </Points>')

    # Point data (landmark indices)
    content.append('      <PointData>')
    content.append('        <DataArray type="Int32" Name="landmark_index" format="ascii">')
    content.append('          ' + ' '.join(str(i) for i in range(n_points)))
    content.append('        </DataArray>')
    content.append('      </PointData>')

    # Close tags
    content.append('    </Piece>')
    content.append('  </PolyData>')
    content.append('</VTKFile>')

    return '\n'.join(content)

def generate_test_subjects(n_subjects=50):
    """Generate test subject information with multiple categories and diseases like the reference format"""
    subjects = []

    for i in range(n_subjects):
        subject_id = f"CID{10000000 + i:08d}"

        # Random demographics with specific age
        sex = random.choice(SEXES)
        age = round(np.random.uniform(20, 80), 2)  # Specific age like 45.11

        # Derive age group from specific age for compatibility
        if age < 30:
            age_group = '20s'
        elif age < 40:
            age_group = '30s'
        elif age < 50:
            age_group = '40s'
        elif age < 60:
            age_group = '50s'
        else:
            age_group = '60s+'

        # Generate initial record with all categories and diseases set to "other"
        subject = {
            'subject_id': subject_id,
            'sex': sex,
            'age': age,  # Specific age like in original JSON
            'age_group': age_group,  # For compatibility with analysis pipeline
            'height': abs(np.random.normal(170, 10)),  # cm (ensure positive)
            'weight': abs(np.random.normal(70, 15)),   # kg (ensure positive)
            'ST_number': 1,
            'group_direction': 'L'  # Default, will be handled per direction later
        }

        # Initialize all category and disease fields to "other"
        for category in ALL_CATEGORIES:
            subject[f'group_category_{category}'] = 'other_category'
        for disease in ALL_DISEASES:
            subject[f'group_disease_{disease}'] = 'other_disease'

        # Determine if subject is normal (30% chance)
        if random.random() < 0.3:
            # Normal subject - set all categories and diseases to 'normal'
            for category in ALL_CATEGORIES:
                subject[f'group_category_{category}'] = 'normal'
            for disease in ALL_DISEASES:
                subject[f'group_disease_{disease}'] = 'normal'
            subject['primary_category'] = 'normal'
            subject['primary_disease'] = 'normal'
        else:
            # Disease subject - can have multiple categories/diseases

            # Choose 1-3 active categories (realistic for medical conditions)
            n_categories = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            non_normal_categories = [cat for cat in ALL_CATEGORIES if cat != 'normal']
            active_categories = np.random.choice(
                non_normal_categories,
                size=min(n_categories, len(non_normal_categories)),
                replace=False
            )

            primary_category = active_categories[0]
            subject['primary_category'] = primary_category

            # Set active categories
            for category in active_categories:
                subject[f'group_category_{category}'] = category

                # For each active category, select diseases
                possible_diseases = CATEGORY_TO_DISEASE_MAPPING[category]

                # Choose 1-2 diseases per category
                n_diseases = np.random.choice([1, 2], p=[0.7, 0.3])
                n_diseases = min(n_diseases, len(possible_diseases))

                selected_diseases = np.random.choice(
                    possible_diseases,
                    size=n_diseases,
                    replace=False
                )

                for disease in selected_diseases:
                    subject[f'group_disease_{disease}'] = disease

            # Set primary disease from the primary category
            primary_category_diseases = [
                disease for disease in CATEGORY_TO_DISEASE_MAPPING[primary_category]
                if subject[f'group_disease_{disease}'] == disease
            ]
            subject['primary_disease'] = primary_category_diseases[0] if primary_category_diseases else 'other_disease'

        subjects.append(subject)

    return subjects

def create_vtp_filename(subject_id, direction, ap, study_num=1, series_num=1):
    """Create VTP filename following the observed pattern"""
    # Pattern: CID{subject}_CR_ST{study}_SE{series}_IM00001_X1_FSAP_{direction}_{ap}_LabelPolyLine.vtp
    return f"{subject_id}_CR_ST{study_num:03d}_SE{series_num:03d}_IM00001_X1_FSAP_{direction}_{ap}_LabelPolyLine.vtp"

def create_clinical_json(subject):
    """Create clinical info JSON matching the exact reference format"""

    # Map back to Korean labels for JSON (opposite of DIAGINOSIS_MAPPING)
    korean_diagnosis_mapping = {v: k for k, v in DIAGINOSIS_MAPPING.items()}
    korean_category_mapping = {v: k for k, v in CATEGORY_MAPPING.items()}

    # Build diagnosis names list (Korean names)
    # If subject is normal, diagnosis names should be empty
    diagnosis_names = []
    diagnosis_codes = []

    if subject.get('primary_category') != 'normal':
        for disease in ALL_DISEASES:
            if subject.get(f'group_disease_{disease}') == disease and disease != 'normal':
                korean_name = korean_diagnosis_mapping.get(disease, disease)
                diagnosis_names.append(korean_name)
                # Generate fake diagnosis code
                diagnosis_codes.append(f"DC{len(diagnosis_codes):03d}")

    # If no diseases found but not normal, add at least one
    if not diagnosis_names and subject.get('primary_category') != 'normal':
        # Add a default disease based on primary category
        primary_cat = subject.get('primary_category', 'foot_disease')
        if primary_cat in CATEGORY_TO_DISEASE_MAPPING:
            default_disease = CATEGORY_TO_DISEASE_MAPPING[primary_cat][0]
            korean_name = korean_diagnosis_mapping.get(default_disease, default_disease)
            diagnosis_names.append(korean_name)
            diagnosis_codes.append("DC000")

    # Determine primary category
    primary_category = subject.get('primary_category', 'normal')
    korean_primary = korean_category_mapping.get(primary_category, primary_category)

    # Build category info (all categories with true/false)
    category_info = {}
    category_rl = {}

    for category in ALL_CATEGORIES:
        korean_cat = korean_category_mapping.get(category, category)
        is_active = subject.get(f'group_category_{category}') == category
        category_info[korean_cat] = is_active

        if is_active:
            # For test data, randomly assign L/R/A (both sides affected)
            category_rl[korean_cat] = random.choice(['L', 'R', 'A'])
        else:
            category_rl[korean_cat] = ""

    # Generate fake operation name
    operations = [
        "Open reduction and internal fixation, pelvis (left)",
        "Arthroscopic surgery, ankle joint",
        "Hallux valgus correction surgery",
        "Plantar fascia release",
        "Tarsal coalition resection",
        "Conservative treatment",
        "Physical therapy intervention"
    ]

    clinical_data = {
        "Category": f"Category0_{korean_primary}",
        "Case_ID": subject['subject_id'],
        "Personal_ID": subject['subject_id'],
        "Category_Info": category_info,
        "Diagnosis_Codes": diagnosis_codes,
        "Diagnosis_Names": diagnosis_names,
        "Category_RL": category_rl,
        "Sex": subject['sex'],
        "Age": str(round(subject['age'], 2)),  # String format like "45.11"
        "Height": str(round(subject['height'], 1)),  # String format like "175.0"
        "Weight": str(round(subject['weight'], 1)),   # String format like "68.0"
        "Operation_Name": random.choice(operations)
    }

    return clinical_data

def main():
    """Generate complete test dataset"""

    # Create output directories
    base_dir = Path("test_data")
    vtp_dir = base_dir / "vtp_files"
    clinical_dir = base_dir / "clinical_info"
    vtp_dir.mkdir(parents=True, exist_ok=True)
    clinical_dir.mkdir(parents=True, exist_ok=True)

    print("Generating test VTP files for foot disease analysis...")

    # Generate test subjects
    subjects = generate_test_subjects(50)

    # Generate clinical JSON files for each subject
    for subject in subjects:
        clinical_data = create_clinical_json(subject)
        json_filename = f"{subject['subject_id']}_ClinicalInfo.json"
        json_filepath = clinical_dir / json_filename

        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(clinical_data, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(subjects)} clinical JSON files")

    # Save subject metadata
    subject_df = pd.DataFrame(subjects)
    subject_df.to_csv(base_dir / "test_subjects.csv", index=False)
    print(f"Generated {len(subjects)} test subjects")

    # Generate VTP files
    total_files = 0
    landmarks_data = []

    for subject in subjects:
        subject_id = subject['subject_id']

        # Create both left and right foot data
        for direction in ['L', 'R']:
            for ap_name, n_landmarks in AP_LANDMARK_COUNTS.items():

                # Generate landmarks for this AP with proper L/R mirroring
                points = generate_foot_landmarks(ap_name, n_landmarks, direction)

                # Create VTP content
                vtp_content = create_vtp_content(points)

                # Save VTP file
                filename = create_vtp_filename(subject_id, direction, ap_name)
                filepath = vtp_dir / filename

                with open(filepath, 'w') as f:
                    f.write(vtp_content)

                total_files += 1

                # Store landmark data for CSV generation
                # For landmark data, we need to pick one disease/category for simplicity
                # If normal, use 'normal', otherwise use first active disease/category
                if subject['primary_category'] == 'normal':
                    label_disease = 'normal'
                    label_category = 'normal'
                else:
                    # Find first active disease
                    active_diseases = [disease for disease in ALL_DISEASES
                                     if subject.get(f'group_disease_{disease}') == disease and disease != 'normal']
                    label_disease = active_diseases[0] if active_diseases else 'other_disease'
                    label_category = subject['primary_category']

                for i, point in enumerate(points):
                    landmarks_data.append({
                        'subject_id': subject_id,
                        'ap': ap_name,
                        'landmark_index': i,
                        'x': point[0],
                        'y': point[1],
                        'z': point[2],
                        'direction': direction,
                        'label_disease': label_disease,
                        'label_category': label_category,
                        'sex': subject['sex'],
                        'age': subject['age'],
                        'age_group': subject['age_group'],
                        'height': subject['height'],
                        'weight': subject['weight'],
                        'filename': filename
                    })

    print(f"Generated {total_files} VTP files")

    # Create landmarks CSV for testing the analysis pipeline
    landmarks_df = pd.DataFrame(landmarks_data)
    landmarks_df.to_csv(base_dir / "test_landmarks.csv", index=False)
    landmarks_df.to_feather(base_dir / "test_landmarks.feather")
    print(f"Generated landmark data with {len(landmarks_df)} points")

    # Create a smaller aligned dataset for testing (simulating the output of preprocessing)
    # This removes the z-coordinate and adds some processing metadata
    aligned_data = landmarks_df.copy()
    aligned_data = aligned_data.drop('z', axis=1)  # 2D analysis
    aligned_data['group_direction'] = aligned_data['direction']

    # Add some synthetic processing flags
    aligned_data['ST_number'] = 1
    aligned_data['processed'] = True

    aligned_data.to_csv(base_dir / "test_aligned_data.csv", index=False)
    aligned_data.to_feather(base_dir / "test_aligned_data.feather")

    print(f"Test data generation complete!")
    print(f"Files created in: {base_dir.absolute()}")
    print(f"- {total_files} VTP files in vtp_files/")
    print(f"- test_subjects.csv: Subject metadata")
    print(f"- test_landmarks.csv/.feather: Raw landmark data")
    print(f"- test_aligned_data.csv/.feather: Preprocessed data for analysis")

    # Print sample statistics
    print("\nDataset statistics:")
    print(f"- Subjects: {len(subjects)}")
    print(f"- Primary category distribution: {subject_df['primary_category'].value_counts().to_dict()}")
    print(f"- Sex distribution: {subject_df['sex'].value_counts().to_dict()}")
    print(f"- Age distribution: {subject_df['age_group'].value_counts().to_dict()}")

    # Count actual diseases per subject
    disease_counts = {}
    for subject in subjects:
        active_diseases = [disease for disease in ALL_DISEASES
                         if subject.get(f'group_disease_{disease}') == disease]
        for disease in active_diseases:
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
    print(f"- Disease distribution: {disease_counts}")

if __name__ == "__main__":
    main()