# Preprocessing Pipeline

This directory contains all preprocessing scripts for the foot disease analysis pipeline.

## Directory Structure

```
preprocessing/
├── data_generation/          # Test data generation
│   └── generate_test_vtp_files.py
├── file_management/          # File organization utilities
│   ├── copy_paste.py
│   ├── file_mover_label.py
│   ├── label_file_to_vtp.py
│   └── move_above_20s.py
├── vtp_alignment/            # VTP file alignment
│   └── align_vtp_unified_pipeline.py
├── particle_conversion/      # Convert VTP to particle format
│   └── 2025_04_26_pycpd_vtp_to_particle_using_gpa_mean_shape.py
├── make_csv_with_landmarks.py    # Create landmark mapping table
├── preprocess_foot_data.py       # Main preprocessing pipeline
└── preprocess_test_data.py       # Test data preprocessing (with fixed Procrustes)
```

## Pipeline Workflow

### 1. Data Generation (Testing Only)
**Script:** `data_generation/generate_test_vtp_files.py`
- Generates synthetic VTP files for testing
- Creates clinical JSON metadata
- Simulates multi-label disease data

### 2. File Management
**Scripts:** `file_management/*.py`
- **copy_paste.py**: Copy files between directories
- **file_mover_label.py**: Organize files by labels
- **label_file_to_vtp.py**: Convert label files to VTP format
- **move_above_20s.py**: Filter patients by age (20+ years)

### 3. VTP Alignment
**Script:** `vtp_alignment/align_vtp_unified_pipeline.py`
- Aligns VTP files for consistent orientation
- Prepares data for landmark extraction

### 4. Particle Conversion
**Script:** `particle_conversion/2025_04_26_pycpd_vtp_to_particle_using_gpa_mean_shape.py`
- Converts aligned VTP files to particle format
- Uses Generalized Procrustes Analysis (GPA) for mean shape
- Applies coherent point drift (CPD) registration

### 5. Data Table Creation
**Script:** `make_csv_with_landmarks.py`
- Creates comprehensive mapping table
- Links VTP files, particles, and clinical metadata
- Generates multi-label disease annotations

### 6. Procrustes Alignment
**Scripts:**
- **preprocess_foot_data.py**: Original preprocessing (legacy)
- **preprocess_test_data.py**: Fixed version with corrected Procrustes bugs

**Two alignment strategies:**
1. **Global alignment**: Preserves inter-bone spatial relationships
2. **Bonewise alignment**: Each bone aligned independently

**Key fixes in test version:**
- Fixed `n_base<2 or n_base<2` → `n_base<2 or n_target<2` (line 285)
- Fixed `mtx *= scale` → `mtx2 *= scale` (undefined variable)
- Added Windows→Linux path normalization
- Fixed age binning with proper NaN handling

## Usage Examples

### Generate Test Data
```bash
python preprocessing/data_generation/generate_test_vtp_files.py
```

### Preprocess Test Data (Recommended)
```bash
python preprocessing/preprocess_test_data.py \
  --metadata_csv test_data/output/vtp_aligned_output_None.csv \
  --output_dir test_data/output/procrustes_results
```

### Full Production Pipeline
```bash
# 1. File organization
python preprocessing/file_management/move_above_20s.py

# 2. VTP alignment
python preprocessing/vtp_alignment/align_vtp_unified_pipeline.py

# 3. Particle conversion
python preprocessing/particle_conversion/2025_04_26_pycpd_vtp_to_particle_using_gpa_mean_shape.py

# 4. Create mapping table
python preprocessing/make_csv_with_landmarks.py

# 5. Procrustes alignment
python preprocessing/preprocess_test_data.py \
  --metadata_csv output/mapping_table.csv \
  --output_dir output/procrustes_results
```

## Output Format

Preprocessed data is saved in Feather format with columns:
```
subject_id, ap, landmark_index, x, y,
label_disease, label_category, sex, age_bin,
sample_id, group_direction
```

### Multi-Label Handling
- **Single disease**: `"hallux valgus"`
- **Multiple diseases**: `"hallux valgus; post traumatic arthritis"`
- **Normal**: `"normal"`
- **Unknown**: `"unknown"` (diseases outside taxonomy)

## Notes

- **Legacy scripts** (from `codes_25_08_19`) preserved for reference
- **Test data generation** creates realistic synthetic data with comorbidities
- **Fixed Procrustes alignment** corrects critical bugs from original implementation
- All scripts handle **multi-label disease data** correctly