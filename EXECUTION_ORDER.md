# Pipeline Execution Order for Server Deployment

> **For offline server environment without internet access**

## 📋 Prerequisites Checklist

### Before Starting
- [ ] Python environment created (Python 3.8+)
- [ ] All dependencies installed (see `requirements.txt`)
- [ ] Raw VTP files available
- [ ] Clinical metadata CSV ready
- [ ] Sufficient disk space (estimate: 3x input data size)
- [ ] Sufficient RAM (minimum 8GB, recommended 16GB+)

---

## 🚀 Complete Pipeline Execution Order

### **STEP 1: File Organization** (Optional - for data cleanup)

```bash
# Filter patients by age (20+ years)
python preprocessing/file_management/move_above_20s.py \
  --input_dir /path/to/raw_data \
  --output_dir /path/to/filtered_data \
  --min_age 20
```

**Input:** Raw VTP files with all age groups
**Output:** Filtered VTP files (age 20+)
**Validation:** Check output directory has expected patient count

---

### **STEP 2: VTP Alignment**

```bash
# Align VTP files for consistent orientation
python preprocessing/vtp_alignment/align_vtp_unified_pipeline.py \
  --input_dir /path/to/filtered_data \
  --output_dir /path/to/aligned_vtp \
  --reference_vtp /path/to/reference.vtp
```

**Input:** Filtered VTP files
**Output:** Aligned VTP files (consistent orientation)
**Validation:** Visually check aligned files using visualization tool

---

### **STEP 3: Particle Conversion** (GPA + CPD Registration)

```bash
# Convert aligned VTP to particle format
python preprocessing/particle_conversion/2025_04_26_pycpd_vtp_to_particle_using_gpa_mean_shape.py \
  --input_dir /path/to/aligned_vtp \
  --output_dir /path/to/particles \
  --n_particles 1023
```

**Input:** Aligned VTP files
**Output:** .particles files (landmark correspondences)
**Validation:** Check each VTP has corresponding .particles file

---

### **STEP 4: Create Mapping Table**

```bash
# Generate comprehensive metadata CSV
python preprocessing/make_csv_with_landmarks.py \
  --vtp_dir /path/to/aligned_vtp \
  --particles_dir /path/to/particles \
  --clinical_metadata /path/to/clinical_data.csv \
  --output_csv /path/to/output/mapping_table.csv
```

**Input:** Aligned VTP + Particles + Clinical metadata
**Output:** Complete mapping CSV with disease labels
**Validation:**
- Check CSV has all expected columns
- Verify patient count matches input
- Check for missing values

---

### **STEP 5: Procrustes Alignment** ⭐ **CRITICAL STEP**

```bash
# Global alignment (preserves inter-bone relationships)
python preprocessing/preprocess_test_data.py \
  --metadata_csv /path/to/output/mapping_table.csv \
  --output_dir /path/to/procrustes_results \
  --alignment_type global \
  --output_format feather

# Bonewise alignment (each bone aligned independently)
python preprocessing/preprocess_test_data.py \
  --metadata_csv /path/to/output/mapping_table.csv \
  --output_dir /path/to/procrustes_results \
  --alignment_type bonewise \
  --output_format feather
```

**Input:** Mapping table CSV
**Output:**
- `aligned_global.feather` - For translation/rotation/relative ratio analysis
- `aligned_bonewise.feather` - For aspect ratio/shape analysis

**Validation:**
```bash
# Check output files exist
ls -lh /path/to/procrustes_results/aligned_*.feather

# Quick validation (count landmarks)
python -c "
import pandas as pd
df = pd.read_feather('/path/to/procrustes_results/aligned_global.feather')
print(f'Total landmarks: {len(df)}')
print(f'Unique patients: {df[\"subject_id\"].nunique()}')
print(f'Columns: {df.columns.tolist()}')
"
```

---

### **STEP 6: Statistical Analysis** (Point-wise Testing)

```bash
# Ultra-optimized mode (recommended for production)
python statistical_analysis/stats_multi_level_optimized.py \
  --input /path/to/procrustes_results/aligned_global.feather \
  --output /path/to/results/pointwise_stats \
  --bootstrap 10000 \
  --permutation 10000 \
  --ultra-optimize \
  --n-jobs -1
```

**Alternative modes:**
```bash
# Adaptive mode (intelligent sampling)
python statistical_analysis/stats_multi_level_optimized.py \
  --input /path/to/procrustes_results/aligned_global.feather \
  --output /path/to/results/pointwise_stats_adaptive \
  --bootstrap 10000 \
  --permutation 10000 \
  --use-adaptive \
  --n-jobs -1

# Traditional mode (exact reproducibility)
python statistical_analysis/stats_multi_level_optimized.py \
  --input /path/to/procrustes_results/aligned_global.feather \
  --output /path/to/results/pointwise_stats_traditional \
  --bootstrap 10000 \
  --permutation 10000 \
  --use-traditional \
  --n-jobs 4
```

**Input:** Aligned feather file (global or bonewise)
**Output:** Statistical results CSV files
**Expected Runtime:**
- 50 patients: 20-30 min (ultra-optimized)
- 700 patients: 2-3 hours (ultra-optimized)

**Validation:**
```bash
# Check output structure
ls -R /path/to/results/pointwise_stats/

# Verify results
python -c "
import pandas as pd
import os
results_dir = '/path/to/results/pointwise_stats'
for root, dirs, files in os.walk(results_dir):
    for f in files:
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(root, f))
            print(f'{os.path.join(root, f)}: {len(df)} rows')
"
```

---

### **STEP 7: Geometric Analysis**

#### Option A: Complete Analysis (All Methods)
```bash
python geometric_analysis/run_complete_geometric_analysis.py \
  --global_data /path/to/procrustes_results/aligned_global.feather \
  --bonewise_data /path/to/procrustes_results/aligned_bonewise.feather \
  --output_dir /path/to/results/geometric_analysis \
  --bootstrap 10000 \
  --permutation 10000
```

#### Option B: Individual Analyses

**Translation Analysis** (Global only):
```bash
python geometric_analysis/geometric_analysis_framework.py \
  --data_path /path/to/procrustes_results/aligned_global.feather \
  --alignment_type global \
  --analysis_type translation \
  --output_dir /path/to/results/geometric/translation
```

**Rotation Analysis** (Global only):
```bash
python geometric_analysis/geometric_analysis_framework.py \
  --data_path /path/to/procrustes_results/aligned_global.feather \
  --alignment_type global \
  --analysis_type rotation \
  --output_dir /path/to/results/geometric/rotation
```

**Relative Scale Ratios** (Global only):
```bash
python geometric_analysis/geometric_analysis_framework.py \
  --data_path /path/to/procrustes_results/aligned_global.feather \
  --alignment_type global \
  --analysis_type relative_ratios \
  --output_dir /path/to/results/geometric/relative_ratios
```

**Aspect Ratios** (Both alignments):
```bash
# Global alignment
python geometric_analysis/geometric_analysis_framework.py \
  --data_path /path/to/procrustes_results/aligned_global.feather \
  --alignment_type global \
  --analysis_type aspect_ratios \
  --output_dir /path/to/results/geometric/aspect_ratios_global

# Bonewise alignment
python geometric_analysis/geometric_analysis_framework.py \
  --data_path /path/to/procrustes_results/aligned_bonewise.feather \
  --alignment_type bonewise \
  --analysis_type aspect_ratios \
  --output_dir /path/to/results/geometric/aspect_ratios_bonewise
```

**Input:** Aligned feather files
**Output:** CSV files with statistical results

---

### **STEP 8: Visualization** (Optional)

```bash
# Visualize particle shapes
python utils/visualization/show_shape_particles.py \
  --particles_file /path/to/particles/sample.particles \
  --output_image /path/to/results/viz/shape_particles.png

# Visualize VTK shapes
python utils/visualization/show_shape_vtk.py \
  --vtk_file /path/to/aligned_vtp/sample.vtp \
  --output_image /path/to/results/viz/shape_vtk.png
```

---

## 🔍 Validation Checklist

After each step, verify:

### Step 1-4 Validation:
- [ ] Output directory created
- [ ] Expected number of files present
- [ ] No error messages in logs
- [ ] File sizes reasonable (not 0 bytes)

### Step 5 Validation (Critical):
```bash
# Check aligned data
python << 'EOF'
import pandas as pd
import numpy as np

# Load aligned data
df_global = pd.read_feather('/path/to/procrustes_results/aligned_global.feather')
df_bonewise = pd.read_feather('/path/to/procrustes_results/aligned_bonewise.feather')

print("=== GLOBAL ALIGNMENT ===")
print(f"Total landmarks: {len(df_global)}")
print(f"Unique patients: {df_global['subject_id'].nunique()}")
print(f"Disease labels: {df_global['label_disease'].nunique()}")
print(f"Has multi-labels: {df_global['label_disease'].str.contains(';').sum() > 0}")
print(f"Columns: {df_global.columns.tolist()}")

print("\n=== BONEWISE ALIGNMENT ===")
print(f"Total landmarks: {len(df_bonewise)}")
print(f"Unique patients: {df_bonewise['subject_id'].nunique()}")

# Check for NaN values
print("\n=== MISSING VALUES ===")
print(df_global.isnull().sum())
EOF
```

Expected output:
- Total landmarks: (n_patients × n_bones × n_landmarks_per_bone)
- No critical NaN values in x, y, label_disease columns
- Multi-labels present (contains ';')

### Step 6-7 Validation:
- [ ] Results CSV files created
- [ ] All expected comparisons present
- [ ] p-values, effect sizes, CIs calculated
- [ ] No excessive missing values

---

## ⚠️ Common Issues & Solutions

### Issue 1: Memory Error
**Solution:** Reduce n-jobs or use adaptive mode
```bash
--n-jobs 2  # Instead of -1 (all cores)
--use-adaptive  # Uses less memory
```

### Issue 2: Missing Disease Labels
**Check:** Multi-label extraction working correctly
```bash
python -c "
import pandas as pd
df = pd.read_feather('/path/to/aligned_global.feather')
print(df['label_disease'].value_counts())
print(f'Multi-label patients: {df[\"label_disease\"].str.contains(\";\").sum()}')
"
```

### Issue 3: Slow Performance
**Solutions:**
- Use ultra-optimized mode: `--ultra-optimize`
- Reduce iterations for testing: `--bootstrap 100 --permutation 100`
- Use fewer cores: `--n-jobs 4`

### Issue 4: Path Issues (Windows → Linux)
**Already handled in preprocessing scripts**
If issues persist, check path separators manually

---

## 📊 Expected Output Structure

```
results/
├── procrustes_results/
│   ├── aligned_global.feather          # For Step 6-7
│   └── aligned_bonewise.feather        # For Step 7
├── pointwise_stats/
│   ├── strata_index.csv
│   ├── feature_layout.json
│   ├── ALL/
│   │   ├── by_disease/point_stats.csv
│   │   └── by_category/point_stats.csv
│   ├── M/, F/, 20-39/, 40-59/, 60+/
│   └── M_20-39/, F_20-39/, ... (12 combinations)
└── geometric_analysis/
    ├── translation/
    ├── rotation/
    ├── relative_ratios/
    └── aspect_ratios/
```

---

## 📝 Execution Log Template

Copy and fill as you execute:

```
[ ] Step 1: File organization completed at: _______
    - Input count: _______
    - Output count: _______

[ ] Step 2: VTP alignment completed at: _______
    - Files processed: _______

[ ] Step 3: Particle conversion completed at: _______
    - Particles generated: _______

[ ] Step 4: Mapping table created at: _______
    - CSV rows: _______
    - Patients: _______

[ ] Step 5: Procrustes alignment completed at: _______
    - Global alignment: _______ landmarks
    - Bonewise alignment: _______ landmarks

[ ] Step 6: Statistical analysis completed at: _______
    - Runtime: _______
    - Results files: _______

[ ] Step 7: Geometric analysis completed at: _______
    - Analyses run: _______
    - Results files: _______

[ ] Step 8: Visualization completed at: _______
```

---

## 🎯 Quick Reference

**Minimum required steps:** 5 → 6
**Full pipeline:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
**Testing pipeline:** 4 → 5 → 6 (with reduced iterations)

**Critical files needed for analysis:**
- `aligned_global.feather` - Required for translation, rotation, relative ratios
- `aligned_bonewise.feather` - Required for bonewise aspect ratios
- Either file works for point-wise statistics

**Estimated total runtime (700 patients):**
- Steps 1-5: ~2-4 hours (data preparation)
- Step 6: ~2-3 hours (ultra-optimized)
- Step 7: ~1-2 hours (all geometric analyses)
- **Total: ~5-9 hours**