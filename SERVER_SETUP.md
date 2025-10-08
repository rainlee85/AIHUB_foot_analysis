# Server Setup and Execution Guide

> **Setup Phase:** Server has internet access
> **Execution Phase:** Data mounted, offline execution

---

## 📦 Phase 1: Initial Setup (With Internet)

### Step 1: Clone Repository
```bash
# Clone the repository
git clone https://github.com/rainlee85/AIHUB_foot_analysis.git
cd AIHUB_foot_analysis/20251008_footanalysis

# Verify all files present
ls -la preprocessing/ statistical_analysis/ geometric_analysis/
```

### Step 2: Create Python Environment
```bash
# Create virtual environment
python3 -m venv foot_analysis_env

# Activate environment
source foot_analysis_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 3: Install All Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt

# Verify critical packages
python -c "
import numpy
import pandas
import scipy
import numba
import pyarrow
print('✓ All dependencies installed successfully')
print(f'NumPy: {numpy.__version__}')
print(f'Pandas: {pandas.__version__}')
print(f'Numba: {numba.__version__}')
"
```

### Step 4: Verify Installation
```bash
# Quick import test
python << 'EOF'
import sys
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api
import numba
import pyarrow
from joblib import Parallel, delayed
import psutil

print("✓ All imports successful!")
print(f"Python: {sys.version}")
print(f"Cores available: {psutil.cpu_count()}")
print(f"RAM available: {psutil.virtual_memory().total / (1024**3):.1f} GB")
EOF
```

### Step 5: Test Pipeline (Optional - Before Data Mount)
```bash
# Generate test data (if needed)
python preprocessing/data_generation/generate_test_vtp_files.py \
  --output_dir test_output \
  --n_patients 10

# Test preprocessing with small dataset
python preprocessing/preprocess_test_data.py \
  --metadata_csv test_output/metadata.csv \
  --output_dir test_results \
  --alignment_type global

# Verify test works
ls -lh test_results/
```

**✓ Setup Complete - Environment Ready for Data**

---

## 💾 Phase 2: Data Mounting

```bash
# Mount your data volume
# (Your specific mount command here)
sudo mount /dev/sdX /mnt/foot_data

# Verify data accessible
ls -lh /mnt/foot_data/
```

**Data Expected Structure:**
```
/mnt/foot_data/
├── vtp_files/              # Raw VTP files
│   ├── patient001_*.vtp
│   ├── patient002_*.vtp
│   └── ...
├── clinical_metadata.csv   # Patient demographics and disease labels
└── particles_files/        # (Optional - if already generated)
```

---

## 🚀 Phase 3: Offline Pipeline Execution

> **From this point, no internet required**

### Quick Start (Recommended Path)

#### If you have VTP files + Clinical CSV:

**1. Create mapping table:**
```bash
source foot_analysis_env/bin/activate

python preprocessing/make_csv_with_landmarks.py \
  --vtp_dir /mnt/foot_data/vtp_files \
  --particles_dir /mnt/foot_data/particles_files \
  --clinical_metadata /mnt/foot_data/clinical_metadata.csv \
  --output_csv /mnt/foot_data/mapping_table.csv
```

**2. Run Procrustes alignment:**
```bash
# Global alignment
python preprocessing/preprocess_test_data.py \
  --metadata_csv /mnt/foot_data/mapping_table.csv \
  --output_dir /mnt/foot_data/results/procrustes \
  --alignment_type global

# Bonewise alignment
python preprocessing/preprocess_test_data.py \
  --metadata_csv /mnt/foot_data/mapping_table.csv \
  --output_dir /mnt/foot_data/results/procrustes \
  --alignment_type bonewise
```

**3. Run statistical analysis:**
```bash
python statistical_analysis/stats_multi_level_optimized.py \
  --input /mnt/foot_data/results/procrustes/aligned_global.feather \
  --output /mnt/foot_data/results/pointwise_stats \
  --bootstrap 10000 \
  --permutation 10000 \
  --ultra-optimize \
  --n-jobs -1
```

**4. Run geometric analysis:**
```bash
python geometric_analysis/run_complete_geometric_analysis.py \
  --global_data /mnt/foot_data/results/procrustes/aligned_global.feather \
  --bonewise_data /mnt/foot_data/results/procrustes/aligned_bonewise.feather \
  --output_dir /mnt/foot_data/results/geometric_analysis \
  --bootstrap 10000 \
  --permutation 10000
```

---

## 📋 Complete Pipeline (All Steps)

For full pipeline from raw VTP files, see [EXECUTION_ORDER.md](EXECUTION_ORDER.md)

### Summary of Steps:
1. **File Organization** (Optional - filter by age)
2. **VTP Alignment** (If VTP not pre-aligned)
3. **Particle Conversion** (VTP → particles)
4. **Create Mapping Table** ⬅️ **START HERE if you have VTP + CSV**
5. **Procrustes Alignment** (Global + Bonewise)
6. **Statistical Analysis** (Point-wise testing)
7. **Geometric Analysis** (Variation decomposition)
8. **Visualization** (Optional)

---

## ⏱️ Estimated Runtime (700 patients)

| Step | Runtime | Disk Space |
|------|---------|------------|
| Mapping table | ~5 min | ~50 MB |
| Procrustes (both) | ~30 min | ~500 MB |
| Statistical analysis | ~2-3 hours | ~1 GB |
| Geometric analysis | ~1-2 hours | ~500 MB |
| **Total** | **~4-6 hours** | **~2 GB** |

---

## 🔍 Monitoring Progress

### Check Progress During Execution:

```bash
# Monitor CPU/Memory usage
htop

# Watch log output
tail -f /mnt/foot_data/results/analysis.log

# Check intermediate files
watch -n 30 'ls -lh /mnt/foot_data/results/procrustes/'

# Count completed analyses
watch -n 60 'find /mnt/foot_data/results/pointwise_stats -name "*.csv" | wc -l'
```

### Validation Commands:

```bash
# After Procrustes alignment
python -c "
import pandas as pd
df = pd.read_feather('/mnt/foot_data/results/procrustes/aligned_global.feather')
print(f'✓ Global alignment: {len(df)} landmarks, {df[\"subject_id\"].nunique()} patients')
print(f'✓ Disease labels: {df[\"label_disease\"].nunique()} unique labels')
print(f'✓ Multi-label patients: {df.groupby(\"subject_id\").first()[\"label_disease\"].str.contains(\";\").sum()}')
"

# After statistical analysis
python -c "
import os
import pandas as pd
results_dir = '/mnt/foot_data/results/pointwise_stats'
csv_files = []
for root, dirs, files in os.walk(results_dir):
    csv_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
print(f'✓ Generated {len(csv_files)} result files')
for f in csv_files[:5]:
    df = pd.read_csv(f)
    print(f'  {os.path.basename(f)}: {len(df)} rows')
"
```

---

## 🛠️ Troubleshooting

### Issue: Out of Memory
```bash
# Reduce parallel jobs
--n-jobs 4  # Instead of -1

# Use adaptive mode (less memory)
--use-adaptive

# Monitor memory
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"
```

### Issue: Slow Performance
```bash
# Test with reduced iterations first
--bootstrap 100 --permutation 100

# Check system resources
nproc  # Number of CPUs
free -h  # Available memory
df -h /mnt/foot_data  # Disk space
```

### Issue: Process Killed
```bash
# Check system logs
dmesg | tail -20

# Likely cause: OOM killer (out of memory)
# Solution: Reduce n-jobs or use adaptive mode
```

### Issue: Missing Dependencies (Should not happen after setup)
```bash
# Verify environment activated
which python  # Should show path to foot_analysis_env

# Reinstall if needed
pip install -r requirements.txt --force-reinstall
```

---

## 📊 Expected Output Structure

```
/mnt/foot_data/results/
├── procrustes/
│   ├── aligned_global.feather          # Main input for analysis
│   ├── aligned_bonewise.feather        # For bonewise methods
│   └── alignment_log.txt
│
├── pointwise_stats/
│   ├── strata_index.csv                # Summary of all analyses
│   ├── feature_layout.json             # Landmark positions
│   ├── ALL/
│   │   ├── by_disease/
│   │   │   └── point_stats.csv         # Disease comparisons
│   │   └── by_category/
│   │       └── point_stats.csv         # Category comparisons
│   ├── M/, F/                          # Sex-stratified
│   ├── 20-39/, 40-59/, 60+/           # Age-stratified
│   └── M_20-39/, F_40-59/, ...        # Combined stratification
│
└── geometric_analysis/
    ├── translation_results.csv
    ├── rotation_results.csv
    ├── relative_ratios_results.csv
    └── aspect_ratios_results.csv
```

---

## ✅ Pre-Execution Checklist

**Before running analysis, verify:**

- [ ] Python environment activated: `source foot_analysis_env/bin/activate`
- [ ] All dependencies installed: `pip list | grep -E "numpy|pandas|scipy|numba"`
- [ ] Data mounted and accessible: `ls /mnt/foot_data`
- [ ] Sufficient disk space: `df -h /mnt/foot_data` (need ~2GB free)
- [ ] Sufficient RAM: `free -h` (need 8GB+, recommend 16GB+)
- [ ] Repository up to date: `git pull` (if internet still available)

---

## 🚀 One-Command Full Pipeline

For experienced users, run entire pipeline:

```bash
#!/bin/bash
# full_pipeline.sh

set -e  # Exit on error

DATA_DIR="/mnt/foot_data"
RESULTS_DIR="${DATA_DIR}/results"

# Activate environment
source foot_analysis_env/bin/activate

# Step 1: Create mapping table
echo "Step 1: Creating mapping table..."
python preprocessing/make_csv_with_landmarks.py \
  --vtp_dir ${DATA_DIR}/vtp_files \
  --particles_dir ${DATA_DIR}/particles_files \
  --clinical_metadata ${DATA_DIR}/clinical_metadata.csv \
  --output_csv ${DATA_DIR}/mapping_table.csv

# Step 2: Procrustes alignment (both types)
echo "Step 2: Running Procrustes alignment..."
python preprocessing/preprocess_test_data.py \
  --metadata_csv ${DATA_DIR}/mapping_table.csv \
  --output_dir ${RESULTS_DIR}/procrustes \
  --alignment_type global

python preprocessing/preprocess_test_data.py \
  --metadata_csv ${DATA_DIR}/mapping_table.csv \
  --output_dir ${RESULTS_DIR}/procrustes \
  --alignment_type bonewise

# Step 3: Statistical analysis
echo "Step 3: Running statistical analysis..."
python statistical_analysis/stats_multi_level_optimized.py \
  --input ${RESULTS_DIR}/procrustes/aligned_global.feather \
  --output ${RESULTS_DIR}/pointwise_stats \
  --bootstrap 10000 \
  --permutation 10000 \
  --ultra-optimize \
  --n-jobs -1

# Step 4: Geometric analysis
echo "Step 4: Running geometric analysis..."
python geometric_analysis/run_complete_geometric_analysis.py \
  --global_data ${RESULTS_DIR}/procrustes/aligned_global.feather \
  --bonewise_data ${RESULTS_DIR}/procrustes/aligned_bonewise.feather \
  --output_dir ${RESULTS_DIR}/geometric_analysis \
  --bootstrap 10000 \
  --permutation 10000

echo "✓ Pipeline complete! Results in: ${RESULTS_DIR}"
```

**Usage:**
```bash
chmod +x full_pipeline.sh
./full_pipeline.sh 2>&1 | tee pipeline_log.txt
```

---

## 📝 Execution Log Template

```
=== FOOT DISEASE ANALYSIS - EXECUTION LOG ===
Date: ___________
Server: ___________
Data mount: /mnt/foot_data
Patient count: ___________

[ ] Setup complete (Phase 1)
    - Environment: foot_analysis_env
    - Dependencies: All installed
    - Test run: ___________

[ ] Data mounted (Phase 2)
    - Mount point: /mnt/foot_data
    - VTP files: ___________
    - Clinical CSV: ___________

[ ] Mapping table created
    - Output: /mnt/foot_data/mapping_table.csv
    - Rows: ___________
    - Time: ___________

[ ] Procrustes alignment
    - Global: _______ landmarks
    - Bonewise: _______ landmarks
    - Time: ___________

[ ] Statistical analysis
    - Input: aligned_global.feather
    - Mode: ultra-optimize
    - Runtime: ___________
    - Files generated: ___________

[ ] Geometric analysis
    - Analyses: translation, rotation, ratios, aspect
    - Runtime: ___________
    - Files generated: ___________

=== COMPLETION ===
Total runtime: ___________
Output location: /mnt/foot_data/results/
Status: Success / Failed
Notes: ___________
```

---

## 🎯 Quick Command Reference

```bash
# Setup (with internet)
git clone https://github.com/rainlee85/AIHUB_foot_analysis.git
cd AIHUB_foot_analysis/20251008_footanalysis
python3 -m venv foot_analysis_env
source foot_analysis_env/bin/activate
pip install -r requirements.txt

# Execution (after data mount)
source foot_analysis_env/bin/activate
./full_pipeline.sh

# Monitoring
htop                    # CPU/Memory
tail -f analysis.log    # Progress
df -h /mnt/foot_data   # Disk space
```
