# AIHUB Foot Disease Analysis

Comprehensive geometric and statistical analysis pipeline for foot disease data from AI Hub.

## Project Overview

This repository contains advanced statistical and geometric analysis tools for analyzing foot disease morphology from 3D landmark data. The pipeline supports multiple analysis strategies optimized for different research questions.

## Repository Structure

```
20251008_footanalysis/
├── preprocessing/          # Data preprocessing and Procrustes alignment
│   └── preprocess_test_data.py
├── statistical_analysis/   # Point-wise statistical testing
│   └── stats_multi_level_optimized.py
├── geometric_analysis/     # Geometric variation analysis
│   ├── geometric_analysis_framework.py
│   ├── test_geometric_analysis.py
│   └── compare_global_vs_bonewise.py
├── utils/                  # Utility functions (to be added)
├── docs/                   # Documentation
│   └── CLAUDE.md
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Key Features

### 1. **Preprocessing Pipeline** (`preprocessing/`)
- **Procrustes alignment**: Global and bonewise strategies
- **Quality control**: Duplicate handling, path normalization
- **Data formats**: Supports .particles and VTP files

### 2. **Statistical Analysis** (`statistical_analysis/`)
- **Multi-level analysis**: Point-wise testing across demographic strata
- **Optimization modes**: Ultra-optimized, Adaptive, Traditional
- **Statistical rigor**:
  - Bootstrap confidence intervals (10K iterations)
  - Permutation tests (adaptive 1K-100K)
  - FDR correction for multiple comparisons
  - Hedges' g and Mahalanobis distance effect sizes

### 3. **Geometric Analysis** (`geometric_analysis/`)
- **Relative scale ratios**: Bone size relationships after normalization
- **Shape deformation**: PCA-based morphological analysis (in development)
- **Statistical framework**: Permutation tests + bootstrap CIs
- **Alignment-aware**: Optimized for global vs bonewise data

## Installation

```bash
# Create Python environment
python -m venv foot_env
source foot_env/bin/activate  # On Linux/Mac
# or
foot_env\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Preprocess Data
```bash
python preprocessing/preprocess_test_data.py \
  --metadata_csv data/metadata.csv \
  --output_dir results/procrustes_results
```

### 2. Statistical Analysis
```bash
python statistical_analysis/stats_multi_level_optimized.py \
  --input results/procrustes_results/aligned_global.feather \
  --output results/stats_output \
  --bootstrap 10000 --permutation 10000 \
  --ultra-optimize --n-jobs -1
```

### 3. Geometric Analysis
```bash
# Quick test
python geometric_analysis/test_geometric_analysis.py

# Compare global vs bonewise
python geometric_analysis/compare_global_vs_bonewise.py
```

## Analysis Types

### Point-wise Analysis
- **When to use**: Detect local morphological differences at specific landmarks
- **Output**: Statistical maps showing significant regions
- **Best for**: Localized disease effects

### Geometric Variation Analysis

#### Relative Scale Ratios
- **When to use**: Understand differential bone growth/atrophy patterns
- **Dataset**: Works with both global and bonewise alignment
- **Output**: Bone-to-bone size ratio changes with statistical significance

#### Translation/Rotation (Global only)
- **When to use**: Study inter-bone positioning changes
- **Dataset**: Requires global alignment
- **Output**: Centroid displacement and orientation patterns

## Data Requirements

### Input Format
- **Landmarks**: .particles files or VTP format
- **Metadata**: CSV with disease labels, demographics
- **Anatomical parts**: 8 foot bones (AP1-AP8)

### Output Format
- **Feather files**: Aligned coordinate data
- **CSV files**: Statistical results with p-values, effect sizes, CIs
- **JSON files**: Feature layouts for visualization

## Performance

| Dataset Size | Traditional | Adaptive | Ultra-Optimized |
|--------------|-------------|----------|-----------------|
| 50 patients  | 45-60 min   | 30-45 min| 20-30 min      |
| 700 patients | 8+ hours    | 3-4 hours| 2-3 hours      |

## Key Dependencies

- numpy >= 1.21.0
- pandas >= 1.4.0
- scipy >= 1.8.0
- statsmodels >= 0.13.0
- numba >= 0.56.0 (for JIT optimization)
- pyarrow >= 10.0.0 (for Feather format)

## Citation

If you use this analysis pipeline in your research, please cite:

```
[Citation to be added]
```

## License

[License to be determined]

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- AI Hub for providing the foot disease dataset
- Statistical methods based on Lee & Young (1996) and other literature
