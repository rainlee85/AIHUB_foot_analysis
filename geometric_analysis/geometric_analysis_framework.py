#!/usr/bin/env python3
"""
Geometric Variation Analysis Framework for Foot Disease Data
Decomposes shape variations into clinically interpretable components:
- Translation (centroid displacement)
- Rotation (orientation changes)
- Scale (size differences)
- Shape deformation (pure morphological changes)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import f as f_dist
from scipy.spatial.distance import pdist, squareform
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
from pathlib import Path
import logging
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
from numba import jit, prange
import psutil
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_effect_size_hedges_g(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Hedges' g effect size for unbalanced samples"""
    n1, n2 = X.shape[0], Y.shape[0]
    if n1 < 2 or n2 < 2:
        return np.nan

    mean1, mean2 = np.mean(X), np.mean(Y)

    # Use manual variance calculation for Numba compatibility
    var1 = np.sum((X - mean1)**2) / (n1 - 1)
    var2 = np.sum((Y - mean2)**2) / (n2 - 1)

    # Pooled standard deviation
    pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_sd == 0:
        return np.nan

    # Hedges' g with correction factor
    cohens_d = (mean1 - mean2) / pooled_sd
    correction = 1 - (3 / (4*(n1+n2-2) - 1))
    return cohens_d * correction

@jit(nopython=True)
def bootstrap_statistic(X_flat: np.ndarray, Y_flat: np.ndarray,
                       n1: int, n2: int, indices1: np.ndarray, indices2: np.ndarray) -> float:
    """Compute bootstrap statistic for geometric measures"""
    X_boot = X_flat[indices1]
    Y_boot = Y_flat[indices2]
    return compute_effect_size_hedges_g(X_boot.reshape(-1, 1), Y_boot.reshape(-1, 1))

class StatisticalFramework:
    """Statistical testing framework for geometric analysis"""

    def __init__(self, n_bootstrap: int = 10000, n_permutation: int = 10000,
                 alpha: float = 0.05, n_jobs: int = -1):
        self.n_bootstrap = n_bootstrap
        self.n_permutation = n_permutation
        self.alpha = alpha
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()

    def permutation_test(self, X: np.ndarray, Y: np.ndarray,
                        statistic_func=None) -> Tuple[float, float, int]:
        """Permutation test for geometric measures"""
        if statistic_func is None:
            statistic_func = lambda x, y: compute_effect_size_hedges_g(x, y)

        # Observed statistic
        observed = statistic_func(X, Y)
        if not np.isfinite(observed):
            return np.nan, np.nan, 0

        # Combined data for permutation
        combined = np.vstack([X, Y])
        n1, n2 = X.shape[0], Y.shape[0]
        n_total = n1 + n2

        # Parallel permutation test
        def single_permutation(seed):
            np.random.seed(seed)
            perm_indices = np.random.permutation(n_total)
            X_perm = combined[perm_indices[:n1]]
            Y_perm = combined[perm_indices[n1:]]
            return statistic_func(X_perm, Y_perm)

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            perm_stats = list(executor.map(single_permutation, range(self.n_permutation)))

        perm_stats = np.array([s for s in perm_stats if np.isfinite(s)])

        if len(perm_stats) < 100:
            return observed, np.nan, len(perm_stats)

        # Two-tailed p-value
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))
        return observed, p_value, len(perm_stats)

    def bootstrap_ci(self, X: np.ndarray, Y: np.ndarray,
                    statistic_func=None) -> Tuple[float, float, int]:
        """Bootstrap confidence intervals for effect sizes"""
        if statistic_func is None:
            statistic_func = lambda x, y: compute_effect_size_hedges_g(x, y)

        n1, n2 = X.shape[0], Y.shape[0]

        def single_bootstrap(seed):
            np.random.seed(seed)
            indices1 = np.random.choice(n1, n1, replace=True)
            indices2 = np.random.choice(n2, n2, replace=True)
            return statistic_func(X[indices1], Y[indices2])

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            bootstrap_stats = list(executor.map(single_bootstrap, range(self.n_bootstrap)))

        finite_stats = np.array([s for s in bootstrap_stats if np.isfinite(s)])

        if len(finite_stats) < 100:
            return np.nan, np.nan, len(finite_stats)

        ci_lower = np.percentile(finite_stats, 2.5)
        ci_upper = np.percentile(finite_stats, 97.5)
        return ci_lower, ci_upper, len(finite_stats)

    def comprehensive_test(self, X: np.ndarray, Y: np.ndarray,
                          statistic_func=None, comparison_name: str = "") -> Dict:
        """Complete statistical analysis with effect size, significance, and CI"""
        if statistic_func is None:
            statistic_func = lambda x, y: compute_effect_size_hedges_g(x, y)

        # Effect size and permutation test
        effect_size, p_value, n_perm_used = self.permutation_test(X, Y, statistic_func)

        # Bootstrap confidence intervals
        ci_lower, ci_upper, n_boot_used = self.bootstrap_ci(X, Y, statistic_func)

        return {
            'comparison': comparison_name,
            'n_base': X.shape[0],
            'n_target': Y.shape[0],
            'effect_size': effect_size,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': p_value < self.alpha if np.isfinite(p_value) else False,
            'n_permutation_used': n_perm_used,
            'n_bootstrap_used': n_boot_used
        }

class GeometricAnalyzer:
    """
    Comprehensive geometric analysis for foot disease data
    Optimized for bonewise-aligned data focusing on:
    - Relative scale ratios between bones
    - Pure shape deformation analysis (PCA-based)
    - Bone-specific morphological patterns
    """

    def __init__(self, data_path: str, alignment_type: str = "bonewise",
                 ap_list: List[str] = None, n_bootstrap: int = 10000, n_permutation: int = 10000):
        """
        Initialize geometric analyzer

        Args:
            data_path: Path to processed foot data (feather format)
            alignment_type: "bonewise" or "global" - determines available analyses
            ap_list: List of anatomical parts (bones) to analyze
            n_bootstrap: Number of bootstrap iterations for CI
            n_permutation: Number of permutation iterations for p-values
        """
        self.data_path = data_path
        self.alignment_type = alignment_type
        self.ap_list = ap_list or [f"AP{i}" for i in range(1, 9)]
        self.data = None
        self.results = {}
        self.stats_framework = StatisticalFramework(n_bootstrap, n_permutation)

        # Validate alignment type and set available analyses
        if alignment_type == "bonewise":
            self.available_analyses = ["aspect_ratios", "shape_deformation", "bone_specific"]
            logger.info("Bonewise alignment: Only aspect ratios available (bones individually aligned)")
        elif alignment_type == "global":
            self.available_analyses = ["translation", "rotation", "aspect_ratios", "relative_ratios", "shape_deformation"]
            logger.info("Global alignment: All geometric analyses available")
        else:
            raise ValueError("alignment_type must be 'bonewise' or 'global'")

    def load_data(self) -> pd.DataFrame:
        """Load and validate foot disease data"""
        logger.info(f"Loading {self.alignment_type} data from {self.data_path}")

        if self.data_path.endswith('.feather'):
            self.data = pd.read_feather(self.data_path)
        else:
            self.data = pd.read_csv(self.data_path)

        # Map column names to standard format
        if 'patient_id' in self.data.columns:
            self.data = self.data.rename(columns={'patient_id': 'subject_id'})

        # Extract disease labels from boolean columns
        self._extract_disease_labels()

        required_cols = ['subject_id', 'ap', 'landmark_index', 'x', 'y']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(f"Loaded {len(self.data)} landmarks from {self.data['subject_id'].nunique()} subjects")
        logger.info(f"Available diseases: {self.data['label_disease'].value_counts().to_dict()}")
        return self.data

    def _extract_disease_labels(self):
        """
        Extract disease and category labels from one-hot encoded columns.
        
        Handles multiple diseases per patient by collecting all active labels.
        Encoding: column value == disease name (patient has it)
                  column value == 'other_disease' (patient doesn't have it)
                  column value == 'normal' (normal for that category)
        """
        disease_cols = [col for col in self.data.columns if col.startswith('group_disease_')]
        category_cols = [col for col in self.data.columns if col.startswith('group_category_')]
        
        if disease_cols:
            # Collect all diseases for each row
            disease_labels = []
            for idx in range(len(self.data)):
                diseases = []
                is_normal = False
                
                for col in disease_cols:
                    val = self.data.iloc[idx][col]
                    if pd.notna(val):
                        if val == 'normal':
                            is_normal = True
                        elif val not in ['other_disease']:
                            # Actual disease name - clean it up
                            disease_name = val.replace('_', ' ').title()
                            diseases.append(disease_name)
                
                # Assign label based on findings
                if is_normal and not diseases:
                    disease_labels.append('Normal')
                elif diseases:
                    # Multiple diseases: join with semicolon
                    disease_labels.append('; '.join(sorted(set(diseases))))
                else:
                    # No normal flag and no diseases found (shouldn't happen)
                    disease_labels.append('Unknown')
            
            self.data['label_disease'] = disease_labels
        
        if category_cols:
            # Collect all categories for each row
            category_labels = []
            for idx in range(len(self.data)):
                categories = []
                is_normal = False
                
                for col in category_cols:
                    val = self.data.iloc[idx][col]
                    if pd.notna(val):
                        if val == 'normal':
                            is_normal = True
                        elif val not in ['other_catergory', 'other_category']:
                            # Actual category name - clean it up
                            category_name = val.replace('_', ' ').title()
                            categories.append(category_name)
                
                # Assign label based on findings
                if is_normal and not categories:
                    category_labels.append('Normal')
                elif categories:
                    # Multiple categories: join with semicolon
                    category_labels.append('; '.join(sorted(set(categories))))
                else:
                    # No normal flag and no categories found (shouldn't happen)
                    category_labels.append('Unknown')
            
            self.data['label_category'] = category_labels


    def extract_bone_coordinates(self, subject_id: str, ap: str) -> np.ndarray:
        """Extract coordinates for a specific bone from a subject"""
        subset = self.data[
            (self.data['subject_id'] == subject_id) &
            (self.data['ap'] == ap)
        ].sort_values('landmark_index')

        if len(subset) == 0:
            return np.array([]).reshape(0, 2)

        return subset[['x', 'y']].values

    def compute_bone_centroids(self) -> pd.DataFrame:
        """
        Compute centroids for each bone in each subject

        Returns:
            DataFrame with columns: subject_id, ap, centroid_x, centroid_y,
                                   label_disease, label_category
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        centroids = []

        for subject_id in self.data['subject_id'].unique():
            subject_data = self.data[self.data['subject_id'] == subject_id]

            # Get metadata for this subject
            metadata = subject_data.iloc[0][['label_disease', 'label_category']]

            for ap in self.ap_list:
                bone_coords = self.extract_bone_coordinates(subject_id, ap)

                if len(bone_coords) > 0:
                    centroid = np.mean(bone_coords, axis=0)
                    centroids.append({
                        'subject_id': subject_id,
                        'ap': ap,
                        'centroid_x': centroid[0],
                        'centroid_y': centroid[1],
                        'label_disease': metadata['label_disease'],
                        'label_category': metadata['label_category']
                    })

        return pd.DataFrame(centroids)

    def analyze_translation_patterns(self) -> Dict:
        """
        Analyze centroid displacement patterns with statistical testing
        Reveals how bone positioning relationships change with disease

        Returns:
            Dict with statistical results for each bone and comparison
        """
        logger.info("Analyzing translation patterns...")

        # Compute centroids for all subjects and bones
        centroids_df = self.compute_bone_centroids()

        if centroids_df.empty:
            raise ValueError("No centroid data computed")

        results = {
            'method': 'Translation Analysis - Centroid Displacement',
            'description': 'Relative bone positioning after Procrustes alignment',
            'comparisons': {}
        }

        # Define comparison groups
        comparisons = [
            ('Normal', 'by_disease'),
            ('Normal', 'by_category')
        ]

        for base_group, comparison_type in comparisons:
            results['comparisons'][comparison_type] = {}

            for ap in self.ap_list:
                bone_data = centroids_df[centroids_df['ap'] == ap].copy()

                if len(bone_data) < 4:  # Need minimum samples
                    continue

                # Get base group (Normal)
                base_mask = bone_data['label_disease'] == base_group
                if not base_mask.any():
                    continue

                base_coords = bone_data[base_mask][['centroid_x', 'centroid_y']].values

                # Test against each disease/category
                target_groups = (bone_data['label_disease'].unique()
                               if comparison_type == 'by_disease'
                               else bone_data['label_category'].unique())

                for target_group in target_groups:
                    if target_group == base_group:
                        continue

                    target_mask = (bone_data['label_disease'] == target_group
                                 if comparison_type == 'by_disease'
                                 else bone_data['label_category'] == target_group)

                    if not target_mask.any():
                        continue

                    target_coords = bone_data[target_mask][['centroid_x', 'centroid_y']].values

                    if len(base_coords) < 2 or len(target_coords) < 2:
                        continue

                    # Statistical analysis for X and Y coordinates separately
                    comparison_name = f"{base_group}_vs_{target_group}"

                    # X-coordinate analysis
                    x_stats = self.stats_framework.comprehensive_test(
                        base_coords[:, 0:1], target_coords[:, 0:1],
                        comparison_name=f"{ap}_centroid_x_{comparison_name}"
                    )

                    # Y-coordinate analysis
                    y_stats = self.stats_framework.comprehensive_test(
                        base_coords[:, 1:2], target_coords[:, 1:2],
                        comparison_name=f"{ap}_centroid_y_{comparison_name}"
                    )

                    # Combined 2D displacement magnitude
                    def displacement_magnitude(coords1, coords2):
                        """Compute mean displacement magnitude between centroids"""
                        mean1 = np.mean(coords1, axis=0)
                        mean2 = np.mean(coords2, axis=0)
                        return np.linalg.norm(mean2 - mean1)

                    magnitude_stats = self.stats_framework.comprehensive_test(
                        base_coords, target_coords,
                        statistic_func=displacement_magnitude,
                        comparison_name=f"{ap}_displacement_{comparison_name}"
                    )

                    # Store results
                    key = f"{ap}_{target_group}"
                    results['comparisons'][comparison_type][key] = {
                        'bone': ap,
                        'comparison': comparison_name,
                        'base_group': base_group,
                        'target_group': target_group,
                        'x_coordinate': x_stats,
                        'y_coordinate': y_stats,
                        'displacement_magnitude': magnitude_stats,
                        'base_centroid': np.mean(base_coords, axis=0).tolist(),
                        'target_centroid': np.mean(target_coords, axis=0).tolist()
                    }

        # Apply FDR correction
        self._apply_fdr_correction(results)

        logger.info(f"Translation analysis complete. Found {len(results['comparisons'])} comparison types.")
        return results

    def _apply_fdr_correction(self, results: Dict) -> None:
        """Apply FDR correction to p-values across all comparisons"""
        all_pvals = []
        pval_locations = []

        # Collect all p-values and their locations
        for comp_type, comparisons in results['comparisons'].items():
            for key, data in comparisons.items():
                for coord_type in ['x_coordinate', 'y_coordinate', 'displacement_magnitude']:
                    if coord_type in data and 'p_value' in data[coord_type]:
                        pval = data[coord_type]['p_value']
                        if np.isfinite(pval):
                            all_pvals.append(pval)
                            pval_locations.append((comp_type, key, coord_type))

        if len(all_pvals) == 0:
            return

        # Apply FDR correction
        rejected, pvals_corrected, _, _ = multipletests(all_pvals, method='fdr_bh')

        # Update results with corrected p-values
        for i, (comp_type, key, coord_type) in enumerate(pval_locations):
            results['comparisons'][comp_type][key][coord_type]['p_value_fdr'] = pvals_corrected[i]
            results['comparisons'][comp_type][key][coord_type]['significant_fdr'] = rejected[i]

    def compute_bone_orientations(self) -> pd.DataFrame:
        """
        Compute principal axis orientation for each bone in each subject
        Uses PCA to find major axis direction

        Returns:
            DataFrame with bone orientations (angles) per subject
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        orientations = []

        for subject_id in self.data['subject_id'].unique():
            subject_data = self.data[self.data['subject_id'] == subject_id]
            metadata = subject_data.iloc[0]

            subject_orientations = {
                'subject_id': subject_id,
                'label_disease': metadata.get('label_disease', 'Unknown'),
                'label_category': metadata.get('label_category', 'Unknown')
            }

            for ap in self.ap_list:
                bone_coords = self.extract_bone_coordinates(subject_id, ap)

                if len(bone_coords) > 2:
                    # Compute PCA to get principal axis
                    centroid = np.mean(bone_coords, axis=0)
                    centered_coords = bone_coords - centroid

                    # Compute covariance matrix
                    cov_matrix = np.cov(centered_coords.T)

                    # Get eigenvectors (principal components)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

                    # Principal axis is eigenvector with largest eigenvalue
                    principal_axis = eigenvectors[:, -1]

                    # Compute orientation angle (in radians)
                    orientation_angle = np.arctan2(principal_axis[1], principal_axis[0])

                    subject_orientations[f'{ap}_orientation'] = orientation_angle
                    subject_orientations[f'{ap}_anisotropy'] = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
                else:
                    subject_orientations[f'{ap}_orientation'] = np.nan
                    subject_orientations[f'{ap}_anisotropy'] = np.nan

            orientations.append(subject_orientations)

        return pd.DataFrame(orientations)

    def analyze_rotation_patterns(self) -> Dict:
        """
        Analyze orientation changes using principal axis analysis
        Reveals how bones rotate relative to anatomical axes

        Returns:
            Dict with statistical results for bone orientations
        """
        if 'rotation' not in self.available_analyses:
            raise ValueError("Rotation analysis only available for global alignment")

        logger.info("Analyzing rotation patterns...")

        # Compute bone orientations
        orientations_df = self.compute_bone_orientations()

        if orientations_df.empty:
            raise ValueError("No orientation data computed")

        results = {
            'method': 'Rotation Analysis - Principal Axis Orientation',
            'description': 'Bone orientation changes relative to anatomical axes',
            'comparisons': {}
        }

        # Define comparison groups
        comparisons = [
            ('Normal', 'by_disease'),
            ('Normal', 'by_category')
        ]

        for base_group, comparison_type in comparisons:
            results['comparisons'][comparison_type] = {}

            for ap in self.ap_list:
                orientation_col = f'{ap}_orientation'
                anisotropy_col = f'{ap}_anisotropy'

                # Get valid data
                valid_data = orientations_df.dropna(subset=[orientation_col]).copy()
                if len(valid_data) < 4:
                    continue

                # Get base group
                base_mask = valid_data['label_disease'] == base_group
                if not base_mask.any():
                    continue

                base_orientations = valid_data[base_mask][orientation_col].values.reshape(-1, 1)

                # Test against each disease/category
                target_groups = (valid_data['label_disease'].unique()
                               if comparison_type == 'by_disease'
                               else valid_data['label_category'].unique())

                for target_group in target_groups:
                    if target_group == base_group:
                        continue

                    target_mask = (valid_data['label_disease'] == target_group
                                 if comparison_type == 'by_disease'
                                 else valid_data['label_category'] == target_group)

                    if not target_mask.any():
                        continue

                    target_orientations = valid_data[target_mask][orientation_col].values.reshape(-1, 1)

                    if len(base_orientations) < 2 or len(target_orientations) < 2:
                        continue

                    # Statistical analysis for orientation
                    comparison_name = f"{base_group}_vs_{target_group}"

                    # Circular statistics for angles (convert to unit vectors)
                    def angular_difference(angles1, angles2):
                        """Compute angular difference statistic"""
                        mean1 = np.arctan2(np.mean(np.sin(angles1)), np.mean(np.cos(angles1)))
                        mean2 = np.arctan2(np.mean(np.sin(angles2)), np.mean(np.cos(angles2)))
                        diff = mean2 - mean1
                        # Normalize to [-pi, pi]
                        diff = np.arctan2(np.sin(diff), np.cos(diff))
                        return np.degrees(diff)  # Return in degrees

                    orientation_stats = self.stats_framework.comprehensive_test(
                        base_orientations, target_orientations,
                        statistic_func=angular_difference,
                        comparison_name=f"{ap}_orientation_{comparison_name}"
                    )

                    # Anisotropy analysis (elongation/shape)
                    base_anisotropy = valid_data[base_mask][anisotropy_col].values.reshape(-1, 1)
                    target_anisotropy = valid_data[target_mask][anisotropy_col].values.reshape(-1, 1)

                    anisotropy_stats = self.stats_framework.comprehensive_test(
                        base_anisotropy, target_anisotropy,
                        comparison_name=f"{ap}_anisotropy_{comparison_name}"
                    )

                    # Store results
                    key = f"{ap}_{target_group}"
                    results['comparisons'][comparison_type][key] = {
                        'bone': ap,
                        'comparison': comparison_name,
                        'base_group': base_group,
                        'target_group': target_group,
                        'orientation_analysis': orientation_stats,
                        'anisotropy_analysis': anisotropy_stats,
                        'base_mean_orientation': np.degrees(np.mean(base_orientations)),
                        'target_mean_orientation': np.degrees(np.mean(target_orientations)),
                        'orientation_difference_deg': orientation_stats.get('effect_size', np.nan),
                        'base_mean_anisotropy': np.mean(base_anisotropy),
                        'target_mean_anisotropy': np.mean(target_anisotropy)
                    }

        # Apply FDR correction
        self._apply_fdr_correction_rotation(results)

        logger.info(f"Rotation analysis complete. Analyzed {len(self.ap_list)} bones.")
        return results

    def _apply_fdr_correction_rotation(self, results: Dict) -> None:
        """Apply FDR correction to p-values for rotation analysis"""
        all_pvals = []
        pval_locations = []

        # Collect all p-values
        for comp_type, comparisons in results['comparisons'].items():
            for key, data in comparisons.items():
                for analysis_type in ['orientation_analysis', 'anisotropy_analysis']:
                    if analysis_type in data and 'p_value' in data[analysis_type]:
                        pval = data[analysis_type]['p_value']
                        if np.isfinite(pval):
                            all_pvals.append(pval)
                            pval_locations.append((comp_type, key, analysis_type))

        if len(all_pvals) == 0:
            return

        # Apply FDR correction
        rejected, pvals_corrected, _, _ = multipletests(all_pvals, method='fdr_bh')

        # Update results
        for i, (comp_type, key, analysis_type) in enumerate(pval_locations):
            results['comparisons'][comp_type][key][analysis_type]['p_value_fdr'] = pvals_corrected[i]
            results['comparisons'][comp_type][key][analysis_type]['significant_fdr'] = rejected[i]

    def compute_bone_aspect_ratios(self) -> pd.DataFrame:
        """
        Compute aspect ratios (height/width) for each bone using PCA eigenvalues

        Returns:
            DataFrame with aspect ratios per subject
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        aspect_ratios = []

        for subject_id in self.data['subject_id'].unique():
            subject_data = self.data[self.data['subject_id'] == subject_id]
            metadata = subject_data.iloc[0]

            subject_ratios = {
                'subject_id': subject_id,
                'label_disease': metadata.get('label_disease', 'Unknown'),
                'label_category': metadata.get('label_category', 'Unknown')
            }

            # TODO(human): Implement aspect ratio computation
            # For each bone (ap), compute the aspect ratio as:
            # 1. Get bone coordinates
            # 2. Center the coordinates
            # 3. Compute covariance matrix
            # 4. Get eigenvalues
            # 5. aspect_ratio = max_eigenvalue / min_eigenvalue
            # This ratio indicates bone elongation (1.0 = circular, >1.0 = elongated)

            for ap in self.ap_list:
                bone_coords = self.extract_bone_coordinates(subject_id, ap)

                if len(bone_coords) > 2:
                    # Center coordinates
                    centroid = np.mean(bone_coords, axis=0)
                    centered = bone_coords - centroid

                    # Compute PCA
                    cov_matrix = np.cov(centered.T)
                    eigenvalues = np.linalg.eigvalsh(cov_matrix)

                    # Aspect ratio = max/min eigenvalue
                    aspect_ratio = eigenvalues[-1] / (eigenvalues[0] + 1e-10)
                    subject_ratios[f'{ap}_aspect_ratio'] = aspect_ratio

                    # Also store width and height for interpretation
                    subject_ratios[f'{ap}_width'] = np.sqrt(eigenvalues[0])
                    subject_ratios[f'{ap}_height'] = np.sqrt(eigenvalues[-1])
                else:
                    subject_ratios[f'{ap}_aspect_ratio'] = np.nan
                    subject_ratios[f'{ap}_width'] = np.nan
                    subject_ratios[f'{ap}_height'] = np.nan

            aspect_ratios.append(subject_ratios)

        return pd.DataFrame(aspect_ratios)

    def analyze_relative_scale_ratios(self) -> Dict:
        """
        Analyze aspect ratios (height/width) for each bone
        Reveals bone elongation patterns in disease

        Returns:
            Dict with aspect ratio statistical comparisons
        """
        logger.info("Analyzing aspect ratios (height/width)...")

        # Compute aspect ratios
        ratios_df = self.compute_bone_aspect_ratios()

        if ratios_df.empty:
            raise ValueError("No aspect ratio data computed")

        results = {
            'method': 'Aspect Ratio Analysis',
            'description': 'Bone height/width ratios revealing elongation patterns',
            'comparisons': {}
        }

        # Define comparison groups
        comparisons = [
            ('Normal', 'by_disease'),
            ('Normal', 'by_category')
        ]

        for base_group, comparison_type in comparisons:
            results['comparisons'][comparison_type] = {}

            for ap in self.ap_list:
                ratio_col = f'{ap}_aspect_ratio'

                # Get valid data
                valid_data = ratios_df.dropna(subset=[ratio_col]).copy()
                if len(valid_data) < 4:
                    continue

                # Get base group
                base_mask = valid_data['label_disease'] == base_group
                if not base_mask.any():
                    continue

                base_ratios = valid_data[base_mask][ratio_col].values.reshape(-1, 1)

                # Test against each disease/category
                target_groups = (valid_data['label_disease'].unique()
                               if comparison_type == 'by_disease'
                               else valid_data['label_category'].unique())

                for target_group in target_groups:
                    if target_group == base_group:
                        continue

                    target_mask = (valid_data['label_disease'] == target_group
                                 if comparison_type == 'by_disease'
                                 else valid_data['label_category'] == target_group)

                    if not target_mask.any():
                        continue

                    target_ratios = valid_data[target_mask][ratio_col].values.reshape(-1, 1)

                    if len(base_ratios) < 2 or len(target_ratios) < 2:
                        continue

                    # Statistical analysis
                    comparison_name = f"{base_group}_vs_{target_group}"

                    ratio_stats = self.stats_framework.comprehensive_test(
                        base_ratios, target_ratios,
                        comparison_name=f"{ap}_aspect_ratio_{comparison_name}"
                    )

                    # Store results
                    key = f"{ap}_{target_group}"
                    results['comparisons'][comparison_type][key] = {
                        'bone': ap,
                        'comparison': comparison_name,
                        'base_group': base_group,
                        'target_group': target_group,
                        'base_mean_ratio': np.mean(base_ratios),
                        'target_mean_ratio': np.mean(target_ratios),
                        'ratio_change': np.mean(target_ratios) / np.mean(base_ratios),
                        'base_mean_width': np.mean(valid_data[base_mask][f'{ap}_width']),
                        'base_mean_height': np.mean(valid_data[base_mask][f'{ap}_height']),
                        'target_mean_width': np.mean(valid_data[target_mask][f'{ap}_width']),
                        'target_mean_height': np.mean(valid_data[target_mask][f'{ap}_height']),
                        'statistical_results': ratio_stats
                    }

        # Apply FDR correction
        self._apply_fdr_correction_ratios(results)

        logger.info(f"Aspect ratio analysis complete. Analyzed {len(self.ap_list)} bones.")
        return results

    def _apply_fdr_correction_ratios(self, results: Dict) -> None:
        """Apply FDR correction to p-values for ratio analysis"""
        all_pvals = []
        pval_locations = []

        # Collect all p-values
        for comp_type, comparisons in results['comparisons'].items():
            for key, data in comparisons.items():
                if 'statistical_results' in data and 'p_value' in data['statistical_results']:
                    pval = data['statistical_results']['p_value']
                    if np.isfinite(pval):
                        all_pvals.append(pval)
                        pval_locations.append((comp_type, key))

        if len(all_pvals) == 0:
            return

        # Apply FDR correction
        rejected, pvals_corrected, _, _ = multipletests(all_pvals, method='fdr_bh')

        # Update results
        for i, (comp_type, key) in enumerate(pval_locations):
            results['comparisons'][comp_type][key]['statistical_results']['p_value_fdr'] = pvals_corrected[i]
            results['comparisons'][comp_type][key]['statistical_results']['significant_fdr'] = rejected[i]

    def compute_bone_sizes(self) -> pd.DataFrame:
        """
        Compute centroid sizes for inter-bone ratio analysis

        Returns:
            DataFrame with bone sizes per subject
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        bone_sizes = []

        for subject_id in self.data['subject_id'].unique():
            subject_data = self.data[self.data['subject_id'] == subject_id]
            metadata = subject_data.iloc[0]

            subject_sizes = {
                'subject_id': subject_id,
                'label_disease': metadata.get('label_disease', 'Unknown'),
                'label_category': metadata.get('label_category', 'Unknown')
            }

            for ap in self.ap_list:
                bone_coords = self.extract_bone_coordinates(subject_id, ap)

                if len(bone_coords) > 1:
                    # Centroid size (RMS distance from centroid)
                    centroid = np.mean(bone_coords, axis=0)
                    distances = np.linalg.norm(bone_coords - centroid, axis=1)
                    centroid_size = np.sqrt(np.mean(distances**2))
                    subject_sizes[f'{ap}_size'] = centroid_size
                else:
                    subject_sizes[f'{ap}_size'] = np.nan

            bone_sizes.append(subject_sizes)

        return pd.DataFrame(bone_sizes)

    def analyze_relative_bone_ratios(self) -> Dict:
        """
        Analyze relative size ratios between bones (e.g., forefoot/hindfoot)
        Reveals regional biomechanical relationships

        Returns:
            Dict with inter-bone ratio statistical comparisons
        """
        logger.info("Analyzing relative bone-to-bone ratios...")

        # Compute bone sizes
        sizes_df = self.compute_bone_sizes()

        if sizes_df.empty:
            raise ValueError("No bone size data computed")

        results = {
            'method': 'Relative Bone Ratio Analysis',
            'description': 'Inter-bone size ratios revealing regional relationships',
            'comparisons': {}
        }

        # Define clinically meaningful bone pairs
        bone_pairs = [
            ('AP1', 'AP2', 'Calcaneus/Talus'),
            ('AP3', 'AP4', 'Cuboid/Navicular'),
            ('AP1', 'AP5', 'Hindfoot/Midfoot'),
            ('AP5', 'AP7', 'Midfoot/Forefoot'),
            ('AP1', 'AP7', 'Hindfoot/Forefoot'),
            ('AP2', 'AP4', 'Talus/Navicular'),
        ]

        # Define comparison groups
        comparisons = [
            ('Normal', 'by_disease'),
            ('Normal', 'by_category')
        ]

        for base_group, comparison_type in comparisons:
            results['comparisons'][comparison_type] = {}

            for ap1, ap2, label in bone_pairs:
                ratio_col = f'{ap1}_{ap2}_ratio'

                # Compute ratios
                valid_data = sizes_df.dropna(subset=[f'{ap1}_size', f'{ap2}_size']).copy()
                if len(valid_data) < 4:
                    continue

                valid_data[ratio_col] = valid_data[f'{ap1}_size'] / valid_data[f'{ap2}_size']

                # Remove invalid ratios
                valid_data = valid_data[np.isfinite(valid_data[ratio_col])]
                if len(valid_data) < 4:
                    continue

                # Get base group
                base_mask = valid_data['label_disease'] == base_group
                if not base_mask.any():
                    continue

                base_ratios = valid_data[base_mask][ratio_col].values.reshape(-1, 1)

                # Test against each disease/category
                target_groups = (valid_data['label_disease'].unique()
                               if comparison_type == 'by_disease'
                               else valid_data['label_category'].unique())

                for target_group in target_groups:
                    if target_group == base_group:
                        continue

                    target_mask = (valid_data['label_disease'] == target_group
                                 if comparison_type == 'by_disease'
                                 else valid_data['label_category'] == target_group)

                    if not target_mask.any():
                        continue

                    target_ratios = valid_data[target_mask][ratio_col].values.reshape(-1, 1)

                    if len(base_ratios) < 2 or len(target_ratios) < 2:
                        continue

                    # Statistical analysis
                    comparison_name = f"{base_group}_vs_{target_group}"

                    ratio_stats = self.stats_framework.comprehensive_test(
                        base_ratios, target_ratios,
                        comparison_name=f"{ratio_col}_{comparison_name}"
                    )

                    # Store results
                    key = f"{ap1}_{ap2}_{target_group}"
                    results['comparisons'][comparison_type][key] = {
                        'bone_pair': f"{ap1}/{ap2}",
                        'bone_pair_label': label,
                        'comparison': comparison_name,
                        'base_group': base_group,
                        'target_group': target_group,
                        'base_mean_ratio': np.mean(base_ratios),
                        'target_mean_ratio': np.mean(target_ratios),
                        'ratio_change': np.mean(target_ratios) / np.mean(base_ratios),
                        'statistical_results': ratio_stats
                    }

        # Apply FDR correction
        self._apply_fdr_correction_ratios(results)

        logger.info(f"Relative bone ratio analysis complete. Analyzed {len(bone_pairs)} bone pairs.")
        return results

    def analyze_shape_deformation(self) -> Dict:
        """
        Analyze pure shape changes after Procrustes alignment
        Reveals true pathological morphological changes
        """
        # Implementation placeholder - waiting for user input
        pass

    def export_results_to_csv(self, results: Dict, output_dir: str = "geometric_analysis_results") -> None:
        """Export analysis results to CSV files for further analysis"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        for analysis_type, data in results.get('comparisons', {}).items():
            # Flatten results for CSV export
            rows = []
            for key, comparison in data.items():
                # Check if this is ratio analysis or translation analysis
                if 'statistical_results' in comparison:
                    # Ratio analysis format
                    stats = comparison['statistical_results']
                    row = {
                        'bone_pair': comparison.get('bone_pair', ''),
                        'ratio_type': comparison.get('ratio_type', ''),
                        'comparison': comparison.get('comparison', ''),
                        'base_group': comparison.get('base_group', ''),
                        'target_group': comparison.get('target_group', ''),
                        'base_mean_ratio': comparison.get('base_mean_ratio', np.nan),
                        'target_mean_ratio': comparison.get('target_mean_ratio', np.nan),
                        'ratio_change': comparison.get('ratio_change', np.nan),
                        'effect_size': stats.get('effect_size', np.nan),
                        'p_value': stats.get('p_value', np.nan),
                        'p_value_fdr': stats.get('p_value_fdr', np.nan),
                        'ci_lower': stats.get('ci_lower', np.nan),
                        'ci_upper': stats.get('ci_upper', np.nan),
                        'significant': stats.get('significant', False),
                        'significant_fdr': stats.get('significant_fdr', False),
                        'n_base': stats.get('n_base', 0),
                        'n_target': stats.get('n_target', 0),
                        'n_bootstrap_used': stats.get('n_bootstrap_used', 0),
                        'n_permutation_used': stats.get('n_permutation_used', 0)
                    }
                    rows.append(row)
                else:
                    # Translation analysis format (original)
                    base_row = {
                        'bone': comparison.get('bone', ''),
                        'comparison': comparison.get('comparison', ''),
                        'base_group': comparison.get('base_group', ''),
                        'target_group': comparison.get('target_group', ''),
                    }

                    # Add results for each coordinate type
                    for coord_type in ['x_coordinate', 'y_coordinate', 'displacement_magnitude']:
                        if coord_type in comparison:
                            coord_data = comparison[coord_type]
                            row = base_row.copy()
                            row.update({
                                'measure_type': coord_type,
                                'effect_size': coord_data.get('effect_size', np.nan),
                                'p_value': coord_data.get('p_value', np.nan),
                                'p_value_fdr': coord_data.get('p_value_fdr', np.nan),
                                'ci_lower': coord_data.get('ci_lower', np.nan),
                                'ci_upper': coord_data.get('ci_upper', np.nan),
                                'significant': coord_data.get('significant', False),
                                'significant_fdr': coord_data.get('significant_fdr', False),
                                'n_base': coord_data.get('n_base', 0),
                                'n_target': coord_data.get('n_target', 0),
                                'n_bootstrap_used': coord_data.get('n_bootstrap_used', 0),
                                'n_permutation_used': coord_data.get('n_permutation_used', 0)
                            })
                            rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                # Determine analysis type for filename
                filename_prefix = "ratio_analysis" if 'bone_pair' in df.columns else "translation_analysis"
                output_file = os.path.join(output_dir, f"{filename_prefix}_{analysis_type}.csv")
                df.to_csv(output_file, index=False)
                logger.info(f"Results exported to {output_file}")

if __name__ == "__main__":
    # Example usage with bonewise data
    analyzer = GeometricAnalyzer(
        "test_data/output/procrustes_results/aligned_bonewise.feather",
        alignment_type="bonewise"
    )
    data = analyzer.load_data()

    print(f"Loaded {len(data)} landmarks from {data['subject_id'].nunique()} subjects")
    print(f"Available bones: {analyzer.ap_list}")
    print(f"Disease groups: {data['label_disease'].value_counts().to_dict()}")
    print(f"Available analyses: {analyzer.available_analyses}")

    # Run relative scale ratio analysis
    print("\nRunning relative scale ratio analysis...")
    ratio_results = analyzer.analyze_relative_scale_ratios()

    # Export results
    analyzer.export_results_to_csv(ratio_results, "bone_ratio_analysis_results")

    print("Relative scale ratio analysis complete!")

    # Display sample results
    by_disease = ratio_results['comparisons'].get('by_disease', {})
    significant_ratios = []
    for key, data in by_disease.items():
        stats = data['statistical_results']
        if stats.get('significant_fdr', False):
            significant_ratios.append({
                'bone_pair': data['bone_pair'],
                'target_group': data['target_group'],
                'ratio_change': data['ratio_change'],
                'effect_size': stats['effect_size'],
                'p_value_fdr': stats['p_value_fdr']
            })

    if significant_ratios:
        print(f"\nFound {len(significant_ratios)} significant bone ratio changes:")
        for result in significant_ratios[:5]:  # Show top 5
            print(f"  {result['bone_pair']} in {result['target_group']}: "
                  f"{result['ratio_change']:.3f}x change "
                  f"(g={result['effect_size']:.3f}, p={result['p_value_fdr']:.4f})")
    else:
        print("\nNo significant ratio changes found after FDR correction.")