#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Multi-level Statistical Analysis for Foot Disease Data
Academic-grade bootstrap and permutation testing with performance optimizations

Features:
- Evidence-based iteration numbers (10K bootstrap, 10K-100K permutation)
- Tiered approach: pilot → standard → high-precision
- Memory-efficient streaming processing
- Robust error handling and checkpointing
- Age binning: 20-39, 40-59, 60+
- Stratified analysis: sex × age combinations
"""

import os, json, argparse, warnings, math, time
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.optimize import brentq
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed, Memory
import psutil
from pathlib import Path
import logging
from dataclasses import dataclass
import pickle
from functools import lru_cache, partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from numba import jit, prange
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Academic-grade iteration settings based on literature review
@dataclass
class IterationConfig:
    """Evidence-based iteration configuration for academic research"""
    # Sampling strategy selection
    use_adaptive_sampling: bool = True  # Enable adaptive vs traditional approach

    # Advanced optimization settings
    use_ultra_optimization: bool = True  # Enable all performance optimizations
    enable_numba_jit: bool = True       # Use Numba JIT compilation
    enable_vectorization: bool = True   # Use vectorized operations
    enable_memory_management: bool = True  # Advanced memory management
    enable_early_stopping: bool = True  # Early stopping for non-significant results

    # Adaptive sampling parameters (Lee & Young, 1996; PMC 2009)
    pilot_bootstrap: int = 1000        # Initial screening
    pilot_permutation: int = 1000      # Quick p-value estimation

    # Decision thresholds for adaptive sampling (literature-supported)
    non_significant_threshold: float = 0.1   # Stop early if p > 0.1
    high_precision_threshold: float = 0.01   # Use max precision if p ≤ 0.01

    # Standard iterations (used in both adaptive and traditional modes)
    standard_bootstrap: int = 10000    # Literature standard (±0.01 precision)
    standard_permutation: int = 10000  # Good balance precision/speed

    # High-precision iterations (for significant results)
    precise_bootstrap: int = 50000     # High-precision CIs
    precise_permutation: int = 100000  # ±0.1% p-value precision

    confidence_level: float = 0.95
    alpha: float = 0.05

    def get_strategy_description(self) -> str:
        """Return description of sampling strategy for methodology reporting"""
        if self.use_adaptive_sampling:
            return (
                f"Adaptive sequential bootstrap sampling (Lee & Young, 1996): "
                f"pilot {self.pilot_bootstrap}→standard {self.standard_bootstrap}→"
                f"precise {self.precise_bootstrap} based on preliminary p-values. "
                f"Permutation tests: {self.pilot_permutation}→{self.standard_permutation}→"
                f"{self.precise_permutation} iterations."
            )
        else:
            return (
                f"Traditional fixed bootstrap sampling: {self.standard_bootstrap} iterations. "
                f"Permutation tests: {self.standard_permutation} iterations."
            )

# Performance and memory settings
MAX_MEMORY_GB = 8  # Adjust based on system
CHECKPOINT_INTERVAL = 50  # Save progress every N comparisons
BATCH_SIZE = 20  # Process landmarks in batches
PRECOMPUTE_CACHE_SIZE = 1000  # LRU cache size for repeated computations
MEMORY_CLEANUP_INTERVAL = 100  # Force garbage collection every N operations
VECTORIZED_BATCH_SIZE = 10000  # Vectorized operations batch size
CHUNK_SIZE_MB = 100  # Memory chunk size for large datasets

def setup_memory_cache(cache_dir: str = './stats_cache'):
    """Setup joblib memory caching for expensive computations"""
    memory = Memory(cache_dir, verbose=0)
    return memory

# Advanced optimization functions
@jit(nopython=True, cache=True)
def numba_hotelling_t2_core(X_flat: np.ndarray, Y_flat: np.ndarray,
                           n1: int, n2: int, p: int) -> float:
    """Numba-optimized core computation for Hotelling's T²"""
    # Reshape flat arrays back to matrices
    X = X_flat.reshape(n1, p)
    Y = Y_flat.reshape(n2, p)

    # Compute means (Numba-compatible way)
    mx = np.zeros(p)
    my = np.zeros(p)
    for j in range(p):
        mx[j] = np.mean(X[:, j])
        my[j] = np.mean(Y[:, j])

    diff = mx - my

    # Compute pooled covariance with ridge regularization
    if n1 > 1:
        X_centered = X - mx
        S1 = np.dot(X_centered.T, X_centered) / (n1 - 1)
    else:
        S1 = np.eye(p) * 1e-6

    if n2 > 1:
        Y_centered = Y - my
        S2 = np.dot(Y_centered.T, Y_centered) / (n2 - 1)
    else:
        S2 = np.eye(p) * 1e-6

    # Pooled covariance with ridge regularization
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    ridge = 1e-6
    for i in range(p):
        Sp[i, i] += ridge

    # Hotelling's T² computation - use manual inverse to avoid try/except
    det = np.linalg.det(Sp)
    if abs(det) < 1e-12:
        return np.nan

    Sp_inv = np.linalg.inv(Sp)
    T2 = (n1 * n2) / (n1 + n2) * np.dot(diff, np.dot(Sp_inv, diff))
    return T2

@jit(nopython=True, parallel=True, cache=True)
def vectorized_bootstrap_batch(X_flat: np.ndarray, Y_flat: np.ndarray,
                              n1: int, n2: int, p: int,
                              indices_1: np.ndarray, indices_2: np.ndarray) -> np.ndarray:
    """Vectorized bootstrap computation using Numba"""
    n_bootstrap = indices_1.shape[0]
    results = np.empty(n_bootstrap, dtype=np.float64)

    for i in prange(n_bootstrap):
        # Bootstrap sample indices
        idx1 = indices_1[i]
        idx2 = indices_2[i]

        # Create bootstrap samples
        X_boot = X_flat.reshape(n1, p)[idx1].flatten()
        Y_boot = Y_flat.reshape(n2, p)[idx2].flatten()

        # Compute T² for bootstrap sample
        results[i] = numba_hotelling_t2_core(X_boot, Y_boot, n1, n2, p)

    return results

@lru_cache(maxsize=PRECOMPUTE_CACHE_SIZE)
def cached_covariance_inverse(cov_key: str) -> np.ndarray:
    """Cached computation of covariance matrix inverse"""
    # This would store pre-computed inverse matrices
    # Implementation depends on specific use case
    pass

class MemoryManager:
    """Advanced memory management for large-scale statistical computations"""

    def __init__(self, max_memory_gb: float = MAX_MEMORY_GB):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.operation_count = 0

    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits"""
        current_memory = psutil.Process().memory_info().rss
        return current_memory < self.max_memory_bytes

    def cleanup_if_needed(self):
        """Force garbage collection if memory usage is high"""
        self.operation_count += 1
        if self.operation_count % MEMORY_CLEANUP_INTERVAL == 0:
            if not self.check_memory_usage():
                gc.collect()
                logger.info("Memory cleanup performed")

    def get_optimal_chunk_size(self, data_size: int, element_size: int = 8) -> int:
        """Calculate optimal chunk size based on available memory"""
        available_memory = self.max_memory_bytes - psutil.Process().memory_info().rss
        safe_memory = available_memory * 0.7  # Use 70% of available memory
        optimal_chunk = int(safe_memory / element_size)
        return min(optimal_chunk, data_size)

class VectorizedStatistics:
    """Vectorized statistical operations for improved performance"""

    @staticmethod
    def batch_hotelling_t2(X_list: List[np.ndarray], Y_list: List[np.ndarray],
                          ridge: float = 1e-6) -> List[float]:
        """Compute Hotelling's T² for multiple pairs simultaneously"""
        results = []
        memory_manager = MemoryManager()

        # Process in chunks to manage memory
        chunk_size = memory_manager.get_optimal_chunk_size(len(X_list))

        for i in range(0, len(X_list), chunk_size):
            chunk_X = X_list[i:i + chunk_size]
            chunk_Y = Y_list[i:i + chunk_size]

            # Vectorized computation for chunk
            chunk_results = []
            for X, Y in zip(chunk_X, chunk_Y):
                if X.shape[1] == Y.shape[1]:  # Same dimensionality
                    n1, n2, p = X.shape[0], Y.shape[0], X.shape[1]
                    T2 = numba_hotelling_t2_core(X.flatten(), Y.flatten(), n1, n2, p)
                    chunk_results.append(float(T2))
                else:
                    chunk_results.append(float('nan'))

            results.extend(chunk_results)
            memory_manager.cleanup_if_needed()

        return results

    @staticmethod
    def parallel_bootstrap_ci(X: np.ndarray, Y: np.ndarray, func,
                             n_bootstrap: int = 10000, n_jobs: int = -1) -> Tuple[float, float]:
        """Highly optimized parallel bootstrap confidence intervals"""
        if n_jobs == -1:
            n_jobs = get_optimal_n_jobs()

        n1, n2, p = X.shape[0], Y.shape[0], X.shape[1]

        # Pre-generate all bootstrap indices
        rng = np.random.default_rng(42)
        indices_1 = rng.integers(0, n1, size=(n_bootstrap, n1))
        indices_2 = rng.integers(0, n2, size=(n_bootstrap, n2))

        # Use vectorized Numba function for bootstrap computation
        if hasattr(func, '__name__') and 'hotelling' in func.__name__:
            bootstrap_vals = vectorized_bootstrap_batch(
                X.flatten(), Y.flatten(), n1, n2, p, indices_1, indices_2
            )
            # Filter finite values
            finite_vals = bootstrap_vals[np.isfinite(bootstrap_vals)]
        else:
            # Fallback to standard approach for other functions
            def single_bootstrap(seed_idx):
                local_rng = np.random.default_rng(seed_idx)
                i1 = local_rng.integers(0, n1, n1)
                i2 = local_rng.integers(0, n2, n2)
                return func(X[i1], Y[i2])

            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                bootstrap_vals = list(executor.map(single_bootstrap, range(n_bootstrap)))
            finite_vals = [v for v in bootstrap_vals if np.isfinite(v)]

        if len(finite_vals) < 100:
            return float('nan'), float('nan')

        return (float(np.percentile(finite_vals, 2.5)),
                float(np.percentile(finite_vals, 97.5)))

def ultra_optimized_bootstrap_ci(X: np.ndarray, Y: np.ndarray, config: IterationConfig,
                                func, seed: int = 42) -> Tuple[float, float, int]:
    """Ultra-optimized bootstrap CI with all performance enhancements"""
    # Use vectorized approach for Hotelling's T² (most common case)
    if hasattr(func, '__name__') and 'hotelling' in func.__name__:
        return VectorizedStatistics.parallel_bootstrap_ci(
            X, Y, func, config.standard_bootstrap
        ) + (config.standard_bootstrap,)

    # Fallback to adaptive approach for other functions
    if config.use_adaptive_sampling:
        return adaptive_bootstrap_ci(X, Y, config, func, seed)
    else:
        return traditional_bootstrap_ci(X, Y, config, func, seed)

def ultra_optimized_permutation_test(X: np.ndarray, Y: np.ndarray, config: IterationConfig,
                                   ridge: float = 1e-6, seed: int = 42) -> Tuple[float, int]:
    """Ultra-optimized permutation test with vectorization and early stopping"""
    memory_manager = MemoryManager()

    # Compute observed statistic using optimized version
    n1, n2, p = X.shape[0], Y.shape[0], X.shape[1]
    T2_observed = numba_hotelling_t2_core(X.flatten(), Y.flatten(), n1, n2, p)

    if not np.isfinite(T2_observed):
        return float('nan'), 0

    pooled_data = np.vstack((X, Y))
    total_n = n1 + n2

    # Use adaptive or traditional approach
    if config.use_adaptive_sampling:
        # Pilot phase
        pilot_size = config.pilot_permutation
        rng = np.random.default_rng(seed)
        pilot_seeds = rng.integers(0, 2**32 - 1, pilot_size)

        pilot_count = 0
        for s in pilot_seeds:
            local_rng = np.random.default_rng(s)
            perm_indices = local_rng.permutation(total_n)
            X_perm = pooled_data[perm_indices[:n1]]
            Y_perm = pooled_data[perm_indices[n1:]]

            T2_perm = numba_hotelling_t2_core(
                X_perm.flatten(), Y_perm.flatten(), n1, n2, p
            )
            if np.isfinite(T2_perm) and T2_perm >= T2_observed:
                pilot_count += 1

        pilot_p = pilot_count / pilot_size

        # Decide on precision level
        if pilot_p > config.non_significant_threshold:
            # Early stopping for non-significant results
            return pilot_p, pilot_size
        elif pilot_p <= config.high_precision_threshold:
            target_iterations = config.precise_permutation
        else:
            target_iterations = config.standard_permutation

        # Continue with remaining iterations
        remaining_iterations = target_iterations - pilot_size
        if remaining_iterations > 0:
            remaining_seeds = rng.integers(0, 2**32 - 1, remaining_iterations)
            remaining_count = 0

            # Process in chunks to manage memory
            chunk_size = memory_manager.get_optimal_chunk_size(remaining_iterations)

            for i in range(0, remaining_iterations, chunk_size):
                chunk_seeds = remaining_seeds[i:i + chunk_size]

                for s in chunk_seeds:
                    local_rng = np.random.default_rng(s)
                    perm_indices = local_rng.permutation(total_n)
                    X_perm = pooled_data[perm_indices[:n1]]
                    Y_perm = pooled_data[perm_indices[n1:]]

                    T2_perm = numba_hotelling_t2_core(
                        X_perm.flatten(), Y_perm.flatten(), n1, n2, p
                    )
                    if np.isfinite(T2_perm) and T2_perm >= T2_observed:
                        remaining_count += 1

                memory_manager.cleanup_if_needed()

            total_count = pilot_count + remaining_count
            return total_count / target_iterations, target_iterations
        else:
            return pilot_p, pilot_size

    else:
        # Traditional fixed approach with optimization
        n_permutation = config.standard_permutation
        rng = np.random.default_rng(seed)
        perm_seeds = rng.integers(0, 2**32 - 1, n_permutation)

        count = 0
        chunk_size = memory_manager.get_optimal_chunk_size(n_permutation)

        for i in range(0, n_permutation, chunk_size):
            chunk_seeds = perm_seeds[i:i + chunk_size]

            for s in chunk_seeds:
                local_rng = np.random.default_rng(s)
                perm_indices = local_rng.permutation(total_n)
                X_perm = pooled_data[perm_indices[:n1]]
                Y_perm = pooled_data[perm_indices[n1:]]

                T2_perm = numba_hotelling_t2_core(
                    X_perm.flatten(), Y_perm.flatten(), n1, n2, p
                )
                if np.isfinite(T2_perm) and T2_perm >= T2_observed:
                    count += 1

            memory_manager.cleanup_if_needed()

        return count / n_permutation, n_permutation

class DataPipelineOptimizer:
    """Advanced data pipeline optimization for large-scale analysis"""

    def __init__(self, config: IterationConfig):
        self.config = config
        self.memory_manager = MemoryManager()
        self.stats_cache = {}

    def optimize_data_grouping(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Optimize data grouping to minimize memory usage and maximize cache efficiency"""
        grouped_data = {}

        # Pre-sort by most frequent grouping patterns to improve cache locality
        df_sorted = df.sort_values(['sex', 'age_bin', 'ap', 'landmark_index'])

        for (sex, age_bin), group in df_sorted.groupby(['sex', 'age_bin']):
            key = f"{sex}_{age_bin}"
            grouped_data[key] = group.copy()

        logger.info(f"Optimized data grouping: {len(grouped_data)} groups created")
        return grouped_data

    def batch_process_landmarks(self, data_groups: Dict[str, pd.DataFrame],
                               batch_size: int = BATCH_SIZE) -> List[Dict]:
        """Process landmarks in optimized batches with memory management"""
        all_comparisons = []

        # Extract all unique landmark comparisons
        landmark_pairs = set()
        for group_data in data_groups.values():
            for ap in group_data['ap'].unique():
                ap_data = group_data[group_data['ap'] == ap]
                for landmark_idx in ap_data['landmark_index'].unique():
                    landmark_pairs.add((ap, landmark_idx))

        landmark_pairs = list(landmark_pairs)
        logger.info(f"Processing {len(landmark_pairs)} landmark comparisons in batches of {batch_size}")

        # Process in batches to manage memory
        for i in range(0, len(landmark_pairs), batch_size):
            batch_pairs = landmark_pairs[i:i + batch_size]
            batch_results = []

            for ap, landmark_idx in batch_pairs:
                # Process all group combinations for this landmark
                for base_group_key in data_groups:
                    base_data = data_groups[base_group_key]
                    base_landmark = base_data[
                        (base_data['ap'] == ap) & (base_data['landmark_index'] == landmark_idx)
                    ]

                    if len(base_landmark) < 2:  # Need at least 2 samples
                        continue

                    # Extract coordinates
                    base_coords = base_landmark[['x', 'y']].values

                    # Compare with other groups
                    for target_group_key in data_groups:
                        if base_group_key != target_group_key:
                            target_data = data_groups[target_group_key]
                            target_landmark = target_data[
                                (target_data['ap'] == ap) & (target_data['landmark_index'] == landmark_idx)
                            ]

                            if len(target_landmark) < 2:  # Need at least 2 samples
                                continue

                            target_coords = target_landmark[['x', 'y']].values

                            # Create comparison data
                            comparison_data = {
                                'index': (ap, landmark_idx),
                                'X': base_coords,
                                'Y': target_coords,
                                'base_group': base_group_key,
                                'target_group': target_group_key
                            }
                            batch_results.append(comparison_data)

            all_comparisons.extend(batch_results)
            self.memory_manager.cleanup_if_needed()

        return all_comparisons

    def parallel_statistical_analysis(self, comparisons: List[Dict],
                                    n_jobs: int = -1) -> List[Dict]:
        """Execute statistical analysis in parallel with optimized resource management"""
        if n_jobs == -1:
            n_jobs = get_optimal_n_jobs()

        logger.info(f"Processing {len(comparisons)} comparisons using {n_jobs} parallel jobs")

        # Define processing function
        def process_single_comparison(comp_data):
            try:
                return process_landmark_comparison(comp_data, self.config)
            except Exception as e:
                logger.warning(f"Failed to process comparison {comp_data.get('index', 'unknown')}: {e}")
                return None

        # Use joblib for parallel processing with memory management
        with Parallel(n_jobs=n_jobs, verbose=1, batch_size='auto') as parallel:
            results = parallel(
                delayed(process_single_comparison)(comp_data)
                for comp_data in comparisons
            )

        # Filter out failed results
        valid_results = [r for r in results if r is not None]
        logger.info(f"Successfully processed {len(valid_results)}/{len(comparisons)} comparisons")

        return valid_results

    def streaming_fdr_correction(self, results: List[Dict], alpha: float = 0.05) -> List[Dict]:
        """Apply FDR correction with streaming approach for memory efficiency"""
        # Extract p-values
        pvals_param = [r['pval_param'] for r in results if np.isfinite(r['pval_param'])]
        pvals_perm = [r['pval_perm'] for r in results if np.isfinite(r['pval_perm'])]

        # Apply FDR correction in chunks to manage memory
        chunk_size = self.memory_manager.get_optimal_chunk_size(len(pvals_param))

        # Parametric FDR correction
        if pvals_param:
            try:
                _, pvals_param_fdr, _, _ = multipletests(pvals_param, alpha=alpha, method='fdr_bh')
            except:
                pvals_param_fdr = pvals_param

        # Permutation FDR correction
        if pvals_perm:
            try:
                _, pvals_perm_fdr, _, _ = multipletests(pvals_perm, alpha=alpha, method='fdr_bh')
            except:
                pvals_perm_fdr = pvals_perm

        # Update results with FDR-corrected p-values
        param_idx = 0
        perm_idx = 0
        for i, result in enumerate(results):
            if np.isfinite(result['pval_param']):
                if pvals_param:
                    result['pval_param_fdr'] = float(pvals_param_fdr[param_idx])
                    param_idx += 1
                else:
                    result['pval_param_fdr'] = float('nan')
            else:
                result['pval_param_fdr'] = float('nan')

            if np.isfinite(result['pval_perm']):
                if pvals_perm:
                    result['pval_perm_fdr'] = float(pvals_perm_fdr[perm_idx])
                    perm_idx += 1
                else:
                    result['pval_perm_fdr'] = float('nan')
            else:
                result['pval_perm_fdr'] = float('nan')

        return results

def get_optimal_n_jobs():
    """Determine optimal number of parallel jobs based on system resources"""
    cpu_count = psutil.cpu_count(logical=False)  # Physical cores
    available_memory = psutil.virtual_memory().available / (1024**3)  # GB

    # Conservative approach: don't overwhelm system
    if available_memory < 4:
        return min(2, cpu_count // 2)
    elif available_memory < 8:
        return min(4, cpu_count)
    else:
        return min(cpu_count, 8)  # Cap at 8 for reasonable resource usage

def create_age_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Create academic-standard age bins: 20-39, 40-59, 60+"""
    df = df.copy()

    # Create age bins
    age_bins = [0, 40, 60, 100]  # Include all ages from 0
    age_labels = ['20-39', '40-59', '60+']

    # First create bins, then handle edge cases
    df['age_bin'] = pd.cut(
        df['group_age'],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True,
        right=False  # [0, 40), [40, 60), [60, 100]
    )

    # Convert to string to handle NaN values easily
    df['age_bin'] = df['age_bin'].astype(str)

    # Handle edge cases: map ages < 20 to '20-39', and handle NaN
    df.loc[df['group_age'] < 20, 'age_bin'] = '20-39'
    df.loc[df['age_bin'] == 'nan', 'age_bin'] = 'Unknown'

    # Also handle ages that might be exactly on boundaries
    df.loc[(df['group_age'] >= 20) & (df['group_age'] < 40), 'age_bin'] = '20-39'
    df.loc[(df['group_age'] >= 40) & (df['group_age'] < 60), 'age_bin'] = '40-59'
    df.loc[df['group_age'] >= 60, 'age_bin'] = '60+'

    logger.info(f"Age bin distribution: {df['age_bin'].value_counts().to_dict()}")
    return df

def ensure_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced label inference with better error handling"""
    df = df.copy()

    # Basic column mappings
    if "patient_id" in df.columns and "subject_id" not in df.columns:
        df = df.rename(columns={'patient_id': 'subject_id'})

    # Create sample_id for L/R foot identification
    df['sample_id'] = df['subject_id'].astype(str) + '_' + df['group_direction'].astype(str)

    # Disease label inference
    disease_cols = [c for c in df.columns if c.startswith('group_disease_')]

    def find_disease_label(row):
        for col in disease_cols:
            disease_name = col.replace('group_disease_', '')
            if row[col] == disease_name:
                return disease_name
        if 'group_disease_normal' in row and row['group_disease_normal'] == 'normal':
            return 'normal'
        return 'unknown'

    if disease_cols:
        df['label_disease'] = df.apply(find_disease_label, axis=1)
    else:
        df['label_disease'] = 'unknown'

    # Category label inference
    category_cols = [c for c in df.columns if c.startswith('group_category_')]

    def find_category_label(row):
        for col in category_cols:
            category_name = col.replace('group_category_', '')
            if row[col] == category_name:
                return category_name
        if 'group_category_normal' in row and row['group_category_normal'] == 'normal':
            return 'normal'
        return 'unknown'

    if category_cols:
        df['label_category'] = df.apply(find_category_label, axis=1)
    else:
        df['label_category'] = 'unknown'

    # Sex standardization
    if 'sex' not in df.columns:
        if 'group_sex' in df.columns:
            df['sex'] = df['group_sex'].astype(str).str.upper().map({'M': 'M', 'F': 'F'}).fillna('U')
        else:
            df['sex'] = 'U'

    # Age binning
    df = create_age_bins(df)

    return df

def pooled_cov(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized pooled covariance computation with ridge regularization"""
    n1, p = X.shape
    n2, _ = Y.shape

    # Compute means
    mx = X.mean(0)
    my = Y.mean(0)

    # Compute covariance matrices
    if n1 > 1:
        S1 = np.cov(X, rowvar=False, bias=False)
    else:
        S1 = np.eye(p) * ridge

    if n2 > 1:
        S2 = np.cov(Y, rowvar=False, bias=False)
    else:
        S2 = np.eye(p) * ridge

    # Pooled covariance
    dof = max(n1 + n2 - 2, 1)
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / dof

    # Ridge regularization for numerical stability
    if ridge > 0:
        Sp = Sp + ridge * np.eye(p)

    return Sp, mx, my

def hotelling_t2(X: np.ndarray, Y: np.ndarray, ridge: float = 1e-6) -> Tuple[float, float, float, float]:
    """Optimized Hotelling's T² test with better numerical stability"""
    n1, p = X.shape
    n2, _ = Y.shape

    if n1 < 2 or n2 < 2:
        return float('nan'), float('nan'), float('nan'), float('nan')

    try:
        Sp, mx, my = pooled_cov(X, Y, ridge=ridge)
        diff = mx - my

        # Use solve instead of pinv for better numerical stability
        try:
            invSp_diff = np.linalg.solve(Sp, diff)
            D2 = float(diff.T @ invSp_diff)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            invSp = np.linalg.pinv(Sp, rcond=1e-10)
            D2 = float(diff.T @ invSp @ diff)

        T2 = float((n1 * n2) / (n1 + n2) * D2)

        # F-statistic and p-value
        df1 = p
        df2 = n1 + n2 - p - 1

        if df2 > 0 and (n1 + n2 - 2) > 0:
            F = ((n1 + n2 - p - 1) / (p * (n1 + n2 - 2))) * T2
            pval = float(1.0 - f_dist.cdf(F, df1, df2))
        else:
            F, pval = float('nan'), float('nan')

        D = float(math.sqrt(max(D2, 0.0)))

        return T2, F, pval, D

    except Exception as e:
        logger.warning(f"Hotelling T² computation failed: {e}")
        return float('nan'), float('nan'), float('nan'), float('nan')

def hedges_g_total(X: np.ndarray, Y: np.ndarray) -> float:
    """Optimized Hedges' g effect size computation"""
    n1, p = X.shape
    n2, _ = Y.shape

    if n1 < 2 or n2 < 2:
        return float('nan')

    mx = X.mean(0)
    my = Y.mean(0)

    # Pooled standard deviation
    s1 = X.std(0, ddof=1) if n1 > 1 else np.ones(p)
    s2 = Y.std(0, ddof=1) if n2 > 1 else np.ones(p)

    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / max(n1 + n2 - 2, 1))
    sp[sp == 0] = 1.0  # Avoid division by zero

    d = (mx - my) / sp

    # Bias correction factor
    J = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0) if (n1 + n2) > 2 else 1.0
    g = J * d

    return float(np.sqrt(np.mean(g**2)))

def traditional_bootstrap_ci(X: np.ndarray, Y: np.ndarray, config: IterationConfig,
                             func, seed: int = 42) -> Tuple[float, float, int]:
    """Traditional fixed-iteration bootstrap for academic reproducibility"""
    rng = np.random.default_rng(seed)
    n1, n2 = X.shape[0], Y.shape[0]
    n_bootstrap = config.standard_bootstrap

    # Fixed number of bootstrap samples (traditional approach)
    bootstrap_vals = []
    bootstrap_seeds = rng.integers(0, 2**32 - 1, n_bootstrap)

    for s in bootstrap_seeds:
        local_rng = np.random.default_rng(s)
        i1 = local_rng.integers(0, n1, n1)
        i2 = local_rng.integers(0, n2, n2)
        val = func(X[i1], Y[i2])
        if np.isfinite(val):
            bootstrap_vals.append(val)

    if len(bootstrap_vals) < 100:  # Minimum for reasonable CI
        return float('nan'), float('nan'), len(bootstrap_vals)

    return (float(np.percentile(bootstrap_vals, 2.5)),
            float(np.percentile(bootstrap_vals, 97.5)),
            len(bootstrap_vals))

def adaptive_bootstrap_ci(X: np.ndarray, Y: np.ndarray, config: IterationConfig,
                         func, seed: int = 42) -> Tuple[float, float, int]:
    """Adaptive bootstrap with tiered precision approach (Lee & Young, 1996)"""
    rng = np.random.default_rng(seed)
    n1, n2 = X.shape[0], Y.shape[0]

    # Pilot run to assess effect size
    pilot_vals = []
    pilot_seeds = rng.integers(0, 2**32 - 1, config.pilot_bootstrap)

    for s in pilot_seeds:
        local_rng = np.random.default_rng(s)
        i1 = local_rng.integers(0, n1, n1)
        i2 = local_rng.integers(0, n2, n2)
        val = func(X[i1], Y[i2])
        if np.isfinite(val):
            pilot_vals.append(val)

    if not pilot_vals:
        return float('nan'), float('nan'), config.pilot_bootstrap

    # Assess variability to determine precision needs
    pilot_std = np.std(pilot_vals)
    pilot_mean = np.mean(pilot_vals)
    cv = pilot_std / max(abs(pilot_mean), 1e-10)  # Coefficient of variation

    # Adaptive sample size based on variability
    if cv < 0.1:  # Low variability
        n_bootstrap = config.standard_bootstrap
    elif cv < 0.5:  # Medium variability
        n_bootstrap = config.standard_bootstrap * 2
    else:  # High variability
        n_bootstrap = config.precise_bootstrap

    # Additional bootstrap samples if needed
    additional_samples = n_bootstrap - config.pilot_bootstrap
    if additional_samples > 0:
        additional_seeds = rng.integers(0, 2**32 - 1, additional_samples)
        for s in additional_seeds:
            local_rng = np.random.default_rng(s)
            i1 = local_rng.integers(0, n1, n1)
            i2 = local_rng.integers(0, n2, n2)
            val = func(X[i1], Y[i2])
            if np.isfinite(val):
                pilot_vals.append(val)

    if len(pilot_vals) < 100:  # Minimum for reasonable CI
        return float('nan'), float('nan'), len(pilot_vals)

    return (float(np.percentile(pilot_vals, 2.5)),
            float(np.percentile(pilot_vals, 97.5)),
            len(pilot_vals))

def traditional_permutation_test(X: np.ndarray, Y: np.ndarray, config: IterationConfig,
                                ridge: float = 1e-6, seed: int = 42) -> Tuple[float, int]:
    """Traditional fixed-iteration permutation test for reproducibility"""
    T2_observed, _, _, _ = hotelling_t2(X, Y, ridge=ridge)

    if not np.isfinite(T2_observed):
        return float('nan'), 0

    n1, n2 = X.shape[0], Y.shape[0]
    pooled_data = np.vstack((X, Y))
    n_permutation = config.standard_permutation

    # Single permutation test function
    def single_permutation(perm_seed):
        local_rng = np.random.default_rng(perm_seed)
        permuted_indices = local_rng.permutation(n1 + n2)
        X_perm = pooled_data[permuted_indices[:n1]]
        Y_perm = pooled_data[permuted_indices[n1:]]
        T2_perm, _, _, _ = hotelling_t2(X_perm, Y_perm, ridge=ridge)
        return 1 if np.isfinite(T2_perm) and T2_perm >= T2_observed else 0

    # Fixed number of permutations (traditional approach)
    rng = np.random.default_rng(seed)
    perm_seeds = rng.integers(0, 2**32 - 1, n_permutation)

    n_jobs = get_optimal_n_jobs()
    results = Parallel(n_jobs=n_jobs, batch_size=1000)(
        delayed(single_permutation)(s) for s in perm_seeds
    )

    count_exceeding = sum(results)
    final_p = (count_exceeding + 1) / (n_permutation + 1)
    return final_p, n_permutation

def adaptive_permutation_test(X: np.ndarray, Y: np.ndarray, config: IterationConfig,
                             ridge: float = 1e-6, seed: int = 42) -> Tuple[float, int]:
    """Adaptive permutation testing with tiered precision (PMC 2009)"""
    T2_observed, _, _, _ = hotelling_t2(X, Y, ridge=ridge)

    if not np.isfinite(T2_observed):
        return float('nan'), 0

    n1, n2 = X.shape[0], Y.shape[0]
    pooled_data = np.vstack((X, Y))
    rng = np.random.default_rng(seed)

    # Pilot permutation test
    pilot_count = 0
    pilot_seeds = rng.integers(0, 2**32 - 1, config.pilot_permutation)

    for s in pilot_seeds:
        local_rng = np.random.default_rng(s)
        permuted_indices = local_rng.permutation(n1 + n2)
        X_perm = pooled_data[permuted_indices[:n1]]
        Y_perm = pooled_data[permuted_indices[n1:]]
        T2_perm, _, _, _ = hotelling_t2(X_perm, Y_perm, ridge=ridge)

        if np.isfinite(T2_perm) and T2_perm >= T2_observed:
            pilot_count += 1

    # Estimate preliminary p-value
    pilot_p = (pilot_count + 1) / (config.pilot_permutation + 1)

    # Determine precision needed based on preliminary result (PMC 2009 methodology)
    if pilot_p > config.non_significant_threshold:
        # Clearly non-significant, pilot is sufficient
        total_iterations = config.pilot_permutation
        total_count = pilot_count
    elif pilot_p > config.high_precision_threshold:
        # Moderately significant, use standard precision
        additional_iterations = config.standard_permutation - config.pilot_permutation
        additional_count = 0

        if additional_iterations > 0:
            additional_seeds = rng.integers(0, 2**32 - 1, additional_iterations)
            for s in additional_seeds:
                local_rng = np.random.default_rng(s)
                permuted_indices = local_rng.permutation(n1 + n2)
                X_perm = pooled_data[permuted_indices[:n1]]
                Y_perm = pooled_data[permuted_indices[n1:]]
                T2_perm, _, _, _ = hotelling_t2(X_perm, Y_perm, ridge=ridge)

                if np.isfinite(T2_perm) and T2_perm >= T2_observed:
                    additional_count += 1

        total_iterations = config.standard_permutation
        total_count = pilot_count + additional_count
    else:
        # Potentially significant, use high precision
        additional_iterations = config.precise_permutation - config.pilot_permutation
        additional_count = 0

        if additional_iterations > 0:
            # Use parallel processing for large number of permutations
            def single_permutation(perm_seed):
                local_rng = np.random.default_rng(perm_seed)
                permuted_indices = local_rng.permutation(n1 + n2)
                X_perm = pooled_data[permuted_indices[:n1]]
                Y_perm = pooled_data[permuted_indices[n1:]]
                T2_perm, _, _, _ = hotelling_t2(X_perm, Y_perm, ridge=ridge)
                return 1 if np.isfinite(T2_perm) and T2_perm >= T2_observed else 0

            additional_seeds = rng.integers(0, 2**32 - 1, additional_iterations)
            n_jobs = get_optimal_n_jobs()
            results = Parallel(n_jobs=n_jobs, batch_size=1000)(
                delayed(single_permutation)(s) for s in additional_seeds
            )
            additional_count = sum(results)

        total_iterations = config.precise_permutation
        total_count = pilot_count + additional_count

    final_p = (total_count + 1) / (total_iterations + 1)
    return final_p, total_iterations

def process_landmark_comparison(landmark_data: Dict, config: IterationConfig,
                              ridge: float = 1e-6) -> Dict:
    """Process statistical comparison for a single landmark with method selection"""
    idx = landmark_data['index']
    X = landmark_data['X']
    Y = landmark_data['Y']

    # Basic statistics
    mean_base = np.mean(X, axis=0)
    mean_target = np.mean(Y, axis=0)

    # Hotelling's T² test (always computed)
    T2, F, pval_param, D = hotelling_t2(X, Y, ridge=ridge)

    # Effect size (always computed)
    g_total = hedges_g_total(X, Y)

    # Choose sampling method based on configuration
    if config.use_ultra_optimization and config.enable_numba_jit:
        # Ultra-optimized approach with all performance enhancements
        pval_perm, n_perm_used = ultra_optimized_permutation_test(X, Y, config, ridge=ridge)
        T2_ci_lo, T2_ci_hi, n_boot_used = ultra_optimized_bootstrap_ci(
            X, Y, config,
            lambda A, B: hotelling_t2(A, B, ridge)[0]
        )
        sampling_method = "ultra_optimized"
    elif config.use_adaptive_sampling:
        # Adaptive sampling approach (Lee & Young, 1996; PMC 2009)
        pval_perm, n_perm_used = adaptive_permutation_test(X, Y, config, ridge=ridge)
        T2_ci_lo, T2_ci_hi, n_boot_used = adaptive_bootstrap_ci(
            X, Y, config,
            lambda A, B: hotelling_t2(A, B, ridge)[0]
        )
        sampling_method = "adaptive"
    else:
        # Traditional fixed sampling approach
        pval_perm, n_perm_used = traditional_permutation_test(X, Y, config, ridge=ridge)
        T2_ci_lo, T2_ci_hi, n_boot_used = traditional_bootstrap_ci(
            X, Y, config,
            lambda A, B: hotelling_t2(A, B, ridge)[0]
        )
        sampling_method = "traditional"

    return {
        'bone': idx[0],
        'landmark_index': idx[1],
        'T2': T2,
        'F': F,
        'pval_param': pval_param,
        'pval_perm': pval_perm,
        'g_total': g_total,
        'D_mahal': D,
        'base_mean_x': mean_base[0],
        'base_mean_y': mean_base[1],
        'target_mean_x': mean_target[0],
        'target_mean_y': mean_target[1],
        'T2_CI_boot_lo': T2_ci_lo,
        'T2_CI_boot_hi': T2_ci_hi,
        'n_bootstrap_used': n_boot_used,
        'n_permutation_used': n_perm_used,
        'sampling_method': sampling_method
    }

def compare_groups_optimized(df: pd.DataFrame, group_col: str, base_label: str,
                           out_dir: str, config: IterationConfig, alpha: float = 0.05,
                           ridge: float = 1e-6, sex_tag: str = 'ALL',
                           age_tag: str = 'ALL') -> None:
    """Optimized group comparison with checkpointing and streaming output"""

    start_time = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # Check if already completed
    output_file = os.path.join(out_dir, 'point_stats.csv')
    checkpoint_file = os.path.join(out_dir, 'checkpoint.pkl')

    if os.path.exists(output_file):
        logger.info(f"Analysis already completed: {out_dir}")
        return

    # Get target groups
    labels = sorted([l for l in df[group_col].dropna().unique() if l != base_label])
    if base_label not in df[group_col].unique():
        logger.warning(f"Base label '{base_label}' not found in {group_col}")
        return

    all_results = []
    completed_comparisons = 0

    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                all_results = checkpoint_data.get('results', [])
                completed_comparisons = checkpoint_data.get('completed', 0)
            logger.info(f"Resumed from checkpoint: {completed_comparisons} comparisons completed")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    for target_idx, target in enumerate(labels):
        if target_idx < completed_comparisons:
            continue  # Skip already completed

        logger.info(f"Processing {base_label} vs {target} ({target_idx+1}/{len(labels)})")

        # Get sample IDs for each group
        base_sample_ids = df.loc[df[group_col] == base_label, 'sample_id'].unique()
        target_sample_ids = df.loc[df[group_col] == target, 'sample_id'].unique()

        n_base, n_target = len(base_sample_ids), len(target_sample_ids)

        # Check minimum sample size (FIXED: was n_base<2 or n_base<2)
        if n_base < 2 or n_target < 2:
            logger.warning(f"Insufficient samples: {base_label}={n_base}, {target}={n_target}")
            continue

        # Prepare data structures
        df_base = df[df['sample_id'].isin(base_sample_ids)].set_index(['ap', 'landmark_index'])
        df_target = df[df['sample_id'].isin(target_sample_ids)].set_index(['ap', 'landmark_index'])

        coords_base = df_base.groupby(level=['ap', 'landmark_index'])[['x', 'y']]
        coords_target = df_target.groupby(level=['ap', 'landmark_index'])[['x', 'y']]

        # Find common landmark indices
        common_indices = sorted(list(
            set(coords_base.groups.keys()).intersection(set(coords_target.groups.keys()))
        ))

        # Prepare data pairs for processing
        data_pairs = []
        for idx in common_indices:
            X_data = coords_base.get_group(idx).to_numpy()
            Y_data = coords_target.get_group(idx).to_numpy()

            if X_data.shape[0] >= 2 and Y_data.shape[0] >= 2:
                data_pairs.append({
                    'index': idx,
                    'X': X_data,
                    'Y': Y_data
                })

        if not data_pairs:
            logger.warning(f"No valid landmark pairs for {base_label} vs {target}")
            continue

        # Process landmarks in parallel batches
        logger.info(f"Processing {len(data_pairs)} landmarks...")
        n_jobs = get_optimal_n_jobs()

        landmark_results = Parallel(n_jobs=n_jobs, batch_size=BATCH_SIZE)(
            delayed(process_landmark_comparison)(pair, config, ridge)
            for pair in data_pairs
        )

        if not landmark_results:
            continue

        # Multiple comparison correction
        pvals_param = np.array([res['pval_param'] for res in landmark_results], dtype=float)
        pvals_perm = np.array([res['pval_perm'] for res in landmark_results], dtype=float)

        # FDR correction for parametric p-values
        mask_param = ~np.isnan(pvals_param)
        if mask_param.sum() > 0:
            corrected_param = np.full(len(pvals_param), float('nan'))
            _, q_param, _, _ = multipletests(pvals_param[mask_param], method='fdr_bh')
            corrected_param[mask_param] = q_param

            for i, res in enumerate(landmark_results):
                res['pval_param_fdr'] = corrected_param[i]

        # FDR correction for permutation p-values
        mask_perm = ~np.isnan(pvals_perm)
        if mask_perm.sum() > 0:
            corrected_perm = np.full(len(pvals_perm), float('nan'))
            _, q_perm, _, _ = multipletests(pvals_perm[mask_perm], method='fdr_bh')
            corrected_perm[mask_perm] = q_perm

            for i, res in enumerate(landmark_results):
                res['pval_perm_fdr'] = corrected_perm[i]

        # Add metadata
        for res in landmark_results:
            res.update({
                'sex': sex_tag,
                'age_group': age_tag,
                'comparison': f"{base_label}_vs_{target}",
                'n_base': n_base,
                'n_target': n_target,
            })

        all_results.extend(landmark_results)
        completed_comparisons = target_idx + 1

        # Save checkpoint
        if completed_comparisons % CHECKPOINT_INTERVAL == 0:
            checkpoint_data = {
                'results': all_results,
                'completed': completed_comparisons,
                'timestamp': time.time()
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"Checkpoint saved: {completed_comparisons}/{len(labels)} completed")

    # Save final results
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Column ordering
        cols = [
            'sex', 'age_group', 'comparison', 'n_base', 'n_target',
            'bone', 'landmark_index', 'T2', 'F',
            'pval_param', 'pval_param_fdr', 'pval_perm', 'pval_perm_fdr',
            'g_total', 'D_mahal',
            'base_mean_x', 'base_mean_y', 'target_mean_x', 'target_mean_y',
            'T2_CI_boot_lo', 'T2_CI_boot_hi',
            'n_bootstrap_used', 'n_permutation_used'
        ]

        results_df = results_df[[c for c in cols if c in results_df.columns]]
        results_df.to_csv(output_file, index=False)

        # Remove checkpoint file
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)

        elapsed = time.time() - start_time
        logger.info(f"Completed {out_dir} in {elapsed:.1f}s with {len(all_results)} comparisons")
    else:
        logger.warning(f"No results generated for {out_dir}")

def run_stratified_analysis(input_path: str, out_dir: str, config: IterationConfig,
                          alpha: float = 0.05, ridge: float = 1e-6) -> None:
    """Run complete stratified analysis with all sex × age combinations"""

    logger.info("Starting stratified foot disease analysis...")
    logger.info(f"Iteration config: {config}")

    os.makedirs(out_dir, exist_ok=True)

    # Load and prepare data
    if input_path.endswith('.feather'):
        df = pd.read_feather(input_path)
    else:
        df = pd.read_csv(input_path)

    df = ensure_labels(df)

    # Save feature layout for downstream PCA analysis
    features = []
    aps = sorted(df['ap'].dropna().unique(),
                key=lambda x: (int(x[2:]) if str(x).startswith('AP') and str(x)[2:].isdigit() else 99, str(x)))

    for ap in aps:
        sub = df[df['ap'] == ap]
        max_landmark = int(sub['landmark_index'].max())
        for j in range(max_landmark + 1):
            features.append((ap, j, 'x'))
            features.append((ap, j, 'y'))

    with open(os.path.join(out_dir, 'feature_layout.json'), 'w', encoding='utf-8') as f:
        json.dump({'features': features}, f, ensure_ascii=False, indent=2)

    # Define stratification combinations
    sexes = ['ALL'] + sorted([s for s in df['sex'].dropna().unique() if s in ('M', 'F')])
    ages = ['ALL'] + sorted([a for a in df['age_bin'].dropna().unique() if a != 'Unknown'])

    strata_index = []

    # Process each stratum
    for sex in sexes:
        for age in ages:
            # Create stratified subset
            sub_df = df.copy()
            if sex != 'ALL':
                sub_df = sub_df[sub_df['sex'] == sex]
            if age != 'ALL':
                sub_df = sub_df[sub_df['age_bin'] == age]

            # Check minimum subjects
            n_subjects = sub_df['subject_id'].nunique()
            if n_subjects < 4:
                logger.warning(f"Skipping {sex}×{age}: insufficient subjects ({n_subjects})")
                continue

            # Create output directory
            tag = "_".join([t for t in [sex, age] if t != 'ALL'])
            if not tag:
                tag = 'ALL'

            # Disease analysis
            disease_dir = os.path.join(out_dir, tag, 'by_disease')
            strata_index.append({
                'sex': sex, 'age_group': age, 'analysis': 'by_disease',
                'path': disease_dir, 'n_subjects': n_subjects
            })

            logger.info(f"Analyzing diseases: {tag} (n={n_subjects})")
            compare_groups_optimized(
                sub_df, group_col='label_disease', base_label='normal',
                out_dir=disease_dir, config=config, alpha=alpha, ridge=ridge,
                sex_tag=sex, age_tag=age
            )

            # Category analysis
            category_dir = os.path.join(out_dir, tag, 'by_category')
            strata_index.append({
                'sex': sex, 'age_group': age, 'analysis': 'by_category',
                'path': category_dir, 'n_subjects': n_subjects
            })

            logger.info(f"Analyzing categories: {tag} (n={n_subjects})")
            compare_groups_optimized(
                sub_df, group_col='label_category', base_label='normal',
                out_dir=category_dir, config=config, alpha=alpha, ridge=ridge,
                sex_tag=sex, age_tag=age
            )

    # Save strata summary
    pd.DataFrame(strata_index).to_csv(os.path.join(out_dir, "strata_index.csv"), index=False)
    logger.info(f"Analysis completed. Results saved to: {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-level statistical analysis for foot disease data')
    parser.add_argument('--input', type=str, required=True, help='Input aligned data (feather/csv)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--alpha', type=float, default=0.05, help='Significance level')
    parser.add_argument('--ridge', type=float, default=1e-6, help='Ridge regularization')
    parser.add_argument('--bootstrap', type=int, default=10000, help='Standard bootstrap iterations')
    parser.add_argument('--permutation', type=int, default=10000, help='Standard permutation iterations')
    parser.add_argument('--precise-bootstrap', type=int, default=50000, help='High-precision bootstrap')
    parser.add_argument('--precise-permutation', type=int, default=100000, help='High-precision permutation')

    # Optimization options
    parser.add_argument('--use-adaptive', action='store_true', default=True, help='Use adaptive sampling (default: True)')
    parser.add_argument('--use-traditional', action='store_true', help='Force traditional fixed sampling')
    parser.add_argument('--ultra-optimize', action='store_true', default=True, help='Enable ultra-optimization (default: True)')
    parser.add_argument('--disable-numba', action='store_true', help='Disable Numba JIT compilation')
    parser.add_argument('--disable-vectorization', action='store_true', help='Disable vectorized operations')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of parallel jobs (-1 for auto)')

    args = parser.parse_args()

    # Create iteration configuration with optimization settings
    config = IterationConfig(
        use_adaptive_sampling=not args.use_traditional,
        use_ultra_optimization=args.ultra_optimize and not args.disable_numba,
        enable_numba_jit=not args.disable_numba,
        enable_vectorization=not args.disable_vectorization,
        enable_memory_management=True,
        enable_early_stopping=not args.use_traditional,
        standard_bootstrap=args.bootstrap,
        standard_permutation=args.permutation,
        precise_bootstrap=args.precise_bootstrap,
        precise_permutation=args.precise_permutation
    )

    # Log optimization settings
    logger.info("Optimization Configuration:")
    logger.info(f"  Sampling method: {'Ultra-optimized' if config.use_ultra_optimization else 'Adaptive' if config.use_adaptive_sampling else 'Traditional'}")
    logger.info(f"  Numba JIT: {'Enabled' if config.enable_numba_jit else 'Disabled'}")
    logger.info(f"  Vectorization: {'Enabled' if config.enable_vectorization else 'Disabled'}")
    logger.info(f"  Parallel jobs: {args.n_jobs if args.n_jobs > 0 else 'Auto-detected'}")
    logger.info(f"  Memory management: {'Enabled' if config.enable_memory_management else 'Disabled'}")

    # Run analysis
    run_stratified_analysis(
        input_path=args.input,
        out_dir=args.output,
        config=config,
        alpha=args.alpha,
        ridge=args.ridge
    )