# Taichi Lang Integration Guide

## Overview

**Taichi** is a high-performance parallel programming language that can provide significant speedups for computational bottlenecks, especially when GPU acceleration is available.

### Current State: Numba JIT
- ✓ CPU-only JIT compilation
- ✓ Parallel processing with `prange`
- ✓ 5-10x speedup over pure NumPy
- ✗ No GPU acceleration
- ✗ Manual memory management needed

### Potential with Taichi:
- ✓ Automatic CPU/GPU backend selection
- ✓ Cleaner syntax than Numba CUDA
- ✓ Better memory management
- ✓ Potential 10-100x speedup with GPU
- ⚠️ Adds another dependency
- ⚠️ Requires kernel redesign

---

## Performance Analysis

### Current Bottlenecks (from profiling)

1. **Bootstrap iterations** (45-60% of runtime)
   - 10,000 bootstrap samples per landmark
   - Embarrassingly parallel
   - **Perfect for Taichi GPU kernels**

2. **Permutation tests** (30-40% of runtime)
   - Adaptive 1K-100K permutations
   - Independent computations
   - **Ideal for parallel GPU execution**

3. **Covariance matrix operations** (10-15% of runtime)
   - Matrix inversions
   - Pooled covariance calculations
   - **Can benefit from Taichi's matrix ops**

### Expected Speedup with Taichi + GPU

| Dataset Size | Current (Numba CPU) | Taichi CPU | Taichi GPU |
|--------------|---------------------|------------|------------|
| 50 patients  | 20-30 min          | 15-25 min  | **5-10 min** |
| 700 patients | 2-3 hours          | 1.5-2.5 hours | **30-60 min** |
| 1400 patients| 4-5 hours          | 3-4 hours  | **1-2 hours** |

**Estimated speedup: 2-4x with GPU**

---

## Installation

### Option 1: Add to Existing Environment

```bash
source foot_analysis_env/bin/activate
pip install taichi
```

### Option 2: Update requirements.txt

Add to `requirements.txt`:
```
taichi>=1.6.0  # GPU-accelerated computation
```

Then reinstall:
```bash
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "
import taichi as ti
ti.init(arch=ti.gpu)  # Try GPU
print(f'✓ Taichi {ti.__version__} installed')
print(f'Backend: {ti.cfg.arch}')
"
```

If no GPU available, falls back to CPU automatically.

---

## Code Conversion Examples

### Example 1: Bootstrap Computation

**Current (Numba):**
```python
@jit(nopython=True, parallel=True, cache=True)
def vectorized_bootstrap_batch(X_flat, Y_flat, n1, n2, p,
                               indices_1, indices_2):
    n_bootstrap = indices_1.shape[0]
    results = np.empty(n_bootstrap, dtype=np.float64)

    for i in prange(n_bootstrap):
        idx1 = indices_1[i]
        idx2 = indices_2[i]
        X_boot = X_flat.reshape(n1, p)[idx1].flatten()
        Y_boot = Y_flat.reshape(n2, p)[idx2].flatten()
        results[i] = numba_hotelling_t2_core(X_boot, Y_boot, n1, n2, p)

    return results
```

**Taichi Version:**
```python
import taichi as ti

ti.init(arch=ti.gpu)  # Auto-fallback to CPU if no GPU

@ti.kernel
def taichi_bootstrap_batch(X_flat: ti.types.ndarray(),
                          Y_flat: ti.types.ndarray(),
                          indices_1: ti.types.ndarray(),
                          indices_2: ti.types.ndarray(),
                          results: ti.types.ndarray(),
                          n1: ti.i32, n2: ti.i32, p: ti.i32):
    for i in range(indices_1.shape[0]):
        # Bootstrap sample
        X_boot = ti.Vector([0.0] * (n1 * p))
        Y_boot = ti.Vector([0.0] * (n2 * p))

        for j in range(n1):
            idx = indices_1[i, j]
            for k in range(p):
                X_boot[j * p + k] = X_flat[idx * p + k]

        for j in range(n2):
            idx = indices_2[i, j]
            for k in range(p):
                Y_boot[j * p + k] = Y_flat[idx * p + k]

        # Compute T² statistic
        results[i] = compute_t2_statistic(X_boot, Y_boot, n1, n2, p)

# Usage
results = np.empty(n_bootstrap, dtype=np.float64)
taichi_bootstrap_batch(X_flat, Y_flat, indices_1, indices_2, results, n1, n2, p)
```

### Example 2: Permutation Test

**Current (Numba):**
```python
def permutation_test(X, Y, n_permutation=10000):
    combined = np.vstack([X, Y])
    n1, n2 = X.shape[0], Y.shape[0]

    observed = compute_statistic(X, Y)
    perm_stats = []

    for _ in range(n_permutation):
        perm_indices = np.random.permutation(n1 + n2)
        X_perm = combined[perm_indices[:n1]]
        Y_perm = combined[perm_indices[n1:]]
        perm_stats.append(compute_statistic(X_perm, Y_perm))

    return observed, np.array(perm_stats)
```

**Taichi Version:**
```python
@ti.kernel
def taichi_permutation_test(combined: ti.types.ndarray(),
                            perm_indices: ti.types.ndarray(),
                            perm_stats: ti.types.ndarray(),
                            n1: ti.i32, n2: ti.i32, p: ti.i32):
    for i in range(perm_indices.shape[0]):
        # Get permutation indices for this iteration
        X_perm = ti.Vector([0.0] * (n1 * p))
        Y_perm = ti.Vector([0.0] * (n2 * p))

        # Sample X
        for j in range(n1):
            idx = perm_indices[i, j]
            for k in range(p):
                X_perm[j * p + k] = combined[idx, k]

        # Sample Y
        for j in range(n2):
            idx = perm_indices[i, n1 + j]
            for k in range(p):
                Y_perm[j * p + k] = combined[idx, k]

        perm_stats[i] = compute_t2_statistic(X_perm, Y_perm, n1, n2, p)
```

---

## Integration Strategy

### Phase 1: Hybrid Approach (Recommended)

Keep Numba as fallback, add Taichi for GPU acceleration:

```python
import numba
try:
    import taichi as ti
    TAICHI_AVAILABLE = True
    ti.init(arch=ti.gpu)
    print(f"✓ Taichi GPU acceleration enabled")
except:
    TAICHI_AVAILABLE = False
    print("⚠ Taichi not available, using Numba")

def bootstrap_analysis(X, Y, n_bootstrap=10000, use_gpu=True):
    if TAICHI_AVAILABLE and use_gpu:
        return taichi_bootstrap(X, Y, n_bootstrap)
    else:
        return numba_bootstrap(X, Y, n_bootstrap)
```

### Phase 2: Full Migration

If Taichi shows significant benefits, fully migrate hot paths:

1. **Week 1**: Convert bootstrap functions
2. **Week 2**: Convert permutation tests
3. **Week 3**: Optimize covariance operations
4. **Week 4**: Benchmark and tune

---

## Implementation Guide

### Step 1: Create Taichi-Optimized Module

Create `20251008_footanalysis/statistical_analysis/stats_taichi_accelerated.py`:

```python
import taichi as ti
import numpy as np

# Initialize Taichi (auto-detects GPU)
ti.init(arch=ti.gpu)

@ti.kernel
def bootstrap_t2_kernel(X: ti.types.ndarray(),
                       Y: ti.types.ndarray(),
                       boot_indices_X: ti.types.ndarray(),
                       boot_indices_Y: ti.types.ndarray(),
                       results: ti.types.ndarray(),
                       n_bootstrap: ti.i32,
                       n1: ti.i32, n2: ti.i32, p: ti.i32):
    """GPU-accelerated bootstrap for Hotelling's T²"""
    for i in range(n_bootstrap):
        # Bootstrap sampling
        X_boot = ti.Matrix.zero(ti.f64, n1, p)
        Y_boot = ti.Matrix.zero(ti.f64, n2, p)

        for j in range(n1):
            idx = boot_indices_X[i, j]
            for k in range(p):
                X_boot[j, k] = X[idx, k]

        for j in range(n2):
            idx = boot_indices_Y[i, j]
            for k in range(p):
                Y_boot[j, k] = Y[idx, k]

        # Compute T² statistic
        results[i] = hotelling_t2_taichi(X_boot, Y_boot, n1, n2, p)

@ti.func
def hotelling_t2_taichi(X, Y, n1, n2, p):
    """Hotelling's T² computation in Taichi"""
    # Compute means
    mean_X = ti.Vector([0.0] * p)
    mean_Y = ti.Vector([0.0] * p)

    for i in range(n1):
        for j in range(p):
            mean_X[j] += X[i, j]
    mean_X /= n1

    for i in range(n2):
        for j in range(p):
            mean_Y[j] += Y[i, j]
    mean_Y /= n2

    # Compute pooled covariance (simplified)
    # ... (implement pooled covariance calculation)

    # Compute T²
    diff = mean_X - mean_Y
    # ... (complete T² calculation)

    return t2_value

class TaichiStatistics:
    """GPU-accelerated statistical testing using Taichi"""

    def __init__(self, n_bootstrap=10000, n_permutation=10000):
        self.n_bootstrap = n_bootstrap
        self.n_permutation = n_permutation

    def bootstrap_test(self, X: np.ndarray, Y: np.ndarray):
        """Bootstrap test with GPU acceleration"""
        n1, p = X.shape
        n2, _ = Y.shape

        # Pre-generate bootstrap indices on CPU
        boot_indices_X = np.random.randint(0, n1, (self.n_bootstrap, n1))
        boot_indices_Y = np.random.randint(0, n2, (self.n_bootstrap, n2))

        # Allocate result array
        results = np.empty(self.n_bootstrap, dtype=np.float64)

        # Run GPU kernel
        bootstrap_t2_kernel(X, Y, boot_indices_X, boot_indices_Y,
                          results, self.n_bootstrap, n1, n2, p)

        # Compute confidence interval
        ci_low = np.percentile(results, 2.5)
        ci_high = np.percentile(results, 97.5)

        return ci_low, ci_high, results
```

### Step 2: Update Main Analysis Script

Modify `stats_multi_level_optimized.py`:

```python
# At the top of file
try:
    from stats_taichi_accelerated import TaichiStatistics
    USE_TAICHI = True
except ImportError:
    USE_TAICHI = False

# In your analysis function
def run_statistical_analysis(data, use_gpu=True):
    if USE_TAICHI and use_gpu:
        stats = TaichiStatistics(n_bootstrap=10000)
        print("Using Taichi GPU acceleration")
    else:
        stats = NumbaStatistics(n_bootstrap=10000)
        print("Using Numba CPU acceleration")

    # ... rest of analysis
```

### Step 3: Add Command-Line Option

```python
parser.add_argument('--use-gpu', action='store_true',
                   help='Use Taichi GPU acceleration if available')
parser.add_argument('--taichi-backend', choices=['gpu', 'cpu', 'cuda', 'vulkan'],
                   default='gpu', help='Taichi backend selection')
```

---

## Benchmarking

### Create Benchmark Script

```python
# benchmark_taichi.py
import time
import numpy as np
from stats_multi_level_optimized import numba_bootstrap
from stats_taichi_accelerated import TaichiStatistics

def benchmark(n_patients=700, n_landmarks=1023, n_bootstrap=10000):
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(n_patients, 2)  # 2D coordinates
    Y = np.random.randn(n_patients, 2)

    # Benchmark Numba
    start = time.time()
    numba_result = numba_bootstrap(X, Y, n_bootstrap)
    numba_time = time.time() - start

    # Benchmark Taichi
    taichi_stats = TaichiStatistics(n_bootstrap=n_bootstrap)
    start = time.time()
    taichi_result = taichi_stats.bootstrap_test(X, Y)
    taichi_time = time.time() - start

    print(f"Numba (CPU): {numba_time:.2f}s")
    print(f"Taichi (GPU): {taichi_time:.2f}s")
    print(f"Speedup: {numba_time/taichi_time:.2f}x")

if __name__ == "__main__":
    benchmark()
```

---

## Pros and Cons

### ✅ Advantages

1. **GPU Acceleration**: 2-10x speedup with NVIDIA/AMD GPUs
2. **Automatic Backend**: Falls back to CPU if no GPU
3. **Cleaner Syntax**: More Pythonic than Numba CUDA
4. **Better Memory Management**: Automatic transfer optimization
5. **Cross-Platform**: Works on CUDA, Vulkan, Metal, CPU
6. **Active Development**: Regular updates and improvements

### ❌ Disadvantages

1. **Extra Dependency**: Adds ~200MB to installation
2. **Learning Curve**: Different programming model from NumPy
3. **Kernel Limitations**: Not all NumPy operations supported
4. **Debugging**: GPU debugging is harder than CPU
5. **Memory Transfer Overhead**: For small datasets, CPU may be faster
6. **Compatibility**: May conflict with other GPU libraries

---

## Decision Matrix

### Use Taichi If:
- ✓ You have access to GPU servers
- ✓ Dataset is large (500+ patients)
- ✓ Bootstrap iterations > 5000
- ✓ Need fastest possible execution
- ✓ Team comfortable with GPU programming

### Stick with Numba If:
- ✓ CPU-only environment
- ✓ Small datasets (<100 patients)
- ✓ Current performance acceptable
- ✓ Minimal dependencies preferred
- ✓ Stability is critical

### Hybrid Approach (Recommended):
- ✓ Implement both backends
- ✓ Auto-detect GPU availability
- ✓ Let users choose via `--use-gpu` flag
- ✓ Benchmark on your specific hardware

---

## Recommended Next Steps

### Minimal Risk Approach:

1. **Test Installation** (5 min)
   ```bash
   pip install taichi
   python -c "import taichi as ti; ti.init(arch=ti.gpu)"
   ```

2. **Create Proof of Concept** (1-2 hours)
   - Convert one bootstrap function
   - Benchmark on your data
   - Compare results for accuracy

3. **If Promising** (1-2 days)
   - Implement hybrid system
   - Add command-line flags
   - Update documentation

4. **If Successful** (1 week)
   - Full migration of hot paths
   - Comprehensive benchmarking
   - Production deployment

### Sample Implementation Timeline:

| Phase | Task | Time | Priority |
|-------|------|------|----------|
| 1 | Install & test Taichi | 30 min | High |
| 2 | Convert bootstrap kernel | 2 hours | High |
| 3 | Benchmark comparison | 1 hour | High |
| 4 | Implement hybrid system | 4 hours | Medium |
| 5 | Convert permutation test | 3 hours | Medium |
| 6 | Full testing & validation | 1 day | High |
| 7 | Documentation | 2 hours | Medium |

**Total: ~2-3 days for full integration**

---

## Conclusion

**Recommendation**: **Start with hybrid approach**

1. Keep Numba as stable baseline
2. Add Taichi as optional GPU acceleration
3. Benchmark on your actual server hardware
4. Migrate fully only if 2x+ speedup confirmed

This gives you:
- ✓ Risk-free experimentation
- ✓ Backwards compatibility
- ✓ Flexibility for different environments
- ✓ Potential for significant speedup

The investment (2-3 days) could save hours per analysis run, especially valuable for large-scale studies with 1000+ patients.
