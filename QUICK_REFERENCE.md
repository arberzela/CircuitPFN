# Causality Lab Integration - Quick Reference

## Installation

```bash
pip install git+https://github.com/IntelLabs/causality-lab.git
```

## Basic Usage

### Import
```python
from baselines import run_rai, run_fci, run_icd, run_brai
from attn_scm.core import AttnSCM
```

### Run Algorithms
```python
import numpy as np

# Your data
X = np.random.randn(500, 20)  # 500 samples, 20 features
y = np.random.randint(0, 2, 500)
feature_names = [f"X{i}" for i in range(20)]

# AttnSCM (attention-based)
model = AttnSCM(top_k_heads=5)
adj = model.fit(X, y)

# RAI (fast, assumes no confounders)
adj_rai = run_rai(X, alpha=0.05, feature_names=feature_names)

# FCI (handles confounders)
adj_fci = run_fci(X, alpha=0.05, feature_names=feature_names)

# ICD (handles confounders, iterative)
adj_icd = run_icd(X, alpha=0.05, feature_names=feature_names)

# B-RAI (uncertainty estimation)
adj_brai, edge_probs = run_brai(X, n_bootstrap=100, feature_names=feature_names)
```

## TabPFN-Enhanced Discovery

```python
from baselines.tabpfn_causality_adapter import CondIndepTabPFN
from causality_lab.learn_structure import LearnStructRAI
from causality_lab.data import Dataset

# Create TabPFN-based CI test
dataset = Dataset(X, var_names=feature_names)
ci_test = CondIndepTabPFN(dataset, threshold=0.05)

# Run RAI with TabPFN test
rai = LearnStructRAI(set(feature_names), ci_test)
rai.learn_structure()
```

## Run Experiments

```bash
# Extended benchmark (7 algorithms)
python experiments/exp_a_causality_lab.py --quick_test

# TabPFN enhancement study
python experiments/exp_e_tabpfn_enhanced_causality.py --quick_test

# Real-world datasets
python experiments/exp_f_real_datasets.py --datasets credit-g diabetes
```

## Evaluate Results

```python
from attn_scm.metrics import compute_graph_metrics

# Compare with ground truth
metrics = compute_graph_metrics(predicted_adj, true_adj)
print(f"F1 Score: {metrics['f1_directed']:.3f}")
print(f"SHD: {metrics['shd']}")
```

## Algorithm Selection Guide

| Use Case | Algorithm | Why |
|----------|-----------|-----|
| Fast discovery, no confounders | RAI | O(d²), assumes causal sufficiency |
| Suspected hidden variables | FCI or ICD | Handle latent confounders |
| Need confidence intervals | B-RAI | Bootstrap uncertainty |
| Zero-shot from TabPFN | AttnSCM | Leverages foundation model |
| Traditional benchmark | PC | Standard constraint-based |

## Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.05 | Significance level for CI tests |
| `indep_test` | 'parcorr' | CI test type ('parcorr' or 'cmi') |
| `feature_names` | None | Optional feature labels |
| `top_k_heads` | 5 | Number of attention heads (AttnSCM) |
| `n_bootstrap` | 100 | Bootstrap samples (B-RAI) |

## Troubleshooting

**Error**: `ImportError: No module named 'causality_lab'`
```bash
pip install git+https://github.com/IntelLabs/causality-lab.git
```

**Error**: TabPFN out of memory
```python
# Use CPU or reduce data size
adj = run_rai(X[:500, :50], device='cpu')
```

**Warning**: Causality-lab not available
- Code will run without causality-lab algorithms
- Install causality-lab to enable RAI, FCI, ICD

## File Locations

- **Wrappers**: `baselines/causality_lab_algorithms.py`
- **CI tests**: `baselines/tabpfn_causality_adapter.py`
- **Experiments**: `experiments/exp_[a|e|f]_*.py`
- **Docs**: `CAUSALITY_LAB_INTEGRATION.md`
- **Tutorial**: `notebooks/causality_lab_quickstart.ipynb`

## Quick Test

```python
# Verify installation
from baselines import CAUSALITY_LAB_AVAILABLE
print(f"Causality Lab: {CAUSALITY_LAB_AVAILABLE}")

# Test basic function
if CAUSALITY_LAB_AVAILABLE:
    from baselines import run_rai
    X = np.random.randn(100, 10)
    adj = run_rai(X, feature_names=[f"X{i}" for i in range(10)])
    print(f"Discovered {adj.sum()} edges")
```

## More Information

- Full documentation: [CAUSALITY_LAB_INTEGRATION.md](CAUSALITY_LAB_INTEGRATION.md)
- Integration summary: [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)
- Interactive tutorial: `notebooks/causality_lab_quickstart.ipynb`
