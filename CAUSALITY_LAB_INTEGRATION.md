# Causality Lab Integration Guide

This document describes the integration of Intel Labs' [causality-lab](https://github.com/IntelLabs/causality-lab) algorithms with the TabPFN-based CircuitPFN codebase for causal discovery on tabular data.

## Overview

This integration enables you to:

1. **Apply causality-lab algorithms to TabPFN downstream tasks** - Use algorithms like RAI, FCI, and ICD on the same tabular datasets that TabPFN processes
2. **Compare causal discovery methods** - Benchmark causality-lab algorithms against AttnSCM (attention-based) and traditional methods (PC, NOTEARS)
3. **Enhance causal discovery with TabPFN** - Use TabPFN's learned representations and attention patterns to improve conditional independence testing
4. **Evaluate on real-world datasets** - Test causal discovery on OpenML tabular datasets

## Installation

### 1. Install Base Requirements

```bash
pip install -r requirements.txt
```

### 2. Install Causality Lab

```bash
pip install git+https://github.com/IntelLabs/causality-lab.git
```

### 3. Verify Installation

```python
from baselines import CAUSALITY_LAB_AVAILABLE
print(f"Causality Lab available: {CAUSALITY_LAB_AVAILABLE}")
```

## Architecture

### New Modules

#### 1. `baselines/causality_lab_algorithms.py`

Wrappers for causality-lab algorithms that conform to the project's API:

- **`run_pc_causality_lab()`** - PC algorithm (causality-lab implementation)
- **`run_rai()`** - Recursive Autonomy Identification
- **`run_brai()`** - Bootstrap/Bayesian RAI with uncertainty estimation
- **`run_fci()`** - Fast Causal Inference (handles latent confounders)
- **`run_icd()`** - Iterative Causal Discovery (handles latent confounders)

All functions take:
- `X: np.ndarray` - Data matrix (n_samples, n_features)
- `alpha: float` - Significance level (default: 0.05)
- `indep_test: str` - Independence test ('parcorr' or 'cmi')
- `feature_names: Optional[list]` - Feature names

And return:
- `np.ndarray` - Binary adjacency matrix (n_features, n_features)

#### 2. `baselines/tabpfn_causality_adapter.py`

Custom conditional independence tests that leverage TabPFN:

- **`CondIndepTabPFN`** - Uses TabPFN predictions to test conditional independence
  - Trains TabPFN to predict X from Z vs. Z∪{Y}
  - Improvement in accuracy indicates dependence

- **`CondIndepAttentionWeighted`** - Weights standard partial correlation tests by TabPFN attention
  - Combines statistical tests with learned attention patterns
  - Prioritizes relationships TabPFN considers important

### New Experiments

#### Experiment A Extended: `exp_a_causality_lab.py`

Extends the original synthetic benchmark to include causality-lab algorithms.

**Usage:**
```bash
# Full benchmark (100 datasets, 3 SCM types)
python experiments/exp_a_causality_lab.py

# Quick test (10 datasets, 1 SCM type)
python experiments/exp_a_causality_lab.py --quick_test

# Without causality-lab (if not installed)
python experiments/exp_a_causality_lab.py --no_causality_lab
```

**Methods compared:**
- AttnSCM (our method)
- PC Algorithm
- NOTEARS
- RAI (causality-lab)
- FCI (causality-lab)
- ICD (causality-lab)
- Random baseline

**Metrics:**
- Structural Hamming Distance (SHD)
- F1 Score (directed and undirected)
- Precision/Recall
- Elapsed time

#### Experiment E: `exp_e_tabpfn_enhanced_causality.py`

Novel experiment showing how TabPFN can enhance traditional causal discovery.

**Usage:**
```bash
# Full experiment
python experiments/exp_e_tabpfn_enhanced_causality.py

# Quick test
python experiments/exp_e_tabpfn_enhanced_causality.py --quick_test
```

**Methods compared:**
- **RAI-Standard** - Standard RAI with partial correlation test
- **RAI-TabPFN** - RAI using `CondIndepTabPFN` (TabPFN-based testing)
- **RAI-Attention** - RAI using `CondIndepAttentionWeighted` (attention-weighted tests)
- **AttnSCM** - Direct attention map decoding

**Key Question:** Does TabPFN's learned representation improve causal discovery?

#### Experiment F: `exp_f_real_datasets.py`

Causal discovery on real-world OpenML tabular datasets.

**Usage:**
```bash
# All datasets
python experiments/exp_f_real_datasets.py

# Specific datasets
python experiments/exp_f_real_datasets.py --datasets credit-g diabetes
```

**Datasets:**
- credit-g (Credit Approval)
- diabetes (Pima Indians Diabetes)
- blood-transfusion (Blood Donation)
- phoneme (Phoneme Classification)

**Evaluation:**
Since ground truth is unknown, we evaluate:
- Graph structure properties (sparsity, degree distribution)
- Method agreement (edge overlap)
- Predictive utility via Markov Blanket feature selection

## Usage Examples

### Basic Usage

```python
import numpy as np
from baselines import run_rai, run_fci, run_icd
from utils import generate_synthetic_dataset

# Generate synthetic data
X, y, true_adj = generate_synthetic_dataset(
    n_nodes=20,
    n_samples=500,
    scm_type='linear_gaussian'
)

# Run RAI
adj_rai = run_rai(X, alpha=0.05)

# Run FCI (handles latent confounders)
adj_fci = run_fci(X, alpha=0.05)

# Run ICD
adj_icd = run_icd(X, alpha=0.05)

# Compare with ground truth
from attn_scm.metrics import compute_graph_metrics
metrics_rai = compute_graph_metrics(adj_rai, true_adj)
print(f"RAI F1 Score: {metrics_rai['f1_directed']:.3f}")
```

### TabPFN-Enhanced Causal Discovery

```python
from baselines.tabpfn_causality_adapter import CondIndepTabPFN
from causality_lab.learn_structure import LearnStructRAI
from causality_lab.data import Dataset

# Create dataset
feature_names = [f"X{i}" for i in range(X.shape[1])]
dataset = Dataset(X, var_names=feature_names)

# Create TabPFN-based conditional independence test
cond_indep_test = CondIndepTabPFN(
    dataset,
    threshold=0.05,
    device='cpu'
)

# Run RAI with TabPFN test
nodes_set = set(feature_names)
rai_learner = LearnStructRAI(nodes_set, cond_indep_test)
rai_learner.learn_structure()

# Access the learned graph
graph = rai_learner.graph
```

### Attention-Weighted Causal Discovery

```python
from attn_scm.core import AttnSCM
from baselines.tabpfn_causality_adapter import CondIndepAttentionWeighted

# Extract attention patterns using AttnSCM
model = AttnSCM(top_k_heads=5)
model.fit(X, y)
attention_matrix = model.get_adjacency_matrix(binarize=False)

# Use attention to weight conditional independence tests
dataset = Dataset(X, var_names=feature_names)
cond_indep_test = CondIndepAttentionWeighted(
    dataset,
    threshold=0.05,
    attention_weights=attention_matrix
)

# Run causal discovery
rai_learner = LearnStructRAI(set(feature_names), cond_indep_test)
rai_learner.learn_structure()
```

### Uncertainty Estimation with B-RAI

```python
from baselines import run_brai

# Run B-RAI to get edge probabilities
adj, edge_probs = run_brai(
    X,
    alpha=0.05,
    n_bootstrap=100,
    feature_names=feature_names
)

# Identify confident edges
confident_edges = edge_probs > 0.7
print(f"Confident edges: {confident_edges.sum()}")
print(f"Total edges: {adj.sum()}")
```

## Algorithm Descriptions

### Causality Lab Algorithms

#### RAI (Recursive Autonomy Identification)
- **Assumption:** Causal sufficiency (no latent confounders)
- **Approach:** Recursively identifies autonomous variables (no parents)
- **Output:** DAG
- **Best for:** Well-specified systems without hidden variables

#### FCI (Fast Causal Inference)
- **Assumption:** Allows latent confounders
- **Approach:** Constraint-based with orientation rules
- **Output:** Partial Ancestral Graph (PAG)
- **Best for:** Real-world data where hidden variables are likely

#### ICD (Iterative Causal Discovery)
- **Assumption:** Allows latent confounders
- **Approach:** Iteratively discovers causal relationships and latent structure
- **Output:** Causal graph with latent variables identified
- **Best for:** Complex systems with suspected latent confounders

#### B-RAI (Bootstrap RAI)
- **Assumption:** Causal sufficiency
- **Approach:** Bootstrap sampling to estimate uncertainty
- **Output:** DAG + edge probabilities
- **Best for:** When you need confidence intervals on edge detection

### AttnSCM (Baseline)
- **Assumption:** TabPFN attention encodes causal structure
- **Approach:** Decode attention maps from pre-trained TabPFN
- **Output:** DAG
- **Best for:** Zero-shot causal discovery leveraging foundation model knowledge

## Results Organization

All experiments save results to `results/`:

```
results/
├── exp_a_causality_lab/       # Extended synthetic benchmark
│   ├── raw_results.csv        # Per-dataset results
│   ├── summary.csv            # Aggregated statistics
│   ├── shd_comparison_*.png   # SHD visualizations
│   └── f1_comparison_*.png    # F1 visualizations
├── exp_e_tabpfn_causality/    # TabPFN-enhanced discovery
│   ├── raw_results.csv
│   ├── summary.csv
│   └── f1_comparison_*.png
└── exp_f_real_datasets/       # Real-world datasets
    ├── results.csv
    └── summary.csv
```

## Performance Considerations

### Computational Complexity

| Method | Time Complexity | Space | Notes |
|--------|----------------|-------|-------|
| RAI | O(n² × d²) | Low | Fast for sparse graphs |
| FCI | O(n² × d³) | Medium | Slower due to orientation rules |
| ICD | O(n² × d²) | Medium | Iterative, may take longer |
| AttnSCM | O(n × d²) | High | TabPFN forward pass dominates |

Where:
- n = number of samples
- d = number of features

### Scalability

**TabPFN Constraints:**
- Max 1000 samples
- Max 100 features
- Automatically enforced in all wrappers

**Recommendations:**
- For d > 50: Use RAI (fastest)
- For suspected confounders: Use ICD or FCI
- For uncertainty estimates: Use B-RAI
- For zero-shot: Use AttnSCM

## Troubleshooting

### Import Errors

**Error:** `ImportError: No module named 'causality_lab'`

**Solution:**
```bash
pip install git+https://github.com/IntelLabs/causality-lab.git
```

### Graph Conversion Issues

**Error:** `AttributeError: 'DirectedEdge' object has no attribute 'source'`

**Solution:** Check causality-lab version. Our wrappers assume the latest API.

### TabPFN Memory Issues

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```python
# Use CPU instead
run_rai(X, device='cpu')

# Or reduce dataset size
X = X[:500, :50]  # Max 500 samples, 50 features
```

### Conditional Independence Test Failures

**Error:** `ValueError: Not enough data for conditional independence test`

**Solution:** Increase sample size or reduce feature set. Minimum ~100 samples recommended.

## Citation

If you use this integration in your research, please cite both:

**AttnSCM/CircuitPFN:**
```bibtex
@article{zela2024attnscm,
  title={Zero-Shot Causal Graph Extraction from Tabular Foundation Models via Attention Map Decoding},
  author={Zela, Arber and others},
  year={2024}
}
```

**Causality Lab:**
```bibtex
@misc{intellabs2023causalitylab,
  title={Causality Lab: Causal Discovery Algorithms and Tools},
  author={Intel Labs},
  year={2023},
  url={https://github.com/IntelLabs/causality-lab}
}
```

## Contributing

To add new causality-lab algorithms:

1. Create wrapper function in `baselines/causality_lab_algorithms.py`
2. Follow the signature: `run_algorithm(X, alpha, indep_test, feature_names, **kwargs) -> np.ndarray`
3. Add to `CAUSALITY_LAB_ALGORITHMS` dictionary
4. Update `baselines/__init__.py`
5. Add tests and documentation

## Contact

For issues related to:
- **Integration:** Open issue in this repository
- **Causality Lab algorithms:** https://github.com/IntelLabs/causality-lab/issues
- **TabPFN:** https://github.com/automl/TabPFN/issues
