# Attn-SCM: Zero-Shot Causal Graph Extraction from TabPFN

This repository implements the **Attn-SCM** framework for extracting Structural Causal Models (SCMs) from pre-trained TabPFN models through attention map decoding.

## Overview

Attn-SCM leverages the self-attention mechanisms in TabPFN (Tabular Prior-Data Fitted Networks) to perform zero-shot causal discovery. By analyzing the attention patterns learned during meta-training on synthetic causal datasets, we can extract directed acyclic graphs (DAGs) representing causal relationships without additional training.

## Key Features

- **Zero-shot causal discovery**: Extract causal graphs without training on target data
- **Structural head identification**: Entropy-based filtering to isolate causally-relevant attention heads
- **Fast inference**: Orders of magnitude faster than traditional causal discovery algorithms
- **Comprehensive experiments**: Four validation experiments demonstrating accuracy, localization, utility, and efficiency
- **Intel Labs Causality Lab Integration**: Apply state-of-the-art causal discovery algorithms (RAI, FCI, ICD) to TabPFN downstream tasks (see [CAUSALITY_LAB_INTEGRATION.md](CAUSALITY_LAB_INTEGRATION.md))

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/attn-scm.git
cd attn-scm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Optional: Install Intel Labs Causality Lab for additional algorithms
pip install git+https://github.com/IntelLabs/causality-lab.git
```

## Quick Start

```python
from attn_scm.core import AttnSCM
import numpy as np

# Load your tabular data
X = np.random.randn(100, 10)  # 100 samples, 10 features
y = np.random.randint(0, 2, 100)  # Binary target

# Extract causal graph
model = AttnSCM(top_k_heads=5, threshold_method='otsu')
adjacency_matrix = model.fit(X, y)

# Get Markov Blanket of target variable
mb_features = model.get_markov_blanket(target_idx=0)
print(f"Markov Blanket features: {mb_features}")
```

## Experiments

This repository includes four key experiments from the paper:

### Experiment A: Synthetic Benchmark Recovery

Tests topological accuracy on 100 synthetic SCMs with varying mechanisms:

```bash
python experiments/exp_a_synthetic.py --num_datasets 100 --n_samples 500 --n_features 20
```

**Metrics**: Structural Hamming Distance (SHD), F1-Score  
**Baselines**: PC Algorithm, NOTEARS, Random

### Experiment B: Layer-Wise Localization

Validates the "Mid-Layer Hypothesis" by testing graph extraction across all TabPFN layers:

```bash
python experiments/exp_b_layers.py --num_datasets 50
```

**Expected**: U-shaped error curve with minimum at mid-layers (4-7)

### Experiment C: Feature Selection

Demonstrates practical utility via Markov Blanket-based feature selection:

```bash
python experiments/exp_c_feature_selection.py --datasets credit-g phoneme adult blood-transfusion diabetes
```

**Metrics**: AUC, Feature Reduction Ratio (FRR)

### Experiment D: Efficiency Profiling

Compares computational efficiency against traditional methods:

```bash
python experiments/exp_d_efficiency.py --max_n 1000 --max_d 50
```

**Expected**: Constant/linear scaling vs exponential for PC algorithm

### NEW: Causality Lab Integration Experiments

Apply Intel Labs causality-lab algorithms to TabPFN downstream tasks:

#### Experiment A Extended: Causality Lab Benchmark
```bash
# Compare AttnSCM with RAI, FCI, ICD on synthetic data
python experiments/exp_a_causality_lab.py --num_datasets 100
```

#### Experiment E: TabPFN-Enhanced Causal Discovery
```bash
# Test if TabPFN attention patterns improve traditional causal discovery
python experiments/exp_e_tabpfn_enhanced_causality.py --num_datasets 50
```

#### Experiment F: Real-World Datasets
```bash
# Apply causal discovery to OpenML tabular datasets
python experiments/exp_f_real_datasets.py --datasets credit-g diabetes
```

**See [CAUSALITY_LAB_INTEGRATION.md](CAUSALITY_LAB_INTEGRATION.md) for detailed documentation.**

## Repository Structure

```
attn_scm/           # Core implementation
├── core.py         # Main AttnSCM class
├── attention.py    # Attention extraction utilities
├── heads.py        # Structural head identification
├── graph.py        # Graph construction and post-processing
└── metrics.py      # Evaluation metrics

experiments/        # Experiment scripts
├── exp_a_synthetic.py
├── exp_b_layers.py
├── exp_c_feature_selection.py
├── exp_d_efficiency.py
├── exp_a_causality_lab.py           # NEW: Extended benchmark
├── exp_e_tabpfn_enhanced_causality.py # NEW: TabPFN enhancement
└── exp_f_real_datasets.py           # NEW: Real-world datasets

baselines/          # Baseline implementations
├── pc_algorithm.py
├── notears.py
├── random_baseline.py
├── causality_lab_algorithms.py      # NEW: Intel Labs algorithms
└── tabpfn_causality_adapter.py      # NEW: TabPFN-enhanced CI tests

utils/              # Utilities
├── data_generation.py
├── visualization.py
└── io_utils.py
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zela2026attnscm,
  title={Zero-Shot Causal Graph Extraction from Tabular Foundation Models via Attention Map Decoding},
  author={Zela, Arber},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details

## Contact

Arber Zela - arber.zela@tue.ellis.eu  
ELLIS Institute Tübingen, Germany
