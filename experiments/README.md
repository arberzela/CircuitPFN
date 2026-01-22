# Attn-SCM Experiment Scripts

This directory contains the four main experiments from the paper.

## Running Experiments

### Experiment A: Synthetic Benchmark Recovery

Tests topological accuracy on synthetic SCMs:

```bash
# Full experiment (100 datasets)
python exp_a_synthetic.py

# Quick test (10 datasets)
python exp_a_synthetic.py --quick_test

# Custom configuration
python exp_a_synthetic.py --num_datasets 50 --n_features 15
```

**Expected runtime**: ~1-2 hours for full experiment (depends on hardware)

### Experiment B: Layer-Wise Localization

Tests the mid-layer hypothesis:

```bash
# Default (50 datasets, 12 layers)
python exp_b_layers.py

# Custom
python exp_b_layers.py --num_datasets 30 --n_layers 12
```

**Expected runtime**: ~2-3 hours

### Experiment C: Feature Selection

Demonstrates practical utility via Markov Blanket:

```bash
# All datasets
python exp_c_feature_selection.py

# Specific datasets only
python exp_c_feature_selection.py --datasets credit-g diabetes

# Available datasets:
# - credit-g
# - phoneme
# - blood-transfusion
# - diabetes
# - kr-vs-kp
```

**Expected runtime**: ~30-60 minutes

### Experiment D: Efficiency Profiling

Compares computational efficiency:

```bash
# Default
python exp_d_efficiency.py

# Custom limits (respect TabPFN constraints)
python exp_d_efficiency.py --max_n 1000 --max_d 50 --n_trials 3
```

**Expected runtime**: ~1-2 hours

## Results

All experiments save results to `results/exp_[a|b|c|d]/`:
- Raw data as CSV files
- Summary statistics
- Visualization plots (PNG)

## Notes

- **TabPFN Constraints**: Maximum 1000 samples and 100 features
- All experiments automatically handle these limits
- Use `--quick_test` or reduce `--num_datasets` for faster testing
