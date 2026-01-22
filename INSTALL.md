# Installation Guide

## Quick Setup

1. **Create virtual environment** (recommended):
   ```bash
   cd /Users/zelaa/Projects/CircuitPFN
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install package in development mode**:
   ```bash
   pip install -e .
   ```

## Conda Environment (Alternative)

If you prefer conda:

```bash
conda create -n attn-scm python=3.10
conda activate attn-scm
pip install -r requirements.txt
pip install -e .
```

## Verify Installation

Run the integration test:

```bash
python tests/test_integration.py
```

## Common Issues

### "ModuleNotFoundError: No module named 'X'"

**Solution**: Make sure you installed all dependencies:
```bash
pip install -r requirements.txt
```

### TabPFN Installation Issues

TabPFN may have specific requirements. If it fails:

```bash
# Try with conda
conda install pytorch -c pytorch
pip install tabpfn
```

### "Mock attention maps" Warning

This is **expected** currently. See `ATTENTION_EXTRACTION.md` for details.

The framework will work but use synthetic attention instead of real TabPFN attention.

## Testing Without Installation

To test the framework logic without installing TabPFN:

1. Install just the core dependencies:
   ```bash
   pip install numpy scipy scikit-learn pandas matplotlib seaborn networkx
   ```

2. Run tests - they'll use mock attention automatically

## Next Steps

After installation, see:
- `README.md` - Main documentation
- `experiments/README.md` - How to run experiments  
- `ATTENTION_EXTRACTION.md` - Attention extraction details
