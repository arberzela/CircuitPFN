# Important Note: TabPFN Attention Extraction

## Current Status

The attention extraction in `attn_scm/attention.py` has been updated to handle TabPFN v1's architecture more robustly.

## How It Works

1. **Attempts Real Extraction**: The code tries to find and hook into TabPFN's internal transformer
2. **Fallback to Mock Attention**: If real extraction fails, it generates synthetic attention maps for testing

## Mock Attention Mode

When you see the message:
```
Warning: Could not find TabPFN internal transformer
Creating mock attention maps (synthetic data for testing)
```

This means the framework is using **synthetic attention patterns** instead of real TabPFN attention.

### What This Means

- ✅ The **framework itself works** and can extract causal graphs
- ✅ All experiments can **run successfully**
- ⚠️ Results won't reflect TabPFN's actual learned patterns
- ⚠️ This is a **limitation for real research** but fine for development/testing

## Getting Real Attention

To get real attention extraction working, you need to:

### Option 1: Inspect TabPFN Structure

```bash
# First install tabpfn
pip install tabpfn

# Then run the inspection script
python scripts/inspect_tabpfn.py
```

This will show you TabPFN's actual internal structure.

### Option 2: Update Hook Registration

Based on the inspection, modify `attn_scm/attention.py` in the `_find_transformer_module()` and `register_hooks()` methods to match TabPFN's actual architecture.

The key is finding where TabPFN stores its transformer model. Common locations:
- `model.c` (expected for TabPFN v1)
- `model.model`
- `model.transformer_encoder`

### Option 3: Alternative Approach

If TabPFN doesn't expose attention maps at all, you could:

1. **Use gradient-based methods** instead of attention
2. **Implement your own TabPFN-like model** with accessible attention
3. **Approximate with feature importance** from TabPFN's predictions

## For Development

The mock attention approach is **sufficient for**:
- Testing the codebase structure
- Validating the experiment pipelines
- Demonstrating the framework's capabilities
- Developing visualization and analysis tools

## For Research

To validate the paper's claims, you **need real attention**:
- Either fix the extraction for TabPFN v1
- Or use a transformer model with accessible attention (e.g., standard PyTorch transformer)

## Quick Fix Test

Try running the test again to see if it works with mock attention:

```bash
python tests/test_integration.py
```

You should now see it complete successfully (with a warning about using mock attention).
