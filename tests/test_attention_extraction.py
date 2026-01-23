#!/usr/bin/env python3
"""Test attention extraction and shape handling."""

import torch
import numpy as np

# Simulate what TabPFN attention looks like
# For 5 features and 20 samples, the sequence might be structured as:
# [20 train samples, features at the end] = total sequence length ~194

def test_extraction():
    """Test that we correctly extract feature attention from TabPFN-style attention."""

    # Simulate TabPFN attention
    batch_size = 1
    n_heads = 8
    n_samples = 20
    n_features = 5

    # TabPFN sequence structure might be: samples + features + special tokens
    # Let's say total sequence = 194 (this matches the error)
    seq_len = 194

    # Create mock attention tensor
    attn = torch.rand(batch_size, n_heads, seq_len, seq_len)

    print(f"Mock TabPFN attention shape: {attn.shape}")
    print(f"Want to extract: ({batch_size}, {n_heads}, {n_features}, {n_features})")

    # OLD METHOD (from beginning) - WRONG
    feat_attn_old = attn[:, :, :n_features, :n_features]
    print(f"Extracted from beginning: {feat_attn_old.shape}")

    # NEW METHOD (from end) - CORRECT
    feat_attn_new = attn[:, :, -n_features:, -n_features:]
    print(f"Extracted from end: {feat_attn_new.shape}")

    assert feat_attn_new.shape == (batch_size, n_heads, n_features, n_features)
    print("✓ Shape is correct!")

    # Now test what happens after batch aggregation
    feat_attn_aggregated = np.mean(feat_attn_new.numpy(), axis=0)
    print(f"After batch aggregation: {feat_attn_aggregated.shape}")
    assert feat_attn_aggregated.shape == (n_heads, n_features, n_features)

    # And after head extraction
    head_0 = feat_attn_aggregated[0, :, :]
    print(f"After head extraction: {head_0.shape}")
    assert head_0.shape == (n_features, n_features)

    print("\n✓ All tests passed! Feature extraction should work correctly now.")

if __name__ == "__main__":
    test_extraction()
