#!/usr/bin/env python3
"""Test that the adjacency matrix validation works correctly."""

import numpy as np
import sys

# Test the validate_dag function
try:
    from attn_scm.graph import validate_dag

    print("Testing validate_dag with various edge cases...")

    # Test 1: Normal binary matrix
    adj1 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    result1 = validate_dag(adj1)
    print(f"Test 1 (normal): {result1}")
    assert result1['is_dag'] == True

    # Test 2: Matrix with NaN
    adj2 = np.array([[0, 1, np.nan], [0, 0, 1], [0, 0, 0]], dtype=float)
    result2 = validate_dag(adj2)
    print(f"Test 2 (with NaN): {result2}")

    # Test 3: Matrix with inf
    adj3 = np.array([[0, 1, np.inf], [0, 0, 1], [0, 0, 0]], dtype=float)
    result3 = validate_dag(adj3)
    print(f"Test 3 (with inf): {result3}")

    # Test 4: Matrix with cycle
    adj4 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    result4 = validate_dag(adj4)
    print(f"Test 4 (with cycle): {result4}")
    assert result4['is_dag'] == False

    # Test 5: Empty matrix
    adj5 = np.zeros((5, 5), dtype=float)
    result5 = validate_dag(adj5)
    print(f"Test 5 (empty): {result5}")
    assert result5['is_dag'] == True
    assert result5['num_edges'] == 0

    print("\nAll tests passed!")
    sys.exit(0)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
