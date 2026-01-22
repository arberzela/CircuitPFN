"""I/O utilities for saving and loading results."""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any


def save_results(results: Dict[str, Any], filepath: str):
    """
    Save experiment results to file.
    
    Args:
        results: Dictionary of results
        filepath: Path to save file (.json or .pkl)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.json':
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
    
    elif filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        raise ValueError(f"Unsupported file extension: {filepath.suffix}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load experiment results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Dictionary of results
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    
    elif filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    else:
        raise ValueError(f"Unsupported file extension: {filepath.suffix}")
