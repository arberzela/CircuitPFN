"""Baseline causal discovery algorithms."""

from baselines.pc_algorithm import run_pc_algorithm
from baselines.notears import run_notears
from baselines.random_baseline import run_random_baseline

# Causality Lab algorithms
try:
    from baselines.causality_lab_algorithms import (
        run_pc_causality_lab,
        run_rai,
        run_brai,
        run_fci,
        run_icd,
        run_causality_lab_algorithm,
        CAUSALITY_LAB_ALGORITHMS
    )
    CAUSALITY_LAB_AVAILABLE = True
except ImportError:
    CAUSALITY_LAB_AVAILABLE = False

# TabPFN-Causality adapters
try:
    from baselines.tabpfn_causality_adapter import (
        CondIndepTabPFN,
        CondIndepAttentionWeighted
    )
    TABPFN_CAUSALITY_ADAPTER_AVAILABLE = True
except ImportError:
    TABPFN_CAUSALITY_ADAPTER_AVAILABLE = False

__all__ = [
    'run_pc_algorithm',
    'run_notears',
    'run_random_baseline',
    'CAUSALITY_LAB_AVAILABLE',
    'TABPFN_CAUSALITY_ADAPTER_AVAILABLE'
]

if CAUSALITY_LAB_AVAILABLE:
    __all__.extend([
        'run_pc_causality_lab',
        'run_rai',
        'run_brai',
        'run_fci',
        'run_icd',
        'run_causality_lab_algorithm',
        'CAUSALITY_LAB_ALGORITHMS'
    ])

if TABPFN_CAUSALITY_ADAPTER_AVAILABLE:
    __all__.extend([
        'CondIndepTabPFN',
        'CondIndepAttentionWeighted'
    ])
