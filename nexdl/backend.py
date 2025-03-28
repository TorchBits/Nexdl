import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
    backend = cp  # Use CuPy if available
except ImportError:
    GPU_AVAILABLE = False
    backend = np  # Default to NumPy

