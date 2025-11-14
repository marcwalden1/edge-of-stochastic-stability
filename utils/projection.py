"""
Random projection utilities for weight trajectory tracking.
Implements k-sparse Johnson-Lindenstrauss (JL) projection for efficient weight distance computation.
"""

import numpy as np
import torch
from typing import Optional, Dict
from pathlib import Path
import json


# Global cache for projection matrices to ensure consistency across runs
_PROJECTION_MATRIX_CACHE: Dict[str, np.ndarray] = {}


def initialize_projection_matrix(
    input_dim: int,
    output_dim: int,
    k: int = 3,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Initialize a k-sparse Johnson-Lindenstrauss projection matrix.
    
    For each column, randomly select k positions and fill with ±1/sqrt(k) values.
    This creates a sparse random matrix suitable for JL projection.
    
    Parameters:
    -----------
    input_dim : int
        Dimension of input vectors (number of model parameters)
    output_dim : int
        Dimension of output vectors (projected dimension, e.g., 10000)
    k : int
        Number of non-zero entries per column (sparsity parameter)
        Default is 3. Can also use sqrt(input_dim) for adaptive sparsity.
    seed : int, optional
        Random seed for reproducibility. If None, uses current RNG state.
        
    Returns:
    --------
    np.ndarray
        Sparse projection matrix of shape (output_dim, input_dim)
        Stored as dense array for efficiency (sparse structure is implicit)
    """
    cache_key = f"{input_dim}_{output_dim}_{k}_{seed}"
    
    if cache_key in _PROJECTION_MATRIX_CACHE:
        return _PROJECTION_MATRIX_CACHE[cache_key]
    
    rng = np.random.default_rng(seed)
    
    # Initialize matrix with zeros
    projection_matrix = np.zeros((output_dim, input_dim), dtype=np.float32)
    
    # For each column (input dimension), randomly select k rows
    for col in range(input_dim):
        # Randomly select k positions in this column
        row_indices = rng.choice(output_dim, size=min(k, output_dim), replace=False)
        
        # Fill with ±1/sqrt(k) with equal probability (Achlioptas-style)
        signs = rng.choice([-1, 1], size=len(row_indices))
        values = signs / np.sqrt(k)
        
        projection_matrix[row_indices, col] = values
    
    # Cache the matrix
    _PROJECTION_MATRIX_CACHE[cache_key] = projection_matrix
    
    return projection_matrix


def project_weights(
    model: torch.nn.Module,
    projection_dim: int = 10000,
    seed: Optional[int] = None,
    k: Optional[int] = None
) -> np.ndarray:
    """
    Project model weights to a lower-dimensional space using k-sparse JL projection.
    
    Parameters:
    -----------
    model : torch.nn.Module
        PyTorch model whose parameters to project
    projection_dim : int
        Target dimension for projection (default: 10000)
    seed : int, optional
        Random seed for projection matrix initialization
    k : int, optional
        Sparsity parameter (non-zero entries per column). 
        If None, uses k=3 or k=sqrt(input_dim) if input_dim is very large.
        
    Returns:
    --------
    np.ndarray
        Projected weight vector of shape (projection_dim,)
    """
    # Flatten all model parameters into a single vector
    from torch.nn.utils import parameters_to_vector
    
    param_vector = parameters_to_vector(model.parameters())
    input_dim = param_vector.numel()
    
    # Convert to numpy for projection
    weights_np = param_vector.detach().cpu().numpy().astype(np.float32)
    
    # Determine k (sparsity parameter)
    if k is None:
        # Use k=3 for small models, sqrt(input_dim) for very large models
        if input_dim < 1000000:
            k = 3
        else:
            k = int(np.sqrt(input_dim))
    
    # Ensure k doesn't exceed output_dim
    k = min(k, projection_dim)
    
    # Initialize or retrieve projection matrix
    projection_matrix = initialize_projection_matrix(
        input_dim=input_dim,
        output_dim=projection_dim,
        k=k,
        seed=seed
    )
    
    # Apply projection: output = projection_matrix @ weights
    projected = projection_matrix @ weights_np
    
    return projected


def get_projection_metadata(
    model: torch.nn.Module,
    projection_dim: int = 10000,
    seed: Optional[int] = None,
    k: Optional[int] = None
) -> Dict:
    """
    Get metadata about a projection (without actually performing it).
    
    Returns:
    --------
    dict
        Metadata dictionary with keys: input_dim, output_dim, k, seed
    """
    from torch.nn.utils import parameters_to_vector
    
    param_vector = parameters_to_vector(model.parameters())
    input_dim = param_vector.numel()
    
    if k is None:
        if input_dim < 1000000:
            k = 3
        else:
            k = int(np.sqrt(input_dim))
    k = min(k, projection_dim)
    
    return {
        'input_dim': int(input_dim),
        'output_dim': int(projection_dim),
        'k': int(k),
        'seed': seed
    }

