import torch as T
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import linalg as LA
import numpy as np
from typing import List, Optional

import wandb
from .lobpcg import torch_lobpcg, _maybe_orthonormalize
# from .hvp import make_param_block_hvp
from torch.func import functional_call

import time
import os
from scipy import stats

__all__ = ['param_vector', 'param_length', 'flatt', 'grads_vector',
           'calculate_all_the_grads', 'compute_eigenvalues', 'compute_grad_H_grad',
           'calculate_averaged_lambdamax', 'create_ntk',
           'compute_fisher_eigenvalues', 'calculate_all_net_grads',
           'calculate_averaged_grad_H_grad', 'calculate_averaged_grad_H_grad_step', 'calculate_gni',
           'calculate_accuracy', 'calculate_param_distance',
           'EigenvectorCache', 'create_hessian_vector_product', 'compute_multiple_eigenvalues_lobpcg',
           'calculate_gradient_norm_squared_mc', 'calculate_expected_one_step_full_loss_change',
           'calculate_expected_one_step_batch_loss_change', 'compute_gradient_projection_ratios',
           'estimate_hessian_trace', 'gimme_new_rng', 'gimme_random_subset_idx',
           'compute_cosine_similarity', 'get_momentum_buffer_vector',
           'get_rmsprop_preconditioner_inv', 'get_rmsprop_momentum_buffer',
           'compute_adaptive_grad_H_grad', 'compute_adaptive_grad_H_grad_momentum',
           'calculate_adaptive_batch_sharpness', 'calculate_adaptive_batch_sharpness_momentum',
           'get_preconditioner_diag', 'create_preconditioned_hessian_vector_product',
           'compute_preconditioned_eigenvalues']


class EigenvectorCache:
    """
    A cache for storing eigenvectors to enable warm starts in power iteration methods.
    Designed to be compatible with future LOBPCG implementations.
    """
    def __init__(self, max_eigenvectors=5):
        self.max_eigenvectors = max_eigenvectors
        self.eigenvectors = []   # List of eigenvectors for multi-eigenvalue computations
        self.eigenvalues = []    # Corresponding eigenvalues
        
    def store_eigenvector(self, eigenvector, eigenvalue=None):
        """Store a single eigenvector (and optionally eigenvalue)"""
        if eigenvalue is not None:
            self.eigenvalues = [eigenvalue]
        self.eigenvectors = [eigenvector]
    
    def store_eigenvectors(self, eigenvectors_list, eigenvalues_list=None):
        """Store multiple eigenvectors (for future LOBPCG compatibility)"""
        self.eigenvectors = [v.detach().clone() for v in eigenvectors_list]
        if eigenvalues_list is not None:
            self.eigenvalues = list(eigenvalues_list)
        
        # Trim to maximum size
        if len(self.eigenvectors) > self.max_eigenvectors:
            self.eigenvectors = self.eigenvectors[:self.max_eigenvectors]
            if self.eigenvalues:
                self.eigenvalues = self.eigenvalues[:self.max_eigenvectors]
    
    def get_warm_start_vectors(self, device=None):
        """Get eigenvectors for warm start, optionally moved to specified device"""
        if not self.eigenvectors:
            return None
        
        if device is not None:
            return [v.to(device) for v in self.eigenvectors]
        return self.eigenvectors
    
    def clear(self):
        """Clear all cached eigenvectors"""
        self.eigenvectors = []
        self.eigenvalues = []
    
    def __len__(self):
        return len(self.eigenvectors)
    
    def __contains__(self, key):
        # For backward compatibility with dict-like access
        return hasattr(self, key) and getattr(self, key) is not None



################################################################################
#                                                                              #
#                               HELPER FUNCTIONS                               #
#                                                                              #
################################################################################


def param_vector(net, clone=True):
    '''
    Returns a vector of all the parameters of the network
    If clone=True, returns a detached clone of the parameters
    '''
    # params = list(net.parameters())
    param_vector = T.cat([p.flatten() for p in net.parameters()])
    if clone:
        return param_vector.detach().clone()
    return param_vector

def param_length(net):
    '''
    Returns the number of parameters in the network
    '''
    params = list(net.parameters())
    return sum([p.numel() for p in params])

def flatt(vectors):
    '''
    Flattens a list of vectors into a single vector
    '''
    return T.cat([v.flatten() for v in vectors])


def grads_vector(net):
    # pull out all the gradients from a network as one vector
    grads = []
    for p in net.parameters():
        grads.append(p.grad.flatten().detach().clone())
    return T.cat(grads)


def compute_cosine_similarity(vec1: T.Tensor, vec2: T.Tensor) -> float:
    """Compute cosine similarity between two flattened vectors."""
    if vec1 is None or vec2 is None:
        return float('nan')
    norm1, norm2 = T.linalg.vector_norm(vec1), T.linalg.vector_norm(vec2)
    if norm1.item() < 1e-12 or norm2.item() < 1e-12:
        return float('nan')
    return (T.dot(vec1, vec2) / (norm1 * norm2)).item()


def get_momentum_buffer_vector(optimizer: T.optim.Optimizer) -> T.Tensor | None:
    """Extract flattened momentum buffer from SGD optimizer."""
    if not isinstance(optimizer, T.optim.SGD):
        return None
    buffers = []
    for group in optimizer.param_groups:
        if group.get('momentum', 0) == 0:
            return None
        for p in group['params']:
            state = optimizer.state.get(p)
            if state is None or 'momentum_buffer' not in state:
                return None
            buffers.append(state['momentum_buffer'].flatten().detach().clone())
    return T.cat(buffers) if buffers else None


def get_rmsprop_preconditioner_inv(optimizer: T.optim.Optimizer) -> T.Tensor | None:
    """Extract flattened P^{-1} vector from RMSProp optimizer state.

    P = diag(sqrt(v_t) + eps), so P^{-1} = diag(1 / (sqrt(v_t) + eps)).
    Returns None if the optimizer is not RMSProp or state is not yet initialized.
    """
    if not isinstance(optimizer, T.optim.RMSprop):
        return None
    eps = optimizer.param_groups[0].get('eps', 1e-8)
    inv_parts = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p)
            if state is None or 'square_avg' not in state:
                return None
            inv_parts.append((1.0 / (state['square_avg'].sqrt() + eps)).flatten().detach().clone())
    return T.cat(inv_parts) if inv_parts else None


def get_rmsprop_momentum_buffer(optimizer: T.optim.Optimizer) -> T.Tensor | None:
    """Extract flattened momentum buffer from RMSProp optimizer state.

    Returns None if the optimizer is not RMSProp, momentum is 0, or state
    is not yet initialized.
    """
    if not isinstance(optimizer, T.optim.RMSprop):
        return None
    if optimizer.param_groups[0].get('momentum', 0) == 0:
        return None
    buffers = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p)
            if state is None or 'momentum_buffer' not in state:
                return None
            buffers.append(state['momentum_buffer'].flatten().detach().clone())
    return T.cat(buffers) if buffers else None


def get_preconditioner_inv(optimizer: T.optim.Optimizer) -> T.Tensor | None:
    """Extract flattened P^{-1} vector for RMSProp or Adam.

    Returns None if the optimizer is not RMSProp/Adam or state is not yet initialized.
    """
    if isinstance(optimizer, T.optim.RMSprop):
        return get_rmsprop_preconditioner_inv(optimizer)
    if isinstance(optimizer, (T.optim.Adam, T.optim.AdamW)):
        p_diag = get_preconditioner_diag(optimizer)
        if p_diag is None:
            return None
        return (1.0 / p_diag).detach().clone()
    return None


def get_preconditioned_momentum_buffer(optimizer: T.optim.Optimizer) -> T.Tensor | None:
    """Extract flattened momentum buffer for adaptive sharpness: RMSProp or Adam.

    For RMSProp: returns momentum_buffer (when momentum > 0).
    For Adam: returns exp_avg (first moment).
    Returns None if not available.
    """
    if isinstance(optimizer, T.optim.RMSprop):
        return get_rmsprop_momentum_buffer(optimizer)
    if isinstance(optimizer, (T.optim.Adam, T.optim.AdamW)):
        buffers = []
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p)
                if state is None or 'exp_avg' not in state:
                    return None
                buffers.append(state['exp_avg'].flatten().detach().clone())
        return T.cat(buffers) if buffers else None
    return None


def get_preconditioned_momentum_beta(optimizer: T.optim.Optimizer) -> float:
    """Return momentum coefficient beta for s = beta*m + P^{-1}*g.

    For RMSProp: param 'momentum' (e.g. 0.9).
    For Adam: betas[0] (e.g. 0.9).
    """
    if isinstance(optimizer, T.optim.RMSprop):
        return optimizer.param_groups[0].get('momentum', 0)
    if isinstance(optimizer, (T.optim.Adam, T.optim.AdamW)):
        return optimizer.param_groups[0]['betas'][0]
    return 0.0


def get_preconditioner_diag(optimizer: T.optim.Optimizer) -> T.Tensor | None:
    """Extract flattened diagonal of the preconditioner P as a 1D tensor.

    For RMSProp: P_diag = sqrt(v_t) + eps
    For Adam:    P_diag = sqrt(v_t / (1 - beta2^step)) + eps  (bias-corrected)
    For SGD or uninitialized state: returns None.
    """
    if isinstance(optimizer, T.optim.RMSprop):
        eps = optimizer.param_groups[0].get('eps', 1e-8)
        parts = []
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p)
                if state is None or 'square_avg' not in state:
                    return None
                parts.append((state['square_avg'].sqrt() + eps).flatten().detach().clone())
        return T.cat(parts) if parts else None

    if isinstance(optimizer, T.optim.Adam) or isinstance(optimizer, T.optim.AdamW):
        eps = optimizer.param_groups[0].get('eps', 1e-8)
        beta2 = optimizer.param_groups[0]['betas'][1]
        parts = []
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p)
                if state is None or 'exp_avg_sq' not in state:
                    return None
                step = state['step']
                if isinstance(step, T.Tensor):
                    step = step.item()
                if step == 0:
                    return None
                bias_correction2 = 1 - beta2 ** step
                v_hat = state['exp_avg_sq'] / bias_correction2
                parts.append((v_hat.sqrt() + eps).flatten().detach().clone())
        return T.cat(parts) if parts else None

    # SGD or any other optimizer
    return None


def gimme_new_rng():
    """
    Create a new random number generator with a unique seed.
    """
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)
    return rng


def gimme_random_subset_idx(dataset_size, subset_size):
    """
    Get random indices for a subset of the dataset.

    Args:
        dataset_size (int): Total size of the dataset.
        subset_size (int): Desired size of the subset.

    Returns:
        Tensor: Random indices for the subset.
    """
    rng = gimme_new_rng()

    shuffle = T.randperm(dataset_size, generator=rng)
    random_idx = shuffle[:subset_size]
    return random_idx


def calculate_param_distance(net, reference_params, p=2):
    """
    Calculate the distance between current network parameters and reference parameters.
    
    Args:
        net (nn.Module): Neural network model
        reference_params (Tensor): Flattened reference parameters (from param_vector())
        p (int, optional): The norm degree. Default: 2 for Euclidean distance
    
    Returns:
        Tensor: The p-norm distance between current and reference parameters
    """
    with torch.no_grad():
        current_params = param_vector(net)
        return T.linalg.vector_norm(current_params - reference_params, ord=p)


def calculate_all_the_grads(net, X, Y, loss_fn, optimizer, storage_device=None):
    # device = net.parameters().__next__().device

    grads = [] # datapoint, parameter
    for x, y in zip(X, Y):
        optimizer.zero_grad()
        y_pred = net(x.unsqueeze(0)).squeeze(dim=-1)
        loss = loss_fn(y_pred, y.unsqueeze(0))
        loss.backward()
        detached_grads = grads_vector(net).detach()
        if storage_device:
            detached_grads = detached_grads.to(storage_device)
        grads.append(detached_grads)
    
    return T.stack(grads)


def calculate_accuracy(predictions, targets):
    """
    Calculate the accuracy given the model predictions and target labels.
    
    Args:
        predictions: tensor of shape (num_samples, num_classes) with model predictions
        targets: tensor of shape (num_samples, num_classes) with one-hot encoded labels
                or tensor of shape (num_samples,) with class indices
    
    Returns:
        accuracy: float representing the accuracy (0.0 to 1.0)
    """
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Get the predicted class (highest value in each row)
        # this is if we have all the classes
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        # Get the predicted class (sign of the prediction)
        # this is if we have only two classes
        pred_classes = torch.sign(predictions).long()

    
    
    # Check if targets are one-hot encoded or class indices
    if len(targets.shape) > 1 and targets.shape[1] > 1:
        # One-hot encoded targets
        true_classes = torch.argmax(targets, dim=1)
    else:
        # Class indices (1D tensor)
        if len(targets.shape) == 1:
            true_classes = torch.round(targets).long()
        else:
            true_classes = targets.long()
    
    # Compare and compute accuracy
    correct = (pred_classes == true_classes).sum().item()
    total = targets.size(0)
    
    return correct / total


def jvp(net, X, Y, loss_fn, vector):
    """
    Computes the Jacobian-vector product (JVP) of the loss with respect to the network parameters.
    
    Args:
        net (nn.Module): The neural network model
        X (Tensor): Input data
        Y (Tensor): Target labels
        loss_fn (callable): Loss function to compute the loss
    
    Returns:
        Tensor: The JVP of the loss with respect to the network parameters
    """
    params = list(net.parameters())
    y_pred = net(X).squeeze(dim=-1)
    loss = loss_fn(y_pred, Y, sampling_vector=vector)
    
    # Compute gradients
    grads = torch.autograd.grad(loss, params, create_graph=True)
    
    # Flatten gradients into a single vector
    grads_vector = flatt(grads).detach()
    return grads_vector



################################################################################
#                                                                              #
#                             EIGENVALUE FUNCTIONS                             #
#                                                                              #
################################################################################


def compute_eigenvalues(loss, 
                        net, 
                        k=1, 
                        max_iterations=100, 
                        reltol=1e-2,
                        init_vectors=None,
                        batched=None,
                        eigenvector_cache=None,
                        return_eigenvectors: bool = False,
                        use_power_iteration: bool = False):
    """
    Computes the top-k eigenvalues of the Hessian of the loss function at the current point.
    
    Uses LOBPCG by default for better performance, with power iteration as fallback for k=1.

    Args:
        loss (Tensor): The loss value at the current point
        net (nn.Module): The neural network model
        k (int, optional): Number of eigenvalues to compute. Defaults to 1.
        max_iterations (int, optional): Maximum number of iterations. Defaults to 1000.
        reltol (float, optional): relative tolerance threshold for eigenvalue computation. Defaults to 1e-2.
        init_vectors (Tensor, optional): Initial vectors. For k=1, can be 1D vector. For k>1, should be [n_params, k]. 
                                        If None, uses cached or random vectors. Defaults to None.
        batched (Any, optional): Unused parameter. Defaults to None.
        eigenvector_cache (EigenvectorCache, optional): Cache to store/retrieve eigenvectors for warm starts. Defaults to None.
        return_eigenvectors (bool, optional): Whether to return the final eigenvectors. Defaults to False.
        use_power_iteration (bool, optional): If True, force use of power iteration (only works for k=1). Defaults to False.

    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]:
            - If k=1 and return_eigenvectors=False: Returns single eigenvalue (scalar Tensor)
            - If k=1 and return_eigenvectors=True: Returns (eigenvalue, eigenvector)
            - If k>1 and return_eigenvectors=False: Returns eigenvalues tensor of shape [k]
            - If k>1 and return_eigenvectors=True: Returns (eigenvalues, eigenvectors) where 
              eigenvalues has shape [k] and eigenvectors has shape [n_params, k]

    Note:
        By default, uses LOBPCG for eigenvalue computation for better performance.
        Falls back to power iteration if use_power_iteration=True (only supported for k=1).
        
        If eigenvector_cache is provided, the function will try to reuse previous eigenvectors
        for warm starts and store the final eigenvector(s) for future use.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    
    if use_power_iteration and k > 1:
        raise ValueError("Power iteration only supports k=1. Use LOBPCG (default) for k>1.")
    
    device = next(net.parameters()).device

    # Choose method: use LOBPCG by default unless explicitly requested to use power iteration
    if use_power_iteration and k == 1:
        # Use the existing power iteration implementation
        return compute_lambdamax_power_iteration(
            loss, net, max_iterations, reltol, init_vectors, batched,
            eigenvector_cache, return_eigenvectors
        )
    else:
        # Use LOBPCG method (default)
        eigenvalues, eigenvectors = compute_multiple_eigenvalues_lobpcg(
            loss, net, k, max_iterations, reltol, init_vectors, 
            eigenvector_cache, return_eigenvectors=True
        )
        
        if k == 1:
            # For backward compatibility with single eigenvalue case
            eigenvalue = eigenvalues[0]
            if return_eigenvectors:
                return eigenvalue, eigenvectors[:, 0]
            else:
                return eigenvalue
        else:
            # Multiple eigenvalues case
            if return_eigenvectors:
                return eigenvalues, eigenvectors
            else:
                return eigenvalues


def create_hessian_vector_product(loss, net):
    """
    Create a Hessian-vector product function for use with LOBPCG.
    
    This function creates a closure that computes the Hessian-vector product
    H @ v where H is the Hessian of the loss function with respect to network parameters.
    
    Args:
        loss (Tensor): The loss value at the current point (must retain computational graph)
        net (nn.Module): The neural network model
        
    Returns:
        callable: A function that takes a vector v and returns H @ v
    """
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grads_vector = flatt(grads)
    
    def hessian_vector_product(v):
        """
        Compute Hessian-vector product H @ v.
        
        Args:
            v (Tensor): Vector(s) to multiply with Hessian. Can be 1D or 2D (for multiple vectors).
            
        Returns:
            Tensor: H @ v (same shape as v)
        """
        # Handle both 1D and 2D inputs for compatibility with LOBPCG
        if v.dim() == 1:
            # Single vector case
            grad_v = torch.dot(grads_vector, v)
            Hv = torch.autograd.grad(grad_v, params, retain_graph=True)
            return flatt(Hv)
        elif v.dim() == 2:
            # Multiple vectors case (for LOBPCG)
            results = []
            for i in range(v.shape[1]):
                vi = v[:, i]
                grad_v = torch.dot(grads_vector, vi)
                Hv = torch.autograd.grad(grad_v, params, retain_graph=True)
                results.append(flatt(Hv))
            return torch.stack(results, dim=1)
        else:
            raise ValueError(f"Input tensor must be 1D or 2D, got {v.dim()}D")
    
    return hessian_vector_product


def create_preconditioned_hessian_vector_product(loss, net, p_inv_sqrt):
    """Create a matvec for the operator P^{-1/2} H P^{-1/2}.

    Since P^{-1}H is not symmetric, we use the similarity transform
    lambda(P^{-1}H) = lambda(P^{-1/2} H P^{-1/2}), which IS symmetric.

    Args:
        loss: Scalar loss (must retain computational graph).
        net: Neural network model.
        p_inv_sqrt: Flattened 1D tensor of element-wise 1/sqrt(P_diag).

    Returns:
        callable: v -> P^{-1/2} H (P^{-1/2} v)  (element-wise multiply).
    """
    hessian_matvec = create_hessian_vector_product(loss, net)

    def preconditioned_matvec(v):
        if v.dim() == 1:
            return p_inv_sqrt * hessian_matvec(p_inv_sqrt * v)
        elif v.dim() == 2:
            # Multiple vectors: broadcast p_inv_sqrt[:, None]
            scaled_v = p_inv_sqrt[:, None] * v
            Hv = hessian_matvec(scaled_v)
            return p_inv_sqrt[:, None] * Hv
        else:
            raise ValueError(f"Input tensor must be 1D or 2D, got {v.dim()}D")

    return preconditioned_matvec


def compute_preconditioned_eigenvalues(loss, net, p_inv_sqrt, k=1, max_iterations=100,
                                       reltol=1e-2, eigenvector_cache=None,
                                       return_eigenvectors=False):
    """Compute top-k eigenvalues of P^{-1/2} H P^{-1/2} via LOBPCG.

    Mirrors compute_multiple_eigenvalues_lobpcg but uses the preconditioned
    operator.  The eigenvector cache lives in a different coordinate space
    from the standard Hessian cache, so callers should supply a separate
    EigenvectorCache instance.

    Returns:
        k=1, return_eigenvectors=False  -> scalar eigenvalue
        k=1, return_eigenvectors=True   -> (scalar, eigenvector)
        k>1, return_eigenvectors=False  -> tensor of shape [k]
        k>1, return_eigenvectors=True   -> (tensor [k], matrix [n, k])
    """
    device = next(net.parameters()).device
    n_params = param_length(net)

    matvec = create_preconditioned_hessian_vector_product(loss, net, p_inv_sqrt)

    # Initialize vectors: cache -> random
    if eigenvector_cache is not None and len(eigenvector_cache) > 0:
        cached_vectors = eigenvector_cache.get_warm_start_vectors(device)
        if cached_vectors:
            n_cached = min(len(cached_vectors), k)
            X_list = cached_vectors[:n_cached]
            if n_cached < k:
                n_random = k - n_cached
                random_vecs = torch.randn(n_params, n_random, device=device)
                X_list.extend([random_vecs[:, i] for i in range(n_random)])
            X = torch.stack(X_list, dim=1)
        else:
            X = torch.randn(n_params, k, device=device)
    else:
        X = torch.randn(n_params, k, device=device)

    X = X.to(device)
    if X.shape != (n_params, k):
        X = X.reshape(n_params, k)

    tol = reltol / (20 * n_params)

    eigenvalues, eigenvectors, iterations = torch_lobpcg(
        matvec, X, max_iter=max_iterations, tol=tol
    )

    try:
        wandb.log({"preconditioned_lobpcg_iterations": iterations}, commit=False)
    except:
        pass

    if eigenvector_cache is not None:
        eigvec_list = [eigenvectors[:, i] for i in range(eigenvectors.shape[1])]
        eigenvector_cache.store_eigenvectors(eigvec_list, eigenvalues.tolist())

    if k == 1:
        ev = eigenvalues[0]
        if return_eigenvectors:
            return ev, eigenvectors[:, 0]
        return ev
    else:
        if return_eigenvectors:
            return eigenvalues, eigenvectors
        return eigenvalues


def compute_multiple_eigenvalues_lobpcg(loss, net, k=5, max_iterations=100, reltol=1e-2,
                                       init_vectors=None, eigenvector_cache=None,
                                       return_eigenvectors=False):
    """
    Compute multiple eigenvalues of the Hessian using LOBPCG algorithm.
    
    This function computes the top-k eigenvalues of the Hessian matrix using the
    LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) algorithm.
    
    Args:
        loss (Tensor): The loss value at the current point (must retain computational graph)
        net (nn.Module): The neural network model
        k (int, optional): Number of eigenvalues to compute. Defaults to 5.
        max_iterations (int, optional): Maximum number of LOBPCG iterations. Defaults to 100.
        reltol (float, optional): Relative tolerance for LOBPCG convergence. Defaults to 2% relative tolerance.

        init_vectors (Tensor, optional): Initial vectors for LOBPCG (shape: [n_params, k]). 
                                       If None, uses random or cached vectors.
        eigenvector_cache (EigenvectorCache, optional): Cache for storing/retrieving eigenvectors.
        return_eigenvectors (bool, optional): Whether to return eigenvectors along with eigenvalues.
        
    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]:
            - If return_eigenvectors is False: Returns eigenvalues tensor of shape [k]
            - If return_eigenvectors is True: Returns tuple of (eigenvalues, eigenvectors)
              where eigenvectors has shape [n_params, k]
    
    Note:
        The eigenvalues are returned in descending order (largest first).
        The function automatically handles the case where k is too large relative to the problem size.
    """
    device = next(net.parameters()).device
    n_params = param_length(net)
    
    # Create Hessian-vector product function
    hessian_matvec = create_hessian_vector_product(loss, net)
    
    # Initialize vectors with priority: init_vectors > cached vectors > random
    if init_vectors is not None:
        X = init_vectors
        if X.shape[1] != k:
            raise ValueError(f"init_vectors must have shape [n_params, {k}], got {X.shape}")
    elif eigenvector_cache is not None and len(eigenvector_cache) > 0:
        # Use cached eigenvectors as initial guess
        cached_vectors = eigenvector_cache.get_warm_start_vectors(device)
        if cached_vectors:
            # Take up to k vectors from cache, pad with random if needed
            n_cached = min(len(cached_vectors), k)
            X_list = cached_vectors[:n_cached]
            
            # Pad with random vectors if we don't have enough cached vectors
            if n_cached < k:
                n_random = k - n_cached
                random_vectors = torch.randn(n_params, n_random, device=device)
                X_list.extend([random_vectors[:, i] for i in range(n_random)])
            
            X = torch.stack(X_list, dim=1)
        else:
            X = torch.randn(n_params, k, device=device)
    else:
        # Use random initialization
        X = torch.randn(n_params, k, device=device)
    
    # Ensure X is on the correct device and has the right shape
    X = X.to(device)
    if X.shape != (n_params, k):
        X = X.reshape(n_params, k)
    
    # Run LOBPCG
    tol = reltol / (20 * n_params)  # Adjust tolerance based on problem size

    eigenvalues, eigenvectors, iterations = torch_lobpcg(
        hessian_matvec, X, max_iter=max_iterations, tol=tol
    )
    
    # Log the number of iterations to wandb (if available)
    try:
        wandb.log({"lobpcg_iterations": iterations}, commit=False)
    except:
        pass  # wandb not initialized or not available
    
    # Store eigenvectors in cache for future use
    if eigenvector_cache is not None:
        eigenvector_list = [eigenvectors[:, i] for i in range(eigenvectors.shape[1])]
        eigenvector_cache.store_eigenvectors(eigenvector_list, eigenvalues.tolist())
    
    # Return results
    if return_eigenvectors:
        return eigenvalues, eigenvectors
    else:
        return eigenvalues



def compute_lambdamax_power_iteration(loss, net, max_iterations, reltol, init_vector,
                                       eigenvector_cache, return_eigenvector):
    """Power iteration implementation of the maximum eigenvalue of the Hessian."""
    device = next(net.parameters()).device

    # compute gradient and keep it
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grads_vector = flatt(grads)

    size = param_length(net)
    
    # Initialize vector with priority: init_vector > cached eigenvector > gradient
    if init_vector is not None:
        v = init_vector
    elif eigenvector_cache is not None:
        # Support both EigenvectorCache objects and dict-style caches
        if isinstance(eigenvector_cache, EigenvectorCache):
            if len(eigenvector_cache) > 0:
                cached_v = eigenvector_cache.eigenvector
                if cached_v.device != device:
                    cached_v = cached_v.to(device)
                v = cached_v.detach()
            else:
                v = T.randn(size, device=device)
        elif isinstance(eigenvector_cache, dict) and 'eigenvector' in eigenvector_cache:
            # Backward compatibility with dict-style cache
            cached_v = eigenvector_cache['eigenvector']
            if cached_v.device != device:
                cached_v = cached_v.to(device)
            v = cached_v.detach()
        else:
            v = T.randn(size, device=device)
    else:
        # Use random vector as initial vector instead of gradient
        v = T.randn(size, device=device)
    
    with torch.no_grad():
        v = v / T.linalg.norm(v)



    # grad_vector as init_vector, since it is very close to the eigenvector
    # v = grads_vector.detach()

    # Power iteration method to find the maximum eigenvalue
    # NEW METHOD
    v = v.detach()
    eigenval = 0.0  # Initialize eigenval to avoid undefined variable error
    for i in range(max_iterations):
        # grad_vector = \nabla L
        grad_v = T.dot(grads_vector, v) # \nabla L . v
        Hv = flatt(T.autograd.grad(grad_v, params, retain_graph=True)).detach() # \nabla (\nabla L . v) = H(L) * v

        v = v.detach()
        with T.no_grad():
            rayleigh_quotient = T.dot(Hv, v) / T.dot(v, v)
            eigenval = rayleigh_quotient  # Update eigenval every iteration
            if T.abs(rayleigh_quotient) < 1e-12:
                break

            residual = Hv - rayleigh_quotient * v
            resid_norm = T.linalg.norm(residual)
            if resid_norm / T.abs(rayleigh_quotient) < reltol:
                break
            
            v = Hv / T.linalg.norm(Hv) # Normalize for next iteration, 
    
    
        
    #### OLD PROCEDURE
    #### This is the old procedure, kept for reference
    # epsilon = 1e-4
    # eigenval = -10
    # eigenvals = [-100, -20]
    # for i in range(max_iterations):
    #     grad_v = T.dot(grads_vector, v)
    #     Hv = flatt(T.autograd.grad(grad_v, params, retain_graph=True))
    #     with T.no_grad():
    #         v = Hv / T.linalg.vector_norm(Hv)
    #         v = v.detach()
    #         eigenval = T.dot(Hv, v) / T.dot(v, v)
    #     old_eigenval = eigenvals[-2]
    #     eigenvals.append(eigenval)

    #     if abs(eigenval - old_eigenval) / eigenval < epsilon:
    #         break

    # Log the number of iterations to wandb
    try:
        wandb.log({"power_iteration_iterations": i + 1}, commit=False)
    except:
        pass



    # Store the final eigenvector in cache for future warm starts
    if eigenvector_cache is not None:
        if isinstance(eigenvector_cache, EigenvectorCache):
            eigenvector_cache.store_eigenvector(v, eigenval)
        else:
            raise ValueError("eigenvector_cache must be an instance of EigenvectorCache")

    # Prepare return values
    results = [eigenval]
    
    if return_eigenvector:
        results.append(v.detach())
    
    # Return single value or tuple based on what was requested
    if len(results) == 1:
        return results[0]
    else:
        return tuple(results)


################################################################################
#                                                                              #
#                         GRAD-H-GRAD (BATCH SHARPNESS)                        #
#                                                                              #
################################################################################


def compute_grad_H_grad(loss, net, grad_already_there: bool = False,
                        return_ghg_gg_separately: bool = False):
    """
    Computes g^T H g / ||g||², the Rayleigh quotient of the Hessian H and gradient g.
    
    This function calculates gradient * Hessian * gradient normalized by the squared gradient norm,
    which represents the curvature of the loss in the gradient direction. If taken on a batch, this is 
    step sharpness. Averaging over many batches gives batch sharpness.
    
    Args:
        loss (Tensor): Loss value (must retain computational graph for Hessian computation)
        net (nn.Module): Neural network model
        grad_already_there (bool, optional): Use existing gradients instead of computing new ones. Defaults to False.
        return_ghg_gg_separately (bool, optional): Return (g^T H g, g^T g) separately instead of ratio. Defaults to False.
    
    Returns:
        Union[Tensor, Tuple[Tensor, Tensor]]: Rayleigh quotient g^T H g / ||g||² or separate components if requested
    """
    
    device = next(net.parameters()).device

    # compute gradient and keep it
    params = list(net.parameters())
    if not grad_already_there:
        grads = torch.autograd.grad(loss, params, create_graph=True)
    else:
        grads = [p.grad for p in params]
    grads_vector = flatt(grads)

    # compute Hessian vector product
    # grads_vector = T.cat([g.flatten() for g in grads])
    step_vector = grads_vector.detach()
    grad_step = T.dot(grads_vector, step_vector)
    Hv = T.autograd.grad(grad_step, params, retain_graph=False)
    Hv = flatt(Hv).detach()

    if return_ghg_gg_separately:
        return T.dot(step_vector, Hv), T.dot(step_vector, step_vector)
    return T.dot(step_vector, Hv) / T.dot(step_vector, step_vector)



def compute_adaptive_grad_H_grad(loss, net, preconditioner_inv):
    """Compute (u^T H u, g^T u) where u = P^{-1} g for a single batch.

    This is the per-batch kernel for adaptive batch sharpness.
    The ratio u^T H u / g^T u gives the preconditioned Rayleigh quotient.

    Args:
        loss: Scalar loss (must retain computational graph).
        net: Neural network model.
        preconditioner_inv: Flattened P^{-1} vector (detached).

    Returns:
        Tuple[Tensor, Tensor]: (u^T H u, g^T u) as scalar tensors.
    """
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g_flat = flatt(grads)

    u = g_flat.detach() * preconditioner_inv  # detached, no graph

    scalar = T.dot(g_flat, u)  # has graph through g_flat
    Hu = torch.autograd.grad(scalar, params, retain_graph=False)
    Hu_flat = flatt(Hu).detach()

    return T.dot(u, Hu_flat), T.dot(g_flat.detach(), u)


def compute_adaptive_grad_H_grad_momentum(loss, net, preconditioner_inv, momentum_buffer, beta):
    """Compute (s^T H s, g^T s) where s = beta*m + P^{-1}*g for a single batch.

    This is the per-batch kernel for adaptive batch sharpness with momentum.
    When beta=0, this reduces to compute_adaptive_grad_H_grad.

    Args:
        loss: Scalar loss (must retain computational graph).
        net: Neural network model.
        preconditioner_inv: Flattened P^{-1} vector (detached).
        momentum_buffer: Flattened momentum buffer m (detached).
        beta: Momentum coefficient.

    Returns:
        Tuple[Tensor, Tensor]: (s^T H s, g^T s) as scalar tensors.
    """
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g_flat = flatt(grads)

    u = g_flat.detach() * preconditioner_inv          # P^{-1} g_B (detached)
    s = beta * momentum_buffer + u                     # s_B = beta*m + P^{-1}*g_B (all detached)

    scalar = T.dot(g_flat, s)                          # g^T s (has graph through g_flat)
    Hs = torch.autograd.grad(scalar, params, retain_graph=False)
    Hs_flat = flatt(Hs).detach()

    return T.dot(s, Hs_flat), T.dot(g_flat.detach(), s)  # (s^T H s, g^T s)


def calculate_averaged_grad_H_grad(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 500,
                              min_estimates = 10,
                              eps = 0.005, # 0.005 approx gives 1% error; 0.005 = 0.01 / 1.96,
                              expectation_inside = False,
                              with_replacement = False,
                              return_confidence_interval: bool = False,
                              confidence_level: float = 0.95
                              ): 
    """
    Computes E[g_b H_b g_b / ||g_b||²], which represents batch sharpness, aka the Rayleigh quotient of the 
    batch Hessian and batch gradient.
    The function uses Monte Carlo sampling with adaptive stopping based on relative standard 
    error to efficiently estimate the expectation.
    Args:
        net: Neural network model whose parameters will be used for gradient computation
        X: Input data tensor
        Y: Target labels tensor  
        loss_fn: Loss function to compute gradients from
        batch_size (int): Size of random batches to sample for each estimate
        n_estimates (int, optional): Maximum number of Monte Carlo estimates. Defaults to 500.
        min_estimates (int, optional): Minimum estimates before checking stopping criterion. Defaults to 10.
        eps (float, optional): Relative standard error threshold for early stopping. Defaults to 0.005.
        expectation_inside (bool, optional): If True, computes E[gHg]/E[g²] instead, mostly used for exploratory purposes. Defaults to False.
        with_replacement (bool, optional): Sample batches with replacement. Defaults to False.
        return_confidence_interval (bool, optional): If True, include a confidence interval and related statistics in the return value. Defaults to False.
        confidence_level (float, optional): Confidence level for the interval when `return_confidence_interval` is True. Defaults to 0.95.
    Returns:
        float or dict: The averaged gradient-Hessian-gradient ratio representing batch sharpness. When
            `return_confidence_interval` is True, returns a dictionary with the estimate, confidence interval,
            standard error, confidence level, and number of Monte Carlo samples used.
    Notes:
        - Uses independent random number generator for true randomness (since it is fixed in the main training loop)
        - Implements adaptive stopping based on relative standard error convergence  
        - Logs the number of estimates to wandb if available
        - eps=0.005 approximately gives 1% estimation error
    """
    gHg_vals = []
    norm_g_vals = []

    x_vals = gHg_vals
    y_vals = norm_g_vals
    

    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)

    # Clear cache once before starting MC estimation (not inside the loop)
    if batch_size > 128 and torch.cuda.is_available():
        torch.cuda.empty_cache()

    for i in range(n_estimates):
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        if with_replacement:
            random_idx = T.randint(0, len(X), (batch_size,), generator=rng)

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]

        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        
        gHg, norm_g = compute_grad_H_grad(loss, net, return_ghg_gg_separately=True)
        gHg = gHg.item()
        norm_g = norm_g.item()
        
        
        gHg_vals.append(gHg)
        norm_g_vals.append(norm_g)

        if i < min_estimates:
            continue    

        mean_x, mean_y = np.mean(x_vals), np.mean(y_vals)
        var_x,  var_y  = np.var(x_vals, ddof=1), np.var(y_vals, ddof=1)
        cov_xy = np.cov(x_vals, y_vals, ddof=1)[0, 1]

        R = mean_x / mean_y

        var_R = (var_x / mean_y**2
                 - 2 * cov_xy * mean_x / mean_y**3
                 + var_y * mean_x**2 / mean_y**4) / i

        rse = np.sqrt(var_R) / abs(R)  # relative standard error

        if rse < eps:                    # stopping rule
            break


    num_samples = len(gHg_vals)

    try:
        wandb.log({"number_of_gHg_estimates": num_samples}, commit=False)
    except:
        pass


    if num_samples == 0:
        raise RuntimeError("calculate_averaged_grad_H_grad received no samples; check dataset and parameters.")

    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    alpha = 1 - confidence_level

    if expectation_inside:
        mean_x = float(np.mean(gHg_vals))
        mean_y = float(np.mean(norm_g_vals))
        if mean_y == 0.0:
            raise ZeroDivisionError("Mean squared gradient is zero; cannot compute batch sharpness.")

        result = mean_x / mean_y

        if not return_confidence_interval:
            return result

        if num_samples < 2:
            stderr = 0.0
            ci = (result, result)
        else:
            var_x = float(np.var(gHg_vals, ddof=1))
            var_y = float(np.var(norm_g_vals, ddof=1))
            cov_xy = float(np.cov(gHg_vals, norm_g_vals, ddof=1)[0, 1])
            var_R = (
                var_x / (mean_y ** 2)
                - 2 * cov_xy * mean_x / (mean_y ** 3)
                + var_y * (mean_x ** 2) / (mean_y ** 4)
            ) / num_samples
            var_R = max(var_R, 0.0)
            stderr = float(np.sqrt(var_R))
            t_multiplier = stats.t.ppf(1 - alpha / 2, df=num_samples - 1) if num_samples > 1 else 0.0
            if not np.isfinite(t_multiplier):
                t_multiplier = 0.0
            half_width = float(t_multiplier * stderr)
            ci = (result - half_width, result + half_width)

        return {
            "mean": result,
            "ci": ci,
            "stderr": stderr,
            "confidence_level": confidence_level,
            "num_samples": num_samples,
        }

    gHg_normalized = np.array(gHg_vals) / np.array(norm_g_vals)
    result = float(np.mean(gHg_normalized))

    if not return_confidence_interval:
        return result

    if num_samples < 2:
        stderr = 0.0
        ci = (result, result)
    else:
        std = float(np.std(gHg_normalized, ddof=1))
        stderr = float(std / np.sqrt(num_samples))
        t_multiplier = stats.t.ppf(1 - alpha / 2, df=num_samples - 1)
        if not np.isfinite(t_multiplier):
            t_multiplier = 0.0
        half_width = float(t_multiplier * stderr)
        ci = (result - half_width, result + half_width)

    return {
        "mean": result,
        "ci": ci,
        "stderr": stderr,
        "confidence_level": confidence_level,
        "num_samples": num_samples,
    }


def calculate_averaged_grad_H_grad_step(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 1000,
                              min_estimates = 10,
                              eps = 0.005,
                              log_the_expectation_outside = False,
                              return_ghg_gg_separately = False,
                              with_replacement = False,
                              return_confidence_interval: bool = False,
                              confidence_level: float = 0.95
                              ):
    """Backward-compatible wrapper for the batch sharpness estimator E[gHg/g²]."""
    if return_ghg_gg_separately:
        raise NotImplementedError("Returning gHg and g² separately is not supported in this refactor.")

    result = calculate_averaged_grad_H_grad(
        net=net,
        X=X,
        Y=Y,
        loss_fn=loss_fn,
        batch_size=batch_size,
        n_estimates=n_estimates,
        min_estimates=min_estimates,
        eps=eps,
        expectation_inside=False,
        with_replacement=with_replacement,
        return_confidence_interval=return_confidence_interval,
        confidence_level=confidence_level,
    )

    return result


def calculate_adaptive_batch_sharpness(net, X, Y, loss_fn, optimizer,
                                       batch_size, n_estimates=1000,
                                       min_estimates=20, eps=0.005):
    """Compute adaptive batch sharpness E[u^T H_B u / g^T u] where u = P^{-1} g.

    This is the preconditioned Rayleigh quotient averaged over mini-batches,
    measuring curvature in RMSProp's adapted coordinate system.

    Args:
        net: Neural network model.
        X: Full input tensor.
        Y: Full target tensor.
        loss_fn: Loss function.
        optimizer: RMSProp optimizer (used to extract preconditioner state).
        batch_size: Mini-batch size for MC sampling.
        n_estimates: Maximum number of MC estimates.
        min_estimates: Minimum estimates before checking convergence.
        eps: Relative standard error threshold for early stopping.

    Returns:
        float: Estimated adaptive batch sharpness, or np.nan if preconditioner
               is not available.
    """
    preconditioner_inv = get_rmsprop_preconditioner_inv(optimizer)
    if preconditioner_inv is None:
        return np.nan

    numerators = []
    denominators = []

    rng = gimme_new_rng()

    if batch_size > 128 and torch.cuda.is_available():
        torch.cuda.empty_cache()

    for i in range(n_estimates):
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]

        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        uHu, gu = compute_adaptive_grad_H_grad(loss, net, preconditioner_inv)
        numerators.append(uHu.item())
        denominators.append(gu.item())

        if i < min_estimates:
            continue

        # Delta-method convergence check (same as batch sharpness)
        x_vals, y_vals = numerators, denominators
        mean_x, mean_y = np.mean(x_vals), np.mean(y_vals)
        var_x = np.var(x_vals, ddof=1)
        var_y = np.var(y_vals, ddof=1)
        cov_xy = np.cov(x_vals, y_vals, ddof=1)[0, 1]

        R = mean_x / mean_y

        var_R = (var_x / mean_y**2
                 - 2 * cov_xy * mean_x / mean_y**3
                 + var_y * mean_x**2 / mean_y**4) / i

        rse = np.sqrt(var_R) / abs(R) if abs(R) > 1e-12 else float('inf')

        if rse < eps:
            break

    try:
        wandb.log({"number_of_adaptive_gHg_estimates": len(numerators)}, commit=False)
    except Exception:
        pass

    if len(numerators) == 0:
        return np.nan

    return float(np.mean(np.array(numerators) / np.array(denominators)))


def calculate_adaptive_batch_sharpness_momentum(net, X, Y, loss_fn, optimizer,
                                                batch_size, n_estimates=1000,
                                                min_estimates=20, eps=0.005):
    """Compute adaptive batch sharpness with momentum: E[s^T H_B s / g^T s].

    Here s = beta*m + P^{-1}*g where m is the momentum buffer (RMSProp momentum_buffer
    or Adam exp_avg) and P^{-1} is the preconditioner inverse. When beta=0 this
    reduces to the non-momentum adaptive batch sharpness.

    Supports RMSProp (with momentum) and Adam.

    Returns:
        float: Estimated value, or np.nan if preconditioner/momentum buffer
               is not available.
    """
    preconditioner_inv = get_preconditioner_inv(optimizer)
    if preconditioner_inv is None:
        return np.nan

    momentum_buffer = get_preconditioned_momentum_buffer(optimizer)
    if momentum_buffer is None:
        return np.nan

    beta = get_preconditioned_momentum_beta(optimizer)

    numerators = []
    denominators = []

    rng = gimme_new_rng()

    if batch_size > 128 and torch.cuda.is_available():
        torch.cuda.empty_cache()

    for i in range(n_estimates):
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]

        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        sHs, gs = compute_adaptive_grad_H_grad_momentum(
            loss, net, preconditioner_inv, momentum_buffer, beta)
        numerators.append(sHs.item())
        denominators.append(gs.item())

        if i < min_estimates:
            continue

        # Delta-method convergence check (same as batch sharpness)
        x_vals, y_vals = numerators, denominators
        mean_x, mean_y = np.mean(x_vals), np.mean(y_vals)
        var_x = np.var(x_vals, ddof=1)
        var_y = np.var(y_vals, ddof=1)
        cov_xy = np.cov(x_vals, y_vals, ddof=1)[0, 1]

        R = mean_x / mean_y

        var_R = (var_x / mean_y**2
                 - 2 * cov_xy * mean_x / mean_y**3
                 + var_y * mean_x**2 / mean_y**4) / i

        rse = np.sqrt(var_R) / abs(R) if abs(R) > 1e-12 else float('inf')

        if rse < eps:
            break

    try:
        wandb.log({"number_of_adaptive_mom_gHg_estimates": len(numerators)}, commit=False)
    except Exception:
        pass

    if len(numerators) == 0:
        return np.nan

    return float(np.mean(np.array(numerators) / np.array(denominators)))


################################################################################
#                                                                              #
#                       GRADIENT–NOISE INTERACTION (GNI)                       #
#                                                                              #
################################################################################


def calculate_gni(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 500,
                              min_estimates = 10,
                              tolerance = 0.01, # st error of mean / mean
                            #   max_hessian_iters = 1000,
                            #   hessian_tolerance = 1e-3,
                              batched = None,
                              compute_gHg: bool = False,
                              use_subset_of_data: int = None # use only a subset of the dataset to calculate H in GNI - speeds up computations!
                              ): 
    sharpnesses = []

    params = list(net.parameters())


    if use_subset_of_data is not None:
        rng = gimme_new_rng()
        # Take random subset of the dataset
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:use_subset_of_data]
        X = X[random_idx]
        Y = Y[random_idx]

    total_loss = loss_fn(net(X).squeeze(dim=-1), Y)

    total_grad = flatt(torch.autograd.grad(total_loss, params, create_graph=True))

    total_grad_detach = total_grad.detach()

    normalizer = T.dot(total_grad_detach, total_grad_detach).item()

    gHg_list = []


    for i in range(n_estimates):
        rng = gimme_new_rng()

        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]


        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        grads_vector = flatt(torch.autograd.grad(loss, params))
        step_vector = grads_vector.detach()

        total_grad_dot_step = T.dot(total_grad, step_vector)

        Hg = flatt(torch.autograd.grad(total_grad_dot_step, params, retain_graph=True))

        Hg = Hg.detach()

        gHg = T.dot(step_vector, Hg)

        gHg_list.append(gHg.item())


    quantity = np.mean(gHg_list) / normalizer

    return quantity



################################################################################
#                                                                              #
#                               MISCELLANEOUS                                  #
#                                                                              #
################################################################################


def compute_gradient_projection_ratios(grad_vector: torch.Tensor,
                                       eigvecs: torch.Tensor,
                                       max_k: int = 20,
                                       eigenvalues: list = None) -> dict:
    """
    Compute cumulative projection ratios of the full-batch gradient onto the
    subspace spanned by the top-i eigenvectors, i = 1..k, where k = min(K, max_k).

    grad_projection_i = ||Proj_{span(v1..vi)}(g)||_2 / ||g||_2

    Args:
        grad_vector: Flattened full-batch gradient g, shape [n]
        eigvecs:   Matrix of eigenvectors, shape [n, K]
        max_k:     Cap on how many cumulative projections to report (default 20)
        eigenvalues: Optional list of eigenvalues (length K) to ensure proper
                     descending ordering; if provided, will sort eigvecs by it.

    Returns:
        dict mapping names 'grad_projection_01', ..., 'grad_projection_{k:02d}',
        and 'grad_projection_residual' to floats in [0, 1].

    Notes:
        - Uses _maybe_orthonormalize to cheaply verify and, if needed,
          re-orthonormalize the eigenvector block prior to projection.
        - If grad_vector has zero norm, returns all zeros.
    """
    if grad_vector is None or eigvecs is None:
        return {}

    # Ensure 2D [n, K]
    if eigvecs.dim() == 1:
        eigvecs = eigvecs.unsqueeze(1)

    n, K = eigvecs.shape
    if n != grad_vector.numel():
        raise ValueError(f"Dimension mismatch: gradient has {grad_vector.numel()} params, eigenvectors have {n}")

    # Limit to at most max_k eigenvectors
    k = min(K, max_k)

    # If eigenvalues are supplied, sort eigenvectors by descending eigenvalue
    if eigenvalues is not None and len(eigenvalues) >= k:
        # Sort pairs (eigenvalue, column index) descending by value
        import math
        order = sorted(range(len(eigenvalues)), key=lambda idx: (-float(eigenvalues[idx]) if not math.isnan(float(eigenvalues[idx])) else float('inf')))
        order = order[:k]
        V = eigvecs[:, order]
    else:
        V = eigvecs[:, :k]

    # Quick orthonormality check; orthonormalize if necessary
    V = _maybe_orthonormalize(V, assume_ortho=True)

    # Compute projection coefficients c = V^T s
    g = grad_vector.reshape(-1)
    g_norm = torch.linalg.vector_norm(g)
    if g_norm.item() == 0.0:
        # Degenerate step; return zeros
        result = {f"grad_projection_{i:02d}": 0.0 for i in range(1, k + 1)}
        result["grad_projection_residual"] = 0.0
        return result

    c = V.T @ g  # shape [k]
    c2 = c.pow(2)
    # Cumulative squared projection norms
    c2_cum = torch.cumsum(c2, dim=0)
    denom = g_norm.pow(2)
    # Convert to ratios in [0,1]
    ratios = torch.sqrt(torch.clamp(c2_cum / denom, min=0.0, max=1.0))

    result = {}
    for i in range(k):
        result[f"grad_projection_{i+1:02d}"] = float(ratios[i].item())

    # Residual norm ratio for the full k-dimensional subspace
    residual_sq = torch.clamp(1.0 - c2_cum[-1] / denom, min=0.0, max=1.0)
    result["grad_projection_residual"] = float(torch.sqrt(residual_sq).item())

    return result


def estimate_hessian_trace(net,
                           X,
                           Y,
                           loss_fn,
                           max_estimates: int = 512,
                           min_estimates: int = 10,
                           eps: float = 0.01,
                           generator: Optional[torch.Generator] = None,
                           probe_type: str = 'rademacher') -> float:
    """
    Estimate the trace of the full-batch loss Hessian via Hutchinson's method.

    Args:
        net: Neural network model.
        X: Full input tensor used to construct the loss.
        Y: Full target tensor used to construct the loss.
        loss_fn: Callable loss function applied on the full batch.
        max_estimates: Maximum number of probe vectors to use.
        min_estimates: Minimum number of probes before adaptive stopping is checked.
        eps: Relative standard error tolerance for adaptive stopping.
        generator: Optional RNG to make the estimator deterministic (useful in tests).
        probe_type: Distribution for probe vectors. Currently only 'rademacher' is supported.

    Returns:
        float: Estimated trace of the Hessian.
    """

    if max_estimates < 1:
        raise ValueError("max_estimates must be positive")
    if min_estimates < 1:
        raise ValueError("min_estimates must be positive")
    if min_estimates > max_estimates:
        raise ValueError("min_estimates cannot exceed max_estimates")
    if probe_type != 'rademacher':
        raise NotImplementedError(f"Unsupported probe_type: {probe_type}")

    first_param = next(net.parameters())
    device = first_param.device
    dtype = first_param.dtype

    # Evaluate full-batch loss and build Hessian-vector product closure
    preds = net(X).squeeze(dim=-1)
    loss = loss_fn(preds, Y)
    hessian_matvec = create_hessian_vector_product(loss, net)

    n_params = param_length(net)

    if generator is None:
        generator = gimme_new_rng()

    trace_estimates: List[float] = []

    for i in range(max_estimates):
        # Sample Rademacher probe vector (entries +/-1)
        probe = torch.randint(0, 2, (n_params,), generator=generator, device='cpu', dtype=torch.float32)
        probe = probe.mul_(2.0).sub_(1.0).to(device=device, dtype=dtype)

        Hz = hessian_matvec(probe)
        if Hz.dim() != 1 or Hz.numel() != n_params:
            raise RuntimeError("Hessian-vector product returned unexpected shape")

        trace_component = torch.dot(probe, Hz).detach().item()
        trace_estimates.append(trace_component)

        num_samples = i + 1
        if num_samples < min_estimates:
            continue

        mean_val = float(np.mean(trace_estimates))
        variance = float(np.var(trace_estimates, ddof=1)) if num_samples > 1 else 0.0

        # Avoid division by zero when the estimate is numerically zero
        if abs(mean_val) < 1e-12:
            continue

        sem = np.sqrt(variance / num_samples)
        if sem / abs(mean_val) < eps:
            break

    try:
        wandb.log({"hessian_trace_estimates": len(trace_estimates)}, commit=False)
    except Exception:
        pass

    return float(np.mean(trace_estimates))


def calculate_gradient_norm_squared_mc(net,
                                     X,
                                     Y,
                                     loss_fn,
                                     batch_size,
                                     n_estimates=1000,
                                     min_estimates=10,
                                     eps=0.005  # 0.005 approx gives 1% error; 0.005 = 0.01 / 1.96
                                     ):
    """
    Computes the Monte Carlo estimate of the expected squared norm of mini-batch gradients.
    
    This function estimates E[||∇f_B||²] where f_B is the loss on a mini-batch B,
    using Monte Carlo sampling over random mini-batches.
    
    Args:
        net (nn.Module): Neural network model
        X (Tensor): Input data tensor
        Y (Tensor): Target labels tensor  
        loss_fn (callable): Loss function that takes (outputs, targets) and returns scalar loss
        batch_size (int): Size of mini-batches to sample
        n_estimates (int, optional): Maximum number of MC estimates. Defaults to 1000.
        min_estimates (int, optional): Minimum number of estimates before checking convergence. Defaults to 10.
        eps (float, optional): Relative standard error threshold for convergence. Defaults to 0.005.
        
    Returns:
        float: Monte Carlo estimate of E[||∇f_B||²]
    """
    gradient_norm_squared_vals = []
    
    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)
    
    params = list(net.parameters())
    
    for i in range(n_estimates):
        # Sample random mini-batch
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        
        X_batch = X[random_idx]
        Y_batch = Y[random_idx]
        
        # Compute loss and gradients
        preds = net(X_batch).squeeze(dim=-1)
        loss = loss_fn(preds, Y_batch)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, params, create_graph=False)
        grads_vector = flatt(grads)
        
        # Compute squared norm of gradient
        grad_norm_squared = torch.dot(grads_vector, grads_vector).item()
        gradient_norm_squared_vals.append(grad_norm_squared)
        
        # Check convergence after minimum estimates
        if i >= min_estimates:
            mean_val = np.mean(gradient_norm_squared_vals)
            var_val = np.var(gradient_norm_squared_vals, ddof=1)
            
            # Relative standard error
            rse = np.sqrt(var_val / len(gradient_norm_squared_vals)) / abs(mean_val)
            
            if rse < eps:  # Convergence criterion
                break
    
    # Log number of estimates to wandb if available
    try:
        wandb.log({"gradient_norm_squared_mc_estimates": len(gradient_norm_squared_vals)}, commit=False)
    except:
        pass
    
    return np.mean(gradient_norm_squared_vals)


def calculate_expected_one_step_full_loss_change(net,
                                          X,
                                          Y,
                                          loss_fn,
                                          optimizer,
                                          batch_size,
                                          n_estimates=500,
                                          min_estimates=10,
                                          eps=0.005,  # 0.005 approx gives 1% error; 0.005 = 0.01 / 1.96
                                          eval_batch_size=None,  # For efficient total loss computation,
                                          use_subset_of_data: int = None # use only a subset of the dataset to calculate total loss - speeds up computations!
                                          ):
    """
    Calculate the expected one-step change in total loss using Monte Carlo estimation.
    
    This function estimates the expected change in total dataset loss when making a 
    gradient step on a randomly sampled mini-batch, then returning to the original parameters.
    
    The process for each estimate:
    1. Compute total loss before step (on entire dataset)
    2. Sample a random mini-batch for gradient computation
    3. Store current parameters
    4. Take one optimization step on the mini-batch
    5. Compute total loss after step (on entire dataset)
    6. Calculate change: (loss_after - loss_before)
    7. Restore original parameters
    
    Args:
        net (nn.Module): Neural network model
        X (Tensor): Input data tensor
        Y (Tensor): Target labels tensor
        loss_fn (callable): Loss function that takes (outputs, targets) and returns scalar loss
        optimizer (torch.optim.Optimizer): Optimizer for taking gradient steps
        batch_size (int): Size of mini-batches to sample for gradient computation
        n_estimates (int, optional): Maximum number of MC estimates. Defaults to 500.
        min_estimates (int, optional): Minimum number of estimates before checking convergence. Defaults to 10.
        eps (float, optional): Relative standard error threshold for convergence. Defaults to 0.005.
        eval_batch_size (int, optional): Batch size for total loss evaluation. If None, uses entire dataset.
        
    Returns:
        float: Monte Carlo estimate of expected total loss change
    """
    loss_changes = []
    
    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)
    
    # Store original parameters
    original_params = param_vector(net).detach().clone()
    
    # Compute total loss before any steps (reused for efficiency)
    # with torch.no_grad():
    if eval_batch_size is None or eval_batch_size >= len(X):
        # Evaluate on entire dataset
        preds_total_before = net(X).squeeze(dim=-1)
        total_loss_before = loss_fn(preds_total_before, Y)
    else:
        raise NotImplementedError("Batched evaluation not implemented")
    # else:
    #     # Evaluate on batches to save memory
    #     total_loss_before = 0.0
    #     n_eval_batches = (len(X) + eval_batch_size - 1) // eval_batch_size
    #     for eval_i in range(n_eval_batches):
    #         start_idx = eval_i * eval_batch_size
    #         end_idx = min((eval_i + 1) * eval_batch_size, len(X))
    #         X_eval = X[start_idx:end_idx]
    #         Y_eval = Y[start_idx:end_idx]
    #         preds_eval = net(X_eval).squeeze(dim=-1)
    #         batch_loss = loss_fn(preds_eval, Y_eval)
    #         total_loss_before += batch_loss.item() * len(X_eval)
    #     total_loss_before = total_loss_before / len(X)

    total_loss_before.backward()
    gradient_norm_squared = sum(p.grad.data.norm(2).item() ** 2 for p in net.parameters())
    eta = optimizer.param_groups[0]['lr']

    for i in range(n_estimates):
        # Sample random mini-batch for gradient step
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        
        X_batch = X[random_idx]
        Y_batch = Y[random_idx]
        
        # Take gradient step on mini-batch
        optimizer.zero_grad()
        preds_batch = net(X_batch).squeeze(dim=-1)
        loss_batch = loss_fn(preds_batch, Y_batch)
        loss_batch.backward()
        optimizer.step()
        
        # Compute total loss after step
        with torch.no_grad():
            if eval_batch_size is None or eval_batch_size >= len(X):
                if use_subset_of_data is not None:
                    random_idx = gimme_random_subset_idx(len(X), use_subset_of_data)

                    X_eval = X[random_idx]
                    Y_eval = Y[random_idx]
                else:
                    X_eval = X
                    Y_eval = Y

                # Evaluate on entire dataset
                preds_total_after = net(X_eval).squeeze(dim=-1)
                total_loss_after = loss_fn(preds_total_after, Y_eval)
            else:
                # Evaluate on batches to save memory
                total_loss_after = 0.0
                n_eval_batches = (len(X) + eval_batch_size - 1) // eval_batch_size
                for eval_i in range(n_eval_batches):
                    start_idx = eval_i * eval_batch_size
                    end_idx = min((eval_i + 1) * eval_batch_size, len(X))
                    X_eval = X[start_idx:end_idx]
                    Y_eval = Y[start_idx:end_idx]
                    preds_eval = net(X_eval).squeeze(dim=-1)
                    batch_loss = loss_fn(preds_eval, Y_eval)
                    total_loss_after += batch_loss.item() * len(X_eval)
                total_loss_after = total_loss_after / len(X)
            
            # Calculate change in total loss
            loss_change = total_loss_after - total_loss_before
            loss_changes.append(loss_change.item() if torch.is_tensor(loss_change) else loss_change)
        
        # Restore original parameters
        with torch.no_grad():
            param_idx = 0
            for param in net.parameters():
                param_size = param.numel()
                param.data.copy_(original_params[param_idx:param_idx + param_size].view_as(param))
                param_idx += param_size
        
        # Check convergence after minimum estimates
        if i >= min_estimates:
            mean_val = np.mean(loss_changes)
            var_val = np.var(loss_changes, ddof=1)
            
            # Relative standard error
            rse = np.sqrt(var_val / len(loss_changes)) / abs(mean_val) if mean_val != 0 else float('inf')
            
            if rse < eps:  # Convergence criterion
                break
    
    # Log number of estimates to wandb if available
    try:
        wandb.log({"one_step_total_loss_change_estimates": len(loss_changes)}, commit=False)
    except:
        pass
    
    return np.mean(loss_changes) / (eta * gradient_norm_squared)



def calculate_expected_one_step_batch_loss_change(net,
                                          X,
                                          Y,
                                          loss_fn,
                                          optimizer,
                                          batch_size,
                                          n_estimates=500,
                                          min_estimates=10,
                                          eps=0.005  # 0.005 approx gives 1% error; 0.005 = 0.01 / 1.96
                                          ):
    """
    Calculate the expected one-step change in loss using Monte Carlo estimation.
    
    This function estimates the expected relative change in loss when making a 
    gradient step on a randomly sampled batch, then returning to the original parameters.
    
    The process for each estimate:
    1. Sample a random batch
    2. Store current parameters
    3. Compute loss before step
    4. Take one optimization step
    5. Compute loss after step  
    6. Calculate relative change: (loss_after - loss_before) / loss_before
    7. Restore original parameters
    
    Args:
        net (nn.Module): Neural network model
        X (Tensor): Input data tensor
        Y (Tensor): Target labels tensor
        loss_fn (callable): Loss function that takes (outputs, targets) and returns scalar loss
        optimizer (torch.optim.Optimizer): Optimizer for taking gradient steps
        batch_size (int): Size of mini-batches to sample
        n_estimates (int, optional): Maximum number of MC estimates. Defaults to 500.
        min_estimates (int, optional): Minimum number of estimates before checking convergence. Defaults to 10.
        eps (float, optional): Relative standard error threshold for convergence. Defaults to 0.005.
        
    Returns:
        float: Monte Carlo estimate of expected relative one-step loss change
    """
    loss_changes = []
    
    # Create independent RNG using current time and process info for true randomness
    entropy_seed = int((time.time() * 1000000) % (2**32)) ^ os.getpid()
    rng = torch.Generator()
    rng.manual_seed(entropy_seed)
    
    # Store original parameters
    original_params = param_vector(net).detach().clone()
    
    for i in range(n_estimates):
        # Sample random mini-batch
        shuffle = T.randperm(len(X), generator=rng)
        random_idx = shuffle[:batch_size]
        
        X_batch = X[random_idx]
        Y_batch = Y[random_idx]
        
        # Compute loss before step
        optimizer.zero_grad()
        preds_before = net(X_batch).squeeze(dim=-1)
        loss_before = loss_fn(preds_before, Y_batch)
        
        # Take gradient step
        loss_before.backward()
        optimizer.step()
        
        # Compute loss after step (on the same batch)
        with torch.no_grad():
            preds_after = net(X_batch).squeeze(dim=-1)
            loss_after = loss_fn(preds_after, Y_batch)
            
            # Calculate relative change in loss
            relative_change = (loss_after - loss_before) #/ loss_before
            loss_changes.append(relative_change.item())
        
        # Restore original parameters
        current_params = param_vector(net)
        with torch.no_grad():
            param_idx = 0
            for param in net.parameters():
                param_size = param.numel()
                param.data.copy_(original_params[param_idx:param_idx + param_size].view_as(param))
                param_idx += param_size
        
        # Check convergence after minimum estimates
        if i >= min_estimates:
            mean_val = np.mean(loss_changes)
            var_val = np.var(loss_changes, ddof=1)
            
            # Relative standard error
            rse = np.sqrt(var_val / len(loss_changes)) / abs(mean_val) if mean_val != 0 else float('inf')
            
            if rse < eps:  # Convergence criterion
                break
    
    # Log number of estimates to wandb if available
    try:
        wandb.log({"one_step_loss_change_estimates": len(loss_changes)}, commit=False)
    except:
        pass
    
    return np.mean(loss_changes)


################################################################################
#                                                                              #
#                        GAUSS–NEWTON (=FIM) MATRIX STUFF                      #
#                                                                              #
################################################################################

def calculate_all_net_grads(net, X):

    gradients = []
    params = list(net.parameters())

    for x in X:
        y = net(x.unsqueeze(0))
        # compute gradient
        grads = torch.autograd.grad(y, params)
        grads_vector = flatt(grads).detach()
        gradients.append(grads_vector)
    
    G = T.stack(gradients)
    del gradients
    return G



def create_ntk(net, X):
    params = list(net.parameters())

    gradients = []

    for x in X:
        y = net(x.unsqueeze(0))
        # compute gradient
        grads = torch.autograd.grad(y, params)
        grads_vector = flatt(grads).detach()
        gradients.append(grads_vector)
    
    G = T.stack(gradients)

    ntk = G @ G.T
    del G
    # f = lambda v: G.T @ (G @ v) / len(X)

    return ntk


def compute_fisher_eigenvalues(net, X):
    '''
    The trick here is that instead of computing the fisher information matrix, we compute the NTK
    They have the same eigenvalues, but NTK is size n_samples x n_samples, while FIM is size n_params x n_params
    '''

    ntk = create_ntk(net, X)
    # size = param_length(net)

    # device = next(net.parameters()).device
    # eigenval = compute_eigenvalues(operator, size, device, iterations=iterations, epsilon=epsilon)

    eigenval = T.lobpcg(ntk, k=1)
    eigenval = 2/len(X) * eigenval[0]
    
    return eigenval




################################################################################
#                                                                              #
#                                LAMBDA^b_MAX                                  #
#                                                                              #
################################################################################


def calculate_averaged_lambdamax(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 500,
                              min_estimates = 10,
                              tolerance = 0.01, # st error of mean / mean
                              max_hessian_iters = 1000,
                              hessian_tolerance = 1e-3,
                              batched = None,
                              compute_gHg: bool = False,
                              eigenvector_cache = None
                              ): 
    
    
    sharpnesses = []

    if compute_gHg:
        gHg_values = []
    
    if batch_size is None:
        batch_size = len(X)
    

    for i in range(n_estimates):
        shuffle = T.randperm(len(X))
        random_idx = shuffle[:batch_size]

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]


        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        sharpness = compute_eigenvalues(loss, 
                        net,
                        max_iterations=max_hessian_iters,
                        reltol=hessian_tolerance,
                        )
        if compute_gHg:
            sharpness, gHg = sharpness
            gHg = gHg.item()
            gHg_values.append(gHg)
        
        sharpness = sharpness.item()
        
        sharpnesses.append(sharpness)

        if batch_size >= len(X):
            break

        if len(sharpnesses) > min_estimates:
            mean = np.mean(sharpnesses)
            sem = np.std(sharpnesses) / np.sqrt(len(sharpnesses))

            if sem / mean < tolerance:
                break
    
    if compute_gHg:
        return sharpnesses, gHg_values
    return sharpnesses
