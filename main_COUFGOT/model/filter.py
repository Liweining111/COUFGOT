import torch
import numpy as np
from numpy import linalg as la  # Kept, but Torch is preferred
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math

def compute_covariance_matrix(X, mode='sample'):
    r"""
    Compute sample covariance or feature covariance matrix
    mode: 'sample' computes L = X @ X.T, 'feature' computes l = X.T @ X
    Supports numpy or Torch
    """
    if isinstance(X, torch.Tensor):
        if mode == 'sample':
            return X @ X.T
        elif mode == 'feature':
            return X.T @ X
        else:
            raise ValueError("mode must be 'sample' or 'feature'")
    else:  # numpy
        if mode == 'sample':
            return X @ X.T
        elif mode == 'feature':
            return X.T @ X
        else:
            raise ValueError("mode must be 'sample' or 'feature'")

def apply_filter(L, filter_type='identity', device='cpu'):
    """
    Apply filter to matrix (based on SVD), supports Torch/CUDA
    L: input covariance matrix (numpy or Torch Tensor)
    filter_type: filter type
    device: 'cuda' or 'cpu' (if L is Torch)
    """
    # Convert to Torch
    if isinstance(L, np.ndarray):
        L = torch.from_numpy(L).float().to(device)
    elif isinstance(L, torch.Tensor):
        L = L.float().to(device)
    else:
        raise ValueError("L must be numpy array or Torch Tensor")

    # Torch SVD (full_matrices=False is equivalent to low-rank form)
    if device == 'cuda' or torch.cuda.is_available():
        U, S, Vh = torch.linalg.svd(L, full_matrices=False)  # Torch 1.9+ supported
    else:
        # Fallback to numpy if no CUDA
        L_np = L.cpu().numpy()
        U, S, Vh = la.svd(L_np, full_matrices=False)
        U = torch.from_numpy(U).float()
        S = torch.from_numpy(S).float()
        Vh = torch.from_numpy(Vh).float()

    # Apply filter to singular values (S is a vector), create a zero matrix with exactly the same shape as S
    sigma_filtered = torch.zeros_like(S)
    for i in range(len(S)):
        if filter_type == 'g1':  # Square root
            sigma_filtered[i] = math.sqrt(S[i].item())
        elif filter_type == 'g2':  # Square
            sigma_filtered[i] = math.pow(S[i].item(), 2)
        elif filter_type == 'g3':  # Exponential decay
            sigma_filtered[i] = math.exp(-0.8 * S[i].item())
        elif filter_type == 'g4':  # Square + square root
            sigma_filtered[i] = math.pow(S[i].item(), 2) + math.sqrt(S[i].item())
        elif filter_type == 'g5':  # Exponential decay + square root
            sigma_filtered[i] = math.exp(-0.8 * S[i].item()) + math.sqrt(S[i].item())
        elif filter_type == 'g6':  # Exponential decay + square
            sigma_filtered[i] = math.exp(-0.8 * S[i].item()) + math.pow(S[i].item(), 2)
        elif filter_type == 'identity':  # Identity filter
            sigma_filtered[i] = S[i]
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    # Reconstruct filtered matrix (U @ diag(sigma) @ Vh)
    sigma_diag = torch.diag(sigma_filtered)
    L_filtered = U @ sigma_diag @ Vh

    # Normalization: Torch version of MinMaxScaler
    L_min = L_filtered.min(dim=0)[0].unsqueeze(0)  # Min per column
    L_max = L_filtered.max(dim=0)[0].unsqueeze(0)  # Max per column
    L_filtered = (L_filtered - L_min.T) / (L_max.T - L_min.T + 1e-8)  # Avoid division by zero

    return L_filtered.cpu().numpy()  # Return numpy

def g1(X, mode='sample'):
    """Apply g1 filter to covariance matrix"""
    L = compute_covariance_matrix(X, mode)
    return apply_filter(L, 'g1')


def g2(X, mode='sample'):
    """Apply g2 filter to covariance matrix"""
    L = compute_covariance_matrix(X, mode)
    return apply_filter(L, 'g2')


def g3(X, mode='sample'):
    """Apply g3 filter to covariance matrix"""
    L = compute_covariance_matrix(X, mode)
    return apply_filter(L, 'g3')


def g4(X, mode='sample'):
    """Apply g4 filter to covariance matrix"""
    L = compute_covariance_matrix(X, mode)
    return apply_filter(L, 'g4')


def g5(X, mode='sample'):
    """Apply g5 filter to covariance matrix"""
    L = compute_covariance_matrix(X, mode)
    return apply_filter(L, 'g5')


def g6(X, mode='sample'):
    """Apply g6 filter to covariance matrix"""
    L = compute_covariance_matrix(X, mode)
    return apply_filter(L, 'g6')


def identity(X, mode='sample'):
    """Identity filter"""
    L = compute_covariance_matrix(X, mode)
    return apply_filter(L, 'identity')