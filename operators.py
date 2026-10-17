"""Proximal maps and projections used by VMOM-Lab."""
import numpy as np

def soft_threshold(x, threshold):
    """Coordinatewise soft-thresholding."""
    x = np.asarray(x, dtype=float)
    if threshold < 0:
        raise ValueError("threshold must be nonnegative")
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

def prox_l1(x, tau=1.0, weight=1.0):
    """Proximal operator of tau * weight * ||.||_1."""
    if tau < 0 or weight < 0:
        raise ValueError("tau and weight must be nonnegative")
    return soft_threshold(x, tau * weight)

def prox_group_lasso(x, groups, tau=1.0, weight=1.0):
    """Block soft-thresholding for disjoint index groups."""
    x = np.asarray(x, dtype=float)
    if tau < 0 or weight < 0:
        raise ValueError("tau and weight must be nonnegative")
    out, seen = x.copy(), set()
    for group in groups:
        idx = np.asarray(list(group), dtype=int)
        if idx.size == 0:
            continue
        if np.any(idx < 0) or np.any(idx >= x.size):
            raise IndexError("group index out of range")
        if any(int(i) in seen for i in idx):
            raise ValueError("groups must be disjoint")
        seen.update(int(i) for i in idx)
        norm = np.linalg.norm(x[idx])
        factor = max(1.0 - tau * weight / norm, 0.0) if norm > 0 else 0.0
        out[idx] = factor * x[idx]
    return out

def project_box(x, lower=0.0, upper=1.0):
    """Euclidean projection onto a coordinatewise box."""
    if np.any(np.asarray(lower) > np.asarray(upper)):
        raise ValueError("lower must not exceed upper")
    return np.clip(np.asarray(x, dtype=float), lower, upper)
