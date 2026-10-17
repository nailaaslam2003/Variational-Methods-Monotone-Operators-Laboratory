"""First-order algorithms for convex composite optimization."""
import numpy as np
from .operators import prox_l1, project_box

def _validate_lasso(A, y, lam, max_iter, tol):
    A, y = np.asarray(A, dtype=float), np.asarray(y, dtype=float).reshape(-1)
    if A.ndim != 2 or A.shape[0] != y.size:
        raise ValueError("A must be 2D and its row count must match y")
    if lam < 0 or max_iter < 1 or tol < 0:
        raise ValueError("lam/tol must be nonnegative and max_iter positive")
    return A, y

def _objective(A, y, x, lam):
    r = A @ x - y
    return 0.5 * np.dot(r, r) / A.shape[0] + lam * np.sum(np.abs(x))

def _lipschitz(A):
    return max(float(np.linalg.norm(A, ord=2)**2 / A.shape[0]), 1e-12)

def ista(A, y, lam=0.1, max_iter=2000, tol=1e-7, x0=None):
    """ISTA for ||Ax-y||²/(2n) + lam*||x||₁."""
    A, y = _validate_lasso(A, y, lam, max_iter, tol)
    p = A.shape[1]
    x = np.zeros(p) if x0 is None else np.asarray(x0, dtype=float).copy()
    if x.shape != (p,):
        raise ValueError("x0 must have one entry per column of A")
    L, history = _lipschitz(A), []
    for _ in range(max_iter):
        grad = A.T @ (A @ x - y) / A.shape[0]
        xn = prox_l1(x - grad/L, tau=lam/L)
        history.append(_objective(A, y, xn, lam))
        if np.linalg.norm(xn-x) <= tol * max(1.0, np.linalg.norm(x)):
            x = xn
            break
        x = xn
    return {"x": x, "objective": np.asarray(history), "iterations": len(history), "L": L}

def fista(A, y, lam=0.1, max_iter=2000, tol=1e-7, x0=None, monotone=False):
    """Accelerated proximal gradient, optionally with monotone restart safeguard."""
    A, y = _validate_lasso(A, y, lam, max_iter, tol)
    p = A.shape[1]
    x = np.zeros(p) if x0 is None else np.asarray(x0, dtype=float).copy()
    if x.shape != (p,):
        raise ValueError("x0 must have one entry per column of A")
    z, t, L, history = x.copy(), 1.0, _lipschitz(A), []
    previous_obj = _objective(A, y, x, lam)
    for _ in range(max_iter):
        grad = A.T @ (A @ z-y) / A.shape[0]
        candidate = prox_l1(z-grad/L, tau=lam/L)
        candidate_obj = _objective(A, y, candidate, lam)
        if monotone and candidate_obj > previous_obj + 1e-12:
            grad = A.T @ (A @ x-y) / A.shape[0]
            candidate = prox_l1(x-grad/L, tau=lam/L)
            candidate_obj = _objective(A, y, candidate, lam)
            tn, zn = 1.0, candidate.copy()
        else:
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0*t*t))
            zn = candidate + ((t-1.0)/tn)*(candidate-x)
        history.append(candidate_obj)
        if np.linalg.norm(candidate-x) <= tol*max(1.0, np.linalg.norm(x)):
            x = candidate
            break
        x, z, t, previous_obj = candidate, zn, tn, candidate_obj
    return {"x": x, "objective": np.asarray(history), "iterations": len(history), "L": L}

def forward_backward(grad_f, prox_g, x0, step, max_iter=1000, tol=1e-8):
    """Generic forward-backward splitting for min f+g."""
    if step <= 0 or max_iter < 1 or tol < 0:
        raise ValueError("invalid step, max_iter, or tol")
    x = np.asarray(x0, dtype=float).copy()
    for k in range(max_iter):
        xn = np.asarray(prox_g(x-step*grad_f(x), step), dtype=float)
        if xn.shape != x.shape:
            raise ValueError("prox_g returned an array with the wrong shape")
        residual = float(np.linalg.norm(xn-x))
        if residual <= tol*max(1.0, np.linalg.norm(x)):
            return {"x": xn, "iterations": k+1, "residual": residual}
        x = xn
    return {"x": x, "iterations": max_iter, "residual": residual}

def projected_gradient_vi(F, x0, lower=0.0, upper=1.0, step=0.1,
                          max_iter=1000, tol=1e-8):
    """Projected iteration for a variational inequality on a box.

    Convergence needs suitable monotonicity/Lipschitz assumptions and step size.
    """
    if step <= 0 or max_iter < 1 or tol < 0:
        raise ValueError("invalid step, max_iter, or tol")
    x = np.asarray(x0, dtype=float).copy()
    for k in range(max_iter):
        xn = project_box(x-step*np.asarray(F(x), dtype=float), lower, upper)
        residual = float(np.linalg.norm(xn-x))
        if residual <= tol*max(1.0, np.linalg.norm(x)):
            return {"x": xn, "iterations": k+1, "residual": residual}
        x = xn
    residual = float(np.linalg.norm(project_box(x-step*np.asarray(F(x)), lower, upper)-x))
    return {"x": x, "iterations": max_iter, "residual": residual}
