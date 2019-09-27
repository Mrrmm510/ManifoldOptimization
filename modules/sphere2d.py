import numpy as np


def distance(X: np.ndarray, Y: np.ndarray) -> float:
    return np.sum(X * Y, axis=-1, keepdims=True)

def exp(X: np.ndarray, V: np.ndarray, t: float = 1.0) -> np.ndarray:
    """
    Compute exp_x(tv)
    """
    norm_V = np.linalg.norm(V, axis=-1, keepdims=True)
    norm_V[norm_V == 0] = 1.0
    return np.cos(t * norm_V) * X + np.sin(t * norm_V) * V / norm_V

def log(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute log_u(v)
    """
    dot = np.clip(np.sum(U * V, axis=-1, keepdims=True), -1, 1)
    vec = V - dot * U
    norm_vec = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm_vec[norm_vec==0] = 1.0
    return vec / norm_vec * np.arccos(dot)