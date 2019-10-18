import numpy as np

from .manifold import Manifold


class Sphere(Manifold):
    def inner_product(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute g_x(u, v)
        """
        return u.dot(v)

    def retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute R_x(v)
        """
        ret = x + v
        norm = np.linalg.norm(ret)
        return ret / norm if norm != 0 else ret

    def vector_transport(self, x: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        w = x + v
        return a - np.dot(v, a) / np.dot(w, w) * w


class Spheres(Manifold):
    def inner_product(self, X: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
        """
        Compute g_x(u, v)
        """
        return np.sum(U * V)

    def distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Arccos distance
        """
        dot = np.clip(np.sum(X * Y, axis=-1, keepdims=True), -1, 1)
        return np.arccos(dot)

    def exp(self, X: np.ndarray, V: np.ndarray, t: float = 1.0) -> np.ndarray:
        """
        Compute exp_x(tv)
        """
        norm_V = np.linalg.norm(V, axis=-1, keepdims=True)
        arg = t * norm_V
        norm_V[norm_V == 0] = 1.0
        ret = np.cos(arg) * X + np.sin(arg) * V / norm_V
        return ret / np.linalg.norm(ret, axis=-1, keepdims=True)

    def log(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Compute log_u(v)
        """
        dot = np.clip(np.sum(U * V, axis=-1, keepdims=True), -1, 1)
        vec = V - dot * U
        norm_vec = np.linalg.norm(vec, axis=-1, keepdims=True)
        norm_vec[norm_vec==0] = 1.0
        return vec / norm_vec * np.arccos(dot)

    def retraction(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Compute R_x(v)
        """
        ret = X + V
        return ret / np.linalg.norm(ret, axis=-1, keepdims=True)
