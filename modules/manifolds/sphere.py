import numpy as np

from .manifold import Manifold


class Sphere(Manifold):
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
