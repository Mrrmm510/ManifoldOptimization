import numpy as np

from .manifold import Manifold


class RealSpace(Manifold):
    def inner_product(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
        return float(np.dot(u, v))

    def retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return x + v

    def vector_transport(self, x: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        return a
