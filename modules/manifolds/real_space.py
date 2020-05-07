import numpy as np

from .manifold import Manifold


class RealSpace(Manifold):
    def inner_product(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
        return float(np.dot(u, v))

    def retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return x + v

    def vector_transport(self, x: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        return a


class RealSpace2D(Manifold):
    def inner_product(self, X: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
        return float(np.vdot(U, V))

    def retraction(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        return X + V

    def exp(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        return X + V

    def log(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return Y - X

    def vector_transport(self, X: np.ndarray, V: np.ndarray, A: np.ndarray) -> np.ndarray:
        return A
