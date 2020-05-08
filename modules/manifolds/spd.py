import numpy as np
from scipy.linalg import eigh

from .manifold import Manifold
from ..symmetric_matrix import expm, logm, symm, triu_indices, v2vec


class SPD(Manifold):
    """
    Symmetric Positive Definite Matrix
    """
    def __init__(self, n: int = None):
        self.n = n

    def retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute R_x(v)
        """
        return symm(x + v + 0.5 * v.dot(np.linalg.inv(x).dot(v)))

    def exp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute exp_x(v)
        """
        xv = np.linalg.inv(x).dot(v)
        return symm(x.dot(expm(xv)))

    def log(self, x: np.ndarray, y: np.ndarray):
        """
        Compute log_x(y)
        """
        xy = np.linalg.inv(x).dot(y)
        return symm(x.dot(logm(xy)))

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute distance from x to y
        """
        w = eigh(x, y, eigvals_only=True)
        return float(np.linalg.norm(np.log(w)**2))

    def inner_product(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute g_x(u, v)
        """
        x_inv = np.linalg.inv(x)
        return float(np.vdot(u.dot(x_inv), v.dot(x_inv)))

    def gradient(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        ∇f(x) -> grad f(x)
        """
        n = self.n if self.n is not None else x.shape[0]
        index = triu_indices(n)
        return (np.kron(x, x).dot(v2vec(d[index], n))).reshape(n, n)


class SPDs(Manifold):
    def __init__(self, n: int = None):
        self.spd = SPD(n=n)

    def retraction(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Compute R_x(v)
        """
        ret = np.array([self.spd.retraction(x, v) for x, v in zip(X, V)])
        return ret

    def exp(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Compute exp_x(v)
        """
        ret = np.array([self.spd.exp(x, v) for x, v in zip(X, V)])
        return ret

    def log(self, X: np.ndarray, Y: np.ndarray):
        """
        Compute log_x(y)
        """
        ret = np.array([self.spd.log(x, y) for x, y in zip(X, Y)])
        return ret

    def inner_product(self, X: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
        """
        Compute g_x(u, v)
        """
        return float(np.sum([self.spd.inner_product(x, u, v) for x, u, v in zip(X, U, V)]))

    def gradient(self, X: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        ∇f(x) -> grad f(x)
        """
        ret = np.array([self.spd.gradient(x, d) for x, d in zip(X, D)])
        return ret
