from typing import Tuple

import numpy as np

from .manifold import Manifold
from ..symmetric_matrix import sqrtm, expm, logm, triu_indices, v2vec


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
        return self.exp(x, v)

    def exp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute exp_x(v)
        """
        x_sqrt = sqrtm(x)
        x_sqrt_inv = np.linalg.inv(x_sqrt)
        xvx = x_sqrt_inv.dot(v.dot(x_sqrt_inv))
        return x_sqrt.dot(expm(xvx).dot(x_sqrt))

    def log(self, x: np.ndarray, y: np.ndarray):
        """
        Compute log_x(y)
        """
        x_sqrt = sqrtm(x)
        x_sqrt_inv = np.linalg.inv(x_sqrt)
        xyx = x_sqrt_inv.dot(y.dot(x_sqrt_inv))
        return x_sqrt.dot(logm(xyx).dot(x_sqrt))

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute distance from x to y
        """
        x_inv = np.linalg.inv(x)
        xy = np.dot(x_inv, y)
        w, _ = np.linalg.eig(xy)
        return np.sum(np.log(w)**2)

    def inner_product(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
        """
        Compute g_x(u, v)
        """
        x_inv = np.linalg.inv(x)
        return np.sum(u.dot(x_inv) * v.dot(x_inv))

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
        return self.exp(X, V)

    def exp(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Compute exp_x(v)
        """
        ret = np.zeros_like(X)
        for i, (x, v) in enumerate(zip(X, V)):
            ret[i] = self.spd.exp(x, v)
        return ret

    def log(self, X: np.ndarray, Y: np.ndarray):
        """
        Compute log_x(y)
        """
        ret = np.zeros_like(X)
        for i, (x, y) in enumerate(zip(X, Y)):
            ret[i] = self.spd.log(x, y)
        return ret

    def inner_product(self, X: np.ndarray, U: np.ndarray, V: np.ndarray) -> float:
        """
        Compute g_x(u, v)
        """
        return np.sum([self.spd.inner_product(x, u, v) for x, u, v in zip(X, U, V)])

    def gradient(self, X: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        ∇f(x) -> grad f(x)
        """
        ret = np.zeros_like(D)
        for i, (x, d) in enumerate(zip(X, D)):
            ret[i] = self.spd.gradient(x, d)
        return ret
