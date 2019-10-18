from typing import Tuple

import numpy as np


def sqrtm(A: np.ndarray, w: np.ndarray = None, v: np.ndarray = None) -> np.ndarray:
    """
    Return a matrix G which satisfies A = GG.
    """
    if w is None or v is None:
        w, v = np.linalg.eig(A)
    return v.dot(np.diag(np.sqrt(w)).dot(v.T))


def expm(A: np.ndarray, w: np.ndarray = None, v: np.ndarray = None) -> np.ndarray:
    """
    Compute Exp(A).
    """
    if w is None or v is None:
        w, v = np.linalg.eig(A)
    return v.dot(np.diag(np.exp(w)).dot(v.T))


def logm(A: np.ndarray, w: np.ndarray = None, v: np.ndarray = None) -> np.ndarray:
    """
    Compute Log(A).
    """
    if w is None or v is None:
        w, v = np.linalg.eig(A)
    return v.dot(np.diag(np.log(w)).dot(v.T))


def triu_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Example:
        Parameters
        ----------
        n: 3

        Returns
        ----------
        : ([0, 1, 2, 0, 2, 0], [0, 1, 2, 1, 2, 2])
    """
    return (np.hstack([np.arange(n)[:n - i] for i in range(n)]),
            np.hstack([np.arange(n)[i:] for i in range(n)]))


def mat2v(A: np.ndarray) -> np.ndarray:
    """
    Example:
        Parameters
        ----------
        A: [[1,7,9],
            [7,2,8],
            [9,8,3]]

        Returns
        ----------
        : [1,2,3,7,8,9]
    """
    n, _ = A.shape
    return A[triu_indices(n)]


def v2mat(v: np.ndarray, size: int) -> np.ndarray:
    """
    Example:
        Parameters
        ----------
        v: [1,2,3,7,8,9]

        size: 3

        Returns
        ----------
        res: [[1,7,9],
              [7,2,8],
              [9,8,3]]
    """
    res = np.zeros((size, size))
    index = triu_indices(size)
    res[index] = v
    res.T[index] = v
    return res


def v2vec(v: np.ndarray, size: int) -> np.ndarray:
    """
    Example:
        Parameters
        ----------
        v: [1,2,3,7,8,9]

        size: 3

        Returns
        ----------
        res: [[1,3.5,4.5,3.5,2,4,4.5,4,3]]
    """
    res = v2mat(v, size)
    res = 0.5 * (res + np.diag(np.diag(res)))
    return res.flatten()
