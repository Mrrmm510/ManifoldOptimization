from typing import Union

import numpy as np
from scipy.linalg import eigh
from sklearn.exceptions import NotFittedError


class TwoDimensionalPCA:
    def __init__(self, n_components: Union[int, None] = None):
        self.n_components_ = n_components

        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, height, width)
        """
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got {X.ndim}D array instead")

        self.mean_ = np.mean(X, axis=0)
        cov = np.mean([x.T @ x for x in X - self.mean_], axis=0)
        n_features = cov.shape[0]

        if self.n_components_ is None or self.n_components_ > n_features:
            self.n_components_ = n_features
        self.components_ = eigh(cov, eigvals=(n_features - self.n_components_, n_features - 1))[1][:, ::-1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, height, width)

        Returns
        -------
        : np.ndarray, shape (n_samples, height, n_components)
        """
        if self.components_ is None:
            raise NotFittedError(
                "This PCA instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator.")

        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got {X.ndim}D array instead")

        return np.array([x @ self.components_ for x in X])

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, height, n_components)

        Returns
        -------
        : np.ndarray, shape (n_samples, height, width)
        """
        return np.array([x @ self.components_.T for x in X])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, height, width)

        Returns
        -------
        : np.ndarray, shape (n_samples, height, n_components)
        """
        self.fit(X)
        return self.transform(X)
