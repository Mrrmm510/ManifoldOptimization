from itertools import product

import numpy as np
from tqdm import trange
from joblib import Parallel, delayed


from ..gradient_descent import GradientDescent
from .sparse_coding import EqualityConstrainedL1QuadraticProgramming
from ..symmetric_matrix import logm, symm_derivative


def _initialize_dictionary(n_components: int, n_features: int) -> np.ndarray:
    """
    Initialize a dictionary.

    Parameters
    ----------
    n_components : int
        The number of atoms

    n_features : int
        The number of features

    Returns
    -------
    : np.ndarray, shape = (n_components, n_features, n_features)
        Initialized dictionary.
    """
    np.random.seed(0)
    x = np.random.randn(n_components, n_features, n_features)
    return np.array([y.dot(y.T) for y in x])


class AffineConstrainedSPDDLSC(GradientDescent):
    """
    Problem
    ----------
    minimize_D,W ∑_i||∑_j w_ij log_x_i(a_j)||_x_i^2 + λ||W||_1
    subject to ∑_j w_ij = 1
    where D = {a_1, ..., a_m}

    Parameters
    ----------
    n_components : int, optional (default=100)
        The number of atoms.

    max_iter : int, optional (default=300)
        The maximum number of iterations.
    """

    def __init__(
            self,
            n_components: int = 100,
            max_iter: int = 300,
            extended_output: bool = False,
            eps: float = 1e-8,
            initial_step: float = 1.0,
            armijo_param: float = 1e-4,
            max_iter_dl: int = 300,
            rho: float = 1.0,
            tau: float = 1.0,
            tol: float = 1e-4,
            max_iter_sp: int = 300
    ):
        super().__init__(
            manifold='spds',
            initial_step=initial_step,
            armijo_param=armijo_param,
            max_iter=max_iter_dl,
            extended_output=False
        )
        self.n_components = n_components
        self.total_iter = max_iter
        self.extended_output_ = extended_output
        self.eps = eps

        # sparse coder
        self.sparse_coder = EqualityConstrainedL1QuadraticProgramming(
            A=np.ones((1, self.n_components)),
            b=np.array([1.]),
            rho=rho,
            tau=tau,
            tol=tol,
            max_iter=max_iter_sp,
            extended_output=False
        )

        self.X_inv = None
        self.D = None
        self.W = None
        self.f = list()

    def _coef_matrix(self, x_inv: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Coefficient matrix.

        Parameters
        ----------
        x_inv : np.ndarray, shape = (n_features, n_features)
            Inverse data.

        D : np.ndarray, shape = (n_components, n_features, n_features)
            Atoms.

        Returns
        -------
        L_trace : np.ndarray, shape = (n_components, n_components)
        """
        n_features = x_inv.shape[0]
        coef = [logm(x_inv.dot(aj)) for aj in D]
        coef_matrix = np.sum(
            np.tile(coef, (self.n_components, 1, 1)) *
            np.tile(coef, (self.n_components, 1)).reshape(
                (self.n_components * self.n_components, n_features, n_features)
            ),
            axis=(1, 2)).reshape(self.n_components, self.n_components)
        return coef_matrix

    def _fi(self, D: np.ndarray, x_inv: np.ndarray, w: np.ndarray) -> float:
        """
        Objective function for one data.

        Parameters
        ----------
        D : np.ndarray, shape = (n_components, n_features, n_features)
            Atoms.

        x_inv : np.ndarray, shape = (n_features, n_features)
            Inverse data.

        w : np.ndarray, shape = (n_components, )
            Weight.

        Returns
        -------
        : float
        """
        return 0.5 * w.dot(self._coef_matrix(x_inv, D).dot(w))

    def _dfi(self, D: np.ndarray, D_inv: np.ndarray, x_inv: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Gradient of objective function for one data.

        Parameters
        ----------
        D : np.ndarray, shape = (n_components, n_features, n_features)
            Atoms.

        D_inv : np.ndarray, shape = (n_components, n_features, n_features)
            Inverse atoms.

        x_inv : np.ndarray, shape = (n_features, n_features)
            Inverse data.

        w : np.ndarray, shape = (n_components, )
            Weight.

        Returns
        -------
        df : np.ndarray, shape = (n_components, n_features, n_features)
        """
        coef_matrix = [logm(x_inv.dot(aj)) for aj in D]
        df = np.zeros_like(D)
        for j, k in product(range(self.n_components), range(self.n_components)):
            df[j] += symm_derivative(coef_matrix[k].T.dot(D_inv[j]) * w[j] * w[k])
        return df

    def _f(self, D: np.ndarray) -> float:
        """
        Objective function.

        Parameters
        ----------
        D : np.ndarray, shape = (n_components, n_features, n_features)
            Atoms.

        Returns
        -------
        : float
        """
        fi = Parallel(n_jobs=-1)(
            [delayed(self._fi)(D, x_inv, w) for x_inv, w in zip(self.X_inv, self.W)]
        )
        return float(np.mean(fi))

    def _df(self, D: np.ndarray) -> np.ndarray:
        """
        Gradient.

        Parameters
        ----------
        D : np.ndarray, shape = (n_components, n_features, n_features)
            Atoms.

        Returns
        -------
        : np.ndarray, shape = (n_components, n_features, n_features)
        """
        D_inv = np.array([np.linalg.inv(a) for a in D])

        dfi = Parallel(n_jobs=-1)(
            [delayed(self._dfi)(D, D_inv, x_inv, w) for x_inv, w in zip(self.X_inv, self.W)]
        )
        return self.manifold.gradient(D, np.mean(dfi, axis=0))

    def _update_one_weight(self, x_inv: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Update a weight.

        Parameters
        ----------
        x_inv : np.ndarray, shape = (n_features, n_features)
            Inverse data.

        D : np.ndarray, shape = (n_components, n_features, n_features)
            Atoms.

        Returns
        -------
        : np.ndarray, shape = (n_components, )
        """
        coef_matrix = self._coef_matrix(x_inv, D)
        self.sparse_coder.fit(coef_matrix, np.zeros(self.n_components))
        return self.sparse_coder.coef_

    def _update_weight(self, X_inv: np.ndarray, D: np.ndarray) -> np.ndarray:
        """
        Update a weight.

        Parameters
        ----------
        X_inv : np.ndarray, shape = (n_samples, n_features, n_features)
            Inverse data.

        D : np.ndarray, shape = (n_components, n_features, n_features)
            Atoms.

        Returns
        -------
        : np.ndarray, shape = (n_samples, n_components)
        """
        W = Parallel(n_jobs=-1)(
            [delayed(self._update_one_weight)(x_inv, D) for x_inv in X_inv]
        )
        # W = [self._update_one_weight(x_inv, D) for x_inv in X_inv]
        return np.array(W)

    def _initialize_data(self, X: np.ndarray) -> None:
        """
        Compute arrays in advance.

        Parameters
        ----------
        X : np.ndarray, shape = (n_samples, n_features, n_features)
        Data.

        Returns
        -------
        None
        """
        self.X_inv = np.array(list(map(np.linalg.inv, X)))

    def fit(self, X: np.ndarray) -> None:
        """
        Fit.

        Parameters
        ----------
        X : np.ndarray, shape = (n_samples, n_features, n_features)
            Data.

        Returns
        -------
        : None
        """
        n_samples, n_features, _ = X.shape
        D = _initialize_dictionary(self.n_components, n_features)

        self._initialize_data(X)

        # main loop
        for _ in trange(self.total_iter):
            # update weight
            self.W = self._update_weight(self.X_inv, D)

            # save objective function value if necessary
            if self.extended_output_:
                self.f.append(self._f(D) + self.sparse_coder.rho * np.sum(np.abs(self.W)))

            # update dictionary
            D = self.optimize(D)
            D += self.eps * np.tile(np.eye(n_features), (self.n_components, 1, 1))

            # save objective function value if necessary
            if self.extended_output_:
                self.f.append(self._f(D) + self.sparse_coder.rho * np.sum(np.abs(self.W)))

        self.D = D

    def transform(self, X: np.ndarray):
        """
        Transform.

        Parameters
        ----------
        X : np.ndarray, shape = (n_samples, n_features, n_features)
            Data.

        Returns
        -------
        : np.ndarray, shape = (n_samples, n_components)
            Weights
        """
        if self.D is None:
            raise Exception('Not Fitted!')
        X_inv = np.array([np.linalg.inv(x) for x in X])
        return self._update_weight(X_inv, self.D)

    def fit_transform(self, X: np.ndarray):
        """
        Fit and transform.

        Parameters
        ----------
        X : np.ndarray, shape = (n_samples, n_features, n_features)
            Data.

        Returns
        -------
        : np.ndarray, shape = (n_samples, n_components)
            Weights
        """
        self.fit(X)
        return self.transform(X)
