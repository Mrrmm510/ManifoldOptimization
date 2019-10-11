from cvxopt import matrix
from cvxopt.solvers import qp
import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import Lasso
from tqdm import trange


class ConstrainedLasso:
    """
    Problem:
        minimize 0.5 * ||Xβ-y||^2 + ρ||β||_1
        subject to Aβ = b, Gβ ≤ h

    Algorithm:
        Alternating Direction Method of Multipliers (ADMM)
    """
    def __init__(
            self,
            A: np.ndarray = None,
            b: np.ndarray = None,
            G: np.ndarray = None,
            h: np.ndarray = None,
            rho: float = 1.0,
            epsilon: float = 1e-4,
            tau: float = None,
            max_iter: int = 300,
            extended_output: bool = False
    ):
        """
        Parameters
        ----------
        A : np.ndarray, optional (default=None)
            The equality constraint matrix.

        b : np.ndarray, optional (default=None)
            The equality constraint vector.

        G : np.ndarray, optional (default=None)
            The inequality constraint matrix.

        h : np.ndarray, optional (default=None)
            The inequality constraint vector.

        rho : float, optional (default=1.0)
            Constant that multiplies the L1 term.

        epsilon : float, optional (default=1e-4)
            Constant that multiplies the small ridge penalty.

        tau : float, optional (default=None)
            Constant that used in augmented Lagrangian function.

        max_iter : int, optional (default=300)
            The maximum number of iterations.

        extended_output : bool, optional (default=False)
            If set to True, objective function value will be saved in `self.f`.
        """
        if (A is None or b is None) and (C is None or d is None):
            raise ValueError('Invalid input for __init__: (A, b) or (C, d) must not be None!')

        if A is None or b is None:
            self.A = None
            self.b = None
            self.G = matrix(G)
            self.h = matrix(h)
        elif G is None or h is None:
            self.A = matrix(A)
            self.b = matrix(b)
            self.G = None
            self.h = None
        else:
            self.A = matrix(A)
            self.b = matrix(b)
            self.G = matrix(G)
            self.h = matrix(h)

        self.rho = rho
        self.epsilon = epsilon
        self.tau = None if tau is None else tau
        self.max_iter = max_iter
        self.extended_output = extended_output

        # Lasso
        self.clf = Lasso(alpha=rho, fit_intercept=False)

        self.f = list()

    def _linear_programming(self, n_features: int) -> np.ndarray:
        """
        Solve following problem.

        Problem:
            minimize ||β||_1 subject to Aβ=b, Gβ≤h

        Solver:
            scipy.optimize.linprog

        Parameters
        ----------
        n_features : int
            The dimension of decision variables

        Returns
        ----------
        : np.ndarray, shape = (n_features, )
            The values of the decision variables that minimizes the objective function while satisfying the constraints.
        """
        # equality constraint matrix and vector
        c = np.hstack((np.zeros(n_features), np.ones(n_features)))
        A_eq = None
        b_eq = None
        if self.A is not None and self.b is not None:
            A, b = np.array(self.A), np.array(self.b).flatten()
            A_eq = np.hstack((A, np.zeros_like(A)))
            b_eq = b

        # inequality constraint matrix and vector
        eye = np.eye(n_features)
        A_ub = np.vstack((np.hstack((eye, -eye)), np.hstack((-eye, -eye))))
        b_ub = np.zeros(n_features * 2)
        if self.G is not None and self.h is not None:
            G = np.array(self.G)
            h = np.array(self.h).flatten()
            A_ub = np.vstack((np.hstack((G, np.zeros_like(G))), A_ub))
            b_ub = np.hstack((h, b_ub))

        return linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)['x'][:n_features]

    def _projection(self, P: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Projection into constraint set

        Problem:
            minimize ||x - q||^2 subject to Ax = b, Gx ≤ h

        Solver:
            cvxopt.solvers.qp

        Parameters
        ----------
        P : np.ndarray, shape = (n_features, n_features)
            Coefficient matrix.

        q: np.ndarray, shape = (n_features, )
            Coefficient vector.

        Returns
        ----------
        : np.ndarray, shape = (n_features, )
        """
        sol = qp(P=matrix(P), q=matrix(-q), G=self.G, h=self.h, A=self.A, b=self.b)
        return np.array(sol['x']).flatten()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Parameters
        ----------
        X : np.ndarray, shape = (n_samples, n_features)
            Data.

        y : np.ndarray, shape = (n_samples, )
            Target.
        """
        n_samples, n_features = X.shape

        # initialize tau if necessary
        if self.tau is None:
            self.tau = 1 / n_samples
        tau = np.sqrt(self.tau)

        # initialize constants
        n_samples_inv = 1 / n_samples
        n_samples_sqrt = np.sqrt(n_samples)
        Q = np.vstack((X, np.eye(n_features) * np.sqrt(self.epsilon), np.eye(n_features) * tau)) * n_samples_sqrt
        p = np.hstack((y * n_samples_sqrt, np.zeros(n_features)))
        P = np.eye(n_features, dtype=np.float)

        # initialize variables
        beta = self._linear_programming(n_features)
        z = np.copy(beta)
        u = np.zeros_like(beta)

        # save objective function value if necessary
        if self.extended_output:
            self.f.append(0.5 * np.linalg.norm(y - X.dot(beta)) ** 2 + np.sum(np.abs(beta)) * self.rho)

        # main loop
        for _ in trange(self.max_iter):
            w = np.hstack((p, (z - u) * tau * n_samples_sqrt))
            self.clf.fit(Q, w)
            beta = self.clf.coef_
            z = self._projection(P=P, q=beta+u)
            u = u + beta - z

            # save objective function value if necessary
            if self.extended_output:
                self.f.append(0.5 * np.linalg.norm(y - X.dot(beta)) ** 2 + np.sum(np.abs(beta)) * self.rho)

        # save result
        self.coef_ = z
