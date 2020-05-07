from cvxopt import matrix
from cvxopt.solvers import qp
import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import Lasso


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

        tau : float, optional (default=None)
            Constant that used in augmented Lagrangian function.

        max_iter : int, optional (default=300)
            The maximum number of iterations.

        extended_output : bool, optional (default=False)
            If set to True, objective function value will be saved in `self.f`.
        """
        if (A is None or b is None) and (G is None or h is None):
            raise ValueError('Invalid input for __init__: (A, b) or (G, h) must not be None!')

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
        self.tau = tau
        self.max_iter = max_iter
        self.extended_output = extended_output

        # Lasso
        self.clf = Lasso(alpha=rho, fit_intercept=False)

        self.f = list()
        self.coef_ = None

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
        A_ub = np.vstack((
            np.hstack((eye, -eye)),
            np.hstack((-eye, -eye)),
            np.hstack((np.zeros((n_features, n_features)), -eye))
        ))
        b_ub = np.zeros(n_features * 3)
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
            self.tau = n_samples
        tau = np.sqrt(self.tau)

        # initialize constants
        n_samples_sqrt = np.sqrt(n_samples)
        Q = np.vstack((X, np.eye(n_features) * tau)) * n_samples_sqrt
        p = y * n_samples_sqrt

        P = np.eye(n_features, dtype=np.float)

        # initialize variables
        beta = self._linear_programming(n_features)
        z = np.copy(beta)
        u = np.zeros_like(beta)

        # save objective function value if necessary
        if self.extended_output:
            self.f.append(0.5 * np.linalg.norm(y - X.dot(beta)) ** 2 + np.sum(np.abs(beta)) * self.rho)

        # main loop
        for _ in range(self.max_iter):
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


class EqualityConstrainedL1QuadraticProgramming:
    """
    Problem
    ----------
    minimize 0.5 * βQβ + pβ + ρ||β||_1
    subject to Aβ = b

    Algorithm
    ----------
    Alternating Direction Method of Multipliers (ADMM)

    Parameters
    ----------
    A : np.ndarray, optional (default=None)
        The equality constraint matrix.

    b : np.ndarray, optional (default=None)
        The equality constraint vector.

    rho : float, optional (default=1.0)
        Constant that multiplies the L1 term.

    tau : float, optional (default=None)
        Constant that used in augmented Lagrangian function.

    tol : float, optional (default=1e-4)
        The tolerance for the optimization.

    max_iter : int, optional (default=300)
        The maximum number of iterations.

    extended_output : bool, optional (default=False)
        If set to True, objective function value will be saved in `self.f`.

    Attributes
    ----------
    f : a list of float
        objective function values.

    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector.

    n_iter_ : int | array-like, shape (n_targets,)
        number of iterations run by admm to reach the specified tolerance.
    """
    def __init__(
            self,
            A: np.ndarray,
            b: np.ndarray,
            rho: float = 1.0,
            tau: float = 1.0,
            tol: float = 1e-4,
            max_iter: int = 300,
            extended_output: bool = False
    ):
        self.A = A
        self.b = b
        self.rho = rho
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter
        self.extended_output = extended_output

        self.f = list()
        self.coef_ = None
        self.n_iter_ = None

    def _linear_programming(self, n_features: int) -> np.ndarray:
        """
        Solve following problem.

        Problem:
            minimize ||β||_1 subject to Aβ=b

        Solver:
            scipy.optimize.linprog

        Parameters
        ----------
        n_features : int
            The dimension of decision variables

        Returns
        -------
        : np.ndarray, shape = (n_features, )
            The values of the decision variables that minimizes the objective function while satisfying the constraints.
        """
        # equality constraint matrix and vector
        c = np.hstack((np.zeros(n_features), np.ones(n_features)))
        A_eq = np.hstack((self.A, np.zeros_like(self.A)))
        b_eq = self.b

        # inequality constraint matrix and vector
        eye = np.eye(n_features)
        A_ub = np.vstack((
            np.hstack((eye, -eye)),
            np.hstack((-eye, -eye)),
            np.hstack((np.zeros((n_features, n_features)), -eye))
        ))
        b_ub = np.zeros(n_features * 3)

        return linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)['x'][:n_features]

    def _prox(self, x: np.ndarray, rho: float) -> np.ndarray:
        """
        Proximal operator for L1 constraint

        Parameters
        ----------
        x : np.ndarray, shape = (n_features, )
            1D array

        rho : float
            a threshold
        """
        return np.sign(x) * np.maximum((np.abs(x) - rho), 0)

    def _objective_function(self, x: np.ndarray, Q: np.ndarray, p: np.ndarray, rho: float):
        """
        Return 1 / 2 * x^T Qx + p^T x + rho * ||x||_1
        """
        return 0.5 * x.dot(Q.dot(x)) + p.dot(x) + np.sum(np.abs(x)) * rho

    def fit(self, Q: np.ndarray, p: np.ndarray) -> None:
        """
        Parameters
        ----------
        Q : np.ndarray, shape = (n_features, n_features)
            Quadratic coefficient.

        p : np.ndarray, shape = (n_features, )
            Linear coefficient.
        """
        n_features = Q.shape[0]
        tau_inv = 1 / self.tau

        # initialize constants
        Q_inv = np.linalg.inv(Q * tau_inv + self.A.T.dot(self.A) + np.eye(n_features))
        p_tau = p * tau_inv
        Ab = self.A.T.dot(self.b)
        rho_tau = self.rho * tau_inv

        # initialize variables
        beta = self._linear_programming(n_features)
        z = np.copy(beta)
        u = np.zeros_like(self.b)
        v = np.zeros_like(beta)

        cost = self._objective_function(beta, Q, p, self.rho)
        # save objective function value if necessary
        if self.extended_output:
            self.f.append(cost)

        # main loop
        k = 0
        for k in range(self.max_iter):
            r = - p_tau - self.A.T.dot(u) - v + Ab + z
            beta = np.dot(Q_inv, r)
            z = self._prox(beta + v, rho_tau)
            u = u + self.A.dot(beta) - self.b
            v = v + beta - z

            pre_cost = cost
            cost = self._objective_function(beta, Q, p, self.rho)
            # save objective function value if necessary
            if self.extended_output:
                self.f.append(cost)
            if np.abs(cost - pre_cost) < self.tol:
                break

        # save result
        self.coef_ = beta
        self.n_iter_ = k


class EqualityConstrainedLasso:
    """
    Problem:
        minimize 0.5 * ||Xβ-y||^2 + ρ||β||_1
        subject to Aβ = b

    Algorithm:
        Alternating Direction Method of Multipliers (ADMM)
    """
    def __init__(
            self,
            A: np.ndarray,
            b: np.ndarray,
            rho: float = 1.0,
            tau: float = None,
            tol: float = 1e-4,
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

        rho : float, optional (default=1.0)
            Constant that multiplies the L1 term.

        tau : float, optional (default=None)
            Constant that used in augmented Lagrangian function.

        max_iter : int, optional (default=300)
            The maximum number of iterations.

        extended_output : bool, optional (default=False)
            If set to True, objective function value will be saved in `self.f`.
        """
        self.clf = EqualityConstrainedL1QuadraticProgramming(
            A=A,
            b=b,
            rho=rho,
            tau=tau,
            tol=tol,
            max_iter=max_iter,
            extended_output=extended_output
        )

        self.f = list()
        self.coef_ = None

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
        if self.clf.tau is None:
            self.clf.tau = n_samples

        self.clf.fit(X.T.dot(X), - X.T.dot(y))
        self.f = self.clf.f
        self.coef_ = self.clf.coef_


class SumOneConstrainedL1QuadraticProgramming(EqualityConstrainedL1QuadraticProgramming):

    """
    Problem
    ----------
    minimize 0.5 * βQβ + pβ + ρ||β||_1
    subject to ∑ß_j = 1

    Algorithm
    ----------
    Alternating Direction Method of Multipliers (ADMM)

    Parameters
    ----------
    n_features : int
        The number of features.

    rho : float, optional (default=1.0)
        Constant that multiplies the L1 term.

    tau : float, optional (default=None)
        Constant that used in augmented Lagrangian function.

    tol : float, optional (default=1e-4)
        The tolerance for the optimization.

    max_iter : int, optional (default=300)
        The maximum number of iterations.

    extended_output : bool, optional (default=False)
        If set to True, objective function value will be saved in `self.f`.

    Attributes
    ----------
    f : a list of float
        objective function values.

    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector.

    n_iter_ : int | array-like, shape (n_targets,)
        number of iterations run by admm to reach the specified tolerance.
    """
    def __init__(
            self,
            n_features: int,
            rho: float = 1.0,
            tau: float = 1.0,
            tol: float = 1e-4,
            max_iter: int = 300,
            extended_output: bool = False
    ):
        super().__init__(
            A=np.ones((1, n_features)),
            b=np.array([1.]),
            rho=rho,
            tau=tau,
            tol=tol,
            max_iter=max_iter,
            extended_output=extended_output
        )

    def _linear_programming(self, n_features: int) -> np.ndarray:
        """
        return [1 / n_features, ... , 1 / n_features].

        Parameters
        ----------
        n_features : int
            The dimension of decision variables

        Returns
        -------
        : np.ndarray, shape = (n_features, )
        """
        return np.full(n_features, 1 / n_features)
