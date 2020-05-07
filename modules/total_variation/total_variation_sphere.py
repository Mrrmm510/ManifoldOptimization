import cv2
import numpy as np
from tqdm import trange

from ..gradient_descent import GradientDescent


class TVonSphere(GradientDescent):
    """
    Problem
    -------
    minimize ∑ d^2(u_i. f_i) + λ ∑_i ∑_j φ(d(u_i, u_j))
    subject to u_i ∈ S^2

    Algorithm
    ---------
    Gradient Descent
    """
    def __init__(
            self,
            c: float = 1.0,
            epsilon: float = 1e-1,
            initial_step: float = 1.0,
            armijo_param: float = 0.5,
            total_max_iter: int = 10,
            max_iter: int = 10,
            tol: float = 1e-4,
            extended_output: bool = False
    ):
        """
        Parameters
        ----------
        c : float, optional (default=1.0)
            Constant that multiplies the TV term.

        epsilon : float, optional (default=1e-1)
            Parameter in function phi.

        initial_step : float, optional (default=1.0)
            Initial step size for armijo condition.

        armijo_param : float, optional (default=0.5)
            Armijo condition parameter.

        total_max_iter : int, optional (default=10)
            The maximum number of iterations.

        max_iter : int, optional (default=10)
            The maximum number of iterations of gradient descent.

        tol : float, optional (default=1e-4)
            The tolerance for the gradient descent.

        extended_output : bool, optional (default=False)
            If set to True, objective function value will be saved in `self.f`.
        """
        super().__init__(
            manifold='spheres',
            initial_step=initial_step,
            armijo_param=armijo_param,
            max_iter=max_iter,
            tol=tol,
            extended_output=extended_output
        )
        self.c = c
        self.epsilon = epsilon
        self.total_max_iter = total_max_iter

        self.mask = None
        self.F = None
        self.V_horizontal_p = None
        self.V_horizontal_n = None
        self.V_vertical_p = None
        self.V_vertical_n = None
        self.f = list()

    def _phi(self, T: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        T : np.ndarray, shape = (height, width, channels)

        Returns
        ----------
        : np.ndarray, shape = (height, width, channels)
        """
        return np.sqrt(T * T + self.epsilon * self.epsilon)

    def _s(self, T: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        T : np.ndarray, shape = (height, width, channels)

        Returns
        ----------
        : np.ndarray, shape = (height, width, channels)
        """
        return 1 / self._phi(T)

    def _horizontal_distance(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Distance of horizontally neighboring pixels.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A 3d image on sphere.

        Returns
        ----------
        ret_p : np.ndarray, shape = (height, width, channels)
            Distance. Last column is zero vector.

        ret_n : np.ndarray, shape = (height, width, channels)
            Distance. First column is zero vector.
        """
        ret_p = np.zeros_like(X)
        ret_n = np.zeros_like(X)
        dist = self.manifold.distance(X[:, :-1], X[:, 1:])
        ret_p[:, :-1] = dist
        ret_n[:, 1:] = dist
        return ret_p, ret_n

    def _vertical_distance(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Distance of vertically neighboring pixels.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A 3d image on sphere.

        Returns
        ----------
        ret_p : np.ndarray, shape = (height, width, channels)
            Distance. Last row is zero vector.

        ret_n : np.ndarray, shape = (height, width, channels)
            Distance. First row is zero vector.
        """
        ret_p = np.zeros_like(X)
        ret_n = np.zeros_like(X)
        dist = self.manifold.distance(X[:-1], X[1:])
        ret_p[:-1] = dist
        ret_n[1:] = dist
        return ret_p, ret_n

    def _horizontal_log(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Logarithm map of horizontally neighboring pixels.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A 3d image on sphere.

        Returns
        ----------
        ret_p : np.ndarray, shape = (height, width, channels)
            Logarithm map. Last column is zero vector.

        ret_n : np.ndarray, shape = (height, width, channels)
            Logarithm map. First column is zero vector.
        """
        ret_p = np.zeros_like(X)
        ret_n = np.zeros_like(X)
        log_p = self.manifold.log(X[:, :-1], X[:, 1:])
        log_n = self.manifold.log(X[:, 1:], X[:, :-1])
        ret_p[:, :-1] = log_p
        ret_n[:, 1:] = log_n
        return ret_p, ret_n

    def _vertical_log(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Logarithm map of vertically neighboring pixels.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A 3d image on sphere.

        Returns
        ----------
        ret_p : np.ndarray, shape = (height, width, channels)
            Logarithm map. Last row is zero vector.

        ret_n : np.ndarray, shape = (height, width, channels)
            Logarithm map. First row is zero vector.
        """
        ret_p = np.zeros_like(X)
        ret_n = np.zeros_like(X)
        log_p = self.manifold.log(X[:-1], X[1:])
        log_n = self.manifold.log(X[1:], X[:-1])
        ret_p[:-1] = log_p
        ret_n[1:] = log_n
        return ret_p, ret_n

    def _f(self, X: np.ndarray) -> float:
        """
        Objective function.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A 3d image on sphere

        Returns
        ----------
        : float
            Objective function value.
        """
        # first term
        dist = self.manifold.distance(X, self.F)
        masked_dist = dist if self.mask is None else dist * self.mask
        masked_dist = 0.5 * np.sum(masked_dist * masked_dist)

        # second term
        horizontal_dist, _ = self._horizontal_distance(X)
        vertical_dist, _ = self._vertical_distance(X)
        tv = self.c * (
            np.sum(horizontal_dist * horizontal_dist * self.V_horizontal_p)
            + np.sum(vertical_dist * vertical_dist * self.V_vertical_p)
        )
        return masked_dist + tv

    def _df(self, X: np.ndarray) -> np.ndarray:
        """
        Gradient.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A 3d image on sphere

        Returns
        ----------
        : np.ndarray, shape = (height, width, channels)
            Gradient of objective function.
        """
        # first term
        d_dist = - self.manifold.log(X, self.F) if self.mask is None else - self.mask * self.manifold.log(X, self.F)

        # second term
        horizontal_log_p, horizontal_log_n = self._horizontal_log(X)
        vertical_log_p, vertical_log_n = self._vertical_log(X)
        d_tv = - 2 * self.c * (
            self.V_horizontal_p * horizontal_log_p + self.V_vertical_p * vertical_log_p
            + self.V_horizontal_n * horizontal_log_n + self.V_vertical_n * vertical_log_n
        )
        return d_dist + d_tv

    def _inpaint(self, F: np.ndarray) -> np.ndarray:
        """
        Initialize.

        Parameters
        ----------
        F : np.ndarray, shape = (height, width, channels)
            A 3d image.

        Returns
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A 3d image on sphere.
        """
        # inpaint if mask is given
        X = F if self.mask is None else cv2.inpaint(F, 255 - np.uint8(self.mask[:, :, 0] > 0) * 255, 3, cv2.INPAINT_TELEA)

        # X on sphere
        X = X.astype(np.float32) / 255.
        norm = np.linalg.norm(X, axis=-1, keepdims=True)
        # [0, 0, 0] -> [1, 1, 1] / sqrt(3)
        X[norm[:, :, 0] == 0] = np.array([1., 1., 1.], dtype=np.float32)
        norm[norm == 0] = np.sqrt(3)
        return X / norm

    def transform(self, F: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Parameters
        ----------
        F : np.ndarray, shape = (height, width, channels)
            A 3d image.

        mask : np.ndarray, shape = (height, width, 1)
            Mask. Zero pixels indicate the area that needs to be inpainted.

        Returns
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A smoothed 3d image on sphere.
        """
        # F on sphere
        norm_F = np.linalg.norm(F, axis=-1, keepdims=True)
        norm_F[norm_F == 0] = 1.0
        self.F = F.astype(np.float32) / norm_F

        self.mask = mask

        # initialize X
        X = self._inpaint(F)

        # main loop
        for _ in trange(self.total_max_iter):
            # update V
            horizontal_dist_p, horizontal_dist_n = self._horizontal_distance(X)
            vertical_dist_p, vertical_dist_n = self._vertical_distance(X)
            self.V_horizontal_p = self._s(horizontal_dist_p)
            self.V_horizontal_n = self._s(horizontal_dist_n)
            self.V_vertical_p = self._s(vertical_dist_p)
            self.V_vertical_n = self._s(vertical_dist_n)

            # update X by gradient descent
            X = self.optimize(X)

            # store objective function value if necessary
            if self.extended_output:
                self.f.append(self._f(X))
        return X


class ApproximatedTVonSphere(TVonSphere):
    """
    Problem
    -------
    minimize ∑ d^2(u_i. f_i) + λ ∑_i ∑_j φ(d(u_i, u_j))
    subject to u_i ∈ S^2

    Algorithm
    ---------
    Gradient Descent
    """
    def __init__(
            self,
            c: float = 1.0,
            epsilon: float = 1e-1,
            initial_step: float = 1.0,
            armijo_param: float = 0.5,
            total_max_iter: int = 10,
            max_iter: int = 10,
            tol: float = 1e-4,
            extended_output: bool = False
    ):
        """
        Parameters
        ----------
        c : float, optional (default=1.0)
            Constant that multiplies the TV term.

        epsilon : float, optional (default=1e-1)
            Parameter in function phi.

        initial_step : float, optional (default=1.0)
            Initial step size for armijo condition.

        armijo_param : float, optional (default=0.5)
            Armijo condition parameter.

        total_max_iter : int, optional (default=10)
            The maximum number of iterations.

        max_iter : int, optional (default=10)
            The maximum number of iterations of gradient descent.

        tol : float, optional (default=1e-4)
            The tolerance for the gradient descent.

        extended_output : bool, optional (default=False)
            If set to True, objective function value will be saved in `self.f`.
        """
        super().__init__(
            c=c,
            epsilon=epsilon,
            initial_step=initial_step,
            armijo_param=armijo_param,
            total_max_iter=total_max_iter,
            max_iter=max_iter,
            tol=tol,
            extended_output=extended_output
        )

    # @staticmethod
    def _distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Distance.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, n_channels)

        Y : np.ndarray, shape = (height, width, n_channels)

        Returns
        -------
        : np.ndarray, shape = (height, width)
        """
        # dot = np.sum(X * Y, axis=-1, keepdims=True)
        # return np.sqrt(np.maximum(np.sum(Y * Y, axis=-1, keepdims=True) - dot * dot, 0))
        return np.linalg.norm(self._logarithm(X, Y), axis=-1, keepdims=True)

    @staticmethod
    def _logarithm(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Logarithm mapping.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, n_channels)

        Y : np.ndarray, shape = (height, width, n_channels)

        Returns
        -------
        : np.ndarray, shape = (height, width)
        """
        dot = np.sum(X * Y, axis=-1, keepdims=True) * X
        return Y - dot

    def _horizontal_distance(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Distance of horizontally neighboring pixels.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A 3d image on sphere.

        Returns
        ----------
        ret_p : np.ndarray, shape = (height, width, channels)
            Distance. Last column is zero vector.

        ret_n : np.ndarray, shape = (height, width, channels)
            Distance. First column is zero vector.
        """
        ret_p = np.zeros_like(X)
        ret_n = np.zeros_like(X)
        dist = self._distance(X[:, :-1], X[:, 1:])
        ret_p[:, :-1] = dist
        ret_n[:, 1:] = dist
        return ret_p, ret_n

    def _vertical_distance(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Distance of vertically neighboring pixels.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A 3d image on sphere.

        Returns
        ----------
        ret_p : np.ndarray, shape = (height, width, channels)
            Distance. Last row is zero vector.

        ret_n : np.ndarray, shape = (height, width, channels)
            Distance. First row is zero vector.
        """
        ret_p = np.zeros_like(X)
        ret_n = np.zeros_like(X)
        dist = self._distance(X[:-1], X[1:])
        ret_p[:-1] = dist
        ret_n[1:] = dist
        return ret_p, ret_n

    def _horizontal_log(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Logarithm map of horizontally neighboring pixels.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A 3d image on sphere.

        Returns
        ----------
        ret_p : np.ndarray, shape = (height, width, channels)
            Logarithm map. Last column is zero vector.

        ret_n : np.ndarray, shape = (height, width, channels)
            Logarithm map. First column is zero vector.
        """
        ret_p = np.zeros_like(X)
        ret_n = np.zeros_like(X)
        log_p = self._logarithm(X[:, :-1], X[:, 1:])
        log_n = self._logarithm(X[:, 1:], X[:, :-1])
        ret_p[:, :-1] = log_p
        ret_n[:, 1:] = log_n
        return ret_p, ret_n

    def _vertical_log(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Logarithm map of vertically neighboring pixels.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A 3d image on sphere.

        Returns
        ----------
        ret_p : np.ndarray, shape = (height, width, channels)
            Logarithm map. Last row is zero vector.

        ret_n : np.ndarray, shape = (height, width, channels)
            Logarithm map. First row is zero vector.
        """
        ret_p = np.zeros_like(X)
        ret_n = np.zeros_like(X)
        log_p = self._logarithm(X[:-1], X[1:])
        log_n = self._logarithm(X[1:], X[:-1])
        ret_p[:-1] = log_p
        ret_n[1:] = log_n
        return ret_p, ret_n

    def _f(self, X: np.ndarray) -> float:
        """
        Objective function.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A 3d image on sphere

        Returns
        ----------
        : float
            Objective function value.
        """
        # first term
        dist = self._distance(X, self.F)
        masked_dist = dist if self.mask is None else dist * self.mask
        masked_dist = 0.5 * np.sum(masked_dist * masked_dist)

        # second term
        horizontal_dist, _ = self._horizontal_distance(X)
        vertical_dist, _ = self._vertical_distance(X)
        tv = self.c * (
            np.sum(horizontal_dist * horizontal_dist * self.V_horizontal_p)
            + np.sum(vertical_dist * vertical_dist * self.V_vertical_p)
        )
        return masked_dist + tv

    def _df(self, X: np.ndarray) -> np.ndarray:
        """
        Gradient.

        Parameters
        ----------
        X : np.ndarray, shape = (height, width, channels)
            A 3d image on sphere

        Returns
        ----------
        : np.ndarray, shape = (height, width, channels)
            Gradient of objective function.
        """
        # first term
        d_dist = - self._logarithm(X, self.F) if self.mask is None else - self.mask * self._logarithm(X, self.F)

        # second term
        horizontal_log_p, horizontal_log_n = self._horizontal_log(X)
        vertical_log_p, vertical_log_n = self._vertical_log(X)
        d_tv = - 2 * self.c * (
            self.V_horizontal_p * horizontal_log_p + self.V_vertical_p * vertical_log_p
            + self.V_horizontal_n * horizontal_log_n + self.V_vertical_n * vertical_log_n
        )
        return d_dist + d_tv
