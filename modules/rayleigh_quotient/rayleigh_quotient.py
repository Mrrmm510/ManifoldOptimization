import numpy as np

from ..gradient_descent import GradientDescent
from ..conjugate_gradient import ConjugateGradient


class RayleighQuotientGD(GradientDescent):
    def __init__(
            self,
            A: np.ndarray,
            initial_step: float = 1.0,
            armijo_param: float = 0.5,
            max_iter: int = 300,
            extended_output: bool = False
    ):
        super().__init__(
            manifold='real',
            initial_step=initial_step,
            armijo_param=armijo_param,
            max_iter=max_iter,
            extended_output=extended_output
        )
        self.A = A

    def _f(self, x: np.ndarray) -> float:
        Ax = np.dot(self.A, x)
        xx = np.dot(x, x)
        return np.dot(Ax, x) / xx

    def _df(self, x: np.ndarray) -> np.ndarray:
        Ax = np.dot(self.A, x)
        xx = np.dot(x, x)
        f = np.dot(Ax, x) / xx
        return 2 * (Ax - f * x) / xx


class RayleighQuotientSphereGD(GradientDescent):
    def __init__(
            self,
            A: np.ndarray,
            initial_step: float = 1.0,
            armijo_param: float = 0.5,
            max_iter: int = 300,
            extended_output: bool = False
    ):
        super().__init__(
            manifold='sphere',
            initial_step=initial_step,
            armijo_param=armijo_param,
            max_iter=max_iter,
            extended_output=extended_output
        )
        self.A = A

    def _f(self, x: np.ndarray) -> float:
        Ax = np.dot(self.A, x)
        return np.dot(Ax, x)

    def _df(self, x: np.ndarray) -> np.ndarray:
        Ax = np.dot(self.A, x)
        f = np.dot(Ax, x)
        return 2 * (Ax - f * x)


class RayleighQuotientCG(ConjugateGradient):
    def __init__(
            self,
            A: np.ndarray,
            initial_step: float = 1.0,
            armijo_param: float = 0.5,
            max_iter: int = 300,
            extended_output: bool = False
    ):
        super().__init__(
            manifold='real',
            initial_step=initial_step,
            armijo_param=armijo_param,
            max_iter=max_iter,
            extended_output=extended_output
        )
        self.A = A

    def _f(self, x: np.ndarray) -> float:
        Ax = np.dot(self.A, x)
        xx = np.dot(x, x)
        return np.dot(Ax, x) / xx

    def _df(self, x: np.ndarray) -> np.ndarray:
        Ax = np.dot(self.A, x)
        xx = np.dot(x, x)
        f = np.dot(Ax, x) / xx
        return 2 * (Ax - f * x) / xx


class RayleighQuotientSphereCG(ConjugateGradient):
    def __init__(
            self,
            A: np.ndarray,
            initial_step: float = 1.0,
            armijo_param: float = 0.5,
            max_iter: int = 300,
            extended_output: bool = False
    ):
        super().__init__(
            manifold='sphere',
            initial_step=initial_step,
            armijo_param=armijo_param,
            max_iter=max_iter,
            extended_output=extended_output
        )
        self.A = A

    def _f(self, x: np.ndarray) -> float:
        Ax = np.dot(self.A, x)
        return np.dot(Ax, x)

    def _df(self, x: np.ndarray) -> np.ndarray:
        Ax = np.dot(self.A, x)
        f = np.dot(Ax, x)
        return 2 * (Ax - f * x)
