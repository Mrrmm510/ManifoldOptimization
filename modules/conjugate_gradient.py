from abc import ABCMeta
import numpy as np

from .gradient_descent import GradientDescent


class ConjugateGradient(GradientDescent, metaclass=ABCMeta):
    def __init__(
            self,
            manifold: str = 'sphere',
            initial_step: float = 1.0,
            armijo_param: float = 1e-4,
            max_iter: int = 300,
            tol: float = 1e-4,
            extended_output: bool = False
    ):
        super().__init__(
            manifold=manifold,
            initial_step=initial_step,
            armijo_param=armijo_param,
            max_iter=max_iter,
            tol=tol,
            extended_output=extended_output
        )

    def _beta(self, x1: np.ndarray, x2: np.ndarray, df1: np.ndarray, df2: np.ndarray) -> float:
        return self.manifold.inner_product(x2, df2, df2) / self.manifold.inner_product(x1, df1, df1)

    def _d(self, x1: np.ndarray, x2: np.ndarray, d: np.ndarray, step_size: float) -> np.ndarray:
        df1, df2 = self._df(x1), self._df(x2)
        return - df2 + self._beta(x1, x2, df1, df2) * self.manifold.vector_transport(x1, step_size * d, d)

    def optimize(self, x: np.ndarray):
        # initialize
        res = np.copy(x)
        d = - self._df(res)

        # main loop
        for _ in range(self.max_iter):
            step_size = self._step_size(res, d)
            # break when step_size == 0
            if step_size == 0:
                break

            # update
            z = self.manifold.retraction(res, step_size * d)
            d = self._d(res, z, d, step_size)
            res = np.copy(z)

            # store objective function value if necessary
            if self.extended_output:
                self.f.append(self._f(res))

            # break if convergence
            if self._convergence(d) < self.tol:
                break
        return res
