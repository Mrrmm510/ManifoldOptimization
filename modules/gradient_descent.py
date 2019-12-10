from abc import ABCMeta, abstractmethod
from logging import getLogger, DEBUG

import numpy as np

from .manifolds.real_space import RealSpace
from .manifolds.sphere import Sphere, Spheres
from .manifolds.spd import SPD, SPDs


logger = getLogger(__name__)
logger.setLevel(DEBUG)


class GradientDescent(metaclass=ABCMeta):
    def __init__(
            self,
            manifold: str = 'sphere',
            initial_step: float = 1.0,
            armijo_param: float = 1e-4,
            max_iter: int = 300,
            tol: float = 1e-4,
            extended_output: bool = False
    ):
        self.initial_step = initial_step
        self.armijo_param = armijo_param
        self.max_iter = max_iter
        self.tol = tol
        self.extended_output = extended_output

        if manifold == 'sphere':
            self.manifold = Sphere()
        elif manifold == 'spheres':
            self.manifold = Spheres()
        elif manifold == 'real':
            self.manifold = RealSpace()
        elif manifold == 'spd':
            self.manifold = SPD()
        elif manifold == 'spds':
            self.manifold = SPDs()
        else:
            logger.warning(f'Manifold {manifold} is not implemented! Use real space instead.')
            self.manifold = RealSpace()

        self.f = list()

    @abstractmethod
    def _f(self, x: np.ndarray):
        raise NotImplementedError('The function _f is not implemented')

    @abstractmethod
    def _df(self, x: np.ndarray):
        raise NotImplementedError('The function _df is not implemented')

    def _step_size(self, x: np.ndarray, d: np.ndarray) -> float:
        """
        Armijo condition with back tracking
        """
        df = self._df(x)
        g = self.manifold.inner_product(x, df, d)
        t = self.initial_step
        f = self._f(x)
        while self._f(self.manifold.retraction(x, t * d)) > f + self.armijo_param * t * g:
            t *= 0.5
            # break when step size 't' becomes too small
            if t <= 1e-16:
                t = 0
                break
        return t

    @staticmethod
    def _initialize(x: np.ndarray) -> np.ndarray:
        return np.copy(x)

    @staticmethod
    def _convergence(d: np.ndarray) -> float:
        return np.linalg.norm(d)

    def optimize(self, x: np.ndarray):
        # initialize
        res = self._initialize(x)

        # main loop
        for _ in range(self.max_iter):
            d = - self._df(res)
            step_size = self._step_size(res, d)
            # break when step_size == 0
            if step_size == 0:
                break

            # update
            res = self.manifold.retraction(res, step_size * d)

            # store objective function value if necessary
            if self.extended_output:
                self.f.append(self._f(res))

            # break if convergence
            if self._convergence(d) < self.tol:
                break
        return res
