from abc import ABCMeta, abstractmethod

import numpy as np


class ConjugateGradient(metaclass=ABCMeta):
    def __init__(self, max_iter: int = 300, extended_output: bool = False):
        self.max_iter = max_iter
        self.extended_output = extended_output
        self.f = list()
    
    @abstractmethod
    def _f(self, x: np.ndarray):
        raise NotImplementedError('The function _f is not implemented')
        
    @abstractmethod
    def _df(self, x: np.ndarray):
        raise NotImplementedError('The function _df is not implemented')
        
    @abstractmethod
    def _step_size(self, x: np.ndarray, d: np.ndarray, iteration: int):
        raise NotImplementedError('The function _step_size is not implemented')
        
    @abstractmethod
    def _retraction(self, x: np.ndarray, v: np.ndarray):
        raise NotImplementedError('The function _retraction is not implemented')
        
    @abstractmethod
    def _vector_transport(self, x: np.ndarray, v: np.ndarray, a: np.ndarray):
        raise NotImplementedError('The function _vector_transport is not implemented')
        
    def _beta(self, x1: np.ndarray, x2: np.ndarray) -> float:
        df1 = self._df(x1)
        df2 = self._df(x2)
        return np.dot(df2, df2) / np.dot(df1, df1)
    
    def _d(self, x1: np.ndarray, x2: np.ndarray, d: np.ndarray, step_size: float) -> np.ndarray:
        return - self._df(x2) + self._beta(x1, x2) * self._vector_transport(x1, step_size * d, d)
        
    def optimize(self, x: np.ndarray):
        res = np.copy(x)
        d = - self._df(x)
        if self.extended_output:
            self.f.append(self._f(res))
        for i in range(1, self.max_iter + 1):
            step_size = self._step_size(res, d, i)
            x = self._retraction(res, step_size * d)
            d = self._d(res, x, d, step_size)
            res = np.copy(x)
            if self.extended_output:
                self.f.append(self._f(res))
        return res

class RayleighQuotientCG(ConjugateGradient):
    def __init__(self, A: np.ndarray, step_size: float = 1.0, c = 0.5, max_iter: int = 300, extended_output: bool = False):
        super().__init__(max_iter=max_iter, extended_output=extended_output)
        self.A = A
        self.step_size = step_size
        self.c = c
        
    def _f(self, x: np.ndarray) -> float:
        Ax = np.dot(self.A, x)
        xx = np.dot(x, x)
        return np.dot(Ax, x) / xx
    
    def _df(self, x: np.ndarray) -> np.ndarray:
        Ax = np.dot(self.A, x)
        xx = np.dot(x, x)
        f = np.dot(Ax, x) / xx
        return 2 * (Ax - f * x) / xx
    
    def _step_size(self, x: np.ndarray, d: np.ndarray, iteration: int) -> float:
        # armijo rule
        df = self._df(x)
        g = np.dot(df, d)
        t = self.step_size
        f = self._f(x)
        while self._f(self._retraction(x, t * d)) > f + self.c * t * g:
            t *= 0.5
        return t
    
    def _retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return x + v
    
    def _vector_transport(self, x: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        return a

class RayleighQuotientSphereCG(ConjugateGradient):
    def __init__(self, A: np.ndarray, step_size: float = 1.0, c = 0.5, max_iter: int = 300, extended_output: bool = False):
        super().__init__(max_iter=max_iter, extended_output=extended_output)
        self.A = A
        self.step_size = step_size
        self.c = c
        
    def _f(self, x: np.ndarray) -> float:
        Ax = np.dot(self.A, x)
        return np.dot(Ax, x)
    
    def _df(self, x: np.ndarray) -> np.ndarray:
        Ax = np.dot(self.A, x)
        f = np.dot(Ax, x)
        return 2 * (Ax - f * x)
    
    def _step_size(self, x: np.ndarray, d: np.ndarray, iteration: int) -> float:
        # armijo rule
        df = self._df(x)
        g = np.dot(df, d)
        t = self.step_size
        f = self._f(x)
        while self._f(self._retraction(x, t * d)) > f + self.c * t * g:
            t *= 0.5
        return t
    
    def _retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x + v)
        return (x + v) / norm if norm != 0 else 0
    
    def _vector_transport(self, x: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        w = x + v
        return a - np.dot(v, a) / np.dot(w, w) * w