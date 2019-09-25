from abc import ABCMeta, abstractmethod

import numpy as np


class GradientDescent(metaclass=ABCMeta):
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
    def _step_size(self, x: np.ndarray, iteration: int):
        raise NotImplementedError('The function _step_size is not implemented')
        
    @abstractmethod
    def _retraction(self, x: np.ndarray, v: np.ndarray):
        raise NotImplementedError('The function _retraction is not implemented')
        
    def optimize(self, x: np.ndarray):
        res = np.copy(x)
        if self.extended_output:
            self.f.append(self._f(res))
        for i in range(1, self.max_iter + 1):
            step_size = self._step_size(res, i)
            res = self._retraction(res, - self._df(res) * step_size)
            if self.extended_output:
                self.f.append(self._f(res))
        return res
    

class RayleighQuotientGD(GradientDescent):
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
    
    def _step_size(self, x: np.ndarray, iteration: int) -> float:
        df = self._df(x)
        d = - np.copy(df)
        g = np.dot(df, d)
        t = self.step_size
        f = self._f(x)
        while self._f(self._retraction(x, t * d)) > f + self.c * t * g:
            t *= 0.5
        return t
    
    def _retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return x + v


class RayleighQuotientSphereGD(GradientDescent):
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
    
    def _step_size(self, x: np.ndarray, iteration: int) -> float:
        df = self._df(x)
        d = - np.copy(df)
        g = np.dot(df, d)
        t = self.step_size
        f = self._f(x)
        while self._f(self._retraction(x, t * d)) > f + self.c * t * g:
            t *= 0.5
        return t
    
    def _retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x + v)
        return (x + v) / norm if norm != 0 else 0