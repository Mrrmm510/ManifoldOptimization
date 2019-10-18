from abc import ABCMeta, abstractmethod

import numpy as np


class Manifold(metaclass=ABCMeta):
    @abstractmethod
    def inner_product(self, x: np.ndarray, u: np.ndarray, v: np.ndarray):
        raise NotImplementedError('The function retraction is not implemented')

    @abstractmethod
    def retraction(self, x: np.ndarray, v: np.ndarray):
        raise NotImplementedError('The function retraction is not implemented')

    def exp(self, x: np.ndarray, v: np.ndarray):
        raise NotImplementedError('The function exp is not implemented')

    def log(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError('The function log is not implemented')
