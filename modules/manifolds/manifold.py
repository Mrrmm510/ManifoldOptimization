from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np


class Manifold(metaclass=ABCMeta):
    @abstractmethod
    def inner_product(self, x: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
        raise NotImplementedError('The function retraction is not implemented')

    @abstractmethod
    def retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError('The function retraction is not implemented')

    def exp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError('The function exp is not implemented')

    def log(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError('The function log is not implemented')

    def vector_transport(self, x: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        raise NotImplementedError('The function vector_transport is not implemented')

    def distance(self, x: np.ndarray, y: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError('The function distance is not implemented')

    def gradient(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        raise NotImplementedError('The function gradient is not implemented')
