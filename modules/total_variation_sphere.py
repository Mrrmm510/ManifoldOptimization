import cv2
import numpy as np
from tqdm import trange

import sys
sys.path += ['..']
from modules import sphere2d


class TVonSphere:
    def __init__(
            self,
            c: float = 1.0, 
            epsilon: float = 1e-1, 
            initial_step: float = 1.0, 
            armijo_param: float = 0.5,
            max_iter: int = 10
    ):
        self.c = c
        self.epsilon = epsilon
        self.initial_step = initial_step
        self.armijo_param = armijo_param
        self.max_iter = max_iter
        
        self.brightness = None
        self.V_horizontal = None
        self.V_vertical = None
        
    def _phi(self, T: np.ndarray) -> np.ndarray:
        return np.sqrt(T * T + self.epsilon * self.epsilon)
    
    def _s(self, T: np.ndarray) -> np.ndarray:
        return 1 / self._phi(T)
    
    def _masked_distance(self, dist: np.ndarray):
        return dist if self.mask is None else dist * self.mask
    
    def _horizontal_distance(self, X: np.ndarray) -> np.ndarray:
        ret = np.zeros_like(X)
        dist = sphere2d.distance(X[:, :-1], X[:, 1:])
        ret[:, :-1] += dist
        ret[:, 1:] += dist
        return ret
    
    def _vertical_distance(self, X: np.ndarray) -> np.ndarray:
        ret = np.zeros_like(X)
        dist = sphere2d.distance(X[:-1], X[1:])
        ret[:-1] += dist
        ret[1:] += dist
        return ret
    
    def _f(self, X: np.ndarray) -> float:
        dist = sphere2d.distance(X, self.F)
        masked_dist = self._masked_distance(dist)
        masked_dist = 0.5 * np.sum(masked_dist * masked_dist)
        horizontal_dist = self._horizontal_distance(X)
        vertical_dist = self._vertical_distance(X)
        tv = self.c * (np.sum(horizontal_dist) + np.sum(vertical_dist))
        return masked_dist + tv
    
    def _df(self, X: np.ndarray) -> np.ndarray:
        d_dist = - sphere2d.log(X, self.F) if self.mask is None else - self.mask * sphere2d.log(X, self.F)
        horizontal_log = self._horizontal_distance(X)
        vertical_log = self._vertical_distance(X)
        d_tv = - 2 * self.c * (self.V_horizontal * horizontal_log + self.V_vertical * vertical_log)
        return d_dist + d_tv
    
    def _step_size(self, X: np.ndarray, iteration: int) -> float:
        df = self._df(X)
        d = - np.copy(df)
        g = np.sum(df * d)
        t = self.initial_step
        f = self._f(X)
        while self._f(sphere2d.exp(X, t * d)) > f + self.armijo_param * t * g:
            t *= 0.5
        return t
    
    def _initialize(self, F: np.ndarray) -> np.ndarray:
        X = F if self.mask is None else cv2.inpaint(F, 255 - np.uint8(self.mask[:, :, 0] > 0) * 255, 3, cv2.INPAINT_TELEA)
        X = X.astype(np.float32) / 255.
        norm = np.linalg.norm(X, axis=-1, keepdims=True)
        X[norm[:,:,0] == 0] = np.array([1.,1.,1.], dtype=np.float32)
        norm[norm==0] = np.sqrt(3)
        return X / norm
    
    def transform(self, F: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        norm_F = np.linalg.norm(F, axis=-1, keepdims=True)
        norm_F[norm_F==0] = 1.0
        self.F = F.astype(np.float32) / norm_F
        self.mask = mask
        X = self._initialize(F)
        for _ in trange(10):
            horizontal_dist = self._horizontal_distance(X)
            vertical_dist = self._vertical_distance(X)
            self.V_horizontal = self._s(horizontal_dist)
            self.V_vertical = self._s(vertical_dist)
            for i in range(self.max_iter):
                step_size = self._step_size(X, i)
                X = sphere2d.exp(X, self._df(X) * step_size)
        return X
    
    