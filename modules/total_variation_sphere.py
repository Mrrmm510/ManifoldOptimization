import cv2
import numpy as np
from tqdm import trange

import sys
sys.path += ['..']
from modules import sphere2d


class TVonSphere:
    """
    $$\text{minimize}\ \sum_{i\in\mathcal{V}}d^2(u_i,f_i)+\lambda\sum_{i\in\mathcal{G}}\sum_{j\in\mathcal{N}^+}\phi(d(u_i,u_j))
    \text{subject to}\ u_i\in S^{2}$$
    """
    def __init__(
            self,
            c: float = 1.0, 
            epsilon: float = 1e-1, 
            initial_step: float = 1.0, 
            armijo_param: float = 0.5,
            max_iter: int = 10,
            max_iter_gd: int = 10,
            extended_output: bool = True
    ):
        self.c = c
        self.epsilon = epsilon
        self.initial_step = initial_step
        self.armijo_param = armijo_param
        self.max_iter = max_iter
        self.max_iter_gd = max_iter_gd
        self.extended_output = extended_output
        
        self.mask = None
        self.V_horizontal_p = None
        self.V_horizontal_n = None
        self.V_vertical_p = None
        self.V_vertical_n = None
        self.f = list()
        
    def _phi(self, T: np.ndarray) -> np.ndarray:
        return np.sqrt(T * T + self.epsilon * self.epsilon)
    
    def _s(self, T: np.ndarray) -> np.ndarray:
        return 1 / self._phi(T)
    
    def _horizontal_distance(self, X: np.ndarray) -> np.ndarray:
        ret_p = np.zeros_like(X)
        ret_n = np.zeros_like(X)
        dist = sphere2d.distance(X[:, :-1], X[:, 1:])
        ret_p[:, :-1] = dist
        ret_n[:, 1:] = dist
        return ret_p, ret_n
    
    def _vertical_distance(self, X: np.ndarray) -> np.ndarray:
        ret_p = np.zeros_like(X)
        ret_n = np.zeros_like(X)
        dist = sphere2d.distance(X[:-1], X[1:])
        ret_p[:-1] = dist
        ret_n[1:] = dist
        return ret_p, ret_n
    
    def _horizontal_log(self, X: np.ndarray) -> np.ndarray:
        ret_p = np.zeros_like(X)
        ret_n = np.zeros_like(X)
        log_p = sphere2d.log(X[:, :-1], X[:, 1:])
        log_n = sphere2d.log(X[:, 1:], X[:, :-1])
        ret_p[:, :-1] = log_p
        ret_n[:, 1:] = log_n
        return ret_p, ret_n
    
    def _vertical_log(self, X: np.ndarray) -> np.ndarray:
        ret_p = np.zeros_like(X)
        ret_n = np.zeros_like(X)
        log_p = sphere2d.log(X[:-1], X[1:])
        log_n = sphere2d.log(X[1:], X[:-1])
        ret_p[:-1] = log_p
        ret_n[1:] = log_n
        return ret_p, ret_n
    
    def _f(self, X: np.ndarray) -> float:
        """
        Objective function
        """
        # first term
        dist = sphere2d.distance(X, self.F)
        masked_dist = dist if self.mask is None else dist * self.mask
        masked_dist = 0.5 * np.sum(masked_dist * masked_dist)
        
        # second term
        horizontal_dist, _ = self._horizontal_distance(X)
        vertical_dist, _ = self._vertical_distance(X)
        tv = self.c * (
            np.sum(horizontal_dist * horizontal_dist * self.V_horizontal_p) + np.sum(vertical_dist * vertical_dist * self.V_vertical_p)
        )
        return masked_dist + tv
    
    def _df(self, X: np.ndarray) -> np.ndarray:
        """
        Gradient of objective function on sphere2d
        """
        # first term
        d_dist = - sphere2d.log(X, self.F) if self.mask is None else - self.mask * sphere2d.log(X, self.F)
        
        # second term
        horizontal_log_p, horizontal_log_n = self._horizontal_log(X)
        vertical_log_p, vertical_log_n = self._vertical_log(X)
        d_tv = - 2 * self.c * (
            self.V_horizontal_p * horizontal_log_p + self.V_vertical_p * vertical_log_p\
            + self.V_horizontal_n * horizontal_log_n + self.V_vertical_n * vertical_log_n
        )
        return d_dist + d_tv
    
    def _step_size(self, X: np.ndarray, iteration: int) -> float:
        """
        Armijo condition
        """
        df = self._df(X)
        d = - np.copy(df)
        g = np.sum(df * d)
        t = self.initial_step
        f = self._f(X)
        while self._f(sphere2d.exp(X, t * d)) > f + self.armijo_param * t * g:
            t *= 0.5
            # break when step size 't' becomes too small
            if t <= 1e-16:
                t = 0
                break
        return t
    
    def _initialize(self, F: np.ndarray) -> np.ndarray:
        # inpaint if mask is given
        X = F if self.mask is None else cv2.inpaint(F, 255 - np.uint8(self.mask[:, :, 0] > 0) * 255, 3, cv2.INPAINT_TELEA)
        
        # X on sphere
        X = X.astype(np.float32) / 255.
        norm = np.linalg.norm(X, axis=-1, keepdims=True)
        # [0, 0, 0] -> [1, 1, 1] / sqrt(3)
        X[norm[:,:,0] == 0] = np.array([1.,1.,1.], dtype=np.float32)
        norm[norm==0] = np.sqrt(3)
        return X / norm
    
    def transform(self, F: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        # F on sphere
        norm_F = np.linalg.norm(F, axis=-1, keepdims=True)
        norm_F[norm_F==0] = 1.0
        self.F = F.astype(np.float32) / norm_F
        
        self.mask = mask
        
        # initialize X
        X = self._initialize(F)
        
        # main loop
        for _ in trange(self.max_iter):
            # update V
            horizontal_dist_p, horizontal_dist_n = self._horizontal_distance(X)
            vertical_dist_p, vertical_dist_n = self._vertical_distance(X)
            self.V_horizontal_p = self._s(horizontal_dist_p)
            self.V_horizontal_n = self._s(horizontal_dist_n)
            self.V_vertical_p = self._s(vertical_dist_p)
            self.V_vertical_n = self._s(vertical_dist_n)
            
            # update U by gradient descent
            for i in range(self.max_iter_gd):
                step_size = self._step_size(X, i)
                X = sphere2d.exp(X, - self._df(X) * step_size)
                
            # save objective function value if necessary
            if self.extended_output:
                self.f.append(self._f(X))
        return X
