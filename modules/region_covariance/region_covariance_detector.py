from typing import List, Tuple
from itertools import product

import cv2
import numpy as np
from scipy.linalg import eigh
from tqdm import tqdm


def calc_distance(x: np.ndarray, y: np.ndarray) -> float:
    w = eigh(x, y, eigvals_only=True)
    return np.linalg.norm(np.log(w) ** 2)


class RegionCovarianceDetector:
    """
    Object Detector Using Region Covariance

    Parameters
    ----------
    coord : bool, optional (default=True)
        Whether use coordinates as features or not.

    color : bool, optional (default=True)
        Whether use color channels as features or not.

    intensity : bool, optional (default=False)
        Whether use intensity as feature or not.

    kernels : a list of np.ndarray, optional (default=None)
        Filters applied to intensity image. If None, no filters are used.

    ratio : float, optional (default=1.15)
        Scaling factor between two consecutive scales of the search window size and step size.

    step : int, optional (default=3)
        The minimum step size.

    n_windows : int, optional (default=9)
        The number of scales of the windows.

    eps : float, optional (default=1e-16)
        Small number to keep covariance matrices in SPD.

    Attributes
    ----------
    object_shape_ : (int, int)
        The object's shape.

    object_covariance_ : np.ndarray, shape (n_features, n_features)
        Covariance matrix of the object.
    """

    def __init__(
            self,
            coord: bool = True,
            color: bool = True,
            intensity: bool = False,
            kernels: List[np.ndarray] = None,
            ratio: float = 1.15,
            step: int = 3,
            n_windows: int = 9,
            eps: float = 1e-16
    ):
        self.coord = coord
        self.color = color
        self.intensity = intensity
        self.kernels = kernels
        self.ratio = ratio
        self.step = step
        self.n_windows = n_windows
        self.eps = eps

        self.object_shape_ = None
        self.object_covariance_ = None

    def _extract_features(self, img: np.ndarray) -> List[np.ndarray]:
        """
        Extract image features.

        Parameters
        ----------
        img : np.ndarray, shape (h, w, c)
            uint8 RGB image

        Returns
        -------
        features : a list of np.ndarray
            Features such as intensity, its gradient and so on.
        """
        h, w, c = img.shape[:3]
        intensity = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2] / 255.
        features = list()

        # use coordinates
        if self.coord:
            features.append(np.tile(np.arange(w, dtype=float), (h, 1)))
            features.append(np.tile(np.arange(h, dtype=float).reshape(-1, 1), (1, w)))

        # use color channels
        if self.color:
            for i in range(c):
                features.append(img[:, :, i].astype(float) / 255.)

        # use intensity
        if self.intensity:
            features.append(intensity)

        # use filtered images
        if self.kernels is not None:
            for kernel in self.kernels:
                features.append(np.abs(cv2.filter2D(intensity, cv2.CV_64F, kernel)))

        return features

    def _calc_integral_images(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate integral images.

        Parameters
        ----------
        img : np.ndarray, shape (h, w, c)
            uint8 RGB image

        Returns
        -------
        P : np.ndarray, shape (h+1, w+1, n_features)
            First order integral images of features.

        Q : np.ndarray, shape (h+1, w+1, n_features, n_features)
            Second order integral images of features.
        """
        h, w = img.shape[:2]
        features = self._extract_features(img)
        length = len(features)

        # first order integral images
        P = cv2.integral(np.array(features).transpose((1, 2, 0)))

        # second order integral images
        Q = cv2.integral(
            np.array(list(map(lambda x: x[0] * x[1], product(features, features)))).transpose((1, 2, 0))
        )
        Q = Q.reshape(h + 1, w + 1, length, length)
        return P, Q

    def _calc_covariance(self, P: np.ndarray, Q: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int]) -> np.ndarray:
        """
        Calculate covariance matrix from integral images.

        Parameters
        ----------
        P : np.ndarray, shape (h+1, w+1, n_features)
            First order integral images of features.

        Q : np.ndarray, shape (h+1, w+1, n_features, n_features)
            Second order integral images of features.

        pt1 : (int, int)
            Left top coordinate.

        pt2 : (int, int)
            Right bottom coordinate.

        Returns
        -------
        covariance : np.ndarray, shape (n_features, n_features)
            Covariance matrix.
        """
        x1, y1 = pt1
        x2, y2 = pt2
        q = Q[y2, x2] + Q[y1, x1] - Q[y1, x2] - Q[y2, x1]
        p = P[y2, x2] + P[y1, x1] - P[y1, x2] - P[y2, x1]
        n = (y2 - y1) * (x2 - x1)
        covariance = (q - np.outer(p, p) / n) / (n - 1) + self.eps * np.identity(P.shape[2])
        return covariance

    def fit(self, img: np.ndarray):
        """
        Calculate the object covariance matrix.

        Parameters
        ----------
        img : np.ndarray, shape (h, w, c)
            uint8 RGB image

        Returns
        -------
        : Fitted detector.
        """
        h, w = img.shape[:2]
        P, Q = self._calc_integral_images(img)

        # normalize about coordinates
        if self.coord:
            for i, size in enumerate((w, h)):
                P[:, :, i] /= size
                Q[:, :, i] /= size
                Q[:, :, :, i] /= size

        # calculate covariance matrix
        self.object_covariance_ = self._calc_covariance(P, Q, (0, 0), (w, h))
        self.object_shape_ = (h, w)
        return self

    def predict(self, img: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], float]:
        """
        Compute object's position in the target image.

        Parameters
        ----------
        img : np.ndarray, shape (h, w, c)
            uint8 RGB image

        Returns
        -------
        pt1 : (int, int)
            Left top coordinate.

        pt2 : (int, int)
            Right bottom coordinate.

        score : float
            Dissimilarity of object and target covariance matrices.
        """
        tar_h, tar_w = img.shape[:2]
        obj_h, obj_w = self.object_shape_
        P, Q = self._calc_integral_images(img)

        # search window's shape and step size
        end = (self.n_windows + 1) // 2
        start = end - self.n_windows
        shapes = [(int(obj_h * self.ratio ** i), int(obj_w * self.ratio ** i)) for i in range(start, end)]
        steps = [int(self.step * self.ratio ** i) for i in range(self.n_windows)]

        distances = list()
        for shape, step in tqdm(zip(shapes, steps)):
            p_h, p_w = shape
            p_P, p_Q = P.copy(), Q.copy()

            # normalize about coordinates
            if self.coord:
                for i, size in enumerate((p_w, p_h)):
                    p_P[:, :, i] /= size
                    p_Q[:, :, i] /= size
                    p_Q[:, :, :, i] /= size

            distance = list()
            y1, y2 = 0, p_h
            while y2 <= tar_h:
                dist = list()
                x1, x2 = 0, p_w
                while x2 <= tar_w:
                    # calculate covariance matrix
                    p_cov = self._calc_covariance(p_P, p_Q, (x1, y1), (x2, y2))

                    # jump horizontally
                    x1 += step
                    x2 += step

                    # calculate dissimilarity of two covariance matrices
                    dist.append(calc_distance(self.object_covariance_, p_cov))

                # jump vertically
                y1 += step
                y2 += step
                distance.append(dist)
            distances.append(np.array(distance))

        # choose the most similar window
        min_indices = list(map(np.argmin, distances))
        min_index = int(np.argmin([dist.flatten()[i] for i, dist in zip(min_indices, distances)]))
        min_step = steps[min_index]
        min_shape = shapes[min_index]
        min_indice = min_indices[min_index]
        b_h, b_w = distances[min_index].shape

        pt1 = ((min_indice % b_w) * min_step, (min_indice // b_w) * min_step)
        pt2 = (pt1[0] + min_shape[1], pt1[1] + min_shape[0])
        score = distances[min_index].flatten()[min_indice]
        return pt1, pt2, score
