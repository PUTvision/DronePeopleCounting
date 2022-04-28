from typing import Tuple, List

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class MaskGenerator:
    def __init__(self, image_shape: Tuple[int, int], sigma: int):
        self._image_shape = image_shape
        self._sigma = sigma

    def __call__(self, keypoints: List[Tuple[int, int]]) -> np.ndarray:
        return self.generate_mask(keypoints, self._image_shape, self._sigma)

    @staticmethod
    def generate_mask(keypoints: List[Tuple[int, int]], image_shape: Tuple[int, int], sigma: int):
        label = np.zeros((image_shape[1], image_shape[0]), dtype=np.float32)

        for key in keypoints:
            label[int(key[1]), int(key[0])] = 100

        label = gaussian_filter(label, sigma=(sigma, sigma), order=0)

        return np.array([label])