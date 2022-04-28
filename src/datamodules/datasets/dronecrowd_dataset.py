from glob import glob
from typing import List, Tuple, Optional

import cv2
from albumentations import Compose
import numpy as np
from pathlib import Path
import torch
import os
import re

from scipy import io
from torch.utils.data import Dataset

from src.datamodules.datasets.density_utils import MaskGenerator


class DronecrowdDataset(Dataset):
    classes = ['0 - density']

    def __init__(self,
                 data_root: Path,
                 images_list: List,
                 image_size: Tuple[int, int],
                 sigma: int,
                 augmentations: Compose
                 ):
        self._images_list = self.get_images_list(data_root, images_list)
        self._image_size = image_size
        self._sigma = sigma
        self._augmentations = augmentations

        self.mask_generate = MaskGenerator(self._image_size, sigma=self._sigma)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, keypoints = self._load_data(index)

        transformed = self._augmentations(image=image, keypoints=keypoints)
        image, keypoints = transformed['image'], transformed['keypoints']

        mask = self.mask_generate(keypoints=keypoints)

        return image, torch.from_numpy(mask)

    def _load_data(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_path = self._images_list[index]
        annotation_path = image_path.replace('images', 'ground_truth').replace("img", "GT_img").replace('.jpg', '.mat')

        frame = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        labels = io.loadmat(str(annotation_path))['image_info'][0, 0][0, 0][0][:, :2]

        labels[:, 0][labels[:, 0] >= 1920] = 1920 - 1e-5
        labels[:, 1][labels[:, 1] >= 1080] = 1080 - 1e-5

        return frame, labels

    def __len__(self) -> int:
        return len(self._images_list)

    @staticmethod
    def get_images_list(data_root: Path, sequences: Optional[List[str]]) -> List[str]:
        return sorted([img_path for seq in sequences for img_path in glob(f'{os.path.join(data_root, seq)}*')])
