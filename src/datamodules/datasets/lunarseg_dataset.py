from typing import List, Tuple

from albumentations import Compose
import cv2
import numpy as np
from pathlib import Path
import torch
import zarr

from torch.utils.data import Dataset


class LunarSegDataset(Dataset):
    classes = ['0 - background', '1 - sky', '2 - rocks', '3 - rover']

    def __init__(self,
                 data_root: Path,
                 images_list: List,
                 augmentations: Compose
                 ):

        self._data_root = zarr.open(str(data_root), mode='r')
        self._images_list = images_list
        self._augmentations = augmentations

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self._load_data(index)

        transformed = self._augmentations(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask']

        return image, torch.multiply(mask.type(torch.float32), 1./255.).permute((2, 0, 1))

    def _load_data(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image_name = self._images_list[index]

        frame = self._data_root[f'{image_name}/frame'][:]
        label = self._data_root[f'{image_name}/label'][:]

        return frame, label

    def __len__(self) -> int:
        return len(self._images_list)
