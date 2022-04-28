from typing import List, Tuple, Optional

from albumentations import Compose
import numpy as np
from pathlib import Path
import torch
import zarr

from torch.utils.data import Dataset

from src.datamodules.datasets.density_utils import MaskGenerator


class CountingSimDataset(Dataset):
    classes = ['0 - density']

    def __init__(self,
                 data_root: Path,
                 images_list: List,
                 image_size: Tuple[int, int],
                 sigma: int,
                 augmentations: Compose
                 ):

        self._data_root = zarr.open(str(data_root), mode='r')
        self._images_list = self.get_images_list(self._data_root, images_list)
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
        image_name = self._images_list[index]

        frame = self._data_root[f'{image_name}/frame'][:]
        label = self._data_root[f'{image_name}/label'][:]

        return frame, label

    def __len__(self) -> int:
        return len(self._images_list)

    @staticmethod
    def get_images_list(data_root: zarr.Group, sequences: Optional[List[str]]) -> List[str]:
        images = []
        if sequences is not None:
            for sequence_name in sequences:
                for image_name in data_root[sequence_name].group_keys():
                    images.append(f'{sequence_name}/{image_name}')
        else:
            for sequence_name, sequence_data in data_root.groups():
                for image_name in sequence_data.group_keys():
                    images.append(f'{sequence_name}/{image_name}')

        return sorted(images)