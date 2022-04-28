import itertools
import os.path
import re
from collections import deque
from pathlib import Path
from random import Random
from typing import Optional, List, Tuple

import albumentations as A
import cv2
import hydra
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class DensityDataModule(LightningDataModule):
    def __init__(self,
                 data_path: Path,
                 dataset: Dataset,
                 augment: bool,
                 batch_size: int,
                 image_size: Tuple[int, int],
                 image_mean: Tuple[float, float, float],
                 image_std: Tuple[float, float, float],
                 mask_gaussian_sigma: int,
                 number_of_workers: int,
                 number_of_splits: int,
                 current_split: int
                 ):
        super().__init__()

        self._data_root = Path(data_path)
        self._dataset = dataset
        self._augment = augment
        self._batch_size = batch_size
        self._image_size = image_size
        self._image_mean = image_mean
        self._image_std = image_std
        self._sigma = mask_gaussian_sigma
        self._number_of_workers = number_of_workers
        self._number_of_splits = number_of_splits
        self._current_split = current_split

        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

        self._transforms = A.Compose([
            A.LongestMaxSize(self._image_size[0]),
            A.PadIfNeeded(self._image_size[1], self._image_size[0], border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=self._image_mean, std=self._image_std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

        self._augmentations = A.Compose([
            # rgb augmentations
            A.RandomGamma(gamma_limit=(80, 120)),
            A.ColorJitter(brightness=0, contrast=0, hue=0.01, saturation=0.5),
            # geometry augmentations
            A.Affine(rotate=(-5, 5), translate_px=(-10, 10), scale=(0.9, 1.1)),
            A.Flip(),
            # transforms
            A.LongestMaxSize(image_size[0]),
            A.PadIfNeeded(image_size[1], image_size[0], border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=self._image_mean, std=self._image_std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))

    @staticmethod
    def partition_sequences(sequences: List[str], n: int) -> List[List[str]]:
        sequences = sequences.copy()
        Random(42).shuffle(sequences)
        return [sequences[i::n] for i in range(n)]

    @staticmethod
    def get_train_valid_test(splits: List[List[str]], current_split: int):
        splits = deque(splits)
        splits.rotate(current_split)
        splits = list(splits)

        return list(itertools.chain.from_iterable(splits[:-2])), splits[-2], splits[-1]

    def setup(self, stage: Optional[str] = None):
        splits = []
        for split_name in ['train_data', 'val_data', 'test_data']:
            seq = []
            for frame_path in Path(os.path.join(self._data_root, split_name, 'images')).iterdir():
                match = re.match(r'img(\d{3})\d+\.jpg', frame_path.name)
                if match is None:
                    raise RuntimeError(f'Unknown file: {frame_path}')
                seq.append(f'{split_name}/images/img{match.group(1)}')

            splits.append(sorted(list(set(seq))))

        if self._number_of_splits is not None and self._current_split is not None:
            splits = [item for sublist in splits for item in sublist]
            splits = self.partition_sequences(splits, self._number_of_splits)
            train_split, valid_split, test_split = self.get_train_valid_test(splits, self._current_split)
        else:
            train_split, valid_split, test_split = splits

        self._train_dataset: Dataset = hydra.utils.instantiate({
            '_target_': self._dataset,
            'data_root': self._data_root,
            'images_list': train_split,
            'image_size': self._image_size,
            'sigma': self._sigma,
            'augmentations': self._augmentations if self._augment else self._transforms,
        })

        self._valid_dataset: Dataset = hydra.utils.instantiate({
            '_target_': self._dataset,
            'data_root': self._data_root,
            'images_list': valid_split,
            'image_size': self._image_size,
            'sigma': self._sigma,
            'augmentations': self._transforms,
        })

        self._test_dataset: Dataset = hydra.utils.instantiate({
            '_target_': self._dataset,
            'data_root': self._data_root,
            'images_list': test_split,
            'image_size': self._image_size,
            'sigma': self._sigma,
            'augmentations': self._transforms,
        })

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True, drop_last=True, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self._valid_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset, batch_size=self._batch_size, num_workers=self._number_of_workers,
            pin_memory=True
        )
