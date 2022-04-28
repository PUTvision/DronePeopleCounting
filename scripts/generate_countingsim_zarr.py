import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import zarr
from numcodecs import Blosc
from scipy import io
from tqdm import tqdm


def generate_countingsim_zarr():
    root_dir = os.path.join('data', 'CountingSim')
    dest_root_path = os.path.join('data', 'CountingSimDataset')

    compressor = Blosc('zstd', clevel=6)
    zarr_data_root = zarr.open(dest_root_path)

    if Path(dest_root_path).is_dir():
        shutil.rmtree(dest_root_path)

    sequences = split_into_sequences(Path(os.path.join(root_dir, 'images')))

    for sequence_name, sequence_paths in sequences.items():

        sequence_root = zarr_data_root.create_group(sequence_name, overwrite=True)

        for img_path in tqdm(sequence_paths):
            label_path = str(img_path).replace('images', 'gt').replace('img', 'GT_img').replace('.jpg', '.mat')

            frame = cv2.imread(str(img_path))
            frame_annotation = io.loadmat(str(label_path))['image_info'][0, 0][0, 0][0]

            frame_annotation[:, 0][frame_annotation[:, 0] >= frame.shape[1]] = frame.shape[1] - 1e-5
            frame_annotation[:, 1][frame_annotation[:, 1] >= frame.shape[0]] = frame.shape[0] - 1e-5

            if frame is None or frame_annotation is None:
                print(f'Loading {img_path.name} failed - skipping')
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_name = img_path.name.replace('.jpg', '')

            frame_group = sequence_root.create_group(frame_name)
            frame_group.array('frame', frame, dtype=np.uint8, chunks=False, compressor=compressor)
            frame_group.array('label', frame_annotation, dtype=np.int32, chunks=False, compressor=compressor)


def split_into_sequences(frames_directory: Path) -> Dict[str, List[Path]]:
    sequences = defaultdict(lambda: [])
    for frame_path in frames_directory.iterdir():
        match = re.match(r'img(\d{3})\d+\.jpg', frame_path.name)
        if match is None:
            raise RuntimeError(f'Unknown file: {frame_path}')

        sequences[match.group(1)].append(frame_path)

    return sequences


if __name__ == '__main__':
    generate_countingsim_zarr()
