import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm


def generate_lunarseg_zarr():
    root_dir = os.path.join('data', 'LunarSeg')
    dest_root_path = os.path.join('data', 'LunarSegDataset')

    compressor = Blosc('zstd', clevel=6)
    zarr_data_root = zarr.open(dest_root_path)

    if Path(dest_root_path).is_dir():
        shutil.rmtree(dest_root_path)

    for img_path in tqdm(Path(os.path.join(root_dir, 'images')).iterdir()):
        label_path = str(img_path).replace('images', 'masks').replace('.jpg', '.png')

        frame = cv2.imread(str(img_path))
        label = cv2.imread(label_path)

        if frame is None or label is None:
            print(f'Loading {img_path.name} failed - skipping')
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_name = img_path.name.replace('.jpg', '')

        frame_group = zarr_data_root.create_group(frame_name)
        frame_group.array('frame', frame, dtype=np.uint8, chunks=False, compressor=compressor)
        frame_group.array('label', label, dtype=np.uint8, chunks=False, compressor=compressor)


if __name__ == '__main__':
    generate_lunarseg_zarr()
