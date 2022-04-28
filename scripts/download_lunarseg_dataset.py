import os
import shutil
import urllib.request
import zipfile
from glob import glob
from pathlib import Path


def download_lunarseg_dataset():
    root_dir = Path('./data/')

    print('Downloading files...')
    download_path = os.path.join(root_dir, 'LunarSeg.zip')
    if not Path(download_path).is_file():
        urllib.request.urlretrieve('https://chmura.put.poznan.pl/s/znxmd6XNEV41Niq/download', download_path)
    else:
        print('Zip file exists, skipping downloading...')

    print('Extracting zip files...')
    extract_path = os.path.join(root_dir, 'LunarSeg')
    if Path(extract_path).is_dir():
        shutil.rmtree(extract_path)

    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(root_dir)

    print('Testing...')
    assert len(glob(os.path.join(extract_path, 'images', '*.jpg'))) == 744
    assert len(glob(os.path.join(extract_path, 'masks', '*.png'))) == 744

    print('Done!')


if __name__ == '__main__':
    download_lunarseg_dataset()
