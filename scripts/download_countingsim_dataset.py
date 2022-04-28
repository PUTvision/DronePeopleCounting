import os
import shutil
import urllib.request
import zipfile
from glob import glob
from pathlib import Path


def download_countingsim_dataset():
    root_dir = Path('./data/')

    print('Downloading files...')
    download_path = os.path.join(root_dir, 'CountingSim.zip')
    if not Path(download_path).is_file():
        urllib.request.urlretrieve('https://chmura.put.poznan.pl/s/wNOZW1aPWS64o7H/download', download_path)
    else:
        print('Zip file exists, skipping downloading...')

    print('Extracting zip files...')
    extract_path = os.path.join(root_dir, 'CountingSim')
    if Path(extract_path).is_dir():
        shutil.rmtree(extract_path)

    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(root_dir)

    print('Testing...')
    assert len(glob(os.path.join(extract_path, 'images', '*.jpg'))) == 2700
    assert len(glob(os.path.join(extract_path, 'gt', '*.mat'))) == 2700

    print('Done!')


if __name__ == '__main__':
    download_countingsim_dataset()
