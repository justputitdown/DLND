import math
import os
import hashlib
from urllib.request import urlretrieve
import zipfile
import gzip
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm


def _read32(bytestream):
    """
    Read 32-bit integer from bytesteam
    :param bytestream: A bytestream
    :return: 32-bit integer
    """
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def _unzip(save_path, _, database_name, data_path):
    """
    Unzip wrapper with the same interface as _ungzip
    :param save_path: The path of the gzip files
    :param database_name: Name of database
    :param data_path: Path to extract to
    :param _: HACK - Used to have to same interface as _ungzip
    """
    print('Extracting {}...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)


def _ungzip(save_path, extract_path, database_name, _):
    """
    Unzip a gzip file and extract it to extract_path
    :param save_path: The path of the gzip files
    :param extract_path: The location to extract the data to
    :param database_name: Name of database
    :param _: HACK - Used to have to same interface as _unzip
    """
    # Get data from save_path
    with open(save_path, 'rb') as f:
        with gzip.GzipFile(fileobj=f) as bytestream:
            magic = _read32(bytestream)
            if magic != 2051:
                raise ValueError('Invalid magic number {} in file: {}'.format(magic, f.name))
            num_images = _read32(bytestream)
            rows = _read32(bytestream)
            cols = _read32(bytestream)
            buf = bytestream.read(rows * cols * num_images)
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)

    # Save data to extract_path
    for image_i, image in enumerate(
            tqdm(data, unit='File', unit_scale=True, miniters=1, desc='Extracting {}'.format(database_name))):
        Image.fromarray(image, 'L').save(os.path.join(extract_path, 'image_{}.jpg'.format(image_i)))


def download_extract(data_name, data_path):
    """
    Download and extract database
    :param database_name: Database name
    """
    DATASET_DOG_NAME = 'dog-data'
    DATASET_HUMAN_NAME = 'human-data'

    if data_name == DATASET_DOG_NAME:
        url = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip'        
        extract_path = os.path.join(data_path, 'dogImages')
        save_path = os.path.join(data_path, 'dog.zip')
        extract_fn = _unzip
    elif data_name == DATASET_HUMAN_NAME:
        url = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip'        
        extract_path = os.path.join(data_path, 'lfw')
        save_path = os.path.join(data_path, 'human.zip')
        extract_fn = _unzip

    if os.path.exists(extract_path):
        print('Found {} Data'.format(data_name))
        return

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(data_name)) as pbar:
            urlretrieve(
                url,
                save_path,
                pbar.hook)

    os.makedirs(extract_path)
    try:
        extract_fn(save_path, extract_path, data_name, data_path)
    except Exception as err:
        shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
        raise err

    # Remove compressed data
    os.remove(save_path)

def dnld_bottleneck(data_name):
    """
    Download and extract database
    :param database_name: Database name
    """
    data_path = './bottleneck_features'
    DATASET_VGG16 = 'vgg16'
    DATASET_VGG19 = 'vgg19'
    DATASET_INCEPV3 = 'inception'
    DATASET_RESNET = 'resnet'
    DATASET_XCEPTION = 'xception'

    if data_name == DATASET_VGG16:
        url = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz'        
        save_path = os.path.join(data_path, 'DogVGG16Data.npz')        
    elif data_name == DATASET_VGG19:
        url = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz'        
        save_path = os.path.join(data_path, 'DogVGG19Data.npz')
    elif data_name == DATASET_INCEPV3:
        url = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz'        
        save_path = os.path.join(data_path, 'DogInceptionV3Data.npz')
    elif data_name == DATASET_RESNET:
        url = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz'        
        save_path = os.path.join(data_path, 'DogResnet50Data.npz')
    elif data_name == DATASET_XCEPTION:
        url = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz'        
        save_path = os.path.join(data_path, 'DogXceptionData.npz')

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(data_name)) as pbar:
            urlretrieve(
                url,
                save_path,
                pbar.hook)
    
    
class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
