"""
Create image dataset folder from text files.
All images in a directory are expected to have the same extension.

USAGE:
    python create_sets.py --src-dir input/people_segmentation/images/ --dest-dir input/train_images --txt input/people_segmentation/segmentation/train.txt
"""

import argparse
import shutil
import os

from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--txt',
    required=True,
    help='path to the text file'
)
parser.add_argument(
    '--src-dir',
    dest='src_dir',
    required=True,
    help='folder path where images are present'
)
parser.add_argument(
    '--dest-dir',
    dest='dest_dir',
    required=True,
    help='folder path to store/copy the images'
)
args = parser.parse_args()

SRC_DIR = args.src_dir
DEST_DIR = args.dest_dir
TXT_FILE = args.txt

all_image_names = [name.rstrip() for name in open(TXT_FILE, 'r').readlines()]

def copy_image(image_name_list, root_path, dest_path):
    """
    :param image_name: List containing image names without extension.
    :param root_path: Root directory path where the images are present.
    :param dest_path: Destination directory path to store the images. 
    """
    root_dir_images = os.listdir(root_path)
    # Find image extension.
    root_image_name = '.' + '.'.join(root_dir_images[0].split('.')[:-1])
    ext = '.' + root_dir_images[0].split('.')[-1]
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for i, image_name in tqdm(enumerate(image_name_list)):
        shutil.copy(
            os.path.join(root_path, image_name+ext),
            os.path.join(dest_path, image_name+ext)
        )

copy_image(all_image_names, SRC_DIR, DEST_DIR)