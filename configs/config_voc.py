"""
VOC.
"""
# Folder containing training images.
TRAIN_IMAGES = '../data/voc_segmentation/voc_2012_segmentation_data/train_images'
# Folder containing training masks.
TRAIN_LABELS = '../data/voc_segmentation/voc_2012_segmentation_data/train_labels'
# Folder containing validation images.
VALID_IMAGES = '../data/voc_segmentation/voc_2012_segmentation_data/valid_images'
# Folder containing validation masks.
VALID_LABELS = '../data/voc_segmentation/voc_2012_segmentation_data/valid_labels'

ALL_CLASSES = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'dining table',
    'dog',
    'horse',
    'motorbike',
    'person',
    'potted plant',
    'sheep',
    'sofa',
    'train',
    'tv/monitor'
]

LABEL_COLORS_LIST = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128), 
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),   
    (128, 192, 0),
    (0, 64, 128)
]

VIS_LABEL_MAP = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128), 
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),   
    (128, 192, 0),
    (0, 64, 128)
]