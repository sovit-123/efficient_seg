# Folder containing training images.
TRAIN_IMAGES = '../data/person_segmentation/train_images'
# Folder containing training masks.
TRAIN_LABELS = '../data/person_segmentation/train_masks'
# Folder containing validation images.
VALID_IMAGES = '../data/person_segmentation/val_images'
# Folder containing validation masks.
VALID_LABELS = '../data/person_segmentation/val_masks'

ALL_CLASSES = ['background', 'person']

LABEL_COLORS_LIST = [
    (0, 0, 0), # Background.
    (255, 255, 255),
]

VIS_LABEL_MAP = [
    (0, 0, 0), # Background.
    (0, 255, 0),
]