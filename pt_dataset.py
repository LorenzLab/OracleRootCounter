from typing import Callable
import os

import numpy as np
from torch.utils.data import Dataset

class Oracle_Dataset(Dataset):
    def __init__(self, img_maker, root_img_path):
        self.img_maker = img_maker
        self.root_img_path = root_img_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, bounding_boxes, classes = self.img_maker(self.root_img_path)
        bounding_boxes = np.array(bounding_boxes)
        classes = np.array(classes)
        labels = np.zeros((len(bounding_boxes), 6))
        labels[:, 1:5] = bounding_boxes
        labels[:, 5] = classes

        # for COCO mAP rescaling
        shapes = 

        return 