import os
import cv2
import glob
import random
import logging
import numpy as np
from PIL import Image

import torch
from torchvision import transforms as trans

from .trans10k_with_fakemix import TransSegmentationWithFakeMix

from ...config import cfg

class MSDWithFakeMix(TransSegmentationWithFakeMix):
    BASE_DIR = 'MSD'
    NUM_CLASS = 2

    def get_pairs(self, img_folder, split='train'):
        img_paths = glob.glob(os.path.join(img_folder,split,"image","*.jpg"))
        mask_paths = []
        for single_image_path in img_paths:
            imgname = os.path.split(single_image_path)[1]
            mask_paths.append(os.path.join(img_folder,split,"mask",imgname.replace('.jpg','.png')))
    
        print(f"datset lenghts for {split} in {img_folder}:{len(img_paths)}")
        return img_paths, mask_paths
    
    def _class_to_index(self, mask):
        mask = np.array(mask)
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        mask[mask==255] = 1
        assert mask.max()<=1, f"error, undefined values in mask: {np.uinque(mask)}"
        mask = Image.fromarray(mask)
        return mask

    @property
    def classes(self):
        """Category names."""
        return ('background', 'mirror')
