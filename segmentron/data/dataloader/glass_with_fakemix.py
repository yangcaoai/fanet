import os
import cv2
import glob
import random
import logging
import numpy as np
from PIL import Image

import torch
from torchvision import transforms as trans

from .msd_with_fakemix import MSDWithFakeMix
from ...config import cfg

class GLASSWithFakeMix(MSDWithFakeMix):
    BASE_DIR = 'GLASS'
    NUM_CLASS = 2

    def get_pairs(self, img_folder, split='train'):
        img_paths = glob.glob(os.path.join(img_folder,split,"image","*.jpg"))
        mask_paths = []
        for single_image_path in img_paths:
            imgname = os.path.split(single_image_path)[1]
            mask_paths.append(os.path.join(img_folder,split,"mask",imgname.replace('.jpg','.png')))
    
        print(f"datset lenghts for {split} in {img_folder}:{len(img_paths)}")
        return img_paths, mask_paths

    @property
    def classes(self):
        """Category names."""
        return ('background', 'glass')
