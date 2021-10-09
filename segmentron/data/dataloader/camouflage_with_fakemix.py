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

class CAMOUFLAGEWithFakeMix(MSDWithFakeMix):
    BASE_DIR = 'CAMOUFLAGE'
    NUM_CLASS = 2

    def get_pairs(self, img_folder, split='train'):
        split = split.capitalize()
        paths = glob.glob(os.path.join(img_folder,split,"Image","*.jpg"))
        img_paths = []
        mask_paths = []
        for single_path in paths:
            if "NonCAM" in single_path: continue
            imgname = os.path.split(single_path)[1]
            mask_path = os.path.join(img_folder,split,"GT_Object",imgname.replace('.jpg','.png'))
            if os.path.exists(mask_path) is True:
                img_paths.append(single_path)
                mask_paths.append(mask_path)
    
        print(f"datset lenghts for {split} in {img_folder}:{len(img_paths)}")
        return img_paths, mask_paths

    @property
    def classes(self):
        """Category names."""
        return ('background', 'camouflage')
