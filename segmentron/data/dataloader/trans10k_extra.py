"""Prepare Trans10K dataset"""
import os
import torch
import numpy as np
import logging

from PIL import Image
from IPython import embed

from ...config import cfg
from .seg_data_base import SegmentationDataset
from .trans10k_with_fakemix import TransSegmentationWithFakeMix

class TransExtraSegmentation(TransSegmentationWithFakeMix):
    def get_pairs(self, folder, split):
        def get_path_pairs(image_folder):
            image_paths = []
            images = os.listdir(image_folder)
            for imagename in images:
                imagepath = os.path.join(image_folder, imagename)
                if os.path.isfile(imagepath):
                    image_paths.append(imagepath)
                else:
                    logging.info('cannot find the image:', imagepath)

            logging.info('Found {} images in the folder {}'.format(len(image_paths), image_folder))
            return image_paths

        image_folder = folder
        image_paths = get_path_pairs(image_folder)

        return image_paths,None


    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        mask = np.zeros_like(np.array(img))[:,:,0]
        assert mask.max()<=2, mask.max()
        mask = Image.fromarray(mask)

        # synchrosized transform
        img, mask = self._val_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask, self.image_paths[index]
