import cv2
import random
import numpy as np

from .trans10k_with_fakemix import TransSegmentationWithFakeMix
from ...config import cfg

class TransSegmentationWithMeansMix(TransSegmentationWithFakeMix):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.image_means = np.reshape(np.array(cfg.DATASET.MEAN),[1,1,-1])
        self.image_stds = np.reshape(np.array(cfg.DATASET.STD),[1,1,-1])

    def _gen_fake_bound(self, img, boundary):
        boundary = boundary[:, :, np.newaxis]
        boundary_content = boundary * (255*np.ones_like(img)*self.image_means).astype(img.dtype)

        dx = random.randint(-1*self.base_size*cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_ALPHA//cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_BETA, self.base_size*cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_ALPHA//cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_BETA)
        dy = random.randint(-1*self.base_size*cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_ALPHA//cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_BETA, self.base_size*cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_ALPHA//cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_BETA)
        move_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        boundary_move = cv2.warpAffine(boundary, move_matrix, (self.base_size, self.base_size))
        boundary_content_move = cv2.warpAffine(boundary_content, move_matrix, (self.base_size, self.base_size)) 
        return boundary_move, boundary_content_move
