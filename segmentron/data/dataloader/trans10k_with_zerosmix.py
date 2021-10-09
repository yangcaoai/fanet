import cv2
import random
import numpy as np
from .trans10k_with_fakemix import TransSegmentationWithFakeMix
from ...config import cfg

class TransSegmentationWithZerosMix(TransSegmentationWithFakeMix):
    def _gen_fake_bound(self, img, boundary):
        boundary = boundary[:, :, np.newaxis]
        boundary_content = boundary * img

        dx = random.randint(-1*self.base_size*cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_ALPHA//cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_BETA, self.base_size*cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_ALPHA//cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_BETA)
        dy = random.randint(-1*self.base_size*cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_ALPHA//cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_BETA, self.base_size*cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_ALPHA//cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_BETA)
        move_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        boundary_move = cv2.warpAffine(boundary, move_matrix, (self.base_size, self.base_size))
        boundary_content_move = np.zeros_like(boundary_content)
        return boundary_move, boundary_content_move
