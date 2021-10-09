"""Prepare Trans10K dataset"""
import os
import torch
import numpy as np
import logging
import random
from PIL import Image
from .seg_data_base import SegmentationDataset
from IPython import embed
import cv2
from torchvision import transforms as trans
from ...config import cfg

class TransSegmentationWithFakeMix(SegmentationDataset):
    BASE_DIR = 'Trans10K'
    NUM_CLASS = 3

    def __init__(self, root=None, split='train', mode=None, transform=None, **kwargs):
        root = cfg.DATASET.PATH if root is None else root
        super().__init__(root, split, mode, transform, **kwargs)
        assert os.path.exists(self.root), f"Please put dataset in {self.root}"
        self.image_paths, self.mask_paths = self.get_pairs(self.root, self.split)

    def _class_to_index(self, mask):
        mask = np.array(mask)[:,:,:3].mean(-1) # in case channel of mask is bigger than 1
        mask[mask==85.0] = 1
        mask[mask==255.0] = 2
        assert mask.max()<=2, f"error, undefined values in mask: {np.uinque(mask)}"
        mask = Image.fromarray(mask)
        return mask

    def get_pairs(self,folder,split='train'):
        def get_path_pairs(img_folder, mask_folder):
            img_paths = []
            mask_paths = []
            imgs = os.listdir(img_folder)

            for imgname in imgs:
                imgpath = os.path.join(img_folder, imgname)
                maskname = imgname.replace('.jpg', '_mask.png')
                maskpath = os.path.join(mask_folder, maskname)
                if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                else:
                    logging.info('cannot find the mask or image:', imgpath, maskpath)
            logging.info('Found {} images in the folder {}'.format(len(img_paths), img_folder))
            return img_paths, mask_paths

        if split == 'train':
            img_folder = os.path.join(folder, split, 'images')
            mask_folder = os.path.join(folder, split, 'masks')
            img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        else:
            assert split == 'validation' or split == 'test'
            easy_img_folder = os.path.join(folder, split, 'easy', 'images')
            easy_mask_folder = os.path.join(folder, split, 'easy', 'masks')
            hard_img_folder = os.path.join(folder, split, 'hard', 'images')
            hard_mask_folder = os.path.join(folder, split, 'hard', 'masks')
            easy_img_paths, easy_mask_paths = get_path_pairs(easy_img_folder, easy_mask_folder)
            hard_img_paths, hard_mask_paths = get_path_pairs(hard_img_folder, hard_mask_folder)
            easy_img_paths.extend(hard_img_paths)
            easy_mask_paths.extend(hard_mask_paths)
            img_paths = easy_img_paths
            mask_paths = easy_mask_paths
        assert len(img_paths) == len(mask_paths), "error, the count of images and the count of masks does not match!"
        assert len(img_paths) > 0, "error, the count of images is 0"
        return img_paths, mask_paths

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')

        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.image_paths[index])

        mask = Image.open(self.mask_paths[index])
        mask = self._class_to_index(mask)
        
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

        boundary = self.get_boundary(mask)
        boundary_move, boundary_content_move = self._gen_fake_bound(img, boundary) 
        boundary = torch.LongTensor(np.array(boundary).astype('int32'))

        # general resize, normalize and toTensor
        if(self.mode == 'val'):
            if self.transform is not None:
                img = self.transform(img)

        return img, mask, boundary, self.image_paths[index], boundary_move, boundary_content_move # img:un_transformed, array. mask: tensor. boundary:tensor. boundary_move:array, boundary_content_move:array

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32'))

    def _gen_fake_bound(self, img, boundary):
        '''
        Input shape:
                   img: [512, 512, 3]
                   boundary: [512, 512]
        Input values:
                   img: [0, 255]
                   boundary: {0, 1} 
        '''
        boundary = boundary[:, :, np.newaxis]
        boundary_content = boundary * img

        dx = random.randint(-1*self.base_size*cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_ALPHA//cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_BETA, self.base_size*cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_ALPHA//cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_BETA)
        dy = random.randint(-1*self.base_size*cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_ALPHA//cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_BETA, self.base_size*cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_ALPHA//cfg.TRAIN.FAKE_BOUNDARY_DISTANCE_BETA)
        move_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

        boundary_move = cv2.warpAffine(boundary, move_matrix, (self.base_size, self.base_size))
        boundary_content_move = cv2.warpAffine(boundary_content, move_matrix, (self.base_size, self.base_size)) 
        return boundary_move, boundary_content_move
 
    def __len__(self):
        return len(self.image_paths)

    def get_boundary(self, mask, thicky=8):
        tmp = mask.data.numpy().astype('uint8')
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        boundary = cv2.drawContours(boundary, contour, -1, 1, thicky)
        return boundary

    @property
    def pred_offset(self):
        return 0

    @property
    def classes(self):
        """Category names."""
        return ('background', 'things', 'stuff')

if __name__ == '__main__':
    dataset = TransSegmentationWithFakeMix()
