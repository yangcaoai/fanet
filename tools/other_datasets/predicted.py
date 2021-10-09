from __future__ import print_function

import os
import sys

path_segments = os.path.abspath(os.path.dirname(__file__)).split( os.path.sep )
root_path = os.path.join('/', *path_segments[:-2] )
sys.path.append(root_path)
print(f"root path:{root_path}")

import time
import cv2
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.utils.distributed import synchronize, make_data_sampler, make_batch_data_sampler
from segmentron.config import cfg
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from progressbar import *

class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])

        # test dataloader
        val_dataset = get_segmentation_dataset(cfg.DATASET.NAME,
                                               split='test',
                                               mode='val',
                                               transform=input_transform,
                                               base_size=cfg.TRAIN.BASE_SIZE)

        val_sampler = make_data_sampler(val_dataset, shuffle=False, distributed=args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=cfg.TEST.BATCH_SIZE, drop_last=False)

        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=cfg.DATASET.WORKERS,
                                          pin_memory=True)
        logging.info('**** number of images: {}. ****'.format(len(self.val_loader)))

        self.classes = val_dataset.classes
        # create network
        self.model = get_segmentation_model().to(self.device)

        if hasattr(self.model, 'encoder') and cfg.MODEL.BN_EPS_FOR_ENCODER:
                logging.info('set bn custom eps for bn in encoder: {}'.format(cfg.MODEL.BN_EPS_FOR_ENCODER))
                self.set_batch_norm_attr(self.model.encoder.named_modules(), 'eps', cfg.MODEL.BN_EPS_FOR_ENCODER)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        self.model.to(self.device)
        num_gpu = args.num_gpus

    def set_batch_norm_attr(self, named_modules, attr, value):
        for m in named_modules:
            if isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.SyncBatchNorm):
                setattr(m[1], attr, value)

    def eval(self, show_scores=False):
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model

        logging.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))

        experiment_id = os.path.basename( self.args.config_file )[:-5]
        output_dir = root_path+f'/result/{cfg.DATASET.NAME}/{experiment_id}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        widgets = ['Inference: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                   ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets, maxval=10 * len(self.val_loader)).start()
        time_start = time.time()
        for i, (image, target, boundary, filename, _, _) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)
            boundary = boundary.to(self.device)

            filename = filename[0]
            with torch.no_grad():
                output,_ = model.evaluate(image) # [b,c,h,w]
                output_softmax = torch.nn.functional.softmax(output,dim=1) # [b,c,h,w]
                saliency_map = output_softmax.cpu().numpy()[0,1] # [h,w]
                saliency_map = (255*saliency_map).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir,os.path.basename(filename[:-3])+"png"),saliency_map)

            pbar.update(10 * i + 1)

        pbar.finish()
        synchronize()

if __name__ == '__main__':
    args = parse_args()
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'test'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    default_setup(args)

    evaluator = Evaluator(args)
    evaluator.eval(show_scores=False)
