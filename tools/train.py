import time
import datetime
import os
import sys
import glob
import random
import numpy as np
import cv2

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

os.environ['TORCH_HOME'] = './init_model/' # put it in the config

import logging
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from torchvision import transforms
from segmentron.data.dataloader import get_segmentation_dataset
from segmentron.models.model_zoo import get_segmentation_model
from segmentron.solver.loss import get_segmentation_loss
from segmentron.solver.optimizer import get_optimizer
from segmentron.solver.lr_scheduler import get_scheduler
from segmentron.utils.distributed import *
from segmentron.utils.score import SegmentationMetric
from segmentron.utils.filesystem import save_checkpoint
from segmentron.utils.options import parse_args
from segmentron.utils.default_setup import default_setup
from segmentron.utils.visualize import show_flops_params
from segmentron.config import cfg
from IPython import embed

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET.MEAN, cfg.DATASET.STD),
        ])
        self.input_transform = input_transform

        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': cfg.TRAIN.BASE_SIZE,
                       'crop_size': cfg.TRAIN.CROP_SIZE}
        train_dataset = get_segmentation_dataset(cfg.DATASET.NAME, split='train', mode='train', **data_kwargs)

        self.iters_per_epoch = len(train_dataset) // (args.num_gpus * cfg.TRAIN.BATCH_SIZE)
        self.max_iters = cfg.TRAIN.EPOCHS * self.iters_per_epoch

        train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        train_batch_sampler = make_batch_data_sampler(train_sampler, cfg.TRAIN.BATCH_SIZE, self.max_iters, drop_last=True)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_sampler=train_batch_sampler,
                                            collate_fn=self.collate_fn,
                                            num_workers=cfg.DATASET.WORKERS,
                                            pin_memory=True)

        # create network
        self.model = get_segmentation_model().to(self.device)
        
        # print params and flops
        if args.num_gpus==1 and get_rank() == 0:
            try:
                show_flops_params(self.model, args.device)
            except Exception as e:
                logging.warning('get flops and params error: {}'.format(e))

        if cfg.MODEL.BN_TYPE not in ['BN']:
            logging.info('Batch norm type is {}, convert_sync_batchnorm is not effective'.format(cfg.MODEL.BN_TYPE))
        elif args.distributed and cfg.TRAIN.SYNC_BATCH_NORM:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logging.info('SyncBatchNorm is effective!')
        else:
            logging.info('Not use SyncBatchNorm!')

        # create criterion
        self.criterion = get_segmentation_loss(cfg.MODEL.MODEL_NAME, use_ohem=cfg.SOLVER.OHEM,
                                               aux=cfg.SOLVER.AUX, aux_weight=cfg.SOLVER.AUX_WEIGHT,
                                               ignore_index=cfg.DATASET.IGNORE_INDEX).to(self.device)

        cfg.SOLVER.LOSS_NAME = 'dice'
        self.criterion_b = get_segmentation_loss(cfg.MODEL.MODEL_NAME).to(self.device)

        # optimizer, for model just includes encoder, decoder(head and auxlayer).
        self.optimizer = get_optimizer(self.model)

        # lr scheduling
        self.lr_scheduler = get_scheduler(self.optimizer, max_iters=self.max_iters,
                                          iters_per_epoch=self.iters_per_epoch)

        # resume checkpoint if needed
        self.start_epoch = 0
        if args.resume is not None:
            loaded_path = None
            if os.path.isdir(args.resume):
                resume_paths = glob.glob( os.path.join( args.resume, "*.pth") ) 
                if len(resume_paths) > 0:
                    loaded_path = resume_paths[0]
                    mtime = os.path.getmtime(loaded_path)
                    for resume_path in resume_paths[1:]:
                        if os.path.getmtime(resume_path) > mtime:
                            loaded_path = resume_path
                            mtime = os.path.getmtime(loaded_path)
            elif os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                loaded_path = args.resume
            
            if loaded_path is not None:
                logging.info('Resuming training, loading {}...'.format(loaded_path))
                resume_sate = torch.load(loaded_path)
                self.model.load_state_dict(resume_sate['state_dict'])
                self.start_epoch = resume_sate['epoch']
                logging.info('resume train from epoch: {}'.format(self.start_epoch))
                if resume_sate['optimizer'] is not None and resume_sate['lr_scheduler'] is not None:
                    logging.info('resume optimizer and lr scheduler from resume state..')
                    self.optimizer.load_state_dict(resume_sate['optimizer'])
                    self.lr_scheduler.load_state_dict(resume_sate['lr_scheduler'])

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank,
                                                             find_unused_parameters=True)


    def collate_fn(self, batch):
        '''
        put fake boundaries cross images in a batch
        '''
        bsize = len(batch)
        imgs, masks, bods, nams, bod_moves, bod_move_contents = zip(*batch)
        imgs = list(imgs)

        for idx, img in enumerate(imgs):
            if(random.randint(0, 1)==1): 
                for tims in range(cfg.TRAIN.FAKE_BOUNDARY_COUNT):
                    fake_id = random.randint(0, bsize-1)
                    img = img*(1-bod_moves[fake_id][:,:,np.newaxis]) + bod_move_contents[fake_id]
                    imgs[idx] = img
            imgs[idx] = self.input_transform(imgs[idx])

        imgs = torch.stack(imgs, dim=0)
        masks = torch.stack(masks, dim=0)
        bods = torch.stack(bods, dim=0) 
        return imgs, masks, bods, nams
        

    def train(self):
        self.save_to_disk = get_rank() == 0
        epochs, max_iters, iters_per_epoch = cfg.TRAIN.EPOCHS, self.max_iters, self.iters_per_epoch
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.iters_per_epoch

        start_time = time.time()
        logging.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        # train the model
        self.model.train()
        iteration = self.start_epoch * iters_per_epoch if self.start_epoch > 0 else 0
        for (images, targets, boundary, filenames) in self.train_loader:
            epoch = iteration // iters_per_epoch + 1
            iteration += 1

            # put data on the device
            images = images.to(self.device)
            targets = targets.to(self.device)
            boundarys = boundary.to(self.device)

            # forward
            outputs, outputs_boundary = self.model(images)

            loss_dict = self.criterion(outputs, targets)
            boundarys = boundarys.float()
            valid = torch.ones_like(boundarys)
            lossb_dict = self.criterion_b(outputs_boundary[0], boundarys, valid)

            weight_boundary = cfg.TRAIN.BLOSS_WEIGHT
            lossb_dict['loss'] = weight_boundary * lossb_dict['loss']
            losses = sum(loss for loss in loss_dict.values()) + \
                     sum(loss for loss in lossb_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            lossb_dict_reduced = reduce_loss_dict(lossb_dict)
            lossesb_reduced = sum(loss for loss in lossb_dict_reduced.values())

            # backward and optimize
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            eta_seconds = ((time.time() - start_time) / iteration) * (max_iters - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % log_per_iters == 0 and self.save_to_disk:
                logging.info(
                    "Epoch: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || "
                    "Loss: {:.4f} || Loss_b: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                        epoch, epochs, iteration % iters_per_epoch, iters_per_epoch,
                        self.optimizer.param_groups[0]['lr'], losses_reduced.item(), lossesb_reduced.item(),
                        str(datetime.timedelta(seconds=int(time.time() - start_time))),
                        eta_string))

            if iteration % self.iters_per_epoch == 0 and self.save_to_disk:
                save_checkpoint(self.model, epoch, self.optimizer, self.lr_scheduler, is_best=False)

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logging.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))



if __name__ == '__main__':
    args = parse_args()

    # get config
    cfg.update_from_file(args.config_file)
    cfg.update_from_list(args.opts)
    cfg.PHASE = 'train'
    cfg.ROOT_PATH = root_path
    cfg.check_and_freeze()

    # setup python train environment, logger, seed..
    default_setup(args)

    # create a trainer and start train
    trainer = Trainer(args)
    trainer.train()
