"""Filesystem utility functions."""
from __future__ import absolute_import
import os
import errno
import torch
import logging

from ..config import cfg

def save_checkpoint(model, epoch, optimizer=None, lr_scheduler=None, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(cfg.TRAIN.MODEL_SAVE_DIR)
    # directory = os.path.join(directory, '{}_{}_{}_{}'.format(cfg.MODEL.MODEL_NAME, cfg.MODEL.BACKBONE,
    #                                                          cfg.DATASET.NAME, cfg.TIME_STAMP))
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}.pth'.format(str(epoch))
    filename = os.path.join(directory, filename)

    state_dict = {
        "state_dict": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        "epoch": epoch,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
    }

    if is_best:
        best_filename = 'best_model.pth'
        best_filename = os.path.join(directory, best_filename)
        torch.save(state_dict, best_filename)
    else:
        if not os.path.exists(filename):
            torch.save(state_dict, filename)
            logging.info('Epoch {} model saved in: {}'.format(epoch, filename))

        # remove last epoch
        pre_filename = '{}.pth'.format(str(epoch - 1))
        pre_filename = os.path.join(directory, pre_filename)
        try:
            if os.path.exists(pre_filename):
                os.remove(pre_filename)
        except OSError as e:
            logging.info(e)

def makedirs(path):
    """Create directory recursively if not exists.
    Similar to `makedir -p`, you can skip checking existence before this function.
    Parameters
    ----------
    path : str
        Path of the desired dir
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

