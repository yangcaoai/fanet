"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .trans10k_extra import TransExtraSegmentation
from .trans10k_with_fakemix import TransSegmentationWithFakeMix
from .trans10k_with_zerosmix import TransSegmentationWithZerosMix
from .trans10k_with_meansmix import TransSegmentationWithMeansMix
from .trans10k_with_imagesmix import TransSegmentationWithImagesMix
from .msd_with_fakemix import MSDWithFakeMix
from .glass_with_fakemix import GLASSWithFakeMix
from .camouflage_with_fakemix import CAMOUFLAGEWithFakeMix

datasets = {
    'trans10k_with_fakemix': TransSegmentationWithFakeMix,
    'trans10k_extra': TransExtraSegmentation,
    'trans10k_with_zerosmix': TransSegmentationWithZerosMix,
    'trans10k_with_meansmix': TransSegmentationWithMeansMix,
    'trans10k_with_imagesmix': TransSegmentationWithImagesMix,
    'msd_with_fakemix': MSDWithFakeMix,
    'glass_with_fakemix': GLASSWithFakeMix,
    'camouflage_with_fakemix': CAMOUFLAGEWithFakeMix,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
