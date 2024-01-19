#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import random

# Libs
import pandas as pd

# Custom
from ..config import IMAGES_REQUIRED
from .image import concatenate_images

##################
# Configurations #
##################

IMAGES_REQUIRED = int(IMAGES_REQUIRED)

#############
# Functions #
#############

def get_samples(data: pd.DataFrame):
    """ Gets a random sample of images from the demo dataset.

    Args:
        data (pd.DataFrame): Dataframe containing image paths and labels.

    Returns:
        sampled_data (pd.DataFrame): Dataframe containing sampled image paths and labels.
    """
    # Get unlabelled data
    null_data = data[data['labels'].isnull()]

    # If unlabelled data is less than required images, get samples from labelled data
    if len(null_data) < IMAGES_REQUIRED:
        samples_required = IMAGES_REQUIRED - len(null_data)
        filled_data = data[data['labels'].notnull()]
        sampled_data = filled_data.sample(samples_required)
        sampled_data = pd.concat([null_data, sampled_data])

    # Else, get samples from unlabelled data
    else:
        sampled_data = null_data.sample(IMAGES_REQUIRED)

    return sampled_data


def permute_sampled_data(sampled_data: pd.DataFrame):
    """ Generates a random permutation of images from the demo dataset and returns
        the concatenated image in bytes, the image paths and true labels.
        
        Args:
            sampled_data (pd.DataFrame): Dataframe containing sampled image paths and labels.

        Returns:
            concat_image (base64): Concatenated image of size 32x320 pixels.
            paths (list): List of 10 image paths used to generate the concatenated image.
            true_labels (list): List of true labels corresponding to the permutation.
    """
    # Store sampled paths and true labels in list
    sampled_paths = sampled_data['paths'].to_list()
    sampled_labels = sampled_data['true_labels'].to_list()

    # Get permutation indices
    permutation_indices = random.sample(range(len(sampled_paths)), IMAGES_REQUIRED)

    # Reorder the paths, true labels and image based on permutation
    paths = [sampled_paths[i] for i in permutation_indices]
    true_labels = [sampled_labels[i] for i in permutation_indices]
    concat_image = concatenate_images(paths)

    return concat_image, paths, true_labels

##########
# Script #
##########

if __name__ == '__main__':
    pass
