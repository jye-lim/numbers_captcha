#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import base64
import io

# Libs
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

# Custom
from ..config import IMAGES_REQUIRED

#############
# Functions #
#############

def concatenate_images(image_paths: list):
    """ Concatenates 10 images into a single image of size 32x320 pixels.
    
    Args:
        image_paths (list): List of 10 image paths.

    Returns:
        encoded_image (base64): Concatenated image of size 32x320 pixels.
    """
    # Create a blank canvas for the concatenated image
    concatenated_image = Image.new('RGB', (320, 32))
    x_offset = 0

    # Open each image and paste it onto the canvas
    for path in image_paths:
        img = Image.open(path)
        concatenated_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Convert the concatenated image to base64
    buffer = io.BytesIO()
    concatenated_image.save(buffer, format="JPEG")    
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return encoded_image


def batch_concat_images(data: pd.DataFrame):
    """ Concatenates images and slices them into batches of 10.

    Args:
        data (pd.DataFrame): Dataframe containing image paths and labels.

    Returns:
        images (list): List of concatenated images in base64.
    """
    # Get paths from dataframe
    paths = data["paths"].to_list()

    # Concatenate images in batches of 10 and append to list
    images = [
        concatenate_images(data["paths"][i:i+IMAGES_REQUIRED]) 
        for i in range(0, len(paths), IMAGES_REQUIRED)
    ]

    return images


def get_img_tensors(data: pd.DataFrame):
    """ Gets tensors of images from input dataframe.
    
    Args:
        data (pd.DataFrame): Dataframe containing image paths and labels.

    Returns:
        tensors (torch.Tensor): Tensor of images.
    """
    # Get images from paths
    images = [Image.open(path) for path in data["paths"]]

    # Convert images to tensors
    tensors = torch.stack(
        [pil_to_tensor(img).to(torch.float32) for img in images]
    )

    return tensors

##########
# Script #
##########

if __name__ == '__main__':
    pass
