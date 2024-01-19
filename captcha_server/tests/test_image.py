import base64
import io
import os

import pandas as pd
import pytest
import numpy as np
import torch
from PIL import Image

from ..src.data import image

IMG_DIR = "captcha_server/tests/data/testing"
TEST_CSV = "captcha_server/tests/data/testing_100.csv"

@pytest.fixture
def img_paths():
    """Fixture to return a list of 10 image paths."""
    img_filenames = [f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")][:10]
    return [os.path.join(IMG_DIR, f) for f in img_filenames]

@pytest.fixture
def dummy_df():
    return pd.read_csv(TEST_CSV)[:50]

def test_concatenate_images(img_paths):
    """Test the concatenate_images function."""

    output_str = image.concatenate_images(img_paths)
    
    assert output_str is not None, "Output is None"
    assert isinstance(output_str, str), "Output is not a string"
    assert output_str.startswith("/9j/"), "Output is not a base64 encoded jpeg"
    assert len(output_str) > 0, "Output is empty"

    output_img = base64.b64decode(output_str, validate=True)
    output_arr = np.array(Image.open(io.BytesIO(output_img)))

    assert output_arr.shape == (32, 320, 3), "Output image is not 32x320x3"

def test_batch_concat_images(dummy_df):
    """Test the batch_concat_images function."""

    output_batches = image.batch_concat_images(dummy_df)
    assert isinstance(output_batches, list), "Output is not a list"
    assert len(dummy_df) == 10 * len(output_batches), \
        "Output is not in batches of 10"
    
def test_get_img_tensors(dummy_df):
    """Test the get_img_tensors function."""
    output_tensors = image.get_img_tensors(dummy_df)
    
    assert len(dummy_df) == len(output_tensors), "Output length is incorrect"
    assert output_tensors.dtype == torch.float32, "Output dtype is incorrect"
    assert output_tensors.shape == (len(dummy_df), 3, 32, 32), \
        "Output shape is incorrect"