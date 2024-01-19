#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os
from typing import List

# Libs
import torch
from pydantic import BaseModel, field_validator, constr
import re

##################
# Configurations #
##################

# Mode
# If True, USE_SAMPLE_DATA will be set to False below.
# This is to avoid the need to create a sample_data folder in the tests folder.
TEST_MODE = os.environ.get("TEST_MODE", "false").lower() == "true"

# Dataset
USE_SAMPLE_DATA = True  # Set to True for sample_data and False for data
SEED = 42
SAMPLE_SIZE = 1000      # Desired sample size
DATASET_SIZE = 5000     # Maximum is 531131 images

# Images
image_csv_filename = "mappings.csv"

# Sampling
IMAGES_REQUIRED = 10
INFERENCE_SAMPLES = 5

# Model
BASE_MODEL_NAME = "base_model_0500.pt"
EXPORT = True
FORCE_EXPORT = True

# Model training
EPOCHS = 5
TRIGGER_CYCLE = 150

# Checkpoint
main_cp_filename = "annotations.csv"
checkpoint_path = "checkpoint_metrics.json"

#############
# Functions #
#############

def get_device():
    """Returns the device to be used for training and inference

    Returns:
        device (torch.device): Device to be used for training and inference
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

###########
# Classes #
###########
class OracleInputs(BaseModel):
    """Inputs from the frontend are expected in this format"""
    paths: List[constr(min_length=1, max_length=255)]
    labels:   str # only allows a string of 10 digits
    username: str
    true_labels: List[int]

    @field_validator("paths", mode="before")
    def check_paths_length(cls, paths):
        assert len(paths) == 10, "Length of paths should be 10"
        return paths

    @field_validator("true_labels", mode="before")
    def check_true_labels_length(cls, true_labels):
        assert len(true_labels) == 10, "Length of true_labels should be 10"
        return true_labels

    @field_validator("username")
    def check_sql_injection(cls, username):
        # BASIC SQL INJECTION CHECK
        forbidden_keywords = ["SELECT", "UPDATE", "DELETE", "INSERT", "DROP"]
        if any(keyword in username.upper() for keyword in forbidden_keywords):
            raise ValueError("Username contains forbidden SQL keywords.")
        return username

    @field_validator('labels', mode="before")
    def validate_labels(cls, v):
        pattern = r"^[0-9]{10}$"
        if not re.match(pattern, v):
            raise ValueError('labels must be a string of 10 digits')
        return v


#############
# Variables #
#############

# Root directory
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
TEST_DIR = os.path.join(ROOT_DIR, "captcha_server", "tests")
BASE_DIR = ROOT_DIR if not TEST_MODE else TEST_DIR

# Dataset
USE_SAMPLE_DATA = False if TEST_MODE else USE_SAMPLE_DATA
DATA_SUBDIR = "sample_data" if USE_SAMPLE_DATA else "data"
DATA_DIR = os.path.join(BASE_DIR, DATA_SUBDIR)
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')

FILENAME = "test_32x32.mat" if USE_SAMPLE_DATA else "extra_32x32.mat"
DATA_URL = f"http://ufldl.stanford.edu/housenumbers/{FILENAME}"
DATA_SAVE_PATH = os.path.join(RAW_DATA_DIR, FILENAME)

DATASET_URL = "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"
DATA_FILE_PATH = "../data/raw/extra_32x32.mat"

# Image data paths
TRAIN_CSV_PATH = os.path.join(BASE_DIR, DATA_SUBDIR, "train", image_csv_filename)
EVAL_CSV_PATH = os.path.join(BASE_DIR, DATA_SUBDIR, "validate", image_csv_filename)
DEMO_CSV_PATH = os.path.join(BASE_DIR, DATA_SUBDIR, "demo", image_csv_filename)

# Model path
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Device
DEVICE = get_device()

# Checkpoint
CP_DIR = os.path.join(BASE_DIR, "models", "losses")
CHECKPOINT_PATH = os.path.join(CP_DIR, checkpoint_path)

# User scoreboard
SCOREBOARD_DIR = os.path.join(BASE_DIR, "inputs")
TEMP_SCOREBOARD_DIR = os.path.join(SCOREBOARD_DIR, "temp_annotations")
MAIN_SCOREBOARD_PATH = os.path.join(SCOREBOARD_DIR, "annotations.csv")

##########
# Script #
##########

if __name__ == '__main__':
    pass
