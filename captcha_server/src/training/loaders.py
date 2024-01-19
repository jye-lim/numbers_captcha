#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import json
import os

# Libs
import torch

# Custom
from ..config import MODEL_DIR, CHECKPOINT_PATH, get_device
from ..model.model import Net
device=get_device()

#############
# Functions #
#############

def load_model(model_dir: str = MODEL_DIR, model_name: str | None = None):
    """
    Load a model from the specified path and return it.

    Parameters:
    model_dir (str): The path to the directory containing the models.
    Defaults to "models".
    model_name (str | None): The name of the model to load. If None, no
    model will be loaded. Defaults to None.

    Returns:
    model: The loaded model.

    Raises:
    FileNotFoundError: If the specified model file is not found.

    """
    model = Net()
    if model_name:
        print(f"Loading '{model_name}' ... ")
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name),
        map_location=device))
    return model

def get_prev_model(model_dir: str = MODEL_DIR):
    """
    Get the previous model from the specified path.

    Args:
        model_dir (str, optional): The path to the directory containing the 
            model files. Defaults to "models".

    Returns:
        The loaded model.
    """
    prev_models = [
        model_name 
        for model_name in sorted(os.listdir(model_dir)) 
        if model_name.endswith(".pt")
    ]
    return prev_models[-1] if prev_models else None


def load_prev_model(model_dir: str = MODEL_DIR):
    """
    Load the previous model from the specified path.

    Args:
        model_dir (str, optional): The path to the directory containing the 
            model files. Defaults to "models".

    Returns:
        The loaded model.
    """
    return load_model(model_dir, get_prev_model(model_dir))


def update_checkpoint_metrics(
        metrics: dict, 
        loss_dir: str, 
        basename: str, 
        json_filename: str = CHECKPOINT_PATH,
):
    """ Updates the metrics of a checkpoint in a JSON file.

    Args:
        metrics (dict): The metrics to be updated.
        loss_dir (str): The directory where the JSON file is located.
        basename (str): The basename of the JSON file.
        json_filename (str, optional): The name of the JSON file. Defaults to 
                                       JSON_FILENAME.

    Returns:
        dict: The updated JSON data with the metrics.

    """
    filename = os.path.join(loss_dir, json_filename)
    with open(filename, 'r') as f:
        data = json.load(f)    
    data[basename] = metrics
    with open(filename, 'w') as f:
        json.dump(data, f)
    return data


def load_loss(
        loss_dir: str, 
        prev_model: str, 
        json_filename: str = CHECKPOINT_PATH,
):
    """
    Load the loss data from a JSON file.

    Args:
        loss_dir (str): The directory containing the loss data file.
        prev_model (str): The name of the previous model.
        json_filename (str, optional): The name of the JSON file. Defaults to 
                                       JSON_FILENAME.

    Returns:
        float: The test loss for the previous model.

    """
    filename = os.path.join(loss_dir, json_filename)
    with open(filename, 'r') as f:
        data = json.load(f)
    return data[prev_model]["test_loss"]

##########
# Script #
##########

