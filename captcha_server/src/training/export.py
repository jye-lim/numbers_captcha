#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import datetime
import os

# Libs
import torch
import torch.nn as nn

# Custom
from ..config import FORCE_EXPORT, MODEL_DIR
from .loaders import get_prev_model, load_loss, update_checkpoint_metrics

#############
# Functions #
#############

def export_model(
        model: nn.Module, 
        metrics: dict, 
        model_dir: str = MODEL_DIR, 
        force_export: bool = FORCE_EXPORT,
):
    """
    Exports a model to a specified directory.

    Args:
        model (object): The model object to be exported.
        model_dir (str): The directory to export the model to.
        metrics (dict): A dictionary containing metrics of the model.
        force_export (bool, optional): Whether to force the export even if 
            conditions are not met. Defaults to EXPORT.

    Returns:
        object: The exported model.

    Raises:
        None
    """
    loss_dir = os.path.join(model_dir, "losses")
    prev_model = get_prev_model(model_dir)

    # Compare with previous loss and exit if conditions not met
    if prev_model:
        prev_loss = load_loss(loss_dir, prev_model)
        if not force_export and metrics["test_loss"] < prev_loss: return None

    # Otherwise export model
    curr_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    basename = f"model_captcha_{curr_time}.pt"

    update_checkpoint_metrics(metrics, loss_dir, basename)
    torch.save(model.state_dict(), os.path.join(model_dir, basename))
    return model

##########
# Script #
##########

if __name__ == '__main__':
    pass
