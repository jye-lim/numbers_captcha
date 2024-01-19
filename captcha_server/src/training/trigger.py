#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in

# Libs
import torch
import torch.nn as nn

# Custom
from ..config import EPOCHS, EXPORT, MODEL_DIR
from ..data.dataload import get_data_loader
from ..evaluation.eval import eval_model
from .export import export_model
from .train import train_model

#############
# Functions #
#############

def trigger_train_loop(
        model: nn.Module, 
        device: torch.device, 
        train_path: str, 
        test_path: str, 
        epochs:int = EPOCHS, 
        model_dir: str = MODEL_DIR, 
        export: bool = EXPORT
) -> tuple[float, float, float]:
    """ Triggers the training loop for the given model.

    Args:
        model (nn.Module): The model to be trained.
        device (torch.device): The device to use for training.
        train_path (str): The path to the training data.
        test_path (str): The path to the test data.
        epochs (int, optional): The number of epochs to train the model for. 
            Defaults to 10.
        model_dir (str, optional): The directory to save the trained model. 
            Defaults to MODEL_DIR.
        export (bool, optional): Whether to export the trained model. Defaults 
            to EXPORT.

    Returns:
        tuple[float, float, float]: A tuple containing the train loss, test 
            loss, and test accuracy.
    """
    print("Training started!")
    
    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.001)
    train_loader = get_data_loader(train_path)
    test_loader = get_data_loader(test_path)
    
    for e in range(epochs):
        train_loss = train_model(model, device, train_loader, optimiser)
    test_loss, test_accuracy = eval_model(model, device, test_loader)

    print(f"Epoch {e+1}/{epochs} // Train loss: {train_loss:.5f} // "
          f"Test loss: {test_loss:.5f} // Test accuracy: {test_accuracy:.5f}")
    
    metrics = {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }
    
    if export: export_model(model, metrics, model_dir)
    return train_loss, test_loss, test_accuracy

##########
# Script #
##########

if __name__ == '__main__':
    pass
