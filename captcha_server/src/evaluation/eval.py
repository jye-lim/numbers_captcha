#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in


# Libs
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom


#############
# Functions #
#############

def eval_model(
        model: nn.Module, 
        device: torch.device, 
        test_loader: torch.utils.data.DataLoader
) -> tuple[float, float]:
    """ Evaluate the performance of a model on a test dataset.

    Args:
        model (nn.Module): The model to evaluate.
        device (str): The device to use for evaluation.
        test_loader (DataLoader): The data loader for the test dataset.

    Returns:
        tuple: Tuple containing test loss (float) and test accuracy (float).
    """
    model.eval()
    loss, correct = 0, 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        loss /= len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)

    return loss, accuracy

##########
# Script #
##########

if __name__ == '__main__':
    pass
