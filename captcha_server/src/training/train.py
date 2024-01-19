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

def train_model(
        model: nn.Module, 
        device: torch.device, 
        train_loader: torch.utils.data.DataLoader, 
        optimiser: torch.optim.Optimizer,
) -> float:
    """ Trains a model using the given device, train loader, and optimizer.

    Parameters:
        model (nn.Module): The model to be trained.
        device (torch.device): The device to be used for training.
        train_loader (DataLoader): The data loader for the training data.
        optimiser (Optimizer): The optimizer to be used for training.

    Returns:
        float: The loss value as a floating-point number.
    """
    model.train()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimiser.step()

    return loss.item()

##########
# Script #
##########

if __name__ == '__main__':
    pass
