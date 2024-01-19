#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in


# Libs
import torch
import torch.nn as nn

# Custom


#############
# Functions #
#############

def predict(
        model: nn.Module, 
        image: torch.Tensor, 
        device: torch.device, 
        return_probs: bool = False,
):
    """
    Predicts the output of a given image using a trained model.

    Args:
        model (nn.Module): The trained model to use for prediction.
        image (torch.Tensor): The input image to predict the output for.
        device (torch.device): The device (CPU or GPU) to perform the 
            prediction on.
        return_probs (bool, optional): Whether to return the output  
            probabilities instead of the predicted labels. Defaults to False.

    Returns:
        torch.Tensor: The predicted output of the model. If `return_probs` is 
            True, it returns the output probabilities; otherwise, it returns 
            the predicted labels.
    """
    model.eval()
    model, image = model.to(device), image.to(device)
    with torch.no_grad():
        output = model(image)
        return output if return_probs else output.argmax(dim=1)

##########
# Script #
##########

if __name__ == '__main__':
    pass
