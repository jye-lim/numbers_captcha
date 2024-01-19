#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in


# Libs


# Custom
from ..config import BASE_MODEL_NAME, IMAGES_REQUIRED
from ..training.loaders import load_model, load_prev_model

##################
# Configurations #
##################

IMAGES_REQUIRED = int(IMAGES_REQUIRED)

#############
# Functions #
#############

def load_base_and_current_models():
    """ Loads the base and current model.
    
    Returns:
        base_model (torch.nn.Module): Base model.
        curr_model (torch.nn.Module): Current model.
    """
    # Load models
    base_model = load_model(model_name=BASE_MODEL_NAME)
    curr_model = load_prev_model()

    return base_model, curr_model


def make_queue():
    """ Creates a queue to store the state of the server."""
    s = [False]
    def queue(state=None):
        """ Gets and sets the state of the server.
        
        Args:
            state (bool, optional): The state to be set. Defaults to None.
        """
        if state is not None:
            s[0] = state        
        return s[0]
    return queue


def batch_predictions(pred):
    """ Slices predictions into batches of 10.

    Args:
        pred (list): List of predictions.

    Returns:
        sliced_pred (list): List of sliced predictions.
    """
    # Slice predictions into batches of 10
    sliced_pred = [
        pred[i:i+IMAGES_REQUIRED] 
        for i in range(0, len(pred), IMAGES_REQUIRED)
    ]

    return sliced_pred


def get_samples(data):
    filtered_data = data[data['labels'].isnull()]
    selected_data = filtered_data if IMAGES_REQUIRED > len(filtered_data) else filtered_data.sample(IMAGES_REQUIRED)
    return selected_data

##########
# Script #
##########

if __name__ == '__main__':
    pass
