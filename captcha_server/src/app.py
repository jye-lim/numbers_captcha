#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import json
from threading import Timer

# Libs
import pandas as pd
from fastapi import FastAPI

# Custom
from .config import (DEMO_CSV_PATH, DEVICE, EPOCHS, EVAL_CSV_PATH,
                     IMAGES_REQUIRED, INFERENCE_SAMPLES, CHECKPOINT_PATH,
                     TRAIN_CSV_PATH, TRIGGER_CYCLE, OracleInputs)
from .data.csv import leaderboard_csv, merge_csv_files, update_data
from .data.image import batch_concat_images, get_img_tensors
from .data.sampling import permute_sampled_data
from .evaluation.predict import predict
from .model.model import Net
from .training.loaders import get_prev_model
from .training.trigger import trigger_train_loop
from .utils.utils import (batch_predictions, get_samples,
                          load_base_and_current_models, make_queue)

##################
# Configurations #
##################

app = FastAPI()
queued = make_queue()
EPOCHS = int(EPOCHS)
IMAGES_REQUIRED = int(IMAGES_REQUIRED)
INFERENCE_SAMPLES = int(INFERENCE_SAMPLES)
TRIGGER_CYCLE = int(TRIGGER_CYCLE)

#############
# Functions #
#############

@app.get("/annotation")
def get_image_permutation():
    """ Generates a random permutation of images from the demo dataset and returns 
        the concatenated image in bytes, the image paths and true labels.

    Returns:
        concatenated_image (base64): Concatenated image of size 32x320 pixels.
        paths (list): List of 10 image paths used to generate the concatenated image.
        true_labels (list): List of true labels corresponding to the permutation.
    """
    # Load samples from data
    data = pd.read_csv(DEMO_CSV_PATH)
    sampled_data = get_samples(data)

    # Get permutated info of sampled data
    concat_image, paths, true_labels = permute_sampled_data(sampled_data)

    return {"image": concat_image, "paths": paths, "true_labels": true_labels}


@app.get("/inference")
def make_inference():
    # Load data and models
    data = pd.read_csv(DEMO_CSV_PATH)
    sampled_data = data.sample(INFERENCE_SAMPLES * IMAGES_REQUIRED)
    base_model, curr_model = load_base_and_current_models()

    # Get tensors of sampled data
    sampled_tensors = get_img_tensors(sampled_data)
    
    # Get predictions
    base_pred = predict(base_model, sampled_tensors, DEVICE).tolist()
    new_pred = predict(curr_model, sampled_tensors, DEVICE).tolist()

    # Slice to batches of 10 to visualize on frontend
    images = batch_concat_images(sampled_data)
    base_infer = batch_predictions(base_pred)
    new_infer = batch_predictions(new_pred)

    return {
        "images": images, 
        "base_model_predictions": base_infer, 
        "new_model_predictions": new_infer
    }


@app.get("/train")
def start_training(
    new_cycle: bool = True, 
    export: bool = True,
    epochs: int = EPOCHS
):
    """ Starts training the model.

    Args:
        new_cycle (bool, optional): Whether to start a new training cycle. 
            Defaults to True.
        export (bool, optional): Whether to export the trained model. Defaults 
            to True.
        epochs (int, optional): Number of epochs to train the model for. Defaults 
            to EPOCHS.

    Returns:
        metrics (tuple[float, float, float]): A tuple containing the train loss,
            test loss, and test accuracy.
    """
    # Load model
    model = Net() if new_cycle else get_prev_model()

    # Start training
    queued(False)
    return trigger_train_loop(
        model, DEVICE, TRAIN_CSV_PATH, EVAL_CSV_PATH, epochs=epochs, export=export
    )


@app.get("/checkpoint_metrics")
def get_checkpoint_metrics():
    """ Gets the checkpoint metrics.

    Returns:
        checkpoint_metrics (dict): Dictionary containing the checkpoint metrics.
    """
    return json.load(open(CHECKPOINT_PATH))


@app.get("/delaytrain")
def start_delay_training():
    """ Starts training the model after a delay."""
    queued(True)
    t = Timer(TRIGGER_CYCLE, start_training)
    t.start()
    print("Scheduler in process ...")
    return True


@app.get("/scoreboard")
def scoreboard():
    """ Gets the scoreboard."""
    print('scoreboard called')
    scores_df=merge_csv_files()
    return scores_df.to_json()


@app.post("/annotation")
def update_csv(oracle_inputs: OracleInputs):
    """ Updates the demo csv with user input and writes user input to new csv for leaderboards.

    Args:
        oracle_inputs (OracleInputs): User input for permutated images.

    Returns:
        user_input (str): User input for permutated images.
    """
    # Update data with user input
    update_data(DEMO_CSV_PATH, oracle_inputs)

    # Write user input to new csv for leaderboards
    leaderboard_csv(oracle_inputs)

    # Check status of training
    train_status(queued)

    return {"Oracle Input Label": oracle_inputs.labels}


def train_status(
    queued: bool = queued,
):
    """ Check queue status and start training if queue is empty.

    Args:
        queued (bool, optional): Whether training is queued.
    """
    if not queued():
        start_delay_training()
        print("Training scheduled!")
    else:
        print("Already scheduled!")

###########
# Scripts #
###########

if __name__ == '__main__':
    pass
