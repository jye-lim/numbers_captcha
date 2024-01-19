#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import csv
import os
import time

# Libs
import pandas as pd

# Custom
from ..config import (DEMO_CSV_PATH, MAIN_SCOREBOARD_PATH, TEMP_SCOREBOARD_DIR, 
                      OracleInputs)

#############
# Functions #
#############

def update_data(
    csv_path: str, 
    inputs: OracleInputs
):
    """
    Write user input to csv.

    Args:
        csv_path (str): Path to csv file.
        inputs (OracleInputs): User input for permutated images.
    
    Returns:
        df (pd.DataFrame): Updated dataframe.
    """
    # Convert user input to integer
    inputs_int = [int(x) for x in inputs.labels]
    # Load csv and update data with user input
    df = pd.read_csv(csv_path)
    for path, input in zip(inputs.paths, inputs_int):
        df.loc[df["paths"] == path, "labels"] = input
    # Save updated data to csv
    df.to_csv(DEMO_CSV_PATH, index=False)


def leaderboard_csv(oracle_inputs: OracleInputs):
    """ Write user input to new csv for leaderboards.

    Args:
        .paths (list): List of image paths.
        .labels (str): User input for permutated images.
        .username (str): Username of user.
        .true_labels (list): List of true labels.
    """
    # Convert user input to integer
    annotations_int = [int(x) for x in oracle_inputs.labels]

    # Use timestamp as filename to prevent overwriting
    timestamp = str(time.time())
    filename = f"{os.path.join(TEMP_SCOREBOARD_DIR, timestamp)}.csv"

    # Write user input to csv
    with open(filename, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["ID", "Annotation", "Username", "true_label"])
        for i, path in enumerate(oracle_inputs.paths):
            csv_writer.writerow(
                [path, annotations_int[i], oracle_inputs.username, oracle_inputs.true_labels[i]]
            )


def merge_csv_files():
    """ Merge all temporary CSV files into the main CSV.

    Returns:
        pd.DataFrame: The merged CSV file as a pandas DataFrame.
    """
    # List all csv files in the temp directory
    temp_files = [
        f
        for f in os.listdir(TEMP_SCOREBOARD_DIR)
        if os.path.isfile(os.path.join(TEMP_SCOREBOARD_DIR, f)) and f.endswith(".csv")
    ]

    # Read each file and append its content to the main CSV file
    for temp_file in temp_files:
        with open(os.path.join(TEMP_SCOREBOARD_DIR, temp_file), "r") as read_obj:
            csv_reader = csv.reader(read_obj)
            next(csv_reader)  # Skip the header

            # Append content to the main CSV file
            with open(MAIN_SCOREBOARD_PATH, "a", newline="") as write_obj:
                csv_writer = csv.writer(write_obj)
                for row in csv_reader:
                    csv_writer.writerow(row)

        # Delete the temporary file after merging
        os.remove(os.path.join(TEMP_SCOREBOARD_DIR, temp_file))

    return pd.read_csv(MAIN_SCOREBOARD_PATH)

##########
# Script #
##########

if __name__ == '__main__':
    pass
