import os
import glob
import pandas as pd
import pytest
from ..src.config import OracleInputs, DEMO_CSV_PATH, TEMP_SCOREBOARD_DIR
from ..src.data.csv import update_data, leaderboard_csv

data = {
    "paths": [
        "data/raw/img_4616.jpg", 
        "data/raw/img_4868.jpg",
        "data/raw/img_4563.jpg",
        "data/raw/img_4241.jpg",
        "data/raw/img_4772.jpg",
        "data/raw/img_4932.jpg",
        "data/raw/img_4212.jpg",
        "data/raw/img_4882.jpg",
        "data/raw/img_4768.jpg",
        "data/raw/img_4991.jpg"
    ],
    "labels": "2235754551",
    "username": "test_user",
    "true_labels": [2, 2, 3, 5, 7, 5, 4, 5, 5, 1]
}

@pytest.fixture
def mock_csv_file_path():
    """Create a mock CSV file for testing."""
    # Create a mock CSV file
    mock_csv_data = data.copy()
    mock_csv_data["labels"] = ""

    # Save the mock CSV file and return the path
    df = pd.DataFrame(mock_csv_data)
    df.to_csv(DEMO_CSV_PATH, index=False)
    return DEMO_CSV_PATH


@pytest.fixture
def mock_oracle_input():
    """Create a mock OracleInputs CSV file."""
    oracle_inputs = OracleInputs(**data)
    return oracle_inputs

def test_update_data(mock_csv_file_path, mock_oracle_input):
    """
    Test that the update_data function correctly updates the CSV file.

    Args:
        mock_csv_file_path (str): Path to the mock CSV file.
        mock_oracle_input (OracleInputs): Mock OracleInputs object.
    """
    # Call the function
    update_data(mock_csv_file_path, mock_oracle_input)
    updated_df = pd.read_csv(mock_csv_file_path)

    # Checks
    assert list(updated_df["labels"]) == mock_oracle_input.true_labels

    # Cleanup file after test
    os.remove(mock_csv_file_path)


def test_leaderboard_csv(mock_oracle_input):
    """
    Test that the leaderboard_csv function correctly creates a new CSV file.

    Args:
        mock_oracle_input (OracleInputs): Mock OracleInputs object.
    """
    # Call the function
    leaderboard_csv(mock_oracle_input)

    # Get the path to the new CSV file, which has the most recent timestamp
    list_of_files = glob.glob(os.path.join(TEMP_SCOREBOARD_DIR, "*.csv"))
    latest_file = max(list_of_files, key=os.path.getctime)
    file_path = os.path.join(TEMP_SCOREBOARD_DIR, latest_file)

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Checks
    assert list(df["ID"]) == mock_oracle_input.paths
    assert list(df["Annotation"]) == [int(x) for x in mock_oracle_input.labels]
    assert list(df["Username"]) == [mock_oracle_input.username] * len(mock_oracle_input.paths)
    assert list(df["true_label"]) == mock_oracle_input.true_labels
    
    # Cleanup file after test
    os.remove(file_path)
