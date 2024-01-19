import os
import pytest
import torch
import json
import shutil
from captcha_server.src.model.model import Net
from captcha_server.src.training.loaders import (
    load_model, get_prev_model, load_prev_model, 
    update_checkpoint_metrics, load_loss, MODEL_DIR, CHECKPOINT_PATH
)

# Constants (dummy values for tests)
DUMMY_JSON_NAME = CHECKPOINT_PATH
DUMMY_METRICS = {
    "test_loss": 0.05,
    "test_accuracy": 95.0
}

@pytest.fixture(scope='module')
def setup_temp_model_dir(tmpdir_factory):
    """Setup a temporary model directory by copying from the actual MODEL_DIR."""
    temp_dir = tmpdir_factory.mktemp("temp_dir")  # Creates a unique temp directory
    temp_model_dir = temp_dir.join("models")  # Defines the desired path without actually creating it
    shutil.copytree(MODEL_DIR, str(temp_model_dir))
    return str(temp_model_dir)

@pytest.fixture
def setup_dummy_json(tmp_path):
    json_path = tmp_path / DUMMY_JSON_NAME
    with open(json_path, 'w') as f:
        json.dump({}, f)
    return json_path


def test_load_model(setup_temp_model_dir):
    all_models = [model_name for model_name in os.listdir(setup_temp_model_dir) if model_name.endswith(".pt")]
    for model_name in all_models:
        model = load_model(setup_temp_model_dir, model_name)
        assert isinstance(model, Net)

def test_get_prev_model(setup_temp_model_dir):
    all_models = sorted(
        [model_name for model_name in os.listdir(setup_temp_model_dir) if model_name.endswith(".pt")]
    )
    # Ensure the function always returns the latest model
    assert get_prev_model(setup_temp_model_dir) == all_models[-1]


def test_load_prev_model(setup_temp_model_dir):
    all_models = [model_name for model_name in os.listdir(setup_temp_model_dir) if model_name.endswith(".pt")]
    for model_name in all_models:
        model = load_prev_model(setup_temp_model_dir)
        assert isinstance(model, Net)

def test_update_checkpoint_metrics(setup_temp_model_dir, setup_dummy_json):
    all_models = [model_name for model_name in os.listdir(setup_temp_model_dir) if model_name.endswith(".pt")]
    for model_name in all_models:
        updated_data = update_checkpoint_metrics(
            DUMMY_METRICS, setup_dummy_json.parent, model_name, DUMMY_JSON_NAME
        )
        assert model_name in updated_data
        assert updated_data[model_name] == DUMMY_METRICS

def test_load_loss(setup_temp_model_dir, setup_dummy_json):
    all_models = [model_name for model_name in os.listdir(setup_temp_model_dir) if model_name.endswith(".pt")]
    for model_name in all_models:
        # First, create a dummy JSON with test loss
        with open(setup_dummy_json, 'w') as f:
            json.dump({model_name: DUMMY_METRICS}, f)
        
        test_loss = load_loss(setup_dummy_json.parent, model_name, DUMMY_JSON_NAME)
        assert test_loss == DUMMY_METRICS["test_loss"]
