import pytest
import os
import numpy as np
import pandas as pd
import requests_mock
from unittest.mock import patch
from ..src.preprocessing.process import DataProcessor

# Mock constants for testing
DATA_DIR = "tests/data"
DATA_URL = "http://example.com/test_data.mat"
DATA_SAVE_PATH = os.path.join(DATA_DIR, "test_data.mat")


@pytest.fixture
def data_processor():
    """Fixture to create a DataProcessor object for testing."""
    return DataProcessor(DATA_DIR, DATA_URL, DATA_SAVE_PATH, True)


class TestDataProcessor:

    def test_split_indices(self, data_processor):
        total = 100
        proportions = [0.7, 0.3]
        split = data_processor._split_indices(total, proportions)

        assert len(split) == 2
        assert len(split[0]) == 70
        assert len(split[1]) == 30
        assert np.intersect1d(split[0], split[1]).size == 0


    def test_save_image_and_get_label(self, data_processor, tmpdir):
        idx = 0
        label = np.array([5])
        subset_images = np.random.randint(0, 256, size=(32, 32, 3, 1),
                                          dtype=np.uint8)
        save_dir = tmpdir.mkdir("images").strpath

        result = data_processor._save_image_and_get_label(
            idx, label, subset_images, save_dir)

        assert result[0].endswith("digit0.png")
        assert result[1] == 5


    def test_download_data(self, data_processor):
        mock_content = b"This is the mock content of the .mat file."

        with requests_mock.Mocker() as m:
            m.get(DATA_URL, content=mock_content)
            data_processor.download_data()

            with open(DATA_SAVE_PATH, 'rb') as f:
                content = f.read()

            assert content == mock_content


    def test_load_data(self, data_processor):
        mock_content = {
            "X": np.random.rand(32, 32, 3, 100),
            "y": np.random.randint(1, 11, size=100)
        }

        with patch('scipy.io.loadmat', return_value=mock_content):
            images, labels = data_processor.load_data()

            assert images.shape == (32, 32, 3, 100)
            assert labels.shape == (100, )


    def test_split_data(self, data_processor, mocker):
        mock_data = (np.random.rand(32, 32, 3, 100),
                     np.array(list(range(1, 101))))

        mocker.patch.object(data_processor, "load_data",
                            return_value=mock_data)
        mocker.patch.object(data_processor, "_save_subset")

        data_processor.split_data((0.7, 0.3), 100)

        assert data_processor._save_subset.call_count == 2


    def test_save_to_csv(self, data_processor, tmpdir):
        subset_name = "test_subset"
        data = [("path1.png", 5), ("path2.png", 6)]
        csv_save_path = os.path.join(tmpdir, subset_name, "mappings.csv")
        os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)

        data_processor.base_dir = tmpdir.strpath
        data_processor._save_to_csv(subset_name, data)

        assert os.path.exists(csv_save_path)

        df = pd.read_csv(csv_save_path)
        assert "paths" in df.columns
        assert "true_labels" in df.columns
        assert len(df) == 2


