#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in
import os

# Libs
import numpy as np
import pandas as pd
import requests
import scipy.io
from PIL import Image
from tqdm import tqdm

# Custom
from ..config import (DATA_DIR, DATA_SAVE_PATH, DATA_URL, SAMPLE_SIZE, SEED,
                      USE_SAMPLE_DATA)

#############
# Functions #
#############


###########
# Classes #
###########

#######################
# DataProcessor Class #
#######################

class DataProcessor:
    """A class dedicated to data processing operations.
    Contains methods to download, load, and split data.
    """
    def __init__(self, base_dir: str, data_url: str, save_path: str,
                 use_sample_data: bool) -> None:
        """
        Initialize the DataProcessor object.

        Args:
            base_dir (str): The base directory for data operations.
            data_url (str): The URL where the data resides.
            save_path (str): The path to save the data.
            use_sample_data (bool): Flag to determine if sampling
                                    should be executed.
        """
        self.base_dir = base_dir
        self.data_url = data_url
        self.save_path = save_path
        self.use_sample_data = use_sample_data

    ###########
    # Helpers #
    ###########

    def _split_indices(self, total: int,
                       proportions: list[float]) -> list[np.ndarray]:
        """Splits indices from total into multiple non-overlapping
        groups based on specified proportions. This method creates a shuffled
        list of indices from 0 to `total-1` and then splits this list into
        several smaller lists based on the given proportions.

        Args:
            total (int): The total number of indices to be generated and split.
            proportions (list[float]): A list of proportions in which the
                                       indices should be split.
                                       The sum of proportions should be 1.

        Returns:
            list[np.ndarray]: A list of numpy arrays, where each array contains
                              a portion of the shuffled indices based on the
                              specified proportions.
        """
        shuffled_indices = np.arange(total)
        np.random.shuffle(shuffled_indices)

        split_limits = np.cumsum(np.array(proportions) * total).astype(int)

        split_indices = []
        start = 0
        for end in split_limits:
            split_indices.append(shuffled_indices[start:end])
            start = end

        return split_indices


    def _save_subset(self, images: np.ndarray, labels: np.ndarray,
                     subset_name: str, indices: list[int]) -> None:
        """
        Save a subset of images and labels.

        Args:
            images: The images array.
            labels: The labels array.
            subset_name: The name of the subset.
            indices: List of indices to select from.
        """
        subset_images = images[:, :, :, indices]
        subset_labels = labels[indices]
        save_dir = os.path.join(self.base_dir, subset_name, 'images')
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        filenames_labels = [
            self._save_image_and_get_label(idx, label, subset_images, save_dir)
            for idx, label in tqdm(enumerate(subset_labels),
                                   total=len(subset_labels),
                                   desc=f"Saving {subset_name} images")
        ]

        self._save_to_csv(subset_name, filenames_labels)


    def _save_image_and_get_label(self,
                                  idx: int,
                                  label: np.ndarray,
                                  subset_images: np.ndarray,
                                  save_dir: str) -> tuple[str, int]:
        """Save a single image and return its filename and label.

        Args:
            idx: The index of the image.
            label: The label of the image.
            subset_images: The subset images array.
            save_dir: Directory to save the image.

        Returns:
            Tuple of filepath and label value.
        """
        label_value = 0 if label[0] == 10 else label[0]
        filename = f"digit{idx}.png"
        filepath = os.path.join(save_dir, filename)
        Image.fromarray(subset_images[:, :, :, idx]).save(filepath)

        prefix = "sample_data" if self.use_sample_data else "data"
        relative_base = os.path.relpath(filepath, start=self.base_dir)
        relative_path = os.path.join(prefix, relative_base)

        return relative_path, label_value


    def _save_to_csv(self, subset_name: str,
                     filenames_labels: list[tuple[str, int]]) -> None:
        """Save the image filenames and labels to a CSV.

        Args:
            subset_name: The name of the subset.
            filenames_labels: List of file paths and labels.
        """
        mapping_path = os.path.join(self.base_dir, subset_name, 'mappings.csv')
        df = pd.DataFrame(filenames_labels, columns=["paths", "true_labels"])
        if subset_name in ["train", "validate"]:
            df["labels"] = df["true_labels"]
        else:
            df["labels"] = ""
        df.to_csv(mapping_path, index=False)

    ##################
    # Core Functions #
    ##################

    def download_data(self) -> None:
        """Download data if not already downloaded."""
        if os.path.exists(self.save_path):
            print(f"File '{self.save_path}' already exists.")
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        response = requests.get(self.data_url, stream=True, timeout=20)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(self.save_path, "wb") as file:
            for chunk in tqdm(response.iter_content(chunk_size=8192),
                              total=total_size // 8192, unit="KB"):
                file.write(chunk)


    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load the data from the .mat file.

        Returns:
            Tuple of images and labels.
        """
        mat_data = scipy.io.loadmat(self.save_path)
        return mat_data["X"], mat_data["y"]


    def split_data(self, proportions: tuple[float], sample_size: int) -> None:
        """Split data into different subsets based on the given proportions.

        Args:
        - proportions (tuple[float]): List of proportions for each subset.
                                      It's in the order of demo, train,
                                      validate, and unlabelled.
        - sample_size (int): Number of images to be extracted from the
                             total dataset.
        """
        images, labels = self.load_data()
        subsets = ["demo", "train", "validate", "unlabelled"]

        total_size = len(labels)
        if not self.use_sample_data:
            sample_size = total_size

        # Randomly sample desired indices
        sampled_indices = np.random.choice(total_size, sample_size,
                                           replace=False)

        # Extract the sampled images and labels
        sampled_images = images[:, :, :, sampled_indices]
        sampled_labels = labels[sampled_indices]

        # Split the sampled data based on proportions
        indices_list = self._split_indices(sample_size, proportions)
        for subset, indices in zip(subsets, indices_list):
            self._save_subset(sampled_images, sampled_labels, subset, indices)

##########
# Script #
##########

if __name__ == "__main__":
    np.random.seed(seed=SEED)
    data_processor = DataProcessor(DATA_DIR, DATA_URL, DATA_SAVE_PATH,
                                   USE_SAMPLE_DATA)
    data_processor.download_data()
    data_processor.split_data((0.2, 0.1, 0.1, 0.6), sample_size=SAMPLE_SIZE)
