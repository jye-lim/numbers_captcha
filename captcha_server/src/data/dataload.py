#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in


# Libs
import torch
from torchvision.io import read_image
import pandas as pd

# Custom


###########
# Classes #
###########

class ImageSetter(torch.utils.data.Dataset):
    def __init__(self, path, labelled=True):
        """ Initializes an instance of the class.
        
        Parameters:
            path (str): The path to the CSV file.
            labelled (bool): Whether to select labelled or unlabelled data.
        
        Returns:
            None
        """
        df = pd.read_csv(path)
        subset = df["labels"].notnull() if labelled else df["labels"].isnull()
        self.data = df[subset]


    def __len__(self):
        """ Returns the length of the object.

        Returns:
            int: The length of the object.
        """
        return len(self.data)
    

    def __getitem__(self, idx):
        """ Get the item at the specified index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
                - img (torch.Tensor): The image tensor.
                - label (int): The label of the image.
        """
        img_path, label = self.data.iloc[idx, 0], self.data.iloc[idx, 1]
        img = read_image(img_path).to(torch.float32)
        return img, label


def get_data_loader(path):
    """ Generate a data loader for the given path.

    Parameters:
        path (str): The path to the data.

    Returns:
        torch.utils.data.DataLoader: The data loader object.

    """
    dataset = ImageSetter(path)
    return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

##########
# Script #
##########

if __name__ == '__main__':
    pass
