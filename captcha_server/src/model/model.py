#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic/Built-in

# Libs
import torch.nn as nn
import torch.nn.functional as F

# Custom


###########
# Classes #
###########

class Net(nn.Module):
    def __init__(self):
        """
        Initializes the neural network model.

        Parameters:
            self: The instance of the class.
        
        Returns:
            None
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)


    def forward(self, x):
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor to the network.

        Returns:
            torch.Tensor: The output tensor of the network.
        """
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

##########
# Script #
##########

if __name__ == '__main__':
    pass
