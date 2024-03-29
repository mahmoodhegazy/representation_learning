
# import libraries
import numpy as np
import random
# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class mlp(nn.Module):

  def __init__(self,
               time_periods, n_classes):
        
        super(mlp, self).__init__()
        self.time_periods = time_periods
        self.n_classes = n_classes
        # WRITE CODE HERE
        # MLP Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(time_periods * 3, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, n_classes)

  def forward(self, x):
      # WRITE CODE HERE
      x = self.flatten(x)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = F.log_softmax(self.fc4(x), dim=1)  # Using log_softmax
      return x
  
# # WRITE CODE HERE

class cnn(nn.Module):

  def __init__(self, time_periods, n_sensors, n_classes):
        super(cnn, self).__init__()
        self.time_periods = time_periods
        self.n_sensors = n_sensors
        self.n_classes = n_classes

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=n_sensors, out_channels=100, kernel_size=10)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=10)
        self.conv3 = nn.Conv1d(in_channels=100, out_channels=160, kernel_size=10)
        self.conv4 = nn.Conv1d(in_channels=160, out_channels=160, kernel_size=10)

        # Max Pooling
        self.pool = nn.MaxPool1d(kernel_size=3)

        # Adaptive Average Pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer
        self.fc = nn.Linear(160, n_classes)


        # WRITE CODE HERE
        
        

  def forward(self, x):
        # Reshape the input to (batch_size, n_sensors, -1)
        x = x.view(-1, self.n_sensors, x.size(-1))

        # Convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # Apply max pooling after the second conv layer
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Global average pooling and dropout
        x = self.adaptive_pool(x)
        x = self.dropout(x)

        # Flatten the tensor for the fully connected layer
        x = torch.flatten(x, 1)

        # Fully connected layer with log_softmax activation
        x = F.log_softmax(self.fc(x), dim=1)

        return x
        