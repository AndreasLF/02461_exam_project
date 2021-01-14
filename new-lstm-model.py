import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import MinMaxScaler from sklearn
from sklearn.preprocessing import MinMaxScaler
# Import floor function from math module
from math import floor


class LSTM(nn.Module):
    """LSTM time series prediction model

    Attributes:
        num_classes (int): Size of output sample for nn.Linear
        input_size (int): Number of features fed to the model
        hidden_size (int): Number of neurons in each layer
        num_layers (int): Number of layers in the network
        fc: Instance of the nn.Linear module
        lstm: Instance of the LSTM module

    """

    def __init__(self, num_classes=1, input_size=1, hidden_size, num_layers=1, seq_length):
        """ Initialize LSTM object

        Args:
            num_classes (int): Size of output sample for nn.Linear
            input_size (int): Number of features fed to the model. Defaults to 1
            hidden_size (int): Number of neurons in each layer
            num_layers (int): Number of layers in the network. Defaults to 1
            seq_length (int): Sequence length for the input

        """
        super(LSTM, self).__init__()
        
        # Set the class attributes
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        # Define the lstm model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        # Define instance of nn.Linear
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """  Propagate through the NN network layers

        Args: 
        x (torch):  is the input features
        """
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)
        
        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out

def create_sequences(data, seq_length):
    """ Create sequences from the data 
    
    Params: 
        data (list): The data you want to split in sequences
        seq_length (int): the length of the sequences
    
    Returns:
        A list of features and a list of targets 
    """
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        # Create a sequence of features
        x = data[i:(i+seq_length)]
        # Take the next value as a target
        y = data[i+seq_length]
        # Append to lists
        xs.append(x)
        ys.append(y)
    # Convert to ndarrays and return
    return np.array(xs), np.array(ys)

# Load flight data from seaborn library
flight_data = sns.load_dataset("flights")
# Convert monthly passengers to float
flight_data = flight_data['passengers'].values.astype(float)

# Percentage of test size
test_size = 0.2

# Split data in training and test
test_data_size = int(floor(len(flight_data)*test_size))
train_data = flight_data[:-test_data_size]
test_data = flight_data[-test_data_size:]

# Define a scaler to normalize the data
scaler = MinMaxScaler(feature_range=(-1, 1))
# Scale data. Data is fit in the range [-1,1]
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

# Create feature sequences and targets
x_train, y_train = create_sequences(train_data_normalized, 12)

# Convert to tensors
x_train = torch.Tensor(np.array(x_train))
y_train = torch.Tensor(np.array(y_train))

