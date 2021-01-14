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
