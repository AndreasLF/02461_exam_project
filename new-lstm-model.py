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

# Create tensors from data
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

print(train_data_normalized)