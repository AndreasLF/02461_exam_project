import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import floor

# Load flight data from seaborn library
flight_data = sns.load_dataset("flights")

# Explore data
print(flight_data)
print(flight_data.shape)

# Plot flight data
plt.title('Flight passengers per month')
plt.ylabel('Passengers')
plt.xlabel('Month')
plt.grid(True)
plt.autoscale(axis='x',tight=True)
plt.plot(flight_data['passengers'])
plt.show()

# Convert monthly passengers to float
flight_data = flight_data['passengers'].values.astype(float)

# Percentage of test size
test_size = 0.2

# Split data in training and test
test_data_size = int(floor(len(flight_data)*test_size))
train_data = flight_data[:-test_data_size]
test_data = flight_data[-test_data_size:]
