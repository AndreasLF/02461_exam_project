import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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