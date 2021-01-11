import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit

from extract_ssi_data import *



def create_sequences(data, seq_length):
    """ Create sequences from the data 
    
    params: 
        data (dataframe): The data you want to split in sequences
        seq_length (int): the length of the sequences
    
    return:
       
    """
    
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        # Her tager han det næste element i dataserien (Indekset er ikke inklusivt så derfor fjerne han blot i:)
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys)
 

data = extract_ssi_data()
all_data = data.values

# Create data sequences
X, y = create_sequences(all_data, 7)

# Make a time series split
tscv = TimeSeriesSplit(n_splits=10)


for train_index, test_index in tscv.split(X):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    # print("TRAIN:", X_train, "TEST:", X_test)
    y_train, y_test = y[train_index], y[test_index]
    print("TRAIN:", y_train, "TEST:", y_test)


# print(X_train)

# print(all_data.shape)