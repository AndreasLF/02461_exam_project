import torch, os
from math import floor
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import seaborn as sns

from pylab import rcParams
import matplotlib.pyplot as plt 
from matplotlib import rc 
from sklearn.preprocessing import MinMaxScaler

from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

# Our module
from extract_ssi_data import *

# Run on GPU
print("GPU Driver is installed: "+str(torch.cuda.is_available()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import SSI Data
df = extract_ssi_data()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

########################
#   Preproccessing     #
########################

# Split data 80 % traning 20 % test
test_data_size = int(floor(len(df)*0.1))
train_data = df[:-test_data_size]
test_data = df[-test_data_size:]

# Data skal squishes til min-max [0:1]
# Dette gør vi for at optimere training speed (blandt andet også derfor man bruger tanh)
# bruger sckit MinMaxScaler

scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(train_data, axis=1))

train_data = scaler.transform(np.expand_dims(train_data, axis=1))

test_data = scaler.transform(np.expand_dims(test_data, axis=1))

# train_data.shape

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


seq_length = 1
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

print(train_data)

print(X_train)