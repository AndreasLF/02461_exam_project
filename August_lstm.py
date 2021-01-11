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

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
from torch import nn, optim

# Our module
from extract_ssi_data import *

# Run on GPU
print("GPU Driver is installed: "+str(torch.cuda.is_available()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import SSI Data
df = get_data()

## Variables ##

num_epochs = 50
learning_rate = 1e-3
features = 1
hidden = 400
layers = 2
test_size = 0.2
seq_length = 1

RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

y = df["NewPositive"].values
X = df.drop(["NewPositive"],axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# TODO: Loop over 5-10 times to create 5-10 different lists to run the LSTM on (for cross-validation)
# print(df.shape)

# print(X_train)
# print(y_train)

# Vi konverterer træningsdata og testdata til fra numpy til pytorch tensors

print("Length of X_train pre tensor: "+str(len(X_train)))
print("Length of y_train pre tensor: "+str(len(y_train)))
print("Shape of X_train pre tensor: "+str(np.shape(X_train)))
print("Shape of y_train pre tensor: "+str(np.shape(y_train)))
print("X Train pre tensor: ")
# print(X_train)
print("X Test pre tensor: ")
# print(X_test)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler = scaler.fit(np.expand_dims(y, axis=1))

X_train = scaler.transform(np.expand_dims(X_train, axis=1))
y_train = scaler.transform(np.expand_dims(y_train, axis=1))

X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)

X_test = torch.from_numpy(X_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

print("Length of X_train post tensor: "+str(len(X_train)))
print("Length of y_train post tensor: "+str(len(y_train)))
print("Shape of X_train post tensor: "+str(np.shape(X_train)))
print("Shape of y_train post tensor: "+str(np.shape(y_train)))
print(np.shape(X_test))
print(np.shape(y_test))

exit()

class CoronaProphet(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(CoronaProphet, self).__init__()

        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = n_hidden,
            num_layers = n_layers,
            dropout = 0.5
        )

        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden).to(device),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden).to(device)
        )

    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1), 
            self.hidden
        )
        last_time_step = \
            lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)

        return y_pred

def train_model(
    model, 
    train_data,
    train_labels,
    test_data = None,
    test_labels = None
):

    loss_fn = torch.nn.MSELoss(reduction="mean")

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        model.reset_hidden_state()

        y_pred = model(X_train)

        loss = loss_fn(y_pred.float(), y_train)

        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = loss_fn(y_test_pred.float(), y_test)

            test_hist[t] = test_loss.item()

            if t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        
        elif t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')

        train_hist[t] = loss.item()

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return model.eval(), train_hist, test_hist

model = CoronaProphet(
  n_features=features,
  n_hidden=hidden,
  seq_len=seq_length,
  n_layers=layers
).to(device)
model, train_hist, test_hist = train_model(
  model,
  X_train,
  y_train,
  X_test,
  y_test
)


plt.plot(train_hist, label="Training loss")
plt.plot(test_hist, label="Test loss")
# data_exploration(extract_ssi_data())
# plt.ylim((0, 5))
plt.legend()
plt.show()