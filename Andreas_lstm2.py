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
df = get_data()

# print(df)

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

cases_train = train_data["NewPositive"]
humidity_train = train_data["Luftfugtighed"]
mid_temp_train = train_data["Middel"]

cases_test = test_data["NewPositive"]
humidity_test = test_data["Luftfugtighed"]
mid_temp_test = test_data["Middel"]

# scaler = scaler.fit(np.expand_dims(train_data, axis=1))

# Scaling training data
cases_scaler = scaler.fit(np.expand_dims(cases_train, axis=1))
cases_train = cases_scaler.transform(np.expand_dims(cases_train, axis=1))

humidity_scaler = scaler.fit(np.expand_dims(humidity_train, axis=1))
humidity_train = humidity_scaler.transform(np.expand_dims(humidity_train, axis=1))

mid_temp_scaler = scaler.fit(np.expand_dims(mid_temp_train, axis=1))
mid_temp_train = mid_temp_scaler.transform(np.expand_dims(mid_temp_train, axis=1))

# Stacking training data in colums
train_data = np.hstack([cases_train, humidity_train, mid_temp_train])

# Scaling training data
# cases_test_scaler = scaler.fit(np.expand_dims(cases_test, axis=1))
cases_test = cases_scaler.transform(np.expand_dims(cases_test, axis=1))

# humidity_test_scaler = scaler.fit(np.expand_dims(humidity_test, axis=1))
humidity_test = humidity_scaler.transform(np.expand_dims(humidity_test, axis=1))

# mid_temp_test_scaler = scaler.fit(np.expand_dims(mid_temp_test, axis=1))
mid_temp_test = mid_temp_scaler.transform(np.expand_dims(mid_temp_test, axis=1))

# Stacking test data in columns
test_data = np.hstack([cases_test, humidity_test, mid_temp_test])


# print(train_data[:,0])
# print(test_data)

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
        x = data[i:(i+seq_length)].ravel()
        # Her tager han det næste element i dataserien (Indekset er ikke inklusivt så derfor fjerne han blot i:)
        y = data[i+seq_length][0]
        xs.append(x)
        ys.append([y])
        
    return np.array(xs), np.array(ys)


# # Vi opdeler vores data i sekvenser på 5 datapoints. Dette skal nok forøges da vi har meget mere end 41 datapunkter
seq_length = 7
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# print(X_test)

# Vi konverterer træningsdata og testdata til fra numpy til pytorch tensors
X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)

X_test = torch.from_numpy(X_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

# print(y_test)
# print(y_test.shape)

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

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 250

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
  n_features=3,
  n_hidden=512,
  seq_len=seq_length,
  n_layers=2
).to(device)
model, train_hist, test_hist = train_model(
  model,
  X_train,
  y_train,
  X_test,
  y_test
)


# plt.plot(train_hist, label="Training loss")
# plt.plot(test_hist, label="Test loss")
# # data_exploration(extract_ssi_data())
# # plt.ylim((0, 5))
# plt.legend()
# plt.show()

with torch.no_grad():
  test_seq = X_test[:1]
  preds = []
  for _ in range(len(X_test)):
    y_test_pred = model(test_seq.to(device))
    pred = torch.flatten(y_test_pred).item()
    # print(pred)
    preds.append(pred)
    new_seq = test_seq.cpu().numpy().flatten()
    new_seq = np.append(new_seq, [pred])
    new_seq = new_seq[1:]
    test_seq = torch.as_tensor(new_seq).view(-1, seq_length*3).float()

true_cases = cases_scaler.inverse_transform(
np.expand_dims(y_test.flatten().cpu().numpy(), axis=0)
).flatten()
predicted_cases = cases_scaler.inverse_transform(
np.expand_dims(preds, axis=0)
).flatten()

print(predicted_cases)

# plt.plot(
#   df.index[:len(train_data)],
#   scaler.inverse_transform(train_data).flatten(),
#   label='Historical Daily Cases'
# )
# plt.plot(
#   df.index[len(train_data):len(train_data) + len(true_cases)],
#   true_cases,
#   label='Real Daily Cases'
# )
# plt.plot(
#   df.index[len(train_data):len(train_data) + len(true_cases)],
#   predicted_cases,
#   label='Predicted Daily Cases'
# )
# plt.legend()
# plt.show()