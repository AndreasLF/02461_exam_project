# Keras model
import tensorflow as tf
from keras import layers
from keras import models
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
import pandas as pd
import datetime as dt
from datetime import datetime
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Activation, Dropout, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
# Our module
from extract_ssi_data import *

# Import Data
pd.set_option('display.max_rows',500)
df = get_data()
train_data = df.astype(float)

# print('Training set shape == {}'.format(train_data.shape))
# print('All timestamps == {}'.format(len(train_data)))
# print('Featured selected: {}'.format(train_data.columns))
# print(df)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
pred_scaled = scaler.fit_transform(train_data.iloc[:,0:1])

# Creating a data structure with 90 timestamps and 1 output
X_train = []
y_train = []

n_future = 5   # Number of days we want top predict into the future
n_past = 356     # Number of past days we want to use to predict the future

for i in range(n_past, len(train_scaled) - n_future +1):
    X_train.append(train_scaled[i - n_past:i, 0:train_data.shape[1] - 1])
    y_train.append(train_scaled[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# print('X_train shape == {}.'.format(X_train.shape))
# print('y_train shape == {}.'.format(y_train.shape))

# Initializing the Neural Network based on LSTM
model = models.Sequential()
# Adding 1st LSTM layer
model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, train_data.shape[1]-1)))
# Adding 2nd LSTM layer
model.add(LSTM(units=10, return_sequences=False))
# Adding Dropout
model.add(Dropout(0.25))
# Output layer
model.add(Dense(units=1, activation='softmax'))
# Compiling the Neural Network
model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')

es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')
history = model.fit(X_train, y_train, shuffle=True, epochs=30, callbacks=[es, rlr, mcp, tb], validation_split=0.2, verbose=1, batch_size=256)
