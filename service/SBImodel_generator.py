import math
import matplotlib
import numpy as np
import pandas as pd

import time
from stockstats import StockDataFrame as Sdf

from datetime import date
from matplotlib import pyplot as plt
from numpy.random import seed

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from keras.models import load_model


import tensorflow.python.keras.backend as K
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Dropout
import yfinance as y

#import tensorflow
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, LSTM
#from keras.utils import plot_model

#### Input params ##################
#start_date = '2017-01-01'
#end_date = '2020-04-23'
#stk_path = yf.download("INFY", start_date,end_date, prepost=True, rounding=True)
stk_path = pd.read_csv(r"E:\Project\Stock_Prediction\ui\public\SBIN.csv")

test_size = 0.2                # proportion of dataset to be used as test set
cv_size = 0.2                  # proportion of dataset to be used as cross-validation set
N = 9
lstm_units = 50                  # lstm param. initial value before tuning.
dropout_prob = 0.5               # lstm param. initial value before tuning.
optimizer = 'adam'               # lstm param. initial value before tuning.
epochs = 10                          # lstm param. initial value before tuning.
batch_size = 1
model_seed = 100
fontsize = 14
ticklabelsize = 14
# Set seeds to ensure same output results
seed(101)
tensorflow.random.set_seed(model_seed)


def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_x_y(data, N, offset):
    """

    Split data into x (features) and y (target)
    """
    x, y = [], []
    for i in range(offset, len(data)):
        x.append(data[i-N:i])
        y.append(data[i])
    x = np.array(x)
    y = np.array(y)

    return x, y


def get_x_scaled_y(data, N, offset):
    """
    Split data into x (features) and y (target)
    We scale x to have mean 0 and std dev 1, and return this.
    We do not scale y here.
    Inputs
        data     : pandas series to extract x and y
        N
        offset
    Outputs
        x_scaled : features used to predict y. Scaled such that each element has mean 0 and std dev 1
        y        : target values. Not scaled
        mu_list  : list of the means. Same length as x_scaled and y
        std_list : list of the std devs. Same length as x_scaled and y
    """
    x_scaled, y, mu_list, std_list = [], [], [], []
    for i in range(offset, len(data)):
        mu_list.append(np.mean(data[i-N:i]))
        std_list.append(np.std(data[i-N:i]))
        x_scaled.append((data[i-N:i]-mu_list[i-offset])/std_list[i-offset])
        y.append(data[i])
    x_scaled = np.array(x_scaled)
    y = np.array(y)
    return x_scaled, y, mu_list, std_list


df = stk_path
Date = df.index
df.reset_index(drop=False, inplace=True)
df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
df['month'] = df['date'].dt.month   # Get month of each sample
# Sort by datetime
df.sort_values(by='date', inplace=True, ascending=True)

num_cv = int(cv_size*len(df))
num_test = int(test_size*len(df))
num_train = len(df) - num_cv - num_test
train = df[:num_train][['date', 'adj_close']]
cv = df[num_train:num_train+num_cv][['date', 'adj_close']]
train_cv = df[:num_train+num_cv][['date', 'adj_close']]
test = df[num_train+num_cv:][['date', 'adj_close']]


# Converting dataset into x_train and y_train
# Here we only scale the train dataset, and not the entire dataset to prevent information leak
scaler = StandardScaler()
train_scaled = scaler.fit_transform(
    np.array(train['adj_close']).reshape(-1, 1))
# Split into x and y
x_train_scaled, y_train_scaled = get_x_y(train_scaled, N, N)
# Scale the cv dataset
# Split into x and y
x_cv_scaled, y_cv, mu_cv_list, std_cv_list = get_x_scaled_y(
    np.array(train_cv['adj_close']).reshape(-1, 1), N, num_train)
# Here we scale the train_cv set, for the final model
scaler_final = StandardScaler()
train_cv_scaled_final = scaler_final.fit_transform(
    np.array(train_cv['adj_close']).reshape(-1, 1))


# Optimized parameters
N_opt = 60
lstm_units_opt = 64
dropout_prob_opt = 0.5
epochs_opt = 10
batch_size_opt = 8
optimizer_opt = 'adam'
x_train_cv_scaled, y_train_cv_scaled = get_x_y(
    train_cv_scaled_final, N_opt, N_opt)

# Split test into x and y
x_test_scaled, y_test, mu_test_list, std_test_list = get_x_scaled_y(
    np.array(df['adj_close']).reshape(-1, 1), N_opt, num_train+num_cv)

model = Sequential()
model.add(LSTM(units=lstm_units_opt, return_sequences=True,
               input_shape=(x_train_cv_scaled.shape[1], 1)))
model.add(Dropout(dropout_prob_opt))  # Add dropout with a probability of 0.5
model.add(LSTM(units=lstm_units_opt))
model.add(Dropout(dropout_prob_opt))  # Add dropout with a probability of 0.5
model.add(Dense(1))

# Compile and fit the LSTM network
model.compile(loss='mean_squared_error', optimizer=optimizer_opt)
model.fit(x_train_cv_scaled, y_train_cv_scaled,
          epochs=epochs_opt, batch_size=batch_size_opt, verbose=0)
# joblib.dump(model,'lstm_1.joblib')

# model1=joblib.load('lstm_1.joblib')
model.save(f'SBI_model.h5')
model1 = load_model('SBI_model.h5')

est_scaled = model1.predict(x_test_scaled)
est = (est_scaled * np.array(std_test_list).reshape(-1, 1)) + \
    np.array(mu_test_list).reshape(-1, 1)


rmse = math.sqrt(mean_squared_error(y_test, est))
mape = get_mape(y_test, est)
print(rmse)
print(mape)

# USED FOR RECOMMENDATION

df_recommend = pd.DataFrame({'close': est.reshape(-1),
                             'date': df[num_train+num_cv:]['date']})
stock_rec = Sdf.retype(df_recommend)
signal = stock_rec['macds']  # signal line
macd = stock_rec['macd']  # macd line
# MACD histogram
macdhist = stock_rec['macdh']
rsi_sig = stock_rec['rsi_6']

# Since you need at least two days in the for loop
listLongShort = ["No data"]

for i in range(1, len(signal)):

    #                         macd crooses upward(sig)
    if macd[i] > signal[i] and macd[i - 1] <= signal[i - 1]:
        listLongShort.append("BUY")
    #                          # The other way around
    elif macd[i] < signal[i] and macd[i - 1] >= signal[i - 1]:
        listLongShort.append("SELL")
    #                          # Do nothing if not crossed
    else:
        listLongShort.append("HOLD")

stock_rec['Advice_macd'] = listLongShort

# The advice column means "Buy/Sell/Hold" at the end of this day or
#  at the beginning of the next day, since the market will be closed

listLongShort = ["No data"]

for i in range(1, len(rsi_sig)):
    #
    if rsi_sig[i] < 30:
        listLongShort.append("BUY")
    #                          # The other way around
    elif rsi_sig[i] > 70:
        listLongShort.append("SELL")
    #                          # Do nothing if not crossed
    else:
        listLongShort.append("HOLD")

stock_rec['Advice_rsi'] = listLongShort
print(stock_rec)
