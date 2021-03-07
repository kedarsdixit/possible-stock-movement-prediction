from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
import numpy as np
import sys
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
import json
import tensorflow.python.keras.backend as K
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Dropout
import yfinance as yf
import datetime
import holidays
from flask import Flask, request, jsonify, make_response

N_opt = 60
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

    x_scaled, y, mu_list, std_list = [], [], [], []
    for i in range(offset, len(data)):
        mu_list.append(np.mean(data[i-N:i]))
        std_list.append(np.std(data[i-N:i]))
        x_scaled.append((data[i-N:i]-mu_list[i-offset])/std_list[i-offset])
        y.append(data[i])
    x_scaled = np.array(x_scaled)
    y = np.array(y)
    return x_scaled, y, mu_list, std_list


def get_x_scaled_y11(data, N, offset):

    x_scaled, y, mu_list, std_list = [], [], [], []
    for i in range(offset, len(data)):

        mu_list.append(np.mean(data[i-N:i]))
        std_list.append(np.std(data[i-N:i]))
        x_scaled.append((data[i-N:i]-mu_list[i-offset])/std_list[i-offset])

    x_scaled = np.array(x_scaled)
    return x_scaled, mu_list, std_list


def dayspredict(data, model):
    for i in range(0, 5):
        index = data.index

        new_row = {'Open': 0, 'High': 0, 'Low': 0,
                   'Close': 0, 'Adj Close': 0, 'Volume': 0}
        data = data.append(new_row, ignore_index=True)

        scaler = StandardScaler()
        train_scaled1 = scaler.fit_transform(
            np.array(data['Close']).reshape(-1, 1))

        x_train_cv_scaled1, y_train_cv_scaled1 = get_x_y(
            train_scaled1, N_opt, N_opt)
        x_test_scaled1, mu_test_list1, std_test_list1 = get_x_scaled_y11(
            np.array(data['Close']).reshape(-1, 1), N_opt, len(index))

        model1 = load_model(model)
        est_scaled1 = model1.predict(x_test_scaled1)
        est = (est_scaled1 * np.array(std_test_list1).reshape(-1, 1)) + \
            np.array(mu_test_list1).reshape(-1, 1)
        data.drop(data.tail(1).index, inplace=True)

        new_row = {'Open': 0, 'High': 0, 'Low': 0,
                   'Close': est[0, 0], 'Adj Close': 0, 'Volume': 0}

        data = data.append(new_row, ignore_index=True)

    return data


def WiproResult():
    N_opt = 60
    data1 = yf.download("WIPRO.NS", period="3mo", prepost=True, rounding=True)
    print(data1.count())
    data1.reset_index()

    data1 = dayspredict(data1, 'Wipro_model.h5')
    df_recommend1 = pd.DataFrame({'close': data1[:]['Close']})
    print(df_recommend1)
    stock_rec = Sdf.retype(df_recommend1)
    signal = stock_rec['macds']  # signal line
    macd = stock_rec['macd']  # macd line
    # MACD histogram
    macdhist = stock_rec['macdh']
    rsi_sig = stock_rec['rsi_6']
    # Since you need at least two days in the for loop
    listLongShort = ["No data"]

    for i in range(1, len(signal)):

        # If the MACD crosses the signal line upward
        if macd[i] > signal[i] and macd[i - 1] <= signal[i - 1]:
            listLongShort.append("BUY")
    #                          # The other way around
        elif macd[i] < signal[i] and macd[i - 1] >= signal[i - 1]:
            listLongShort.append("SELL")
    #                          # Do nothing if not crossed
        else:
            listLongShort.append("HOLD")

    stock_rec['Advice_macd'] = listLongShort
    print(stock_rec)
    final_result = pd.DataFrame({'close': stock_rec[-5:]['close'],
                                 'advice': stock_rec.iloc[-5:]['Advice_macd']})
    final_result = final_result.reset_index(drop=True)

    print(final_result)

    return (final_result)


def ONGCResult():
    N_opt = 60
    data1 = yf.download("ONGC.NS", period="3mo", prepost=True, rounding=True)
    print(data1.count())

    data1.reset_index()

    data1 = dayspredict(data1, 'ONGC_model.h5')
    df_recommend1 = pd.DataFrame({'close': data1[:]['Close']})
    print(df_recommend1)

    stock_rec = Sdf.retype(df_recommend1)
    signal = stock_rec['macds']  # signal line
    macd = stock_rec['macd']  # macd line
# MACD histogram
    macdhist = stock_rec['macdh']
    rsi_sig = stock_rec['rsi_6']
    # Since you need at least two days in the for loop
    listLongShort = ["No data"]

    for i in range(1, len(signal)):

        #                          # If the MACD crosses the signal line upward
        if macd[i] > signal[i] and macd[i - 1] <= signal[i - 1]:
            listLongShort.append("BUY")
    #                          # The other way around
        elif macd[i] < signal[i] and macd[i - 1] >= signal[i - 1]:
            listLongShort.append("SELL")
    #                          # Do nothing if not crossed
        else:
            listLongShort.append("HOLD")

    stock_rec['Advice_macd'] = listLongShort
    print(stock_rec)
    final_result = pd.DataFrame({'close': stock_rec[-5:]['close'],
                                 'advice': stock_rec.iloc[-5:]['Advice_macd']})
    final_result = final_result.reset_index(drop=True)

    print(final_result)

    return (final_result)


def TataMotorsResult():
    N_opt = 60
    data1 = yf.download("TATAMOTORS.NS", period="3mo",
                        prepost=True, rounding=True)
    print(data1.count())

    data1.reset_index()

    data1 = dayspredict(data1, 'TataMotors_model.h5')
    df_recommend1 = pd.DataFrame({'close': data1[:]['Close']})
    print(df_recommend1)

    stock_rec = Sdf.retype(df_recommend1)
    signal = stock_rec['macds']  # signal line
    macd = stock_rec['macd']  # macd line
# MACD histogram
    macdhist = stock_rec['macdh']
    rsi_sig = stock_rec['rsi_6']
    # Since you need at least two days in the for loop
    listLongShort = ["No data"]

    for i in range(1, len(signal)):

        #                          # If the MACD crosses the signal line upward
        if macd[i] > signal[i] and macd[i - 1] <= signal[i - 1]:
            listLongShort.append("BUY")
    #                          # The other way around
        elif macd[i] < signal[i] and macd[i - 1] >= signal[i - 1]:
            listLongShort.append("SELL")
    #                          # Do nothing if not crossed
        else:
            listLongShort.append("HOLD")

    stock_rec['Advice_macd'] = listLongShort
    print(stock_rec)
    final_result = pd.DataFrame({'close': stock_rec[-5:]['close'],
                                 'advice': stock_rec.iloc[-5:]['Advice_macd']})
    final_result = final_result.reset_index(drop=True)

    print(final_result)

    return (final_result)


def SbiResult():
    N_opt = 60
    data1 = yf.download("SBIN.NS", period="3mo", prepost=True, rounding=True)
    print(data1.count())

    data1.reset_index()

    data1 = dayspredict(data1, 'SBI_model.h5')
    df_recommend1 = pd.DataFrame({'close': data1[:]['Close']})
    print(df_recommend1)

    stock_rec = Sdf.retype(df_recommend1)
    signal = stock_rec['macds']  # signal line
    macd = stock_rec['macd']  # macd line
# MACD histogram
    macdhist = stock_rec['macdh']
    rsi_sig = stock_rec['rsi_6']
    # Since you need at least two days in the for loop
    listLongShort = ["No data"]

    for i in range(1, len(signal)):

        #                          # If the MACD crosses the signal line upward
        if macd[i] > signal[i] and macd[i - 1] <= signal[i - 1]:
            listLongShort.append("BUY")
    #                          # The other way around
        elif macd[i] < signal[i] and macd[i - 1] >= signal[i - 1]:
            listLongShort.append("SELL")
    #                          # Do nothing if not crossed
        else:
            listLongShort.append("HOLD")

    stock_rec['Advice_macd'] = listLongShort
    print(stock_rec)
    final_result = pd.DataFrame({'close': stock_rec[-5:]['close'],
                                 'advice': stock_rec.iloc[-5:]['Advice_macd']})
    final_result = final_result.reset_index(drop=True)

    print(final_result)

    return (final_result)


def CiplaResult():
    N_opt = 60
    data1 = yf.download("CIPLA.NS", period="3mo", prepost=True, rounding=True)
    print(data1.count())

    data1.reset_index()

    data1 = dayspredict(data1, 'Cipla_model.h5')
    df_recommend1 = pd.DataFrame({'close': data1[:]['Close']})
    print(df_recommend1)

    stock_rec = Sdf.retype(df_recommend1)
    signal = stock_rec['macds']  # signal line
    macd = stock_rec['macd']  # macd line
# MACD histogram
    macdhist = stock_rec['macdh']
    rsi_sig = stock_rec['rsi_6']
    # Since you need at least two days in the for loop
    listLongShort = ["No data"]

    for i in range(1, len(signal)):

        #                          # If the MACD crosses the signal line upward
        if macd[i] > signal[i] and macd[i - 1] <= signal[i - 1]:
            listLongShort.append("BUY")
    #                          # The other way around
        elif macd[i] < signal[i] and macd[i - 1] >= signal[i - 1]:
            listLongShort.append("SELL")
    #                          # Do nothing if not crossed
        else:
            listLongShort.append("HOLD")

    stock_rec['Advice_macd'] = listLongShort
    print(stock_rec)
    final_result = pd.DataFrame({'close': stock_rec[-5:]['close'],
                                 'advice': stock_rec.iloc[-5:]['Advice_macd']})
    final_result = final_result.reset_index(drop=True)

    print(final_result)

    return (final_result)


def dateinsert(final_result):
    print(final_result)
    x = datetime.date.today()

    holiday = holidays.India()

    cnt = 0
    lst = []
    for i in range(1, 18):
        x += datetime.timedelta(days=1)
        if x.weekday() < 5 and x not in holiday:
            cnt = cnt+1
            lst.append(str(x))
            print(str(x))
        if cnt > 4:
            break
    print(lst)
    final_result.insert(0, "Date", lst, True)
    return final_result


flask_app = Flask(__name__)
app = Api(app=flask_app,
          version="1.0",
          title="Stock Prediction",
          description="Predict the value of Company stock and display Technical Indicator graphs")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params',
                  {'Company Name': fields.Float(required=True,
                                                description="Company Name for stock value prediction ",
                                                help="Company Name cannot be blank"),

                   })


@name_space.route("/")
class MainClass(Resource):

    def options(self):
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        return response

    @app.expect(model)
    def post(self):
        try:
            formData = request.json
            data = [val for val in formData.values()]

            print(data)
            if data == ['WIPRO']:
                final_result = WiproResult()
                print("1")
                print(final_result)
                final_result = dateinsert(final_result)
                print("2")
                print(final_result)
            elif data == ['TATAMOTORS']:
                final_result = TataMotorsResult()
                final_result = dateinsert(final_result)
            elif data == ['ONGC']:
                final_result = ONGCResult()
                final_result = dateinsert(final_result)
            elif data == ['CIPLA']:
                final_result = CiplaResult()
                final_result = dateinsert(final_result)
            elif data == ['SBIN']:
                final_result = SbiResult()
                final_result = dateinsert(final_result)

            response = jsonify({
                "statusCode": 200,
                "status": "Prediction made",
                "result": final_result.to_json(orient='records')

            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        except Exception as error:
            return jsonify({
                "statusCode": 500,
                "status": "Could not make prediction",
                "error": str(error)
            })
