import pandas as pd
import numpy as np
import os
import time
import warnings
from numpy import newaxis
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from fbprophet import Prophet
import seaborn as sns


bitcoin_data = pd.read_csv(r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\Bitcoin1.csv')
ethereum_data = pd.read_csv(r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\Etherum1.csv')
bitcoincash_data = pd.read_csv(r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\Bitcoincash1.csv')
xrpripple_data = pd.read_csv(r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\xrpripple1.csv')
litecoin_data = pd.read_csv(r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\litecoin1.csv')
tether_data = pd.read_csv(r'C:\Users\Arbak Aydemir\Desktop\Dersler\M.Sc. Business Analytics\MS5103 Business Analytics Project\2nd\Facebook Prophet\tether1.csv')


cryptocurrencies_dict = {
    "bitcoin" : bitcoin_data,
    "ethereum" : ethereum_data,
    "bitcoincash" : bitcoincash_data,
    "xrpripple" : xrpripple_data,
    "litecoin" : litecoin_data,
    "tether" : tether_data
}

bitcoin_data = cryptocurrencies_dict["bitcoin"][['Date', 'Close**']]
ethereum_data = cryptocurrencies_dict["ethereum"][['Date', 'Close**']]
bitcoincash_data = cryptocurrencies_dict["bitcoincash"][['Date', 'Close**']]
xrpripple_data = cryptocurrencies_dict["xrpripple"][['Date', 'Close**']]
litecoin_data = cryptocurrencies_dict["litecoin"][['Date', 'Close**']]
tether_data = cryptocurrencies_dict["tether"][['Date', 'Close**']]

bitcoin_data.head(5)

dateclose_dict = {
    "bitcoin" : bitcoin_data,
    "ethereum" : ethereum_data,
    "bitcoincash" : bitcoincash_data,
    "xrpripple" : xrpripple_data,
    "litecoin" : litecoin_data,
    "tether" : tether_data
}

dateclose_dict['bitcoin'].head(5)

###########################################

bitcoin_data
bitcoin_data['Date'] = pd.to_datetime(bitcoin_data['Date'])
bitcoin_data.loc[0:, 'Date'] = pd.to_datetime(bitcoin_data['Date'])
bitcoin_data.head(5)

for currency in dateclose_dict:
    dateclose_dict[currency].loc[0:, 'Date'] = pd.to_datetime(dateclose_dict[currency]['Date'])

ethereum_data.head(5)

bitcoin_ts = dateclose_dict['bitcoin'].set_index('Date')
ethereum_ts = dateclose_dict['ethereum'].set_index('Date')
bitcoincash_ts = dateclose_dict['bitcoincash'].set_index('Date')
xrpripple_ts = dateclose_dict['xrpripple'].set_index('Date')
litecoin_ts = dateclose_dict['litecoin'].set_index('Date')
tether_ts = dateclose_dict['tether'].set_index('Date')

dateclose_time_series = {
    "bitcoin" : bitcoin_ts,
    "ethereum" : ethereum_ts,
    "bitcoincash" : bitcoincash_ts,
    "xrpripple" : xrpripple_ts,
    "litecoin" : litecoin_ts,
    "tether" : tether_ts
}

bitcoin_ts.head(5)
tether_ts.head(5)

for ts in dateclose_time_series:
#     plt.subplot(2, 2, 2)
#     plt.subplots(1, 2, figsize=(20, 4))
    fig, ax = plt.subplots(figsize=(7,4))
    plt.plot(dateclose_time_series[ts])
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    title_str = "Closing price distribution of " + ts
    plt.title(title_str, fontsize=15)
    plt.show()

#Calculating rolling statistics to check for a trend/seasonality
for ts in dateclose_time_series:
    rolling_mean = dateclose_time_series[ts].rolling(window=20,center=False).mean()
    rolling_std = dateclose_time_series[ts].rolling(window=20,center=False).std()

    #Plot rolling statistics:
    fig, ax = plt.subplots(figsize=(8,5))
    orig = plt.plot(dateclose_time_series[ts], color='blue',label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation for ' + ts)
    plt.show(block=False)

dateclose_time_series['bitcoin']['Close**'].hist()
plt.title('Data distribution for bitcoin')
plt.show()

bitcoin_log = np.log(dateclose_time_series['bitcoin'])
bitcoin_log['Close**'].hist()
plt.title('Log Transformed Data distribution for bitcoin')
plt.show()
##########################################################################################################################

def split_data(data):
    splitIndex = int(np.floor(data.shape[0]*0.90))
    trainDataset, testDataset = data[:splitIndex], data[splitIndex:]
    return (trainDataset, testDataset)


dataforFBPROPHET = cryptocurrencies_dict['bitcoin']
dataforFBPROPHET = dataforFBPROPHET.reset_index()

dataforFBPROPHETD_DATE_CLOSE = dataforFBPROPHET[['Date', 'Close**']]
dataforFBPROPHETD_DATE_CLOSE = dataforFBPROPHETD_DATE_CLOSE.rename(columns={"Date": "ds", "Close**": "y"})

dataforFBPROPHETD_DATE_CLOSE.head(5)

dataforFBPROPHETD_DATE_CLOSE['y_orig'] = dataforFBPROPHETD_DATE_CLOSE['y']

dataforFBPROPHETD_DATE_CLOSE['y'] = np.log(dataforFBPROPHETD_DATE_CLOSE['y'])
dataforFBPROPHETD_DATE_CLOSE.head(5)

splitIndex = int(np.floor(dataforFBPROPHETD_DATE_CLOSE.shape[0]*0.90))
X_train_prophet, X_test_prophet = dataforFBPROPHETD_DATE_CLOSE[:splitIndex], dataforFBPROPHETD_DATE_CLOSE[splitIndex:]
print ("No. of samples in the training set: ", len(X_train_prophet))
print ("No. of samples in the test set", len(X_test_prophet))

model=Prophet(daily_seasonality=True)
model.fit(X_train_prophet)

futureFORECAST_data = model.make_future_dataframe(periods=90)
futureFORECAST_data.tail() ##To show days that will be predicted

forecasting_data = model.predict(futureFORECAST_data)

forecasting_data.head(5)

from sklearn.metrics import mean_absolute_error, mean_squared_error

testdataframe = X_test_prophet
del testdataframe['y_orig']
testdataframe.set_index('ds')


test1 = model.predict(testdataframe)
MSE = mean_squared_error(np.exp(testdataframe['y']), np.exp(test1['yhat']))
print ("Mean Squared Error: ", MSE)

model.plot(forecasting_data)
model.plot_components(forecasting_data)


def fbProphetmodelforallcurrencies(data):
    dataProphet = data
    dataProphet = dataProphet.reset_index()
    dataforFBPROPHETD_DATE_CLOSE = dataProphet[['Date', 'Close**']]
    dataforFBPROPHETD_DATE_CLOSE = dataforFBPROPHETD_DATE_CLOSE.rename(columns={"Date": "ds", "Close**": "y"})
    dataforFBPROPHETD_DATE_CLOSE['y_orig'] = dataforFBPROPHETD_DATE_CLOSE['y'] # to save a copy of the original data
    #log transform y
    dataforFBPROPHETD_DATE_CLOSE['y'] = np.log(dataforFBPROPHETD_DATE_CLOSE['y'])
    splitIndex = int(np.floor(dataforFBPROPHETD_DATE_CLOSE.shape[0]*0.90))
    X_train_prophet, X_test_prophet = dataforFBPROPHETD_DATE_CLOSE[:splitIndex], dataforFBPROPHETD_DATE_CLOSE[splitIndex:]
    model=Prophet(yearly_seasonality=True, daily_seasonality=True)
    # model.fit(dataProphetRed)
    model.fit(X_train_prophet)
    test = X_test_prophet
    del test['y_orig']
    test.set_index('ds')
    prediction = model.predict(test)
    MSE = mean_squared_error(np.exp(test['y']), np.exp(prediction['yhat']))
    return MSE

for currency in cryptocurrencies_dict:
    original_data = cryptocurrencies_dict[currency]
    mse = fbProphetmodelforallcurrencies(original_data)
    print ("MSE using FB Prophet for " + currency + " :", mse)