# First of all we have to import the libraries that we are going to use

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from sklearn.model_selection import train_test_split # This function is commonly used to split a dataset into training and testing sets.
from sklearn.metrics import classification_report, accuracy_score # These functions are commonly used to evaluate the performance of a machine learning model.
from sklearn.ensemble import RandomForestClassifier # This class is an implementation of the random forest algorithm.

yf.pdr_override()

# Then we have to download the data

ticker = ['^GSPC']
startdate = '2020-01-01'
enddate = '2023-02-18'

data = pdr.get_data_yahoo(ticker, start=startdate, end=enddate)

# Returns

returns = data['Close'].pct_change() 
log_returns = np.log(1+data['Close'].pct_change())

# Independent variables

# 1 Common variables

data['Open-Close'] = (data.Open - data.Close)/data.Open    # Daily variation between Open-Close
data['High-Low'] = (data.High - data.Low)/data.Low         # Daily variation between High-Low
data['std'] = returns.rolling(30).std()                    # 30 days standard deviation
data['mean'] = returns.rolling(30).mean()                  # 30 days mean

# 2 EMAS

ema_5 = data['Close'].ewm(span=5).mean()
ema_10 = data['Close'].ewm(span=10).mean()
ema_30 = data['Close'].ewm(span=30).mean()

# 3 Conditionals

data['ema_5>ema_10'] = np.where(ema_5 > ema_10, 1, -1)
data['ema_10>ema_30'] = np.where(ema_10 > ema_30, 1, -1)
data['Close>ema_5'] = np.where(data['Close'] > ema_5, 1, -1)
data['Close>ema_10'] = np.where(data['Close'] > ema_10, 1, -1)

# 4 MACD

exp_1 = data['Close'].ewm(span=12).mean()
exp_2 = data['Close'].ewm(span=26).mean()
macd = exp_1 - exp_2
macd_signal = macd.ewm(span=9).mean()
data['MACD'] = macd_signal - macd

# 5 RSI

up = returns.clip(lower=0)
down = returns.clip(upper=0)
ema_up = up.ewm(com=14, adjust=False).mean()
ema_down = down.ewm(com=14, adjust=False).mean()
rs = ema_up / ema_down
data['RSI'] = 100-(100/(1+rs))

# 6 Stochastic oscillator

high_14 = data['High'].rolling(14).max()
low_14 = data['Low'].rolling(14).min()
data['%K'] = (data['Close'] - low_14) * 100 / (high_14 - low_14)

# 7 ROC

ct_n = data['Close'].shift(6)
data['ROC'] = (data['Close'] - ct_n) / ct_n

# Dependent variables (The questions that we want to be answered)

data['Class'] = np.where(returns>0, 1,0)                                       # If the next day returns are going to be positives or negatives
data['Class_2'] = np.where(data['Close'].shift(-1)>data['Close'], 1, 0)        # If the next day returns are going to be higher or lower than the previous ones

# Cleaning the data

data = data.dropna()

# We indicate the model which data is independant and which one is dependant

predictors = ['Open-Close', 'High-Low', 'std', 'mean', 'ema_5>ema_10', 'ema_10>ema_30', 'Close>ema_5', 'Close>ema_10', 'MACD', 'RSI', '%K', 'ROC']
X = data[predictors]
y = data['Class_2']

# Then we have to separate the data between the data that we are going to use to train the model and the data that we are going to use to test it

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # We use the train_test_split function to split our train and test data. 70/30

# Model: Training data

rfc = RandomForestClassifier(random_state=0)
rfc = rfc.fit(X_train, y_train)

# Model: Testing data

y_pred = rfc.predict(X_test)

# Results

report = classification_report(y_test, y_pred)

# We print this result

print("\nModel Accuracy: ", accuracy_score(y_test, y_pred, normalize=True))
print("\n", report)