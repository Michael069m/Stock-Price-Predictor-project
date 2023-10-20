import requests


def get_data(stock_symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={stock_symbol}&interval=5min&apikey=3KAFN4YVX9POWDJ7'

    r = requests.get(url)

    if r.status_code == 200:
        data = r.json()
        return data
    else:

        return None


stock_symbol = 'TCS'

data = get_data(stock_symbol)

if data:
    print(data)
else:
    print(f"Failed to retrieve data for {stock_symbol}. Check your API key or the symbol.")



import pandas as pd


df = pd.DataFrame(data['Time Series (5min)']).T


df.index = pd.to_datetime(df.index)


df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
print(df)

import matplotlib.pyplot as plt
df.columns
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = df[['Open', 'High', 'Low', 'Volume']]
print(X)

y = df['Close']
print(y)

def score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train)
    print(model.score(X_test,y_test))

le = LinearRegression()
lgr = LogisticRegression()
scv = SVC()
frst = RandomForestClassifier(n_estimators=50)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
score(le, X_train, X_test, y_train, y_test)
score(lgr, X_train, X_test, y_train, y_test)
score(scv, X_train, X_test, y_train, y_test)
score(frst, X_train, X_test, y_train, y_test)

y_pred = le.predict(X)
df['pred'] = pd.DataFrame(data={'pred': y_pred}, index=df.index)
print(df)


df['Close'] = df['Close'].astype(float)

%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(df.index, df['Close'], color="green", label="Actual")
plt.plot(df.index, df['pred'], color="red", label="Predicted")




plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Actual vs. Predicted Close Prices')
plt.legend()
plt.xticks(rotation=75)

plt.show()
