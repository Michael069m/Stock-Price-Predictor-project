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

# Convert the JSON data into a Pandas DataFrame
df = pd.DataFrame(data['Time Series (5min)']).T

# Convert timestamps to datetime objects
df.index = pd.to_datetime(df.index)

# Rename columns for clarity
df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
print(df)
