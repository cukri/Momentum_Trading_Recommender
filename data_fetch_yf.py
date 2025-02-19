import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests


def get_tickers_from_api():
    """Dynamic list of tickers from API"""
    url = 'https://financialmodelingprep.com/api/v3/search?query=AA&apikey=oIlfUS3C0X2DGhm3Lh0CA71GqWbmnMSc'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        tickers = [stock['symbol'] for stock in data[:10]]
        return tickers
    else:
        print('Error during downloading tickers.')
        return []

def get_stock_data(tickers, start = '2024-01-01', end = '2024-12-31', save_csv=True):
    all_data = []

    for ticker in tickers:
        print(f'Downloading data for {ticker}')
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            print(f'No data {ticker}, going next...')
            continue

            df['Ticker'] = ticker

        if 'Close' in df.columns:

            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.macd(df['Close'], fast= 12, slow=26, signal=9).values.T
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['SMA_200'] = ta.sma(df['Close'], length = 200)

            #Calcualting momentum
            df['ROC_30'] = ta.roc(df['Close'], length=30)
            df['ROC_90'] = ta.roc(df['Close'], length=90)
            df['ROC_120'] = ta.roc(df['Close'], length=120)
            df['ROC_180'] = ta.roc(df['Close'], length=180)

            df.dropna(inplace=True)
            all_data.append(df)

            if save_csv:
                combined_df = pd.concat(all_data)
                combined_df.to_csv('stocks_data.csv')
                print(f'All data saved to stocks_data.csv')

        return all_data

tickers = get_tickers_from_api()

if tickers:
    data = get_stock_data(tickers)
    print(data.tail())