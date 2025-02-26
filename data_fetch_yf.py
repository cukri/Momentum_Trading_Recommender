import requests
import pandas as pd
import pandas_ta as ta
import yfinance as yf


def get_nasdaq_tickers():
    """Downloading list of tickers from NASDAQ"""
    url = "https://financialmodelingprep.com/api/v3/stock/list?apikey=oIlfUS3C0X2DGhm3Lh0CA71GqWbmnMSc"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        nasdaq_tickers = [stock["symbol"] for stock in data if stock["exchange"] == "NASDAQ"]
        return nasdaq_tickers
    else:
        print(f"Error during downloading tickers NASDAQ: {response.status_code}")
        return []


def get_sp500_tickers():
    """Downloading list of tickers from S&P 500"""
    url = "https://financialmodelingprep.com/api/v3/sp500_constituent?apikey=oIlfUS3C0X2DGhm3Lh0CA71GqWbmnMSc"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        sp500_tickers = [stock["symbol"] for stock in data]
        return sp500_tickers
    else:
        print(f"Error during downloading tickers S&P 500: {response.status_code}")
        return []


def get_all_tickers():
    """Combining tickers from NASDAQ and S&P 500"""
    nasdaq_tickers = get_nasdaq_tickers()
    sp500_tickers = get_sp500_tickers()

    all_tickers = list(set(nasdaq_tickers + sp500_tickers))
    return all_tickers


def get_stock_data(ticker, start="2024-01-01", end="2024-06-31"):
    """Downloading stock data using yfinance."""
    df = yf.download(ticker, start=start, end=end)

    if df.empty:
        print(f"No data for {ticker} – check if the ticker is correct.")
        return None

    df.reset_index(inplace=True)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.rename(
        columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"},
        inplace=True)
    df["Ticker"] = ticker
    print(f"Data downloaded for {ticker}.")
    return df


def calculate_technical_indicators(df):
    if len(df) < 14:
        print(f"Insufficient data to calculate indicators, only {len(df)} days available.")
        return df

    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    if not macd.empty:
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['MACD_hist'] = macd['MACDh_12_26_9']
    df['SMA'] = ta.sma(df['close'], length=30)
    df["ROC_30"] = ta.roc(df["close"], length=30)
    df["ROC_90"] = ta.roc(df["close"], length=90)
    df["ROC_120"] = ta.roc(df["close"], length=120)
    df["ROC_180"] = ta.roc(df["close"], length=180)
    return df


def save_data_to_csv(all_data, filename="stocks_data_half24.csv"):
    if all_data:
        combined_df = pd.concat(all_data)
        combined_df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print("No data to save.")


if __name__ == "__main__":
    tickers = get_all_tickers()

    if tickers:
        all_data = []

        for ticker in tickers:
            print(f"Getting data for {ticker}...")
            df = get_stock_data(ticker)

            if df is not None:
                df = calculate_technical_indicators(df)
                all_data.append(df)
                print(f"Metrics for {ticker} downloaded and calculated.")
            else:
                print(f"No data for {ticker}.")
            print("-")

        save_data_to_csv(all_data)
    else:
        print("No tickers available.")
