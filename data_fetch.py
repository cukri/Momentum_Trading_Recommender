import requests
import pandas as pd
import pandas_ta as ta

API_KEY = "oIlfUS3C0X2DGhm3Lh0CA71GqWbmnMSc"
BASE_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/"


def get_nasdaq_tickers():
    """Downloading list of tickers from NASDAQ"""
    url = "https://financialmodelingprep.com/api/v3/stock/list?apikey=" + API_KEY
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
    url = "https://financialmodelingprep.com/api/v3/sp500_constituent?apikey=" + API_KEY
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

    all_tickers = list(set(nasdaq_tickers + sp500_tickers))  # Usuwa duplikaty
    return all_tickers


def get_fmp_data(ticker, start="2024-01-01", end="2024-12-31"):
    """Downloading data about stocks from FinancialModelingPrep."""
    url = f"{BASE_URL}{ticker}?from={start}&to={end}&apikey={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "historical" in data and data["historical"]:
            df = pd.DataFrame(data["historical"])
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df["Ticker"] = ticker
            print(f"Data downloaded for {ticker}.")
            return df
        else:
            print(f"No data for {ticker} – it is possible that FinancialModelingPrep does not support this ticker.")
            return None
    else:
        print(f"Error downloading data for {ticker}: {response.status_code}")
        return None


def calculate_technical_indicators(df):
    # Minimum data lengths for each indicator
    min_data_length_rsi = 14
    min_data_length_macd = 26
    min_data_length_sma = 30
    min_data_length_roc_30 = 30
    min_data_length_roc_90 = 90
    min_data_length_roc_120 = 120
    min_data_length_roc_180 = 180

    # Checking the data length for each indicator
    if len(df) < min_data_length_rsi:
        print(f"Insufficient data to calculate RSI, only {len(df)} days available.")
        df['RSI'] = None
    else:
        df['RSI'] = ta.rsi(df['close'], length=min_data_length_rsi)

    if len(df) < min_data_length_macd:
        print(f"Insufficient data to calculate MACD, only {len(df)} days available.")
        df['MACD'] = None
        df['MACD_signal'] = None
        df['MACD_hist'] = None
    else:
        macd = ta.macd(df['close'])
        df['MACD'] = macd['MACD_12_26_9'] if 'MACD_12_26_9' in macd.columns else None
        df['MACD_signal'] = macd['MACDs_12_26_9'] if 'MACDs_12_26_9' in macd.columns else None
        df['MACD_hist'] = macd['MACDh_12_26_9'] if 'MACDh_12_26_9' in macd.columns else None

    if len(df) < min_data_length_sma:
        print(f"Insufficient data to calculate SMA, only {len(df)} days available.")
        df['SMA'] = None
    else:
        df['SMA'] = ta.sma(df['close'], length=min_data_length_sma)

    # ROC for every period
    if len(df) < min_data_length_roc_30:
        df["ROC_30"] = None
    else:
        df["ROC_30"] = ta.roc(df["close"], length=min_data_length_roc_30)

    if len(df) < min_data_length_roc_90:
        df["ROC_90"] = None
    else:
        df["ROC_90"] = ta.roc(df["close"], length=min_data_length_roc_90)

    if len(df) < min_data_length_roc_120:
        df["ROC_120"] = None
    else:
        df["ROC_120"] = ta.roc(df["close"], length=min_data_length_roc_120)

    if len(df) < min_data_length_roc_180:
        df["ROC_180"] = None
    else:
        df["ROC_180"] = ta.roc(df["close"], length=min_data_length_roc_180)

    return df


def save_data_to_csv(all_data, filename="stocks_data.csv"):
    """Saves collected data to a CSV file."""
    if all_data:
        combined_df = pd.concat(all_data)
        combined_df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print("No data to save.")


# Download all tickers from NASDAQ and S&P 500
tickers = get_all_tickers()

if tickers:

    all_data = []

    for ticker in tickers:
        print(f"Getting data for {ticker}...")
        df = get_fmp_data(ticker)

        if df is not None:
            # Calculation of technical indicators
            df = calculate_technical_indicators(df)
            all_data.append(df)
            print(f"Metrics for {ticker} downloaded and calculated.")
        else:
            print(f"No data for {ticker}.")
        print("-")

    save_data_to_csv(all_data)
else:
    print("No tickers available.")
