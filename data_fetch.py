import requests
import pandas as pd
import datetime
import json
import os

def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Brak pliku konfiguracyjnego: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

config = load_config()  # Wczytanie ustawień z config.json
API_KEY = config.get("API_KEY")
BASE_URL = config.get("BASE_URL")

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

def get_fmp_data(ticker, start=None, end=None):
    """Downloading stock data from FinancialModelingPrep."""
    if start is None or end is None:
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        start = end = today  # Pobieranie danych tylko z dzisiaj

    url = f"{BASE_URL}{ticker}?from={start}&to={end}&apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "historical" in data and data["historical"]:
            df = pd.DataFrame(data["historical"])
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df["Ticker"] = ticker
            print(f"Data downloaded for {ticker} from {start} to {end}.")
            return df
        else:
            print(f"No data for {ticker}. It may not be supported by FinancialModelingPrep.")
            return None
    else:
        print(f"Error downloading data for {ticker}: {response.status_code}")
        return None

def load_tickers(ticker_file="tickers.txt"):
    """Wczytuje listę tickerów z pliku tekstowego."""
    try:
        with open(ticker_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    except Exception as e:
        print(f"Błąd wczytywania tickerów: {e}")
        return []
