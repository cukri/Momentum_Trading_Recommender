import pandas as pd
import pandas_ta as ta
import yfinance as yf


def get_gpw_tickers():
    """Zwraca listę tickerów z GPW."""
    return ["XTB.WA", "KRU.WA", "DCR.WA", "CDR.WA", "ARP.WA", "ACP.WA", "ASE.WA", "COG.WA"]


import pandas as pd
import yfinance as yf

def get_stock_data(tickers, start="2020-01-01", end="2025-01-01"):
    """Pobiera dane giełdowe i zapisuje je do jednego DataFrame, bez append/concat."""

    # Lista przechowująca wszystkie wiersze
    rows = []

    for ticker in tickers:
        print(f"Pobieranie danych dla {ticker}...")
        stock_data = yf.download(ticker, start=start, end=end)

        if stock_data.empty:
            print(f"Brak danych dla {ticker}")
            continue

        stock_data.reset_index(inplace=True)
        stock_data = stock_data[["Date", "Open", "High", "Low", "Close", "Volume"]]
        stock_data.columns = ["date", "open", "high", "low", "close", "volume"]
        stock_data["ticker"] = ticker

        # Obliczanie wskaźników technicznych
        stock_data = calculate_technical_indicators(stock_data)

        # Zapisujemy wiersze do listy (nie do df)
        rows.extend(stock_data.to_dict("records"))

    # Tworzymy DataFrame dopiero na końcu
    df = pd.DataFrame(rows)
    return df



def calculate_technical_indicators(df):
    """Oblicza wskaźniki techniczne dla podanego DataFrame."""
    print(f"Obliczanie wskaźników technicznych dla {df['ticker'].iloc[0]}...")

    df["RSI"] = ta.rsi(df["close"], length=14) if len(df) >= 14 else None
    macd = ta.macd(df["close"]) if len(df) >= 26 else None
    df["MACD"] = macd["MACD_12_26_9"] if macd is not None else None
    df["MACD_signal"] = macd["MACDs_12_26_9"] if macd is not None else None
    df["MACD_hist"] = macd["MACDh_12_26_9"] if macd is not None else None
    df["SMA"] = ta.sma(df["close"], length=30) if len(df) >= 30 else None
    df["ROC_30"] = ta.roc(df["close"], length=30) if len(df) >= 30 else None
    df["ROC_90"] = ta.roc(df["close"], length=90) if len(df) >= 90 else None
    df["ROC_120"] = ta.roc(df["close"], length=120) if len(df) >= 120 else None
    df["ROC_180"] = ta.roc(df["close"], length=180) if len(df) >= 180 else None

    return df


if __name__ == "__main__":
    tickers = get_gpw_tickers()
    final_df = get_stock_data(tickers)
    final_df.to_csv("stocks_data_gpw.csv", index=False)
    print("Dane zapisane do stocks_data_gpw.csv")
