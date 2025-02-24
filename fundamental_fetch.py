import requests
import pandas as pd

API_KEY = "oIlfUS3C0X2DGhm3Lh0CA71GqWbmnMSc"
BASE_URL = "https://financialmodelingprep.com/api/v3/profile/"  # Endpoint do profili firm


def get_fundamental_data(ticker):
    """Pobiera podstawowe dane fundamentalne dla pojedynczego tickera."""
    url = f"{BASE_URL}{ticker}?apikey={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data:
            stock = data[0]
            return {
                "Ticker": ticker,
                "Company Name": stock.get("companyName"),
                "Sector": stock.get("sector"),
                "Industry": stock.get("industry"),
                "Market Cap": stock.get("mktCap"),
                "P/E Ratio": stock.get("pe"),
                "Price to Book": stock.get("pb"),
                "Dividend Yield": stock.get("lastDiv"),
                "EPS": stock.get("eps"),
                "ROE": stock.get("roe"),
                "Debt to Equity": stock.get("debtToEquity"),
                "Revenue": stock.get("revenue"),
                "Net Income": stock.get("netIncome"),
            }
        else:
            print(f"Brak danych dla {ticker}.")
            return None
    else:
        print(f"Błąd pobierania danych dla {ticker}: {response.status_code}")
        return None


def get_fundamentals_from_csv(input_csv="stocks_data.csv", output_csv="fundamentals_data.csv"):
    """Pobiera listę tickerów z pliku CSV i pobiera dla nich dane fundamentalne."""
    try:
        df = pd.read_csv(input_csv)
        if "Ticker" not in df.columns:
            print(f"Plik {input_csv} nie zawiera kolumny 'Ticker'. Sprawdź strukturę pliku.")
            return

        tickers = df["Ticker"].unique()
        all_fundamentals = []

        for ticker in tickers:
            print(f"Pobieranie danych fundamentalnych dla {ticker}...")
            data = get_fundamental_data(ticker)
            if data:
                all_fundamentals.append(data)

        if all_fundamentals:
            df_fundamentals = pd.DataFrame(all_fundamentals)
            df_fundamentals.to_csv(output_csv, index=False)
            print(f"Dane fundamentalne zapisane do {output_csv}")
        else:
            print("Nie pobrano żadnych danych fundamentalnych.")

    except FileNotFoundError:
        print(f"Plik {input_csv} nie został znaleziony. Upewnij się, że istnieje.")


if __name__ == "__main__":
    get_fundamentals_from_csv()
