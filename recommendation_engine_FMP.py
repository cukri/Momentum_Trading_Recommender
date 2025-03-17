import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime

API_KEY = "oIlfUS3C0X2DGhm3Lh0CA71GqWbmnMSc"
BASE_URL = "https://financialmodelingprep.com/api/v3/historical-price-full/"


def load_model(model_path):
    """Wczytuje zapisany model ML z pliku."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_tickers_from_csv(csv_path):
    """Wczytuje listę tickerów z pliku CSV."""
    df = pd.read_csv(csv_path)
    return df['Ticker'].unique().tolist()


def fetch_today_features(tickers):
    """Pobiera dane techniczne dla dzisiejszego dnia."""
    features = []
    today = datetime.today().strftime('%Y-%m-%d')

    for ticker in tickers:
        url = f"{BASE_URL}{ticker}?from={today}&to={today}&apikey={API_KEY}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Błąd pobierania danych dla {ticker}: {response.status_code}")
            continue

        data = response.json()
        if "historical" not in data or not data["historical"]:
            print(f"Brak danych dla {ticker}")
            continue

        stock_data = data["historical"][0]
        row = {
            'Ticker': ticker,
            'Close': stock_data['close'],
            'Volume': stock_data['volume'],
            'ROC_30': np.nan,  # Te wartości wymagają dłuższego okresu danych
            'ROC_90': np.nan,
            'ROC_120': np.nan,
            'ROC_180': np.nan
        }
        features.append(row)
        print(f"Pobrano dane dla {ticker}.")

    df = pd.DataFrame(features)

    # Zapisanie danych do pliku CSV
    filename = f"processed_stocks_{today}.csv"
    df.to_csv(filename, index=False)
    print(f"Dane zapisano do pliku: {filename}")

    return df


def generate_predictions(model, X):
    """Generuje przewidywane zwroty dla podanych danych wejściowych."""
    return model.predict(X.drop(columns=['Ticker']))


def generate_recommendations(predictions, tickers, top_n=5):
    """Generuje rekomendacje na podstawie przewidywanych zwrotów."""
    recommendations = pd.DataFrame({
        'Ticker': tickers,
        'Predicted_Return': predictions
    })

    # Sortowanie po najwyższych przewidywanych zwrotach
    recommendations = recommendations.sort_values(by='Predicted_Return', ascending=False)

    return recommendations.head(top_n)


def main(model_path, csv_path, top_n=5):
    """Główna funkcja generująca rekomendacje."""
    # Wczytujemy model
    model = load_model(model_path)

    # Wczytujemy listę tickerów
    tickers = load_tickers_from_csv(csv_path)

    # Pobieramy dane dla dzisiejszego dnia
    X = fetch_today_features(tickers)

    if X.empty:
        print("Brak danych do analizy.")
        return

    # Generujemy przewidywane zwroty
    predictions = generate_predictions(model, X)

    # Tworzymy rekomendacje
    recommendations = generate_recommendations(predictions, X['Ticker'].tolist(), top_n)

    print("\n===== Najlepsze rekomendacje =====")
    print(recommendations.to_string(index=False))

    return recommendations


if __name__ == "__main__":
    # Ścieżka do modelu
    MODEL_PATH = "xgboost_model.pkl"

    # Ścieżka do pliku CSV z tickerami
    CSV_PATH = "processed_stocks_data.csv"

    # Generowanie rekomendacji
    recommendations = main(MODEL_PATH, CSV_PATH)
