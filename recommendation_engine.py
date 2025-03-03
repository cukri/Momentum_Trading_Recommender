import pandas as pd
import numpy as np
import pickle


def load_model(model_path):
    """Wczytuje zapisany model ML z pliku."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def generate_predictions(model, X):
    """Generuje przewidywane zwroty dla podanych danych wejściowych."""
    return model.predict(X)


def generate_recommendations(predictions, tickers, top_n=5):
    """Generuje rekomendacje na podstawie przewidywanych zwrotów."""
    recommendations = pd.DataFrame({
        'ticker': tickers,
        'predicted_return': predictions
    })

    # Sortujemy po najwyższych przewidywanych zwrotach
    recommendations = recommendations.sort_values(by='predicted_return', ascending=False)

    return recommendations.head(top_n)


def main(model_path, features_path, tickers_path, top_n=5):
    """Główna funkcja generująca rekomendacje."""
    # Wczytujemy model
    model = load_model(model_path)

    # Wczytujemy dane cech
    X = pd.read_csv(features_path)

    # Wczytujemy tickery
    tickers = pd.read_csv(tickers_path)['ticker'].tolist()

    # Generujemy przewidywane zwroty
    predictions = generate_predictions(model, X)

    # Tworzymy rekomendacje
    recommendations = generate_recommendations(predictions, tickers, top_n)

    print("Najlepsze rekomendacje:")
    print(recommendations)

    return recommendations


if __name__ == "__main__":
    # Ścieżki do plików
    MODEL_PATH = "trained_model.pkl"
    FEATURES_PATH = "processed_features.csv"
    TICKERS_PATH = "tickers.csv"

    # Generowanie rekomendacji
    recommendations = main(MODEL_PATH, FEATURES_PATH, TICKERS_PATH)
