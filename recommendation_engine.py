import pandas as pd
import pickle

def load_model(model_path):
    """Wczytuje zapisany model ML z pliku."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku modelu w ścieżce: {model_path}")
        return None
    except Exception as e:
        print(f"Błąd podczas ładowania modelu: {e}")
        return None


def generate_predictions(model, X):
    # Lista oczekiwanych cech użytych podczas treningu
    expected_features = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA',
                         'ROC_30', 'ROC_90', 'ROC_120', 'ROC_180']

    # Jeśli kolumna 'Ticker' lub 'date' istnieje, usuń je
    cols_to_drop = []
    for col in ['Ticker', 'date']:
        if col in X.columns:
            cols_to_drop.append(col)
    X_numeric = X.drop(columns=cols_to_drop, errors='ignore')

    # Wybierz tylko oczekiwane cechy
    X_final = X_numeric[expected_features]

    return model.predict(X_final)


def generate_recommendations(predictions, tickers, top_n=5):
    """Generuje rekomendacje na podstawie przewidywanych zwrotów."""
    recommendations = pd.DataFrame({
        'Ticker': tickers,
        'Predicted_Return': predictions
    })
    recommendations_sorted = recommendations.sort_values(by='Predicted_Return', ascending=False)
    return recommendations_sorted.head(top_n)

def main_recommendation_flow(model_path, csv_path, top_n=5):
    """Główna funkcja generująca rekomendacje na podstawie modelu i danych z CSV."""
    model = load_model(model_path)

    if model is None:
        return

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku CSV w ścieżce: {csv_path}")
        return
    except Exception as e:
        print(f"Błąd podczas ładowania danych z CSV: {e}")
        return

    if df.empty:
        print("Brak danych w pliku CSV.")
        return

    # Usuwamy puste wiersze (jeśli takie istnieją)
    df = df.dropna()

    # Generowanie przewidywań
    predictions = generate_predictions(model, df)

    # Generowanie rekomendacji
    recommendations = generate_recommendations(predictions, df['Ticker'].tolist(), top_n)

    print("\n===== Najlepsze rekomendacje =====")
    print(recommendations.to_string(index=False))

    return recommendations
