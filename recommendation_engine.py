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

    # Upewnij się, że kolumna 'Ticker' istnieje
    if 'Ticker' not in X.columns:
        raise ValueError("Brakuje kolumny 'Ticker' w danych wejściowych.")

    tickers = X['Ticker'].copy()

    # Jeśli kolumna 'Ticker' lub 'date' istnieje, usuń je
    cols_to_drop = []
    for col in ['Ticker', 'date']:
        if col in X.columns:
            cols_to_drop.append(col)
    X_numeric = X.drop(columns=cols_to_drop, errors='ignore')

    # Wybierz tylko oczekiwane cechy
    X_final = X_numeric[expected_features]

    preds = model.predict(X_final)

    # Stwórz tymczasowy DataFrame do grupowania
    df_preds = pd.DataFrame({
        'Ticker': tickers,
        'Predicted_Return': preds
    })

    # Grupowanie po tickerze (średnia predykcja na spółkę)
    grouped = df_preds.groupby('Ticker', as_index=False).mean()

    return grouped  # Zwraca DataFrame z kolumnami: Ticker, Predicted_Return


def generate_recommendations(predictions_df, top_n=5):
    return predictions_df.sort_values(by="Predicted_Return", ascending=False).head(top_n)

def main_recommendation_flow(model_path, csv_path, target_horizon=30, top_n=5):
    """Główna funkcja generująca rekomendacje na podstawie modelu i danych z CSV."""
    import pandas as pd
    from data_engineering import prepare_features
    from recommendation_engine import load_model, generate_predictions, generate_recommendations

    print(f"\n🔄 Generowanie rekomendacji dla horyzontu {target_horizon} dni...")

    model = load_model(model_path)
    if model is None:
        return

    # Wczytaj dane
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

    # Przygotuj cechy techniczne z uwzględnieniem target_horizon
    df = prepare_features(df, target_horizon=target_horizon)

    # Usuń brakujące wartości
    df = df.dropna()

    # Generowanie przewidywań
    predictions = generate_predictions(model, df)

    # Generowanie rekomendacji
    recommendations = generate_recommendations(predictions, top_n)

    print("\n✅ ===== Najlepsze rekomendacje =====")
    print(recommendations.to_string(index=False))

    return recommendations

