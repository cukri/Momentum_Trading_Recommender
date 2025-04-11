import pandas as pd
import pickle
from data_engineering import prepare_features
import numpy as np
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
    # Lista wymaganych cech
    expected_features = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA',
                         'ROC_30', 'ROC_90', 'ROC_120', 'ROC_180']

    if 'Ticker' not in X.columns:
        raise ValueError("Brakuje kolumny 'Ticker' w danych wejściowych.")

    # Zabezpieczenie - kopiujemy Ticker
    tickers = X['Ticker'].copy().reset_index(drop=True)

    stochastic = X['Stochastic'].copy().reset_index(drop=True) if 'Stochastic' in X.columns else pd.Series(
        [np.nan] * len(X))

    # Usuwamy zbędne kolumny
    cols_to_drop = [col for col in ['Ticker', 'date'] if col in X.columns]
    X_numeric = X.drop(columns=cols_to_drop, errors='ignore')

    # Upewniamy się, że tylko oczekiwane cechy są brane do predykcji
    X_final = X_numeric[expected_features].copy()

    # Reset indeksów
    X_final = X_final.reset_index(drop=True)

    # Predykcja
    preds = model.predict(X_final)

    # Zaokrąglenie do 2 miejsc po przecinku i formatowanie w procenty
    preds = np.round(preds * 100, 2)

    # Zbudowanie DataFrame
    df_preds = pd.DataFrame({
        'Ticker': tickers,
        'Predicted_Return': preds,
        'Stochastic': stochastic
    })

    # Średnia predykcja per ticker (na wypadek duplikatów)
    grouped = df_preds.groupby('Ticker', as_index=False).mean()

    return grouped




def generate_recommendations(df, top_n=10):
    df = df.copy()

    # Określenie sygnału na podstawie oscylatora Stochastic
    df['Signal'] = df['Stochastic'].apply(
        lambda x: 'Buy' if x < 20 else ('Sell' if x > 80 else 'Hold')
    )

    # Posortuj po najwyższych prognozowanych zwrotach
    df_sorted = df.sort_values(by='Predicted_Return', ascending=False)

    # Wybierz top N
    return df_sorted[['Ticker', 'Predicted_Return', 'Stochastic', 'Signal']].head(top_n)



def main_recommendation_flow(model_path, csv_path, target_horizon, top_n=10):
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

