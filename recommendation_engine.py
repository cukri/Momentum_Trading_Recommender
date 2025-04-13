import pandas as pd
import pickle
from data_engineering import prepare_features
import numpy as np
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"Błąd: No model file found in path: {model_path}")
        return None
    except Exception as e:
        print(f"Error while loading model: {e}")
        return None


def generate_predictions(model, X):
    # List of features for model
    expected_features = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA',
                         'ROC_30', 'ROC_90', 'ROC_120', 'ROC_180']

    if 'Ticker' not in X.columns:
        raise ValueError("Brakuje kolumny 'Ticker' w danych wejściowych.")

    tickers = X['Ticker'].copy().reset_index(drop=True)

    stochastic = X['Stochastic'].copy().reset_index(drop=True) if 'Stochastic' in X.columns else pd.Series(
        [np.nan] * len(X))

    # Removing not necessary columns
    cols_to_drop = [col for col in ['Ticker', 'date'] if col in X.columns]
    X_numeric = X.drop(columns=cols_to_drop, errors='ignore')

    X_final = X_numeric[expected_features].copy()

    X_final = X_final.reset_index(drop=True)

    preds = model.predict(X_final)

    preds = np.round(preds * 100, 2)

    # DF for recommendations
    df_preds = pd.DataFrame({
        'Ticker': tickers,
        'Predicted_Return': preds,
        'Stochastic': stochastic
    })

    grouped = df_preds.groupby('Ticker', as_index=False).mean()

    return grouped




def generate_recommendations(df, top_n=10):
    df = df.copy()

    # Setting up signal based on stochastic
    df['Signal'] = df['Stochastic'].apply(
        lambda x: 'Buy' if x < 20 else ('Sell' if x > 80 else 'Hold')
    )

    # Sorting by predicted return
    df_sorted = df.sort_values(by='Predicted_Return', ascending=False)

    return df_sorted[['Ticker', 'Predicted_Return', 'Stochastic', 'Signal']].head(top_n)



def main_recommendation_flow(model_path, csv_path, target_horizon, top_n=10):
    model = load_model(model_path)
    if model is None:
        return

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found in path: {csv_path}")
        return
    except Exception as e:
        print(f"Error while loading data from CSV: {e}")
        return

    if df.empty:
        print("No data in CSV file.")
        return

    # Preparing technical features with target_horizon
    df = prepare_features(df, target_horizon=target_horizon)

    df = df.dropna()

    predictions = generate_predictions(model, df)

    recommendations = generate_recommendations(predictions, top_n)

    print("\n===== Best recommendations =====")
    print(recommendations.to_string(index=False))

    return recommendations

