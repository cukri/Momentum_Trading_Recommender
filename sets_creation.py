import pandas as pd
from sklearn.model_selection import train_test_split
from data_engineering import prepare_features

def create_train_test_sets(file_path="stocks_data.csv", test_size=0.2, random_state=42, target_horizon=30):
    """Wczytuje dane, przygotowuje cechy i dzieli na zbiory treningowe i testowe."""
    df = pd.read_csv(file_path)
    df_prepared = prepare_features(df, target_horizon=target_horizon)

    feature_cols = ["RSI", "MACD", "MACD_signal", "MACD_hist", "SMA",
                    "ROC_30", "ROC_90", "ROC_120", "ROC_180"]

    X = df_prepared[feature_cols]
    y = df_prepared["return"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)