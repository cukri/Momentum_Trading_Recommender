import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engineering import prepare_features

def create_train_test_sets(file_path="stocks_data.csv", test_size=0.2, random_state=42):
    """Wczytuje dane, przygotowuje cechy i dzieli na zbiory treningowe i testowe."""
    # Wczytanie danych
    df = pd.read_csv(file_path)

    # Przygotowanie cech
    df_prepared = prepare_features(df)

    # Wybór cech technicznych
    X = df_prepared[["RSI", "MACD", "MACD_signal", "MACD_hist", "SMA", "ROC_30", "ROC_90", "ROC_120", "ROC_180"]]  # Wszystkie wskaźniki techniczne
    y = df_prepared["return"]  # Zmienna celu - procentowy zwrot oparty na cenie 'Close'

    # Podział na zbiory treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
