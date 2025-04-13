import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler


def calculate_technical_indicators(df):
    """Calculates technical indicators for data according to historical data availability."""

    df = df.sort_values(by='date', ascending=True)

    min_data_length_rsi = 14
    min_data_length_macd = 26
    min_data_length_sma = 30
    min_data_length_roc_30 = 30
    min_data_length_roc_90 = 90
    min_data_length_roc_120 = 120
    min_data_length_roc_180 = 180

    # RSI
    if len(df) < min_data_length_rsi:
        print(f"Insufficient data to calculate RSI, only {len(df)} days available.")
        df['RSI'] = None
    else:
        df['RSI'] = ta.rsi(df['close'], length=min_data_length_rsi)

    # MACD
    if len(df) < min_data_length_macd:
        print(f"Insufficient data to calculate MACD, only {len(df)} days available.")
        df['MACD'] = None
        df['MACD_signal'] = None
        df['MACD_hist'] = None
    else:
        macd = ta.macd(df['close'])
        df['MACD'] = macd['MACD_12_26_9'] if 'MACD_12_26_9' in macd.columns else None
        df['MACD_signal'] = macd['MACDs_12_26_9'] if 'MACDs_12_26_9' in macd.columns else None
        df['MACD_hist'] = macd['MACDh_12_26_9'] if 'MACDh_12_26_9' in macd.columns else None

    # SMA
    if len(df) < min_data_length_sma:
        print(f"Insufficient data to calculate SMA, only {len(df)} days available.")
        df['SMA'] = None
    else:
        df['SMA'] = ta.sma(df['close'], length=min_data_length_sma)

    # ROC dla różnych okresów
    df["ROC_30"] = ta.roc(df["close"], length=min_data_length_roc_30) if len(df) >= min_data_length_roc_30 else None
    df["ROC_90"] = ta.roc(df["close"], length=min_data_length_roc_90) if len(df) >= min_data_length_roc_90 else None
    df["ROC_120"] = ta.roc(df["close"], length=min_data_length_roc_120) if len(df) >= min_data_length_roc_120 else None
    df["ROC_180"] = ta.roc(df["close"], length=min_data_length_roc_180) if len(df) >= min_data_length_roc_180 else None

    # Obliczenie wskaźnika Stochastic Oscillator
    df = calculate_stochastic_oscillator(df)

    return df

def calculate_stochastic_oscillator(df, k_period=14, d_period=3):
    """Oblicza wskaźnik Stochastic Oscillator (SO)"""
    df["low_min"] = df["low"].rolling(window=k_period).min()
    df["high_max"] = df["high"].rolling(window=k_period).max()

    df["Stochastic"] = 100 * ((df["close"] - df["low_min"]) / (df["high_max"] - df["low_min"]))  # %K
    df["%D"] = df["Stochastic"].rolling(window=d_period).mean()

    df.drop(columns=["low_min", "high_max"], inplace=True)
    return df


def clean_data(df):

    # Remove columns that are completely empty
    df = df.dropna(axis=1, how='all')

    # Remove only those rows that have NaN in the remaining columns
    df = df.dropna()

    return df


def check_missing_values(df):
    """Displays columns with missing data."""
    missing_data = df.isnull().sum()
    if missing_data.any():
        print("Missing values found in the following columns:")
        print(missing_data[missing_data > 0])
    else:
        print("No missing values found in the dataset.")
    return df


def normalize_features(df, feature_columns):
    """Normalization of selected features using MinMaxScaler."""
    scaler = MinMaxScaler()
    df.loc[:, feature_columns] = scaler.fit_transform(df[feature_columns])
    return df


def prepare_features(df, target_horizon=30):
    """Prepares data for the model: calculates return, technical indicators, cleansing and normalization."""
    df = calculate_technical_indicators(df)

    print(f"Data before cleaning: {len(df)} rows")

    #Calculation of refund (e.g. in 30 days)
    df["return"] = df["close"].pct_change(periods=target_horizon).shift(-target_horizon)

    df = clean_data(df)
    print(f"Data after cleaning: {len(df)} rows")

    df = check_missing_values(df)  # Checking for missing data

    feature_columns = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA',
                       'ROC_30', 'ROC_90', 'ROC_120', 'ROC_180']
    df = normalize_features(df, feature_columns)
    return df


def save_data_to_csv(all_data, filename="processed_stocks_data.csv"):
    """Combines data (a list of DataFrames or a single DataFrame), sorts it and saves it to a CSV file."""
    if isinstance(all_data, list):
        combined_df = pd.concat(all_data)
    else:
        combined_df = all_data

    combined_df.sort_values(by="date", ascending=True, inplace=True)
    combined_df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def run_data_engineering(all_data):
    """Concatenates a list of DataFrames, processes the data, and writes the result to a CSV file."""
    if not all_data:
        print("No data to process.")
        return
    combined_df = pd.concat(all_data)
    processed_df = prepare_features(combined_df)
    save_data_to_csv(processed_df)
    return processed_df

