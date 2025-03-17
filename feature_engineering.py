import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_data(df):
    """Removing missing values."""
    df = df.dropna()
    return df

def check_missing_values(df):
    """Check for any missing values in the dataframe."""
    missing_data = df.isnull().sum()
    if missing_data.any():
        print("Missing values found in the following columns:")
        print(missing_data[missing_data > 0])

        # Display rows with missing values
        for column in df.columns:
            if df[column].isnull().any():
                print(f"\nRows with missing values in column '{column}':")
                print(df[df[column].isnull()])
    else:
        print("No missing values found in the dataset.")
    return df


def normalize_features(df, feature_columns):
    """Scaling/normalization of data."""
    scaler = MinMaxScaler()
    df.loc[:, feature_columns] = scaler.fit_transform(df[feature_columns])
    return df


def prepare_features(df):
    """Preparing data for model."""
    df["return"] = df["close"].pct_change()  # Calculate percentage return based on the 'Close' price
    df = clean_data(df.copy())  # Clean missing values (including those caused by 'Return')
    df = check_missing_values(df)  # Check for any missing values in the dataset after adding 'Return'
    feature_columns = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA', 'ROC_30', 'ROC_90', 'ROC_120', 'ROC_180']
    df = normalize_features(df, feature_columns)
    return df


if __name__ == "__main__":
    df = pd.read_csv("stocks_data.csv")  # Load the raw data
    df = prepare_features(df)  # Prepare the data, including the Return column
    df.to_csv("processed_stocks_data.csv", index=False)  # Save the processed data
    print("Data has been processed and saved.")
