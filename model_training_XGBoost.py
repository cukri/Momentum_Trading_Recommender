import pandas as pd
import pickle
import xgboost as xgb
from sets_creation import create_train_test_sets

def train_and_save_model(target_horizon):
    """Trains an XGBoost model for a given target_horizon and saves it to a file."""
    print(f"\nTraining model for target_horizon = {target_horizon} days...")
    X_train, X_test, y_train, y_test = create_train_test_sets(target_horizon=target_horizon)

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    model_filename = f"xgboost_model_{target_horizon}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model saved to: {model_filename}")


if __name__ == "__main__":
    for horizon in [1, 30]:
        train_and_save_model(target_horizon=horizon)
