import pickle
import xgboost as xgb
from sets_creation import create_train_test_sets

# Wczytanie danych i podział na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = create_train_test_sets()

# Tworzenie i trenowanie modelu XGBoost
xgboost_model = xgb.XGBRegressor()
xgboost_model.fit(X_train, y_train)

# Zapisanie modelu
with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(xgboost_model, f)

print("XGBoost model trained and saved.")
