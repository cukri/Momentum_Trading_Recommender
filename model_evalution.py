import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from sets_creation import create_train_test_sets


def evaluate_model(X_train, y_train, X_test, y_test, model, model_name):
    """A function that evaluates the model and generates graphs and metrics."""

    # Prediction on training and test sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Displaying metrics results
    print(f"{model_name} Evaluation:")
    print(f"Mean Squared Error (Train): {mean_squared_error(y_train, y_pred_train)}")
    print(f"R-squared Score (Train): {r2_score(y_train, y_pred_train)}")
    print(f"Mean Squared Error (Test): {mean_squared_error(y_test, y_pred_test)}")
    print(f"R-squared Score (Test): {r2_score(y_test, y_pred_test)}")

    # Creating graphs comparing actual values ​​with predictions
    plt.figure(figsize=(12, 6))

    # Graph for training data
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, color='blue', alpha=0.5)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', lw=2)
    plt.title(f"{model_name} - Train Data")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")

    # Chart for test data
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, color='green', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
    plt.title(f"{model_name} - Test Data")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")

    plt.tight_layout()
    plt.show()

    #
    if isinstance(model, xgb.XGBRegressor):
        print(f"\n{model_name} - Feature Importance:")
        xgb.plot_importance(model, importance_type='weight', max_num_features=10)
        plt.title(f"{model_name} - Feature Importance")
        plt.show()


with open('linear_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('xgboost_model.pkl', 'rb') as f:
    xgboost_model = pickle.load(f)


X_train, X_test, y_train, y_test = create_train_test_sets()

# Evaluation of models
evaluate_model(X_train, y_train, X_test, y_test, lr_model, "Linear Regression")
evaluate_model(X_train, y_train, X_test, y_test, xgboost_model, "XGBoost")
