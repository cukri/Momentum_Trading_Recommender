import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from sets_creation import create_train_test_sets
import pickle

X_train, X_test, y_train, y_test = create_train_test_sets()

gru_model = Sequential([
    GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    GRU(32),
    Dropout(0.2),
    Dense(1, activation='linear')
])

gru_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

gru_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

gru_model.save("gru_model.keras")

print("GRU model trained and saved.")