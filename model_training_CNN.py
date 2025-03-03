from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from sets_creation import create_train_test_sets

X_train, X_test, y_train, y_test = create_train_test_sets()
# Tworzenie modelu CNN
cnn_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')

# Kompilacja modelu
cnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Trenowanie modelu
cnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
