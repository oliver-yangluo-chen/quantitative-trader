import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collectdata import *


# Part 3: Build Model
def build_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main Execution
if __name__ == "__main__":
    # Collect and preprocess data
    data = collect_data(STOCK, START_DATE, END_DATE, INTERVAL)
    processed_data = preprocess_data(data)

    # Prepare data for training
    X = processed_data[:-1]  # Features (e.g., previous hour's price)
    y = processed_data[1:]  # Target (e.g., next hour's price)
    
    if DEBUG: print(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train the model
    model = build_model((X_train_scaled.shape[1],))
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))

    # Evaluate the model
    print("Model evaluation on test data:", model.evaluate(X_test_scaled, y_test))
