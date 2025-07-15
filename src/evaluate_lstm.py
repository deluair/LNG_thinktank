# src/evaluate_lstm.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# No longer importing preprocess_data here
from train_lstm import create_dataset

def evaluate_lstm_model():
    """
    Evaluates the performance of the LSTM model.
    """
    # Load the preprocessed data
    data = pd.read_csv('data/preprocessed_data.csv', index_col='observation_date', parse_dates=True)

    # Load the trained model and scaler
    model = load_model('../models/lstm_model_multivariate.h5')
    scaler = joblib.load('../models/lstm_scaler_multivariate.pkl')

    # Select features for scaling
    features = data[['price', 'oil_price', 'coal_price']].values

    # Scale the data
    dataset = scaler.transform(features)

    # Create the test dataset
    look_back = 12
    testX, testY = create_dataset(dataset, look_back)

    # Reshape input to be [samples, time steps, features]
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], features.shape[1]))

    # Make predictions
    testPredict = model.predict(testX)

    # Invert predictions
    # Create a dummy array to inverse transform the predictions correctly
    dummy_array = np.zeros((len(testPredict), features.shape[1]))
    dummy_array[:, 0] = testPredict[:, 0]  # Put the predicted price back into the first column
    testPredict = scaler.inverse_transform(dummy_array)[:, 0]

    dummy_array_y = np.zeros((len(testY), features.shape[1]))
    dummy_array_y[:, 0] = testY  # Put the actual price back into the first column
    testY_actual = scaler.inverse_transform(dummy_array_y)[:, 0]

    # Calculate evaluation metrics
    mae = mean_absolute_error(testY_actual, testPredict)
    rmse = np.sqrt(mean_squared_error(testY_actual, testPredict))

    print("\n--- LSTM Model Evaluation (Multivariate) ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(testPredict):], testPredict, label='Predicted Prices')
    plt.plot(data.index[-len(testY_actual):], testY_actual, label='Actual Prices')
    plt.title('Multivariate LSTM LNG Price Forecast vs. Actual')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per Million BTU)')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/lstm_multivariate_evaluation_plot.png') # Changed path here
    print("\nMultivariate LSTM evaluation plot saved as data/lstm_multivariate_evaluation_plot.png")

if __name__ == "__main__":
    evaluate_lstm_model()