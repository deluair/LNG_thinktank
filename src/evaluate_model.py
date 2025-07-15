# src/evaluate_model.py

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

from data_preprocessing import preprocess_data
from train_model import train_arima_model

def evaluate_model():
    """
    Evaluates the performance of the ARIMA model.
    """
    # Load and preprocess the data
    data = pd.read_csv('data/preprocessed_data.csv', index_col='observation_date', parse_dates=True)

    # Split data into training and testing sets (e.g., 80% train, 20% test)
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[0:train_size], data[train_size:]

    # Train the model on the training data
    print("Training model on the training set...")
    trained_model = train_arima_model(train_data)

    # Make predictions on the test set
    print("Making predictions on the test set...")
    predictions = trained_model.forecast(steps=len(test_data))

    # Create a new dataframe for comparison
    test_predictions = test_data.copy()
    test_predictions['predicted_price'] = predictions.values

    # Calculate evaluation metrics
    mae = mean_absolute_error(test_predictions['price'], test_predictions['predicted_price'])
    rmse = np.sqrt(mean_squared_error(test_predictions['price'], test_predictions['predicted_price']))

    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # You can also visualize the results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data['price'], label='Training Data')
    plt.plot(test_data.index, test_data['price'], label='Actual Prices')
    plt.plot(test_predictions.index, test_predictions['predicted_price'], label='Predicted Prices', linestyle='--')
    plt.title('LNG Price Forecast vs. Actual')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per Million BTU)')
    plt.legend()
    plt.grid(True)
    plt.savefig('evaluation_plot.png')
    print("\nEvaluation plot saved as evaluation_plot.png")

if __name__ == "__main__":
    evaluate_model()
