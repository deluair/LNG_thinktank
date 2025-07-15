# src/main.py

import argparse
import pandas as pd
import joblib
from keras.models import load_model
import numpy as np

from data_preprocessing import preprocess_data

def get_arima_forecast(months):
    """
    Generates an ARIMA forecast for the given number of months.

    Args:
        months (int): The number of months to forecast.

    Returns:
        numpy.ndarray: The forecasted prices.
    """
    # Load the trained model
    model = joblib.load('../models/arima_model.pkl')

    # Make a forecast
    forecast = model.forecast(steps=months)

    return forecast

def get_lstm_forecast(months):
    """
    Generates an LSTM forecast for the given number of months.

    Args:
        months (int): The number of months to forecast.

    Returns:
        numpy.ndarray: The forecasted prices.
    """
    # Load the trained model and scaler
    model = load_model('../models/lstm_model_multivariate.h5')
    scaler = joblib.load('../models/lstm_scaler_multivariate.pkl')

    # Load the preprocessed data to get the last 'look_back' values
    data = pd.read_csv('data/preprocessed_data.csv', index_col='observation_date', parse_dates=True)

    look_back = 12
    # Select features for scaling
    features = data[['price', 'oil_price', 'coal_price']].values

    last_data_points = features[-look_back:]
    last_data_points_scaled = scaler.transform(last_data_points)

    # Make predictions for the next n months
    predictions = []
    current_batch = last_data_points_scaled.reshape(1, look_back, features.shape[1])

    for _ in range(months):
        next_prediction_scaled = model.predict(current_batch) # This predicts only LNG price

        # Create a dummy array of the correct shape for inverse transformation
        dummy_array_for_inverse = np.zeros((1, features.shape[1]))
        dummy_array_for_inverse[0, 0] = next_prediction_scaled[0, 0] # Place predicted LNG price

        # Inverse transform only the predicted LNG price for the output
        next_prediction_unscaled_lng = scaler.inverse_transform(dummy_array_for_inverse)[0, 0]
        predictions.append(next_prediction_unscaled_lng)

        # Get the scaled oil and coal prices from the last step of the current batch
        scaled_oil_price = current_batch[0, -1, 1]
        scaled_coal_price = current_batch[0, -1, 2]

        # Construct the new scaled row for the next time step
        # Predicted LNG price (scaled), and persistence for oil and coal (scaled)
        new_scaled_row = np.array([next_prediction_scaled[0, 0], scaled_oil_price, scaled_coal_price])

        # Update current_batch with the new scaled row
        current_batch = np.append(current_batch[:, 1:, :], new_scaled_row.reshape(1, 1, features.shape[1]), axis=1)

    return np.array(predictions).flatten()

def main():
    """
    Main function to run the AI Think Tank CLI.
    """
    parser = argparse.ArgumentParser(description="AI Think Tank for the LNG Industry")
    parser.add_argument('--model', type=str, default='lstm', choices=['arima', 'lstm'], help='The model to use for forecasting.')
    parser.add_argument('--months', type=int, default=12, help='The number of months to forecast.')

    args = parser.parse_args()

    print(f"Using {args.model.upper()} model to forecast for the next {args.months} months.")

    if args.model == 'arima':
        forecast = get_arima_forecast(args.months)
    else:
        forecast = get_lstm_forecast(args.months)

    print("\n--- LNG Price Forecast ---")
    # Create a date range for the forecast
    last_date = pd.read_csv('data/preprocessed_data.csv', index_col='observation_date', parse_dates=True).index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=args.months, freq='MS')

    for date, price in zip(forecast_dates, forecast):
        print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

if __name__ == "__main__":
    main()