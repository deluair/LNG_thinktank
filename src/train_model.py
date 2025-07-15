# src/train_model.py

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
from data_preprocessing import preprocess_data

def train_arima_model(data):
    """
    Trains an ARIMA model on the given time-series data.

    Args:
        data (pandas.DataFrame): The preprocessed time-series data.

    Returns:
        statsmodels.tsa.arima.model.ARIMAResultsWrapper: The trained ARIMA model.
    """
    # Fit the ARIMA model
    model = ARIMA(data['price'], order=(5, 1, 0))
    model_fit = model.fit()

    return model_fit

if __name__ == "__main__":
    # Preprocess the data
    preprocessed_data = pd.read_csv('data/preprocessed_data.csv', index_col='observation_date', parse_dates=True)

    # Train the ARIMA model
    trained_model = train_arima_model(preprocessed_data)

    # Save the trained model
    joblib.dump(trained_model, 'models/arima_model.pkl')

    print("ARIMA model trained and saved successfully.")
