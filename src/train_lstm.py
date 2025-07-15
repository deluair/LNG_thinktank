# src/train_lstm.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import joblib

# No longer importing preprocess_data here, as we expect preprocessed_data.csv

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        # Use all features for X, and only 'price' for Y
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])  # Assuming 'price' is the first column after scaling
    return np.array(dataX), np.array(dataY)

def train_lstm_model(data):
    """
    Trains an LSTM model on the given time-series data with multiple features.

    Args:
        data (pandas.DataFrame): The preprocessed time-series data with 'price', 'oil_price', and 'coal_price'.

    Returns:
        keras.models.Sequential: The trained LSTM model.
        sklearn.preprocessing.MinMaxScaler: The scaler used to transform the data.
    """
    # Select features for scaling
    features = data[['price', 'oil_price', 'coal_price']].values

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(features)

    # Create the training dataset
    look_back = 12  # Use 12 months of historical data to predict the next month
    trainX, trainY = create_dataset(dataset, look_back)

    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], features.shape[1]))

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, features.shape[1])))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=2) # Increased epochs to 200

    return model, scaler

if __name__ == "__main__":
    # Load the preprocessed data
    preprocessed_data = pd.read_csv('data/preprocessed_data.csv', index_col='observation_date', parse_dates=True)

    # Train the LSTM model
    trained_model, scaler = train_lstm_model(preprocessed_data)

    # Save the trained model and scaler
    trained_model.save('../models/lstm_model_multivariate.h5')
    joblib.dump(scaler, '../models/lstm_scaler_multivariate.pkl')

    print("Multivariate LSTM model trained and saved successfully.")