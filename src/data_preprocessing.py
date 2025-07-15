# src/data_preprocessing.py

import pandas as pd

def preprocess_data(lng_price_file_path, oil_price_file_path, coal_price_file_path, output_file_path):
    """
    Preprocesses the LNG price, oil price, and coal price data.

    Args:
        lng_price_file_path (str): The path to the LNG price data CSV file.
        oil_price_file_path (str): The path to the oil price data CSV file.
        coal_price_file_path (str): The path to the coal price data CSV file.
        output_file_path (str): The path to save the merged and preprocessed data.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    # Load the LNG price dataset
    lng_df = pd.read_csv(lng_price_file_path)
    lng_df['observation_date'] = pd.to_datetime(lng_df['observation_date'])
    lng_df.set_index('observation_date', inplace=True)
    lng_df.rename(columns={'PNGASJPUSDM': 'price'}, inplace=True)

    # Load the oil price dataset
    oil_df = pd.read_csv(oil_price_file_path)
    oil_df['observation_date'] = pd.to_datetime(oil_df['observation_date'])
    oil_df.set_index('observation_date', inplace=True)
    oil_df.rename(columns={'DCOILBRENTEU': 'oil_price'}, inplace=True)

    # Resample oil price data to monthly frequency (mean)
    oil_df = oil_df['oil_price'].resample('MS').mean().to_frame()

    # Load the coal price dataset
    coal_df = pd.read_csv(coal_price_file_path)
    coal_df['observation_date'] = pd.to_datetime(coal_df['observation_date'])
    coal_df.set_index('observation_date', inplace=True)
    coal_df.rename(columns={'PCOALAUUSDM': 'coal_price'}, inplace=True)

    # Resample coal price data to monthly frequency (mean)
    coal_df = coal_df['coal_price'].resample('MS').mean().to_frame()

    # Merge the datasets
    merged_df = pd.merge(lng_df, oil_df, left_index=True, right_index=True, how='inner')
    merged_df = pd.merge(merged_df, coal_df, left_index=True, right_index=True, how='inner')

    # Handle missing values (if any)
    merged_df['price'] = merged_df['price'].interpolate(method='linear')
    merged_df['oil_price'] = merged_df['oil_price'].interpolate(method='linear')
    merged_df['coal_price'] = merged_df['coal_price'].interpolate(method='linear')

    # Create time-based features
    merged_df['year'] = merged_df.index.year
    merged_df['month'] = merged_df.index.month
    merged_df['day_of_week'] = merged_df.index.dayofweek

    # Save the preprocessed data
    merged_df.to_csv(output_file_path)

    return merged_df

if __name__ == "__main__":
    preprocessed_data = preprocess_data(
        'data/LNG_prices.csv',
        'data/brent_oil_prices.csv',
        'data/coal_prices.csv',
        'data/preprocessed_data.csv'
    )
    print(preprocessed_data.head())
