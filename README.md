# AI Think Tank for the LNG Industry

This project is an AI-powered think tank designed to provide insights and analysis for the Liquefied Natural Gas (LNG) industry. It leverages machine learning and natural language processing to analyze industry data, predict trends, and offer strategic recommendations.

## Features

- **Data-driven Insights:** Analyze historical and real-time data to identify key trends and patterns in the LNG market.
- **Predictive Analytics:** Forecast LNG prices, demand, and supply using advanced machine learning models.
- **Risk Assessment:** Identify and assess potential risks in the LNG value chain, from production to delivery.
- **Sentiment Analysis:** Gauge market sentiment and public opinion on LNG projects and developments.
- **Knowledge Base:** A comprehensive knowledge base of LNG-related information, accessible through a natural language interface.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ai_thinktank_lng.git
   ```
2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare Data:**
   - **LNG Prices:** The `LNG_prices.csv` file will be automatically downloaded to the `data/` directory when you run the preprocessing script.
   - **Brent Oil Prices:** The `brent_oil_prices.csv` file will be automatically downloaded to the `data/` directory when you run the preprocessing script.
   - **Coal Prices:** The `coal_prices.csv` file will be automatically downloaded to the `data/` directory when you run the preprocessing script.

4. **Run Data Preprocessing:**
   ```bash
   python src/data_preprocessing.py
   ```
   This will generate `data/preprocessed_data.csv` which combines LNG, Brent oil, and coal prices.

5. **Run the application:**
   ```bash
   # Get a forecast for the next 12 months using the LSTM model (default)
   python src/main.py

   # Get a forecast for the next 6 months using the ARIMA model
   python src/main.py --model arima --months 6
   ```

## Models

This project includes two forecasting models:

- **ARIMA:** A classical time-series model that provides a baseline forecast.
- **LSTM:** A Long Short-Term Memory neural network that provides more accurate forecasts, now incorporating multiple features (LNG price, Brent oil price, and coal price).

The LSTM model is the default model. You can choose which model to use with the `--model` flag.

## Contributing

We welcome contributions from the community. Please see our [contributing guidelines](CONTRIBUTING.md) for more information.