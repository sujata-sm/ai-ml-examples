# Time Series Forecasting

Predicting future values based on historical sequential data.

## Installation
```bash
pip install -r requirements.txt
```

## What is Time Series?
Data points indexed in time order. Examples: stock prices, weather, sales, energy consumption.

## Techniques Used

| Technique | Description |
|-----------|-------------|
| **LSTM** | Long Short-Term Memory - captures long-term dependencies |
| **RNN** | Recurrent Neural Network - sequential pattern learning |
| **MLP** | Multi-Layer Perceptron - basic neural network |

## Common Preprocessing
- **Normalization:** MinMaxScaler (0-1 range)
- **Lag Features:** Use past values (t-1, t-2, ...) as input
- **Sliding Window:** Create sequences for training

## Evaluation Metrics
- **MSE / RMSE:** Mean Squared Error
- **MAE:** Mean Absolute Error
- **R2 Score:** Coefficient of determination

## Projects

| Project | Description |
|---------|-------------|
| [Airline Passengers](./air_passanger_keras/) | Monthly passenger forecasting |
| [Bitcoin Prediction](./bit_coins_closing_price_prediction/) | Crypto price prediction |
| [Solar Power](./solar_power_predition_with_LSTM/) | Power generation forecasting |
