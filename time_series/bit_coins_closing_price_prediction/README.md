# Bitcoin Closing Price Prediction

## What
Predict Bitcoin closing price using Dense Neural Network with lag features.

## How
- Load Bitcoin OHLCV time series data
- Create lag features: `close_(t-1)` to `close_(t-15)` (15 previous days)
- Normalize with MinMaxScaler (range: 0.01-0.99)
- Train Dense Neural Network with Dropout
- Evaluate with R2 Score

## Tech Stack
- Python, TensorFlow, Keras
- sklearn (MinMaxScaler, r2_score)
- Pandas, NumPy, Matplotlib

## Model Architecture
```
Input(15) → Dense(60) → Dense(60) → Dropout(0.2) → Dense(1)
```
- Optimizer: Adam
- Loss: MSE

## Feature Engineering
- Lag features from historical closing prices
- 15 time steps look-back window
- MinMaxScaler normalization

## Data Split
- Train: 90%
- Validation: 5%
- Test: 5%

## Run Notebook
```bash
# Google Colab - Upload notebook + bitcoin.csv

# VS Code - Install Jupyter extension, run notebook

# Jupyter CLI
jupyter notebook bitcoin_Time_Series_Prediction_with_Keras.ipynb
```
