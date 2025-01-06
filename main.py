from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from binance.client import Client
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import io
import base64
import os

load_dotenv()

app = FastAPI()
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))

@app.get('/')
def read_root():
    return {"message": "Crypto Price Predictor!"}

@app.get("/historical-data/{symbol}")
def get_historical_data(symbol: str, interval: str = "1h", limit: int = 100):
    """
    Fetch historical candlestick data for a given symbol.
    :param symbol: Cryptocurrency pair (e.g., 'BTCUSDT')
    :param interval: Time interval (e.g., '1h', '1d')
    :param limit: Number of data points to fetch
    :return: List of candlestick data
    """
    try:
        candlesticks = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        data = [
            {
                "open_time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
                "close_time": c[6],
            }
            for c in candlesticks
        ]
        return {"symbol": symbol, "interval": interval, "data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data: {str(e)}")
    
@app.get("/preprocessed-data/{symbol}")
def get_preprocessed_data(symbol: str, interval: str = "1h", limit: int = 100):
    """
    Fetch and preprocess historical candlestick data for a given symbol.
    :param symbol: Cryptocurrency pair (e.g., 'BTCUSDT')
    :param interval: Time interval (e.g., '1h', '1d')
    :param limit: Number of data points to fetch
    :return: Preprocessed and normalized data
    """
    try:
        # Fetch raw data
        candlesticks = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        data = [
            {
                "open_time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
                "close_time": c[6],
            }
            for c in candlesticks
        ]

        # Convert to DataFrame
        df = pd.DataFrame(data)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        # Select features to normalize
        features = ["open", "high", "low", "close", "volume"]
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        # Return normalized data
        return {
            "symbol": symbol,
            "interval": interval,
            "normalized_data": df[["open_time", "open", "high", "low", "close", "volume", "close_time"]].to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing data: {str(e)}")

@app.post("/predict-price/{symbol}")
def predict_price(symbol: str, interval: str = "1h", limit: int = 100):
    """
    Predict the next closing price for a given symbol using Linear Regression.
    :param symbol: Cryptocurrency pair (e.g., 'BTCUSDT')
    :param interval: Time interval (e.g., '1h', '1d')
    :param limit: Number of data points to use for training
    :return: Predicted closing price
    """
    try:
        # Fetch and preprocess data
        candlesticks = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        data = [
            {
                "open_time": c[0],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
                "close_time": c[6],
            }
            for c in candlesticks
        ]

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Use the 'close' prices as the target variable
        df["close_shifted"] = df["close"].shift(-1)  # Next close price
        df.dropna(inplace=True)  # Remove rows with NaN

        # Define features (X) and target (y)
        X = df[["open", "high", "low", "volume"]].values
        y = df["close_shifted"].values

        # Split into training and test sets
        split_index = int(0.8 * len(X))  # 80% training, 20% testing
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Train the Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the next closing price
        last_row = np.array([df.iloc[-1][["open", "high", "low", "volume"]].values])
        predicted_price = model.predict(last_row)[0]

        return {
            "symbol": symbol,
            "interval": interval,
            "predicted_closing_price": predicted_price,
            "last_closing_price": df.iloc[-1]["close"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")