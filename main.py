from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI()
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Utility function to fetch and preprocess raw data
def fetch_data(symbol, interval, limit):
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
        return pd.DataFrame(data)
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error fetching data: {str(e)}")

@app.get('/')
def read_root():
    return {"message": "Crypto Price Predictor!"}

@app.get("/historical-data/{symbol}")
def get_historical_data(symbol: str, interval: str = "1h", limit: int = 100):
    try:
        df = fetch_data(symbol, interval, limit)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        return {"symbol": symbol, "interval": interval, "data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/preprocessed-data/{symbol}")
def get_preprocessed_data(symbol: str, interval: str = "1h", limit: int = 100):
    try:
        df = fetch_data(symbol, interval, limit)
        features = ["open", "high", "low", "close", "volume"]
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        return {"symbol": symbol, "interval": interval, "normalized_data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-price/{symbol}")
def predict_price(symbol: str, interval: str = "1h", limit: int = 100):
    """
    Predict the next closing price for a given symbol using Linear Regression with probabilities.
    :param symbol: Cryptocurrency pair (e.g., 'BTCUSDT')
    :param interval: Time interval (e.g., '1h', '1d')
    :param limit: Number of data points to use for training
    :return: Predicted closing price with probabilities
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
        last_closing_price = df.iloc[-1]["close"]

        # Determine up/down prediction
        price_change = predicted_price - last_closing_price
        up_probability = 1 / (1 + np.exp(-price_change))  # Sigmoid function
        down_probability = 1 - up_probability

        return {
            "symbol": symbol,
            "interval": interval,
            "predicted_closing_price": predicted_price,
            "last_closing_price": last_closing_price,
            "prediction": "up" if price_change > 0 else "down",
            "probabilities": {
                "up": round(up_probability * 100, 2),  # Convert to percentage
                "down": round(down_probability * 100, 2),  # Convert to percentage
            },
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making prediction: {str(e)}")


@app.get("/backtest/{symbol}")
def backtest_model(symbol: str, interval: str = "1h", limit: int = 100):
    try:
        df = fetch_data(symbol, interval, limit)
        df["close_shifted"] = df["close"].shift(-1)
        df.dropna(inplace=True)

        X = df[["open", "high", "low", "volume"]].values
        y = df["close_shifted"].values

        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        split_index = int(0.8 * len(X))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(y_test)), y_test, label="Actual", color="blue")
        plt.plot(range(len(y_pred)), y_pred, label="Predicted", color="orange")
        plt.legend()
        plt.title("Actual vs Predicted Prices")
        plt.xlabel("Data Points")
        plt.ylabel("Price")
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        return {
            "symbol": symbol,
            "interval": interval,
            "mae": mae,
            "mse": mse,
            "visualization": f"data:image/png;base64,{img_base64}",
        }
    except Exception as e:
        logging.error(f"Backtesting error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
