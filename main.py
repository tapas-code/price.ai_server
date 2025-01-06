from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from binance.client import Client
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