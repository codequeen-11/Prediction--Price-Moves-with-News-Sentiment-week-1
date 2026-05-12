    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib


def load_and_prepare_data(file_path):
    """
    Load and clean historical stock price data.
    Expected columns: Date, Open, High, Low, Close, Volume
    """

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date")
    df = df.dropna()
    df.set_index("Date", inplace=True)

    return df


def compute_indicators(df):
    """
    Compute technical indicators using TA-Lib.
    """

    # Simple Moving Averages
    df["SMA_20"] = talib.SMA(df["Close"], timeperiod=20)
    df["SMA_50"] = talib.SMA(df["Close"], timeperiod=50)

    # Exponential Moving Averages
    df["EMA_20"] = talib.EMA(df["Close"], timeperiod=20)
    df["EMA_50"] = talib.EMA(df["Close"], timeperiod=50)

    # Relative Strength Index
    df["RSI"] = talib.RSI(df["Close"], timeperiod=14)

    # MACD
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = talib.MACD(
        df["Close"],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )

    # Financial metrics
    df["Daily_Return"] = df["Close"].pct_change()
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1
    df["Volatility_20"] = df["Daily_Return"].rolling(window=20).std()

    return df


def plot_stock_analysis(df, stock_name):
    """
    Plot price with moving averages, RSI, and MACD.
    """

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 12),
        sharex=True
    )

    # Price + Moving Averages
    axes[0].plot(df.index, df["Close"], label="Close Price")
    axes[0].plot(df.index, df["SMA_20"], label="SMA 20")
    axes[0].plot(df.index, df["SMA_50"], label="SMA 50")
    axes[0].plot(df.index, df["EMA_20"], label="EMA 20")

    axes[0].set_title(f"{stock_name} Closing Price with Moving Averages")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(True)

    # RSI
    axes[1].plot(df.index, df["RSI"], label="RSI 14")
    axes[1].axhline(70, linestyle="--", label="Overbought 70")
    axes[1].axhline(30, linestyle="--", label="Oversold 30")

    axes[1].set_title(f"{stock_name} Relative Strength Index")
    axes[1].set_ylabel("RSI")
    axes[1].legend()
    axes[1].grid(True)

    # MACD
    axes[2].plot(df.index, df["MACD"], label="MACD")
    axes[2].plot(df.index, df["MACD_Signal"], label="Signal Line")
    axes[2].bar(df.index, df["MACD_Hist"], label="MACD Histogram")

    axes[2].set_title(f"{stock_name} MACD")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("MACD")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()