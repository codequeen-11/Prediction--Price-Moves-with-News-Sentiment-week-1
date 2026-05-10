import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_prepare_data(file_path):

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(df["Date"])

    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date")

    df.dropna(inplace=True)

    df.set_index("Date", inplace=True)

    return df


def compute_indicators(df):

    # SMA
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # EMA
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def plot_stock_analysis(df, stock_name):

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 12),
        sharex=True
    )

    # PRICE + SMA
    axes[0].plot(df.index, df["Close"], label="Close")
    axes[0].plot(df.index, df["SMA_20"], label="SMA 20")
    axes[0].plot(df.index, df["SMA_50"], label="SMA 50")

    axes[0].set_title(f"{stock_name} Price and Moving Averages")
    axes[0].legend()
    axes[0].grid(True)

    # RSI
    axes[1].plot(df.index, df["RSI"], label="RSI")

    axes[1].axhline(70, linestyle="--")
    axes[1].axhline(30, linestyle="--")

    axes[1].set_title(f"{stock_name} RSI")
    axes[1].grid(True)

    # MACD
    axes[2].plot(df.index, df["MACD"], label="MACD")
    axes[2].plot(df.index, df["MACD_Signal"], label="Signal")

    axes[2].set_title(f"{stock_name} MACD")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()