import pandas as pd


def load_stock_data(file_path):

    stock_df = pd.read_csv(file_path)

    stock_df["Date"] = pd.to_datetime(
        stock_df["Date"],
        errors="coerce"
    )

    stock_df = stock_df.sort_values("Date")

    stock_df["daily_return"] = (
        stock_df["Close"].pct_change() * 100
    )

    stock_df["date_only"] = stock_df["Date"].dt.date

    stock_df = stock_df.dropna(
        subset=["Date", "Close", "daily_return"]
    )

    return stock_df