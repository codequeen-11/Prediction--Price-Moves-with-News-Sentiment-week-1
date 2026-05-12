import pandas as pd

from scipy.stats import pearsonr


def merge_sentiment_with_returns(
    stock_df,
    daily_sentiment
):

    merged_df = pd.merge(
        stock_df,
        daily_sentiment,
        on="date_only",
        how="inner"
    )

    return merged_df


def calculate_correlation(merged_df):

    correlation, p_value = pearsonr(
        merged_df["avg_sentiment"],
        merged_df["daily_return"]
    )

    return correlation, p_value

def align_news_to_trading_days(news_df, stock_df):

    trading_days = sorted(stock_df["date_only"].unique())

    def get_next_trading_day(news_date):

        for trading_day in trading_days:
            if trading_day >= news_date:
                return trading_day

        return None

    news_df = news_df.copy()

    news_df["aligned_date"] = news_df["date_only"].apply(
        get_next_trading_day
    )

    news_df = news_df.dropna(subset=["aligned_date"])

    return news_df