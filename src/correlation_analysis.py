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