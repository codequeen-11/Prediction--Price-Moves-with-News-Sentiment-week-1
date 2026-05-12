# import pandas as pd
# import nltk

# from nltk.sentiment import SentimentIntensityAnalyzer

# nltk.download("vader_lexicon")


# def classify_sentiment(score):
#     if score > 0.05:
#         return "Positive"
#     elif score < -0.05:
#         return "Negative"
#     else:
#         return "Neutral"


# def load_news_data(file_path):
#     news_df = pd.read_csv(file_path)

#     news_df["date"] = pd.to_datetime(
#         news_df["date"],
#         utc=True,
#         errors="coerce"
#     )

#     news_df = news_df.dropna(
#         subset=["date", "headline", "stock"]
#     )

#     news_df["date_only"] = news_df["date"].dt.date

#     return news_df


# def add_sentiment_scores(news_df):

#     sia = SentimentIntensityAnalyzer()

#     news_df = news_df.copy()

#     news_df["sentiment_score"] = news_df["headline"].apply(
#         lambda text: sia.polarity_scores(str(text))["compound"]
#     )

#     news_df["sentiment_label"] = news_df["sentiment_score"].apply(
#         classify_sentiment
#     )

#     return news_df


# def aggregate_daily_sentiment(news_df, stock_symbol):

#     stock_news = news_df[
#         news_df["stock"] == stock_symbol
#     ].copy()

#     daily_sentiment = (
#         stock_news.groupby("date_only")
#         .agg(
#             avg_sentiment=("sentiment_score", "mean"),
#             article_count=("headline", "count")
#         )
#         .reset_index()
#     )

#     daily_sentiment["sentiment_category"] = (
#         daily_sentiment["avg_sentiment"]
#         .apply(classify_sentiment)
#     )

#     return daily_sentiment

import os
import pandas as pd
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer


def download_vader_lexicon():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


def classify_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"


def validate_news_columns(news_df):
    required_columns = ["date", "headline", "stock"]

    missing_columns = [
        column for column in required_columns
        if column not in news_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required news columns: {missing_columns}"
        )


def load_news_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"News file not found: {file_path}"
        )

    news_df = pd.read_csv(file_path)

    validate_news_columns(news_df)

    news_df["date"] = pd.to_datetime(
        news_df["date"],
        utc=True,
        errors="coerce"
    )

    news_df = news_df.dropna(
        subset=["date", "headline", "stock"]
    )

    news_df["date_only"] = news_df["date"].dt.date

    if news_df.empty:
        raise ValueError(
            "News dataframe is empty after cleaning."
        )

    return news_df


def add_sentiment_scores(news_df):
    download_vader_lexicon()

    news_df = news_df.copy()

    sia = SentimentIntensityAnalyzer()

    news_df["sentiment_score"] = news_df["headline"].apply(
        lambda text: sia.polarity_scores(str(text))["compound"]
    )

    news_df["sentiment_label"] = news_df["sentiment_score"].apply(
        classify_sentiment
    )

    return news_df


def aggregate_daily_sentiment(news_df, stock_symbol):
    news_df = news_df.copy()

    date_column = "aligned_date" if "aligned_date" in news_df.columns else "date_only"

    stock_news = news_df[
        news_df["stock"] == stock_symbol
    ].copy()

    if stock_news.empty:
        raise ValueError(
            f"No news articles found for stock symbol: {stock_symbol}"
        )

    daily_sentiment = (
        stock_news.groupby(date_column)
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            article_count=("headline", "count")
        )
        .reset_index()
    )

    daily_sentiment = daily_sentiment.rename(
        columns={date_column: "date_only"}
    )

    daily_sentiment["sentiment_category"] = (
        daily_sentiment["avg_sentiment"]
        .apply(classify_sentiment)
    )

    return daily_sentiment