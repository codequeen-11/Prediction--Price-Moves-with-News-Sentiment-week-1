import pandas as pd
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")


def classify_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"


def load_news_data(file_path):
    news_df = pd.read_csv(file_path)

    news_df["date"] = pd.to_datetime(
        news_df["date"],
        utc=True,
        errors="coerce"
    )

    news_df = news_df.dropna(
        subset=["date", "headline", "stock"]
    )

    news_df["date_only"] = news_df["date"].dt.date

    return news_df


def add_sentiment_scores(news_df):

    sia = SentimentIntensityAnalyzer()

    news_df = news_df.copy()

    news_df["sentiment_score"] = news_df["headline"].apply(
        lambda text: sia.polarity_scores(str(text))["compound"]
    )

    news_df["sentiment_label"] = news_df["sentiment_score"].apply(
        classify_sentiment
    )

    return news_df


def aggregate_daily_sentiment(news_df, stock_symbol):

    stock_news = news_df[
        news_df["stock"] == stock_symbol
    ].copy()

    daily_sentiment = (
        stock_news.groupby("date_only")
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            article_count=("headline", "count")
        )
        .reset_index()
    )

    daily_sentiment["sentiment_category"] = (
        daily_sentiment["avg_sentiment"]
        .apply(classify_sentiment)
    )

    return daily_sentiment