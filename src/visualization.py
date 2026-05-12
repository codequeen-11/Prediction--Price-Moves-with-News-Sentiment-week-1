import matplotlib.pyplot as plt
import seaborn as sns


def plot_sentiment_vs_return(
    merged_df,
    stock_symbol,
    correlation
):

    plt.figure(figsize=(10, 6))

    sns.regplot(
        data=merged_df,
        x="avg_sentiment",
        y="daily_return",
        scatter_kws={"alpha": 0.6}
    )

    plt.title(
        f"{stock_symbol} Sentiment vs Daily Return\n"
        f"Correlation = {correlation:.2f}"
    )

    plt.xlabel("Average Daily Sentiment")
    plt.ylabel("Daily Return (%)")

    plt.grid(True)

    plt.show()


def plot_average_return_by_sentiment(
    merged_df,
    stock_symbol
):

    avg_returns = (
        merged_df.groupby("sentiment_category")[
            "daily_return"
        ]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=avg_returns,
        x="sentiment_category",
        y="daily_return"
    )

    plt.title(
        f"{stock_symbol} Average Return by Sentiment"
    )

    plt.xlabel("Sentiment Category")
    plt.ylabel("Average Daily Return (%)")

    plt.grid(axis="y")

    plt.show()