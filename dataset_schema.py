import pandas as pd
import pandera.pandas as pa

historical_prices_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(pd.Timestamp, nullable=False),
        "symbol": pa.Column(str, nullable=False),
        "open": pa.Column(float, nullable=False),
        "high": pa.Column(float, nullable=False),
        "low": pa.Column(float, nullable=False),
        "close": pa.Column(float, nullable=False),
        "volume": pa.Column(int, nullable=False),
    },
    coerce=True
)

all_news_schema = pa.DataFrameSchema(
    {
        "id": pa.Column(int, nullable=False),
        "headline": pa.Column(str, nullable=False),
        "URL": pa.Column(str, nullable=False),
        "article": pa.Column(str, nullable=False),
        "publisher": pa.Column(str, nullable=False),
        "date": pa.Column(pd.Timestamp, nullable=False),
        "symbol": pa.Column(str, nullable=True),  # Symbol can be null for general news
    },
    coerce=True
)

# Modified aggregated_news schema
aggregated_news_schema = pa.DataFrameSchema(
    {
        "symbol": pa.Column(str, nullable=False),
        "date": pa.Column(pd.Timestamp, nullable=False),
        "news": pa.Column(str, nullable=False),
    },
    coerce=True
)

historical_prices_impact_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(pd.Timestamp, nullable=False),
        "symbol": pa.Column(str, nullable=False),
        "open": pa.Column(float, nullable=False),
        "high": pa.Column(float, nullable=False),
        "low": pa.Column(float, nullable=False),
        "close": pa.Column(float, nullable=False),
        "volume": pa.Column(int, nullable=False),
        "daily_return": pa.Column(float, nullable=False),
        "daily_volatility": pa.Column(float, nullable=False),
        "market_return": pa.Column(float, nullable=False),
        "beta": pa.Column(float, nullable=False),
        "alpha": pa.Column(float, nullable=False),
        "idiosyn_return": pa.Column(float, nullable=False),
        "idiosyn_volatility": pa.Column(float, nullable=False),
        "market_adj_return": pa.Column(float, nullable=False),
        "market_adj_volatility": pa.Column(float, nullable=False),
        "impact_score": pa.Column(float, nullable=False),
    },
    coerce=True
)

trade_log_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(pd.Timestamp, nullable=False),
        "symbol": pa.Column(str, nullable=False),
        "trade_type": pa.Column(str, pa.Check.isin(["Buy", "Sell"]), nullable=False),
        "shares": pa.Column(int, pa.Check.ge(0), nullable=False),
        "price": pa.Column(float, pa.Check.ge(0), nullable=False),
        "trans_amount": pa.Column(float, nullable=False),
        "cash_after_trade": pa.Column(float, nullable=False),
        "news_headline": pa.Column(str, nullable=True),
        "impact_score": pa.Column(float, nullable=True),
    },
    coerce=True
)

trade_log_summary_schema = pa.DataFrameSchema(
    {
        "total_gain_loss": pa.Column(float, nullable=False),
        "avg_annual_return_pct": pa.Column(float, nullable=False),
        "total_return_pct": pa.Column(float, nullable=False),
        "final_balance": pa.Column(float, nullable=False),
    },
    coerce=True
)

vectorized_news_schema = pa.DataFrameSchema(
    {
        "symbol": pa.Column(str, nullable=False),
        "date": pa.Column(pd.Timestamp, nullable=False),
        "news_vector": pa.Column(object, nullable=False),  # Assuming news_vector can be a list/array
        "impact_score": pa.Column(float, nullable=False),
    },
    coerce=True
)
