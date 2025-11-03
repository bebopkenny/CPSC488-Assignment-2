import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("datasets")
SRC = DATA / "historical_prices.csv"
OUT = DATA / "historical_prices_impact.csv"

def load_prices():
    df = pd.read_csv(SRC, parse_dates=["date"])
    # Normalize symbol casing
    df["symbol"] = df["symbol"].astype(str)

    # Map common S&P tickers to "s&p"
    sp_map = {
        "^GSPC": "s&p",
        "GSPC": "s&p",
        "^SPX": "s&p",
        "SPX": "s&p",
        "S&P": "s&p",
        "s&p": "s&p",
    }
    df["symbol"] = df["symbol"].replace(sp_map)

    # If none of those exist but SPY exists, use it as a proxy, and rename to s&p
    if not (df["symbol"] == "s&p").any():
        if (df["symbol"] == "SPY").any():
            df.loc[df["symbol"] == "SPY", "symbol"] = "s&p"
            print("[note] Using SPY as a proxy for S&P. Mention this in limitations.")
        else:
            raise SystemExit(
                "No S&P benchmark found. Add ^GSPC or GSPC to historical_prices.csv, "
                "or temporarily allow SPY as a proxy."
            )

    # Sort and compute per symbol daily return
    df = df.sort_values(["symbol", "date"])
    df["daily_return"] = df.groupby("symbol")["close"].pct_change()

    # Simple 20-day rolling volatility
    df["daily_volatility"] = (
        df.groupby("symbol")["daily_return"]
          .rolling(window=20, min_periods=5).std().reset_index(level=0, drop=True)
    )

    return df

def compute_market_adjusted(df: pd.DataFrame) -> pd.DataFrame:
    # Pull market series
    sp = (
        df[df["symbol"] == "s&p"][["date", "daily_return"]]
        .rename(columns={"daily_return": "market_return"})
    )

    # Merge market return onto every row
    out = df.merge(sp, on="date", how="left")

    # Market-adjusted return
    out["market_adj_return"] = out["daily_return"] - out["market_return"]

    # Rolling market-adjusted volatility
    out["market_adj_volatility"] = (
        out.groupby("symbol")["market_adj_return"]
           .rolling(window=20, min_periods=5).std().reset_index(level=0, drop=True)
    )

    # Per symbol alpha and beta over the whole sample (OLS without intercept for beta, then alpha)
    def alpha_beta(g):
        g2 = g.dropna(subset=["daily_return", "market_return"])
        if len(g2) < 10:
            return pd.Series({"beta": np.nan, "alpha": np.nan})
        mr = g2["market_return"].to_numpy()
        sr = g2["daily_return"].to_numpy()
        # beta = cov(sr, mr) / var(mr)
        var_m = np.var(mr, ddof=1)
        if var_m == 0 or np.isnan(var_m):
            return pd.Series({"beta": np.nan, "alpha": np.nan})
        beta = np.cov(sr, mr, ddof=1)[0,1] / var_m
        # alpha = mean(sr - beta*mr)
        alpha = float(np.mean(sr - beta*mr))
        return pd.Series({"beta": beta, "alpha": alpha})

    ab = (
        out.groupby("symbol")[["daily_return", "market_return"]]
        .apply(lambda g: alpha_beta(pd.DataFrame(g)))
        .reset_index()
    )
    out = out.merge(ab, on="symbol", how="left")

    # Idiosyncratic residual and its 20-day rolling vol
    out["idiosyn_return"] = out["daily_return"] - (out["alpha"] + out["beta"]*out["market_return"])
    out["idiosyn_volatility"] = (
        out.groupby("symbol")["idiosyn_return"]
           .rolling(window=20, min_periods=5).std().reset_index(level=0, drop=True)
    )

    # Keep columns expected by your schema, preserving any existing ones if present
    wanted = [
        "date","symbol","open","high","low","close","volume",
        "daily_return","daily_volatility",
        "market_return","beta","alpha",
        "idiosyn_return","idiosyn_volatility",
        "market_adj_return","market_adj_volatility",
    ]

    # If you already had an impact_score computed elsewhere, try to bring it over
    if "impact_score" in out.columns:
        wanted.append("impact_score")
    else:
        out["impact_score"] = np.nan
        wanted.append("impact_score")

    # Drop rows that can't have valid stats yet (first few per symbol)
    required = [
        "daily_return", "daily_volatility",
        "market_return", "market_adj_return", "market_adj_volatility",
        "idiosyn_return", "idiosyn_volatility",
    ]
    out = out.dropna(subset=required).copy()

    # If your schema forbids nulls in impact_score, ensure it's filled
    if "impact_score" in out.columns:
        out["impact_score"] = out["impact_score"].fillna(0.0)

    # Final ordering, dropping duplicates if any merge created them
    out = out[wanted].drop_duplicates(subset=["symbol","date"])
    return out

def main():
    df = load_prices()
    out = compute_market_adjusted(df)
    out.to_csv(OUT, index=False)
    print(f"✔ wrote {OUT} with {len(out):,} rows, symbols={out['symbol'].nunique()}")

if __name__ == "__main__":
    main()
