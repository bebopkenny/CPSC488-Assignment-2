import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH  = Path("../historical_prices.csv")
OUT_PATH = Path("../historical_prices_impact.csv")

def _log_return(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1))

def _ols_alpha_beta(r_stock: pd.Series, r_mkt: pd.Series) -> tuple[float, float]:
    df = pd.DataFrame({"y": r_stock, "x": r_mkt}).dropna()
    if len(df) < 10 or df["x"].var() == 0:
        return (0.0, 0.0)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    beta = np.cov(y, x, ddof=1)[0,1] / np.var(x, ddof=1)
    alpha = y.mean() - beta * x.mean()
    return (float(alpha), float(beta))

def main():
    if not IN_PATH.exists():
        raise SystemExit(f"Missing input: {IN_PATH}")

    df = pd.read_csv(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol","date"]).reset_index(drop=True)

    # Compute per-symbol daily log returns
    df["daily_return"] = df.groupby("symbol", group_keys=False)["close"].apply(_log_return)

    # 3-day rolling volatility of returns (per symbol)
    df["daily_vol_3d"] = (
        df.groupby("symbol", group_keys=False)["daily_return"]
          .apply(lambda s: s.rolling(window=3, min_periods=2).std())
    )

    # Extract market return from s&p
    mkt = (
        df.loc[df["symbol"].str.lower() == "s&p", ["date", "daily_return"]]
          .rename(columns={"daily_return": "market_return"})
    )

    # Join market_return by date to every row
    df = df.merge(mkt, on="date", how="left")

    # Compute alpha/beta per non-market symbol over the full sample window
    alpha_beta = []
    for sym, g in df.groupby("symbol"):
        if sym.lower() == "s&p":
            alpha_beta.append({"symbol": sym, "alpha": 0.0, "beta": 0.0})
            continue
        a, b = _ols_alpha_beta(g["daily_return"], g["market_return"])
        alpha_beta.append({"symbol": sym, "alpha": a, "beta": b})
    ab = pd.DataFrame(alpha_beta)

    df = df.merge(ab, on="symbol", how="left")

    # Market-adjusted return and its 3-day volatility
    df["market_adj_return"] = df["daily_return"] - (df["alpha"] + df["beta"] * df["market_return"])
    df.loc[df["symbol"].str.lower() == "s&p", "market_adj_return"] = np.nan

    df["market_adj_vol_3d"] = (
        df.groupby("symbol", group_keys=False)["market_adj_return"]
          .apply(lambda s: s.rolling(window=3, min_periods=2).std())
    )

    # Save
    KEEP = [
        "date","symbol","open","high","low","close","volume",
        "daily_return","daily_vol_3d","market_return","alpha","beta",
        "market_adj_return","market_adj_vol_3d"
    ]
    df[KEEP].to_csv(OUT_PATH, index=False)
    print(f"[OK] Wrote {len(df)} rows → {OUT_PATH}")
    # small peek for sanity
    print(df.loc[df['symbol']!='s&p', KEEP].head(8).to_string(index=False))

if __name__ == "__main__":
    main()