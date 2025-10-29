import pandas as pd
import numpy as np
from pathlib import Path

INOUT = Path("../historical_prices_impact.csv")

def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        # avoid div by zero; return NaNs → will become zeros later
        return (s - mu) * np.nan
    return (s - mu) / sd

def score_block(g: pd.DataFrame) -> pd.DataFrame:
    # Compute z-scores per symbol on available history
    zr = zscore(g["market_adj_return"])
    zv = zscore(g["market_adj_vol_3d"])

    # Base: neutral
    impact = np.zeros(len(g), dtype=int)

    # Non-neutral where |zr| > 0.5
    strong_mask = zr.abs() > 0.5
    sign = np.sign(zr.fillna(0)).astype(int)

    # magnitude = 1 + I(|zr|>1) + I(zv>1)
    mag = 1 + (zr.abs() > 1).astype(int) + (zv > 1).astype(int)

    impact[strong_mask] = (sign[strong_mask] * mag[strong_mask]).astype(int)

    out = g.copy()
    out["impact_score"] = impact
    return out

def main():
    if not INOUT.exists():
        raise SystemExit(f"Missing input: {INOUT}")

    df = pd.read_csv(INOUT, parse_dates=["date"])

    # Only score non-market rows; keep S&P rows with score = 0
    non_mkt = df["symbol"].str.lower() != "s&p"

    df_scored = (
        df.loc[non_mkt]
          .groupby("symbol", group_keys=False)
          .apply(score_block)
          .reset_index(drop=True)
    )

    # Merge back market rows unchanged
    final = pd.concat([df_scored, df.loc[~non_mkt].assign(impact_score=0)], ignore_index=True)
    final = final.sort_values(["symbol","date"]).reset_index(drop=True)

    final.to_csv(INOUT, index=False)
    print(f"[OK] Updated {INOUT} with impact_score.")
    # Peek a few non-market rows
    print(final.loc[final["symbol"].str.lower()!="s&p", ["date","symbol","market_adj_return","market_adj_vol_3d","impact_score"]].head(10).to_string(index=False))

if __name__ == "__main__":
    main()