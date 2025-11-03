import pandas as pd
from pathlib import Path
import numpy as np

DS = Path("datasets")
TRADE = DS / "trade_log.csv"
OUT = DS / "final_summary.csv"

def main():
    df = pd.read_csv(TRADE, parse_dates=["date"])

    # Use first/last non-null equity
    if "equity" not in df.columns:
        raise SystemExit("trade_log.csv is missing 'equity' column.")

    eq = df["equity"].dropna()
    if eq.empty:
        raise SystemExit("No non-null equity values found.")

    equity_start = float(eq.iloc[0])
    equity_end = float(eq.iloc[-1])

    # Dates for CAGR
    dmin = df["date"].min()
    dmax = df["date"].max()
    days = (dmax - dmin).days if pd.notna(dmin) and pd.notna(dmax) else 0

    total_gain_loss = equity_end - equity_start
    total_return_pct = (equity_end / equity_start - 1.0) * 100.0

    if days > 0:
        cagr = (equity_end / equity_start) ** (365.0 / days) - 1.0
        avg_annual_return_pct = cagr * 100.0
    else:
        avg_annual_return_pct = np.nan

    out = pd.DataFrame(
        {
            "total_gain_loss": [total_gain_loss],
            "avg_annual_return_pct": [avg_annual_return_pct],
            "total_return_pct": [total_return_pct],
            "final_balance": [equity_end],
        }
    )

    # Ensure correct order and dtypes
    cols = ["total_gain_loss","avg_annual_return_pct","total_return_pct","final_balance"]
    out = out[cols].astype("float64")

    out.to_csv(OUT, index=False)
    print(f"✔ wrote {OUT} with corrected totals")
    print(out.round(6).to_string(index=False))

if __name__ == "__main__":
    main()
