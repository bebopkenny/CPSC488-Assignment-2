from pathlib import Path
import pandas as pd
import numpy as np

THIS = Path(__file__).resolve()
ROOT = THIS.parent.parent  # project root

def main():
    preds_path  = ROOT / "phase_3" / "predictions.csv"
    prices_path = ROOT / "historical_prices.csv"

    preds  = pd.read_csv(preds_path)
    prices = pd.read_csv(prices_path)

    
    # Preferred path: merge on (date, symbol)
    
    have_keys = (
        {"date","symbol"}.issubset(preds.columns) and
        {"date","symbol","close"}.issubset(prices.columns)
    )

    if have_keys:
        preds["date"]  = pd.to_datetime(preds["date"], errors="coerce")
        prices["date"] = pd.to_datetime(prices["date"], errors="coerce")

        # Restrict to symbols we actually predicted for
        wanted = preds["symbol"].dropna().unique().tolist()
        prices = prices[prices["symbol"].isin(wanted)].copy()

        # Merge close onto predictions
        df = preds.merge(
            prices[["date","symbol","close"]],
            on=["date","symbol"], how="left"
        ).sort_values(["symbol","date"]).reset_index(drop=True)
    else:
        
        # Fallback: index-aligned attach of 'close'
        
        if "date" in prices.columns and "symbol" in prices.columns:
            prices = prices.sort_values(["date","symbol"]).reset_index(drop=True)
        closes = prices["close"].values
        take = min(len(closes), len(preds))
        df = preds.iloc[:take].copy().reset_index(drop=True)
        df["close"] = closes[:take]
        # Create synthetic keys so downstream still works
        if "symbol" not in df.columns:
            df["symbol"] = "A"  # assume single-ticker case
        if "date" not in df.columns:
            df["date"] = pd.RangeIndex(start=0, stop=len(df), step=1)

    # Safety: ensure we have a trading signal column in {-1,0,1}
    if "pred_class" not in df.columns:
        raise RuntimeError("predictions.csv must contain a 'pred_class' column")
    df["signal"] = df["pred_class"].clip(-1, 1)

    
    # Compute next-day returns per symbol
    
    df = df.sort_values(["symbol","date"]).reset_index(drop=True)

    # Shift next close within each symbol
    df["close_next"] = df.groupby("symbol", group_keys=False)["close"].shift(-1)
    df["ret_next"] = (df["close_next"] - df["close"]) / df["close"]

    # PnL per row (no costs)
    df["pnl"] = df["signal"] * df["ret_next"]

    # Equity curve: combine across symbols by chronological order
    # (When multiple symbols, this is a simple compounding of per-row trade returns)
    df["equity"] = (1.0 + df["pnl"].fillna(0.0)).cumprod()

    
    # Outputs
    
    trade_log_path = ROOT / "phase_3" / "trade_log.csv"
    summary_path   = ROOT / "phase_3" / "final_summary.csv"
    trade_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Save trade log
    df.to_csv(trade_log_path, index=False)
    print("[OK] wrote", trade_log_path)

    tau = 0.5  # try 0.25, 0.5, 0.75
    if "margin" in df.columns:
        gated = (df["margin"].abs() >= tau).astype(int)
    else:
        gated = 1  # no margin present -> trade all (current behavior)

    raw_signal = df["pred_class"].clip(-1, 1)
    signal = raw_signal * gated
    df["signal"] = signal

    # Simple costs: pay a small fee when position changes
    cost_bps = 0.0005  # 5 bps per change (example)
    df["position"] = signal.shift(0).fillna(0)     # enter same day
    df["prev_position"] = df["position"].shift(1).fillna(0)
    df["turnover"] = (df["position"] != df["prev_position"]).astype(int)
    df["ret_next"] = (df["close_next"] - df["close"]) / df["close"]

    gross = df["position"] * df["ret_next"]
    costs = df["turnover"] * cost_bps
    df["pnl"] = gross - costs

    df["equity"] = (1.0 + df["pnl"].fillna(0)).cumprod()

    # Summary stats (exclude final NaN ret_next rows)
    valid = df["pnl"].notna()
    n_trades = int(valid.sum())
    if n_trades > 0:
        # Use last valid equity (ignore the very last row if ret_next is NaN)
        last_valid_equity = float(df.loc[valid, "equity"].iloc[-1])
        total_return = last_valid_equity - 1.0
        hit_rate = float((df.loc[valid, "pnl"] > 0).mean())
        avg_pnl = float(df.loc[valid, "pnl"].mean())
        std_pnl = float(df.loc[valid, "pnl"].std(ddof=0))
    else:
        total_return = 0.0
        hit_rate = 0.0
        avg_pnl = 0.0
        std_pnl = 0.0

    summary = pd.DataFrame([{
        "total_return": total_return,
        "hit_rate": hit_rate,
        "avg_pnl": avg_pnl,
        "std_pnl": std_pnl,
        "n_trades": n_trades
    }])
    summary.to_csv(summary_path, index=False)
    print("[OK] wrote", summary_path)

if __name__ == "__main__":
    main()
