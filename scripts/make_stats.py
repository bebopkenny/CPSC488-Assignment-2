# scripts/make_stats.py
import pandas as pd
import numpy as np
from pathlib import Path

DS = Path("datasets")

def safe_to_datetime(s):
    """Coerce to datetime; return Series of datetime64 with NaT for bad rows."""
    if s.dtype.kind in ("M",):  # already datetime64
        return s
    return pd.to_datetime(s, errors="coerce", utc=False)

def date_range_strings(dates: pd.Series):
    """Return ('YYYY-MM-DD','YYYY-MM-DD') or ('N/A','N/A') if empty/NaT."""
    if dates.isna().all() or len(dates.dropna()) == 0:
        return "N/A", "N/A"
    dmin = pd.to_datetime(dates.min())
    dmax = pd.to_datetime(dates.max())
    try:
        return str(dmin.date()), str(dmax.date())
    except Exception:
        return "N/A", "N/A"

# ---------- Load core files ----------
final_summary = pd.read_csv(DS/"final_summary.csv")
trade = pd.read_csv(DS/"trade_log.csv")
hp_imp = pd.read_csv(DS/"historical_prices_impact.csv")

# Parse dates safely
for df, col in [(trade, "date"), (hp_imp, "date")]:
    if col in df.columns:
        df[col] = safe_to_datetime(df[col])

# Ensure optional columns exist
for c in ["ret_next", "pnl", "equity", "impact_score", "signal", "pred_class", "trade_type"]:
    if c not in trade.columns:
        trade[c] = np.nan

# ---------- Final summary (direct) ----------
fs = {
    "total_gain_loss": float(final_summary.loc[0, "total_gain_loss"]),
    "avg_annual_return_pct": float(final_summary.loc[0, "avg_annual_return_pct"]),
    "total_return_pct": float(final_summary.loc[0, "total_return_pct"]),
    "final_balance": float(final_summary.loc[0, "final_balance"]),
}

# ---------- Trade stats ----------
n_trades = len(trade)
win_rate = float((trade["ret_next"] > 0).mean()) if trade["ret_next"].notna().any() else np.nan
avg_pnl = float(trade["pnl"].mean()) if trade["pnl"].notna().any() else np.nan
med_pnl = float(trade["pnl"].median()) if trade["pnl"].notna().any() else np.nan

# groupby that won’t fail if trade_type is NaN
by_type = (
    trade.assign(trade_type=trade["trade_type"].astype("string"))
         .groupby("trade_type", dropna=False)["pnl"]
         .agg(count="count", mean="mean", sum="sum")
         .reset_index()
)

def max_drawdown(equity_ser: pd.Series):
    equity_ser = equity_ser.astype("float64")
    if equity_ser.notna().sum() == 0:
        return np.nan
    run_max = equity_ser.cummax()
    dd = (equity_ser - run_max) / run_max
    return float(dd.min())

equity_start = float(trade["equity"].dropna().iloc[0]) if trade["equity"].notna().any() else np.nan
equity_end   = float(trade["equity"].dropna().iloc[-1]) if trade["equity"].notna().any() else np.nan
mdd          = max_drawdown(trade["equity"]) if trade["equity"].notna().any() else np.nan

# ---------- Impact/return relationships ----------
stats_imp = {}
if "impact_score" in hp_imp.columns and hp_imp["impact_score"].notna().any():
    stats_imp["impact_score_mean"] = float(hp_imp["impact_score"].mean())
    stats_imp["impact_score_std"]  = float(hp_imp["impact_score"].std())

# If trade has both impact_score and ret_next, compute correlation & deciles
if trade[["impact_score","ret_next"]].notna().all(axis=1).sum() > 0:
    sub = trade[["impact_score","ret_next"]].dropna()
    corr = sub.corr().iloc[0,1]
    stats_imp["impact_ret_corr"] = float(corr) if np.isfinite(corr) else np.nan
    try:
        dec = pd.qcut(trade["impact_score"], 10, labels=False, duplicates="drop")
        qtab = trade.assign(_dec=dec).groupby("_dec")["ret_next"] \
                     .agg(count="count", mean="mean", median="median") \
                     .reset_index() \
                     .rename(columns={"_dec":"decile"})
    except Exception:
        qtab = pd.DataFrame(columns=["decile","count","mean","median"])
else:
    qtab = pd.DataFrame(columns=["decile","count","mean","median"])

# ---------- Volatility / market-adjusted stats ----------
madj_mean = float(hp_imp["market_adj_return"].mean())
madj_std  = float(hp_imp["market_adj_return"].std())
madj_med  = float(hp_imp["market_adj_return"].median())

# ---------- News vector coverage ----------
def vec_summary(fname):
    p = DS / fname
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # Coerce date if present
    if "date" in df.columns:
        df["date"] = safe_to_datetime(df["date"])
        dmin, dmax = date_range_strings(df["date"])
    else:
        dmin, dmax = "N/A", "N/A"
    symbols = int(df["symbol"].nunique()) if "symbol" in df.columns else 0
    return {"file": fname, "rows": len(df), "symbols": symbols, "date_min": dmin, "date_max": dmax}

vecs = list(filter(None, [
    vec_summary("vectorized_news_dtm.csv"),
    vec_summary("vectorized_news_tfidf.csv"),
    vec_summary("vectorized_news_curated.csv"),
]))

# ---------- Print a compact report ----------
print("\n=== FINAL SUMMARY ===")
for k,v in fs.items():
    print(f"{k}: {v:,.4f}" if isinstance(v, float) else f"{k}: {v}")

print("\n=== TRADES ===")
print(f"n_trades: {n_trades}")
print(f"win_rate: {win_rate:.3f}" if np.isfinite(win_rate) else "win_rate: N/A")
print(f"avg_pnl: {avg_pnl:.6f}" if np.isfinite(avg_pnl) else "avg_pnl: N/A")
print(f"med_pnl: {med_pnl:.6f}" if np.isfinite(med_pnl) else "med_pnl: N/A")
print(f"equity_start: {equity_start:.2f}" if np.isfinite(equity_start) else "equity_start: N/A")
print(f"equity_end: {equity_end:.2f}" if np.isfinite(equity_end) else "equity_end: N/A")
print(f"max_drawdown: {mdd:.3%}" if np.isfinite(mdd) else "max_drawdown: N/A")
print("\nBy trade_type (count, mean pnl, sum pnl):")
print(by_type.to_string(index=False))

print("\n=== MARKET-ADJUSTED STATS (historical_prices_impact.csv) ===")
print(f"market_adj_return mean: {madj_mean:.6f}")
print(f"market_adj_return median: {madj_med:.6f}")
print(f"market_adj_return std: {madj_std:.6f}")

if stats_imp:
    print("\nImpact score distribution:")
    for k,v in stats_imp.items():
        print(f"{k}: {v:.6f}")

print("\nImpact score vs next-day return (trade_log):")
print(qtab.to_string(index=False) if not qtab.empty else "N/A")

print("\n=== NEWS VECTORS COVERAGE ===")
if vecs:
    for v in vecs:
        print(f"{v['file']}: rows={v['rows']}, symbols={v['symbols']}, range={v['date_min']} → {v['date_max']}")
else:
    print("No vectorized files found.")
