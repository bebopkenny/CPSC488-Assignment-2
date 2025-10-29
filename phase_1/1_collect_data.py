from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional, Set

import pandas as pd
import numpy as np
from dateutil import parser as dtparser

import time
def _download_single_with_retry(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    for k in range(max(1, retries)):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, group_by='ticker', threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                # When a single ticker still returns multi-index, select it
                if ticker in df.columns.get_level_values(0):
                    df = df[ticker]
            if not df.empty:
                return df
        except Exception as e:
            pass
        # backoff
        sleep_s = min(2 ** k, 8)
        time.sleep(sleep_s)
    return pd.DataFrame()


# yfinance import is only needed when executing the download; users may install it in their own env
try:
    import yfinance as yf
except Exception as e:
    yf = None


# -----------------------------
# Flexible CSV loading utilities
# -----------------------------

CANDIDATE_COLS = {
    "date": ["date", "datetime", "published_at", "timestamp", "pub_date"],
    "symbol": ["symbol", "ticker", "ric", "sid"],
    "headline": ["headline", "title", "headlines", "news_title"],
    "url": ["url", "link", "article_url"],
    "publisher": ["publisher", "source", "outlet"],
}


def _find_first_col(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    # also try stripped variations
    for cand in candidates:
        for c in cols:
            if c.lower().strip() == cand:
                return c
    return None



# -----------------------------
# Symbol extraction from free-text headlines
# -----------------------------
import re

TICKER_PATTERNS = [
    re.compile(r"\((?P<ticker>[A-Z]{1,5})\)"),                    # e.g., "Agilent (A) Gears Up..."
    re.compile(r"\bNYSE[:\s]+(?P<ticker>[A-Z]{1,5})\b"),         # e.g., "NYSE: A"
    re.compile(r"\bNASDAQ[:\s]+(?P<ticker>[A-Z]{1,5})\b"),       # e.g., "NASDAQ: MSFT"
    re.compile(r"\bNasdaq[:\s]+(?P<ticker>[A-Z]{1,5})\b"),
    re.compile(r"\bAMEX[:\s]+(?P<ticker>[A-Z]{1,5})\b"),
    re.compile(r"\bTicker[:\s]+(?P<ticker>[A-Z]{1,5})\b"),
]

# Words that are ALL-CAPS 1–5 letters but are not tickers in this context
TICKER_DENYLIST = set("""
EPS CEO FDA ETF FOMC GDP CPI PPI WHO CDC EPSGA GAAP NONGAAP
COVID Q1 Q2 Q3 Q4 FY FY20 FY21 YOY MOM DOJ FTC SEC DOJ DCF EBITDA
EBIT EBITA ROE ROI ROI ROA
""".split())

def extract_first_ticker(text: str) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    for pat in TICKER_PATTERNS:
        m = pat.search(text)
        if m:
            t = m.group("ticker").strip().upper().rstrip(".,;:")
            if t and t not in TICKER_DENYLIST:
                return t
    # Fallback: find any ALL-CAPS token 1–5 letters inside parens (e.g., "(A)")
    m = re.search(r"\(([A-Z]{1,5})\)", text)
    if m:
        t = m.group(1).upper()
        if t not in TICKER_DENYLIST:
            return t
    return None

def load_news_like_csv(path: Optional[str]) -> pd.DataFrame:
    """Load a CSV that likely contains {date, symbol, headline, url, publisher} in any naming variant.
    Returns empty DataFrame if path is None or file missing.
    """
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        print(f"[WARN] File not found: {path}", file=sys.stderr)
        return pd.DataFrame()

    df = pd.read_csv(p)
    if df.empty:
        return df

    # map best-effort columns
    mapping = {}
    for key, cands in CANDIDATE_COLS.items():
        col = _find_first_col(df.columns, cands)
        if col is not None:
            mapping[key] = col

    # Required for symbol discovery is just 'symbol' (and optional date for sanity)
    if "symbol" not in mapping:
        # If no explicit symbol column, try to extract ticker from headlines
        headline_col = _find_first_col(df.columns, CANDIDATE_COLS["headline"]) or _find_first_col(df.columns, ["title"])
        if headline_col:
            syms = df[headline_col].astype(str).map(extract_first_ticker)
            if syms.notnull().any():
                mapping["symbol"] = "__derived_symbol__"
                df["__derived_symbol__"] = syms
        if "symbol" not in mapping:
            # Attempt recovery: look for any column that looks like a ticker (all-cap strings, <=6 chars)
            tick_like_cols = [c for c in df.columns if df[c].dtype == object]
            recovered = None
            for c in tick_like_cols:
                sample = df[c].dropna().astype(str).str.strip().head(100)
                frac_tickers = (sample.str.fullmatch(r"[A-Za-z.\-^]{1,6}")).mean() if len(sample) > 0 else 0
                if frac_tickers > 0.6:
                    recovered = c
                    break
            if recovered:
                mapping["symbol"] = recovered
            else:
                print(f"[WARN] Could not find a symbol column in {path}; proceeding without.", file=sys.stderr)

    # Build unified dataframe with at least 'symbol'
    out = pd.DataFrame()
    if "symbol" in mapping:
        out["symbol"] = df[mapping["symbol"]].astype(str).str.strip().str.upper()
    if "date" in mapping:
        # Try to parse dates
        out["date"] = pd.to_datetime(df[mapping["date"]], errors="coerce").dt.date

    # Optional info
    if "headline" in mapping:
        out["headline"] = df[mapping["headline"]].astype(str)
    if "url" in mapping:
        out["url"] = df[mapping["url"]].astype(str)
    if "publisher" in mapping:
        out["publisher"] = df[mapping["publisher"]].astype(str)

    # Drop rows missing symbol
    if "symbol" in out.columns:
        out = out.dropna(subset=["symbol"])

    return out


# -----------------------------
# Symbol normalization
# -----------------------------

def normalize_symbols_for_yahoo(symbols: Iterable[str]) -> Dict[str, str]:
    """
    Convert generic tickers to Yahoo Finance ticker format for download.
    Returns mapping: original_symbol -> yahoo_symbol
    - Replace '.' with '-' (e.g., BRK.B -> BRK-B; RDS.A -> RDS-A)
    - Keep ^GSPC as-is
    """
    mapping = {}
    for s in symbols:
        s_clean = s.strip().upper()
        if s_clean == "S&P" or s_clean == "SNP":
            # Not a real symbol; users sometimes write S&P; we will handle S&P via ^GSPC separately
            continue
        if s_clean.startswith("^"):
            mapping[s_clean] = s_clean
        else:
            mapping[s_clean] = s_clean.replace(".", "-")
    return mapping


def gather_symbols(analyst_df: pd.DataFrame, headlines_df: pd.DataFrame, extra_symbols: List[str]) -> List[str]:
    symbols: Set[str] = set()
    for df in (analyst_df, headlines_df):
        if "symbol" in df.columns:
            syms = df["symbol"].dropna().astype(str).str.strip().str.upper()
            symbols.update(syms.unique().tolist())
    for s in extra_symbols or []:
        if s:
            symbols.add(s.strip().upper())
    # Remove any "S&P" placeholders; market index handled separately
    symbols.difference_update({"S&P", "SNP"})
    # Deduplicate and sort
    syms = sorted(symbols)
    return syms


# -----------------------------
# Price downloading
# -----------------------------


def _safe_price_frame(df_sym: pd.DataFrame) -> pd.DataFrame:
    """Normalize a single-symbol price frame to required columns with unique names."""
    if df_sym is None or df_sym.empty:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    # Drop duplicate columns if any
    df_sym = df_sym.loc[:, ~df_sym.columns.duplicated()].copy()

    # Build close from Adj Close if present else Close
    if "Adj Close" in df_sym.columns:
        df_sym["close"] = df_sym["Adj Close"]
    elif "Close" in df_sym.columns:
        df_sym["close"] = df_sym["Close"]
    elif "close" in df_sym.columns:
        df_sym["close"] = df_sym["close"]
    else:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])

    # Map other standard columns if present
    rename_map = {}
    if "Open" in df_sym.columns: rename_map["Open"] = "open"
    if "High" in df_sym.columns: rename_map["High"] = "high"
    if "Low" in df_sym.columns:  rename_map["Low"]  = "low"
    if "Volume" in df_sym.columns: rename_map["Volume"] = "volume"
    df_sym = df_sym.rename(columns=rename_map)

    # Create missing columns if absent
    for c in ["open","high","low","volume"]:
        if c not in df_sym.columns:
            df_sym[c] = pd.NA

    df_sym["date"] = df_sym.index.date
    return df_sym[["date","open","high","low","close","volume"]].reset_index(drop=True)


def download_prices(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Please 'pip install yfinance' in your environment.")
    if not symbols:
        raise ValueError("No symbols to download. Provide --symbols or valid CSVs.")

    # Map symbols to Yahoo-friendly tickers
    sym_map = normalize_symbols_for_yahoo(symbols)
    yahoo_symbols = sorted(set(sym_map.values()))

    # Always include S&P 500 ^GSPC for market returns
    if "^GSPC" not in yahoo_symbols:
        yahoo_symbols.append("^GSPC")

    print(f"[INFO] Downloading {len(yahoo_symbols)} tickers from {start} to {end} ...")

    # Batch to reduce rate limits / memory
    batch_size = 100
    records = []

    for i in range(0, len(yahoo_symbols), batch_size):
        batch = yahoo_symbols[i:i+batch_size]
        try:
            data = yf.download(batch, start=start, end=end, auto_adjust=True, progress=True, group_by='ticker', threads=True)
        except Exception as e:
            print(f"[WARN] yfinance batch failed ({batch[0]}..{batch[-1]}): {e}")
            continue

        if isinstance(data.columns, pd.MultiIndex):
            # Multi-ticker
            for ysym in batch:
                if ysym not in data.columns.get_level_values(0):
                    # Try per-ticker fallback with retries
                    df_try = _download_single_with_retry(ysym, start, end, retries=args.retries if "args" in globals() else 3)
                    if df_try.empty:
                        print(f"[WARN] No data for {ysym}")
                        continue
                    df_sym = df_try
                else:
                    df_sym = data[ysym]
                df_sym = data[ysym]
                out_sym = _safe_price_frame(df_sym)
                if out_sym.empty:
                    continue
                originals = [orig for orig, y in sym_map.items() if y == ysym]
                if ysym == "^GSPC":
                    symbol_out = "s&p"
                elif originals:
                    symbol_out = originals[0]
                else:
                    symbol_out = ysym
                out_sym.insert(1, "symbol", symbol_out)
                records.append(out_sym)
        else:
            # Single-ticker case
            df_sym = data if not data.empty else _download_single_with_retry(yahoo_symbols[i], start, end, retries=args.retries if "args" in globals() else 3)
            out_sym = _safe_price_frame(df_sym)
            if not out_sym.empty:
                symbol_out = symbols[0] if symbols else "UNKNOWN"
                out_sym.insert(1, "symbol", symbol_out)
                records.append(out_sym)

    prices = pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=["date","symbol","open","high","low","close","volume"])
    prices = prices.dropna(subset=["close"]).sort_values(["symbol", "date"]).reset_index(drop=True)

    bad = prices[(prices["close"] <= 0) | (prices["open"] <= 0)]
    if not bad.empty:
        print(f"[WARN] Found {len(bad)} rows with non-positive prices; kept for inspection.")

    return prices



def main():
    ap = argparse.ArgumentParser(description="Phase 1A: Build historical_prices.csv")
    ap.add_argument("--analyst_csv", type=str, default=None, help="Path to analyst ratings CSV (optional)")
    ap.add_argument("--headlines_csv", type=str, default=None, help="Path to headlines CSV (optional)")
    ap.add_argument("--symbols", type=str, default=None, help="Comma-separated extra symbols to include (optional)")
    ap.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--out", type=str, default="historical_prices.csv", help="Output CSV path")
    ap.add_argument("--max_symbols", type=int, default=50, help="Cap number of symbols to download (after discovery)")
    ap.add_argument("--retries", type=int, default=3, help="Retries per batch or ticker on download errors")
    args = ap.parse_args()

    # Load CSVs (best-effort)
    analyst_df = load_news_like_csv(args.analyst_csv)
    headlines_df = load_news_like_csv(args.headlines_csv)

    extra_symbols = []
    if args.symbols:
        extra_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    symbols = gather_symbols(analyst_df, headlines_df, extra_symbols)
    print(f"[INFO] Discovered {len(symbols)} symbols from CSVs and args.")
    if args.max_symbols and len(symbols) > args.max_symbols:
        print(f"[INFO] Capping to first {args.max_symbols} symbols for this run. Use --max_symbols to change.")
        symbols = symbols[:args.max_symbols]
    if len(symbols) <= 3:
        print(f"[HINT] If this looks too small, pass --symbols AAPL,MSFT,GOOGL,...", file=sys.stderr)

    # Download prices
    prices = download_prices(symbols, args.start, args.end)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(prices):,} rows to {out_path}")

    # Print a small sample for quick verification
    print(prices.head().to_string(index=False))


if __name__ == "__main__":
    main()
