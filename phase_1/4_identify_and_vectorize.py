from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ROOT = Path(__file__).resolve().parent
NEWS_DIR = (ROOT / "../news_datasets").resolve()
PRICES_WITH_IMPACT = (ROOT / "../historical_prices_impact.csv").resolve()
OUT_DIR = (ROOT / "../datasets").resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- ticker extraction (mirrors step 1) ----------
TICKER_PATTERNS = [
    re.compile(r"\((?P<ticker>[A-Z]{1,5})\)"),
    re.compile(r"\bNYSE[:\s]+(?P<ticker>[A-Z]{1,5})\b"),
    re.compile(r"\bNASDAQ[:\s]+(?P<ticker>[A-Z]{1,5})\b", re.IGNORECASE),
    re.compile(r"\bNasdaq[:\s]+(?P<ticker>[A-Z]{1,5})\b"),
    re.compile(r"\bAMEX[:\s]+(?P<ticker>[A-Z]{1,5})\b"),
    re.compile(r"\bTicker[:\s]+(?P<ticker>[A-Z]{1,5})\b"),
]
TICKER_DENYLIST = set("""
EPS CEO FDA ETF FOMC GDP CPI PPI WHO CDC GAAP EBITDA EBIT EBITA ROE ROI ROA
Q1 Q2 Q3 Q4 FY FY20 FY21 YOY MOM DOJ FTC SEC DCF
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
    m = re.search(r"\(([A-Z]{1,5})\)", text)
    if m:
        t = m.group(1).upper()
        if t not in TICKER_DENYLIST:
            return t
    return None

# ---------- flexible CSV loader ----------
CANDIDATE_COLS = {
    "date": ["date", "datetime", "published_at", "timestamp", "pub_date"],
    "symbol": ["symbol", "ticker", "ric", "sid"],
    "headline": ["headline", "title", "headlines", "news_title"],
    "url": ["url", "link", "article_url"],
    "publisher": ["publisher", "source", "outlet"],
}

def _find_first_col(cols: List[str], cands: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in cols}
    for cand in cands:
        if cand in lower_map:
            return lower_map[cand]
    for cand in cands:
        for c in cols:
            if c.lower().strip() == cand:
                return c
    return None

def load_news_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date","symbol","headline","url","publisher"])
    df = pd.read_csv(path)
    mapping = {}
    for key, cands in CANDIDATE_COLS.items():
        col = _find_first_col(df.columns.tolist(), cands)
        if col is not None:
            mapping[key] = col

    out = pd.DataFrame()
    # headline first (we need this for ticker derivation)
    if "headline" in mapping:
        out["headline"] = df[mapping["headline"]].astype(str)
    else:
        # if no headline/title column, bail
        return pd.DataFrame(columns=["date","symbol","headline","url","publisher"])

    # symbol: use provided or derive from headline
    if "symbol" in mapping:
        out["symbol"] = (
            df[mapping["symbol"]].astype(str).str.strip().str.upper()
        )
    else:
        out["symbol"] = out["headline"].map(extract_first_ticker)

    # date if present (not strictly needed for aggregation, but helpful)
    if "date" in mapping:
        out["date"] = pd.to_datetime(df[mapping["date"]], errors="coerce").dt.date
    else:
        out["date"] = pd.NaT

    if "url" in mapping:
        out["url"] = df[mapping["url"]].astype(str)
    if "publisher" in mapping:
        out["publisher"] = df[mapping["publisher"]].astype(str)

    # keep only rows with a symbol (others don't map to tickers)
    out = out.dropna(subset=["symbol"])
    return out

# ---------- simple text cleaning ----------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
PUNCT_RE = re.compile(r"[^a-z0-9\s]")

NEGATIONS = {"no", "not", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere", "hardly", "scarcely", "barely"}

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = HTML_TAG_RE.sub(" ", s)
    # Remove punctuation but keep spaces; later collapse
    s = PUNCT_RE.sub(" ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s
