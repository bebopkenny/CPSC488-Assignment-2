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

CURATED = [
    "upgrade","downgrade","beat","miss","surge","plunge","strong","weak","guidance","cut"
]

def build_aggregated_news() -> pd.DataFrame:
    """Returns DataFrame: date (datetime64[D]), symbol, news (cleaned, possibly empty)."""
    # Load both files (best-effort union)
    a = load_news_csv(NEWS_DIR / "analyst_ratings.csv")
    h = load_news_csv(NEWS_DIR / "headlines.csv")
    news = pd.concat([a, h], ignore_index=True)
    if news.empty:
        return pd.DataFrame(columns=["date","symbol","news"])

    # Ensure types
    news["symbol"] = news["symbol"].astype(str).str.upper().str.strip()
    # If date missing or invalid, drop (we need dates to align)
    news["date"] = pd.to_datetime(news["date"], errors="coerce")
    news = news.dropna(subset=["date", "symbol"])
    news["date"] = news["date"].dt.normalize()

    # Build a minimal text field (headline only for this assignment)
    news["text"] = news["headline"].fillna("")

    # Clean
    news["text"] = news["text"].map(clean_text)

    # Aggregate per (date, symbol)
    agg = (news.groupby(["date","symbol"], as_index=False)["text"]
                .apply(lambda s: " . ".join([t for t in s.tolist() if t])))
    agg = agg.rename(columns={"text":"news"})
    return agg

def main():
    if not PRICES_WITH_IMPACT.exists():
        raise SystemExit(f"Missing input: {PRICES_WITH_IMPACT}")

    prices = pd.read_csv(PRICES_WITH_IMPACT, parse_dates=["date"])
    # trading calendar per symbol (we only vectorize for non-market rows)
    prices = prices.sort_values(["symbol","date"]).reset_index(drop=True)
    non_mkt = prices["symbol"].str.lower() != "s&p"
    base = prices.loc[non_mkt, ["date","symbol","impact_score"]].copy()

    # All trading dates per symbol indexed
    base["date"] = base["date"].dt.normalize()

    # News aggregated per calendar day
    day_news = build_aggregated_news()

    # Build 3-trading-day rolling text per (symbol, date)
    # 1) wide dict of news per (symbol,date)
    key_to_text = {(r.symbol, r.date): r.news for r in day_news.itertuples(index=False)} if not day_news.empty else {}

    rows = []
    for sym, g in base.groupby("symbol"):
        dates = g["date"].tolist()  # trading dates for this symbol
        for i, d in enumerate(dates):
            # take d, d-1, d-2 in TRADING order (not calendar)
            window = dates[max(0, i-2): i+1]
            texts = []
            for wd in window:
                t = key_to_text.get((sym, wd), "")
                if t:
                    texts.append(t)
            combined = " . ".join(texts)
            rows.append({"date": d, "symbol": sym, "impact_score": int(g.iloc[i].impact_score), "news": combined})

    agg3 = pd.DataFrame(rows)
    # Clean again to be safe
    agg3["news"] = agg3["news"].map(clean_text)

    # (Optional) remove truly empty news rows to avoid empty docs
    # But we will keep them; vectorizers can handle empty -> all zeros.

    # --- Vectorizers ---
    # Common params
    token_pattern = r"(?u)\b[a-z0-9]{2,}\b"  # keep simple alnum tokens length>=2
    stop_words = "english"  # we kept negations at cleaning by not removing 'no'/'not' explicitly

    # 1) DTM
    cv = CountVectorizer(min_df=5, token_pattern=token_pattern, stop_words=stop_words)
    X_dtm = cv.fit_transform(agg3["news"].values)
    # 2) TF-IDF
    tfv = TfidfVectorizer(min_df=5, token_pattern=token_pattern, stop_words=stop_words)
    X_tfidf = tfv.fit_transform(agg3["news"].values)
    # 3) Curated (fixed small vocab counts)
    vocab_idx = {w:i for i,w in enumerate(CURATED)}
    rows_idx, cols_idx, data_vals = [], [], []
    for r, text in enumerate(agg3["news"].values):
        counts = {}
        for tok in text.split():
            if tok in vocab_idx:
                counts[tok] = counts.get(tok, 0) + 1
        for tok, c in counts.items():
            rows_idx.append(r); cols_idx.append(vocab_idx[tok]); data_vals.append(c)
    X_cur = sparse.csr_matrix((data_vals, (rows_idx, cols_idx)), shape=(len(agg3), len(CURATED)), dtype=np.float32)

    # Save matrices and index files
    def dump(name: str, X: sparse.csr_matrix):
        from scipy.sparse import save_npz
        save_npz(OUT_DIR / f"{name}_features.npz", X)
        agg3[["date","symbol","impact_score"]].to_csv(OUT_DIR / f"{name}_index.csv", index=False)

    dump("vector_dtm", X_dtm)
    dump("vector_tfidf", X_tfidf)
    dump("vector_curated", X_cur)

    print(f"[OK] Vector files written to {OUT_DIR}")
    print("DTM shape:", X_dtm.shape, "| TFIDF shape:", X_tfidf.shape, "| Curated shape:", X_cur.shape)
    # quick label check
    print("Label counts:", agg3["impact_score"].value_counts().sort_index().to_dict())

if __name__ == "__main__":
    main()