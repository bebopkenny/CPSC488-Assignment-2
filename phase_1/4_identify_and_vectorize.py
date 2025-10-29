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
