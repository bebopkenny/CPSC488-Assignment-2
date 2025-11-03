# scripts/make_vectorized_csvs.py
import json
import numpy as np
import pandas as pd
from pathlib import Path

DS = Path("datasets")
PROC = DS / "processed"

def load_first_array(npz_path: Path) -> np.ndarray:
    """Load the first available array from an .npz, regardless of key name."""
    with np.load(npz_path, allow_pickle=True) as z:
        # Prefer common names if present, else take the first key
        for k in ("X", "features", "data", "arr_0"):
            if k in z.files:
                return z[k]
        return z[z.files[0]]

def ensure_2d(name: str, X: np.ndarray) -> np.ndarray | list:
    """
    Return either a proper 2D numpy array (N, D) or a Python list-of-lists (N, D)
    that we can serialize row-wise. Handles:
      - 2D numeric arrays
      - 1D object arrays of vectors
      - 1D numeric arrays that need reshape using meta input_dim
    """
    # Case 1: already 2D
    if getattr(X, "ndim", 1) == 2:
        return X

    # Case 2: 1D object array where each element is already a vector
    if X.ndim == 1 and X.dtype == object and len(X) > 0 and hasattr(X[0], "__len__"):
        return [np.asarray(v).ravel().tolist() for v in X]

    # Case 3: 1D numeric — try to reshape using meta info
    if X.ndim == 1 and np.issubdtype(X.dtype, np.number):
        meta_path = PROC / f"{name}.json"
        input_dim = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            input_dim = meta.get("input_dim")

        if input_dim and X.size % input_dim == 0:
            return X.reshape(-1, int(input_dim))

        raise ValueError(
            f"{name}: 1D numeric features could not be reshaped. "
            f"Expected {meta_path} with 'input_dim' to be present."
        )

    # Fallback
    raise ValueError(f"{name}: Unsupported feature array shape/dtype: {getattr(X, 'shape', None)}, {X.dtype}")

def export(name: str):
    """
    name ∈ {"vector_dtm","vector_tfidf","vector_curated"}
    Reads:
      - datasets/<name>_features.npz
      - datasets/<name>_index.csv
    Writes:
      - datasets/vectorized_news_{dtm|tfidf|curated}.csv
    """
    feat_path = DS / f"{name}_features.npz"
    idx_path  = DS / f"{name}_index.csv"

    if not feat_path.exists():
        raise FileNotFoundError(f"Missing features: {feat_path}")
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing index: {idx_path}")

    # Load features robustly
    X_raw = load_first_array(feat_path)
    X = ensure_2d(name, X_raw)

    # Load index
    index = pd.read_csv(idx_path)
    if "date" not in index.columns or "symbol" not in index.columns:
        raise ValueError(f"{idx_path} must have 'date' and 'symbol' columns")
    index["date"] = pd.to_datetime(index["date"], errors="coerce")

    # Align lengths
    N = min(len(index), (len(X) if isinstance(X, list) else X.shape[0]))
    index = index.iloc[:N].copy()
    if isinstance(X, list):
        vecs = X[:N]
    else:
        vecs = [X[i, :].tolist() for i in range(N)]

    # impact_score: optional in index; default to 0.0
    if "impact_score" in index.columns:
        impact = pd.to_numeric(index["impact_score"], errors="coerce").fillna(0.0).values[:N]
    else:
        impact = np.zeros(N, dtype=float)

    out = pd.DataFrame({
        "symbol": index["symbol"].astype(str).values,
        "date": index["date"].values,
        "news_vector": vecs,
        "impact_score": impact,
    })

    suffix = name.split("_", 1)[1] if "_" in name else name
    out_path = DS / f"vectorized_news_{suffix}.csv"
    out.to_csv(out_path, index=False)
    print(f"✔ wrote {out_path} with {len(out)} rows")

if __name__ == "__main__":
    export("vector_dtm")
    export("vector_tfidf")
    export("vector_curated")
