import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parent
DATASETS = (ROOT / "../datasets").resolve()
OUT = (ROOT / "../datasets/processed").resolve()
OUT.mkdir(parents=True, exist_ok=True)

CLASS_ORDER = [-3, -2, -1, 0, 1, 2, 3]
CLASS_TO_IDX = {c:i for i,c in enumerate(CLASS_ORDER)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=["vector_dtm","vector_tfidf","vector_curated"], required=True)
    ap.add_argument("--train_frac", type=float, default=0.8)
    args = ap.parse_args()

    X = load_npz(DATASETS / f"{args.name}_features.npz")
    meta = pd.read_csv(DATASETS / f"{args.name}_index.csv", parse_dates=["date"])
    # Sort by time for chronological split
    meta = meta.sort_values("date").reset_index(drop=True)
    # Align X rows with sorted meta
    order = meta.index.values
    X = X[order, :]

    y_raw = meta["impact_score"].astype(int).tolist()
    y = np.array([CLASS_TO_IDX[v] for v in y_raw], dtype=np.int64)

    n = len(meta)
    n_train = int(n * args.train_frac)
    idx_train = np.arange(n_train)
    idx_test = np.arange(n_train, n)

    # Save splits (as dense float32; safe for your current sizes)
    X_train = X[idx_train].astype(np.float32).toarray()
    y_train = y[idx_train]
    X_test  = X[idx_test].astype(np.float32).toarray()
    y_test  = y[idx_test]

    prefix = OUT / args.name
    np.save(prefix.with_suffix(".X_train.npy"), X_train)
    np.save(prefix.with_suffix(".y_train.npy"), y_train)
    np.save(prefix.with_suffix(".X_test.npy"),  X_test)
    np.save(prefix.with_suffix(".y_test.npy"),  y_test)

    summary = {
        "name": args.name,
        "n_total": n,
        "n_train": int(idx_train.size),
        "n_test": int(idx_test.size),
        "input_dim": int(X.shape[1]),
        "class_order": CLASS_ORDER
    }
    with open(prefix.with_suffix(".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("[OK] Wrote:", prefix.with_suffix(".X_train.npy").name, prefix.with_suffix(".y_train.npy").name,
          prefix.with_suffix(".X_test.npy").name, prefix.with_suffix(".y_test.npy").name)
    print("Summary:", summary)

if __name__ == "__main__":
    main()