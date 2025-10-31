#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# --- Paths
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "datasets" / "processed"
PH3  = ROOT / "phase_3"
PH3.mkdir(parents=True, exist_ok=True)

def load_splits(prefix: Path):
    Xtr = np.load(prefix.with_suffix(".X_train.npy"))
    ytr = np.load(prefix.with_suffix(".y_train.npy"))
    Xte = np.load(prefix.with_suffix(".X_test.npy"))
    yte = np.load(prefix.with_suffix(".y_test.npy"))
    meta = json.loads(prefix.with_suffix(".json").read_text())
    return Xtr, ytr, Xte, yte, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name",
                    choices=["vector_dtm", "vector_tfidf", "vector_curated"],
                    default="vector_dtm")
    ap.add_argument("--balanced", action="store_true",
                    help="Use class_weight='balanced' in LinearSVC")
    ap.add_argument("--C", type=float, default=1.0,
                    help="Regularization strength (LinearSVC)")
    args = ap.parse_args()

    # Load data
    prefix = PROC / args.name
    Xtr, ytr, Xte, yte, meta = load_splits(prefix)

    # Train baseline Linear SVM
    class_weight = "balanced" if args.balanced else None
    clf = LinearSVC(C=args.C, class_weight=class_weight, max_iter=10000)
    clf.fit(Xtr, ytr)

    # Predict
    ypred = clf.predict(Xte)

    # Decision margins (confidence-ish)
    dec = clf.decision_function(Xte)
    if dec.ndim == 1:
        margin = np.abs(dec)
    else:
        sortd = np.sort(dec, axis=1)
        margin = sortd[:, -1] - sortd[:, -2]

    # Report
    acc = accuracy_score(yte, ypred)
    macro_f1 = f1_score(yte, ypred, average="macro")
    print(f"Acc: {acc}")
    print(f"Macro F1: {macro_f1}")
    print(classification_report(yte, ypred))

    # Build predictions output (attach date,symbol if index exists)
    idx_csv = prefix.with_suffix(".index_test.csv")
    if idx_csv.exists():
        idx_df = pd.read_csv(idx_csv)
        out = idx_df.copy()
        out["pred_class"] = ypred
        out["margin"] = margin
    else:
        out = pd.DataFrame({
            "row_id": np.arange(len(ypred)),
            "pred_class": ypred,
            "margin": margin
        })

    # Write where Phase 3 expects it
    out_path = PH3 / "predictions.csv"
    out.to_csv(out_path, index=False)
    print("[OK] wrote", out_path)

if __name__ == "__main__":
    main()
