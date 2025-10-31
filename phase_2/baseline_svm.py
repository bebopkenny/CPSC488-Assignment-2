#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.sparse import load_npz
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=["vector_dtm","vector_tfidf","vector_curated"], required=True)
    ap.add_argument("--ternary", action="store_true")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parent
    DS   = (ROOT / "../datasets").resolve()

    X = load_npz(DS / f"{args.name}_features.npz")
    meta = pd.read_csv(DS / f"{args.name}_index.csv", parse_dates=["date"]).sort_values("date")
    y = meta["impact_score"].astype(int).to_numpy()
    if args.ternary:
        y = np.where(y <= -1, -1, np.where(y >= 1, 1, 0))

    n = len(y); n_tr = int(0.8*n)
    Xtr, Xte = X[:n_tr], X[n_tr:]
    ytr, yte = y[:n_tr], y[n_tr:]
    clf = LinearSVC(class_weight="balanced", max_iter=6000)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print("Acc:", accuracy_score(yte, pred))
    print("Macro F1:", f1_score(yte, pred, average="macro"))
    print(classification_report(yte, pred, zero_division=0))

if __name__ == "__main__":
    main()
