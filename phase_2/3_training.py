#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import importlib.util, sys
from pathlib import Path

_model_path = Path(__file__).with_name("2_model.py")
_spec = importlib.util.spec_from_file_location("model2", _model_path)
_model2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model2)
MLP = _model2.MLP


def load_splits(prefix: Path):
    X_train = np.load(prefix.with_suffix(".X_train.npy"))
    y_train = np.load(prefix.with_suffix(".y_train.npy"))
    X_test  = np.load(prefix.with_suffix(".X_test.npy"))
    y_test  = np.load(prefix.with_suffix(".y_test.npy"))
    with open(prefix.with_suffix(".json")) as f:
        meta = json.load(f)
    return X_train, y_train, X_test, y_test, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=["vector_dtm","vector_tfidf","vector_curated"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parent
    PROC = (ROOT / "../datasets/processed").resolve()
    MODELS = (ROOT / "../models").resolve(); MODELS.mkdir(parents=True, exist_ok=True)

    prefix = PROC / args.name
    X_train, y_train, X_test, y_test, meta = load_splits(prefix)
    input_dim = meta["input_dim"]; num_classes = 7

    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train)

    class_counts = np.bincount(y_train, minlength=num_classes)
    weights = class_counts.sum() / np.maximum(class_counts, 1)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32)

    model = MLP(input_dim=input_dim, num_classes=num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
        print(f"Epoch {epoch:02d} | loss {total / len(Xtr):.4f}")

    out_path = MODELS / f"{args.name}_mlp.pth"
    torch.save({"model_state": model.state_dict(), "meta": meta}, out_path)
    print("[OK] saved", out_path)

if __name__ == "__main__":
    main()
