#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

import importlib.util, sys
from pathlib import Path

_model_path = Path(__file__).with_name("2_model.py")
_spec = importlib.util.spec_from_file_location("model2", _model_path)
_model2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model2)
MLP = _model2.MLP

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # tensor of class weights (shape [C]) or None
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        # logits: [B,C], targets: [B]
        ce = nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.softmax(logits, dim=1).gather(1, targets.view(-1,1)).squeeze(1).clamp_min(1e-6)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


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
    input_dim = meta["input_dim"]
    num_classes = len(meta.get("class_order", [-3, -2, -1, 0, 1, 2, 3]))


    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train)

    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(class_counts, 1.0))
    inv_sqrt /= inv_sqrt.mean()
    class_weights = torch.tensor(inv_sqrt, dtype=torch.float32)

    # sample weights per example -> oversample rare classes
    sample_weights = class_weights[ytr].numpy()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    model = MLP(input_dim=input_dim, num_classes=num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # focal loss with our class weights
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, sampler=sampler)

    model.train()
    best_loss = float("inf"); patience = 5; strikes = 0
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
            avg = total / len(Xtr)
        print(f"Epoch {epoch:02d} | loss {avg:.4f}")
        if avg + 1e-4 < best_loss:
            best_loss = avg; strikes = 0
        else:
            strikes += 1
            if strikes >= patience:
                print("Early stopping."); break

        print(f"Epoch {epoch:02d} | loss {total / len(Xtr):.4f}")

    out_path = MODELS / f"{args.name}_mlp.pth"
    torch.save({"model_state": model.state_dict(), "meta": meta}, out_path)
    print("[OK] saved", out_path)

if __name__ == "__main__":
    main()
