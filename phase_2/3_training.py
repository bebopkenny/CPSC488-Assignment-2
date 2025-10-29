#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from importlib.machinery import SourceFileLoader

# Load MLP class directly from 2_model.py (no renames needed)
_MLP = SourceFileLoader("model2", str(Path(__file__).with_name("2_model.py"))).load_module()
MLP = _MLP.MLP




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

    print(f"[SUMMARY] name={args.name}")
    print(f"[SUMMARY] X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"[SUMMARY] y_train={y_train.shape}, y_test={y_test.shape}")

    print(f"[SUMMARY] saving to: {MODELS / f'{args.name}_mlp.pth'}")


    # carve chronological validation from the end of TRAIN (10%)
    n_tr = len(X_train)
    n_val = max(1, int(0.1 * n_tr))
    Xtr, ytr = X_train[:-n_val], y_train[:-n_val]
    Xval, yval = X_train[-n_val:], y_train[-n_val:]

    input_dim = meta["input_dim"]
    num_classes = len(meta.get("class_order", [-3, -2, -1, 0, 1, 2, 3]))

   # tensors
    Xtr_t  = torch.from_numpy(Xtr).float()
    ytr_t  = torch.from_numpy(ytr).long()
    Xval_t = torch.from_numpy(Xval).float()
    yval_t = torch.from_numpy(yval).long()

    # ----- class weights (balanced-ish) -----
    # Support ternary labels {-1,0,1} by mapping to indices {0,1,2}
    if ytr.min() == -1 and ytr.max() == 1:
        counts = np.array([
            (ytr == -1).sum(),
            (ytr ==  0).sum(),
            (ytr ==  1).sum()
        ], dtype=np.float32)
        idx_for_weights = ytr + 1          # {-1,0,1} -> {0,1,2}
    else:
        counts = np.bincount(ytr, minlength=num_classes).astype(np.float32)
        idx_for_weights = ytr              # already 0..K-1

    # inverse-frequency style weights, normalized to mean = 1
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32)

    # ----- oversampling via per-sample weights -----
    sample_weights = class_weights[idx_for_weights]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True
    )

    # model / opt / loss
    model = MLP(input_dim=input_dim, num_classes=num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    # use sampler (do NOT also pass shuffle=True)
    train_loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False
    )

    # ----- train with val early stopping -----
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_val = float("inf"); patience = 8; strikes = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
        train_avg = total / len(Xtr_t)

        # validate
        model.eval()
        with torch.no_grad():
            logits = model(Xval_t)
            val_loss = float(criterion(logits, yval_t).item())

        print(f"Epoch {epoch:02d} | train {train_avg:.4f} | val {val_loss:.4f}")

        if val_loss + 1e-4 < best_val:
            best_val = val_loss
            strikes = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            strikes += 1
            if strikes >= patience:
                print("Early stopping on validation loss.")
                break

    # restore best and save
    model.load_state_dict(best_state)
    out_path = MODELS / f"{args.name}_mlp.pth"
    torch.save({"model_state": model.state_dict(), "meta": meta}, out_path)
    print("[OK] saved", out_path)

    print(f"[SUMMARY] name={args.name}")
    print(f"[SUMMARY] X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"[SUMMARY] y_train={y_train.shape}, y_test={y_test.shape}")
    print(f"[SUMMARY] saving to: {MODELS / f'{args.name}_mlp.pth'}")

if __name__ == "__main__":
    main()

