import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import importlib.util, sys
from pathlib import Path

_model_path = Path(__file__).with_name("2_model.py")
_spec = importlib.util.spec_from_file_location("model2", _model_path)
_model2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_model2)
MLP = _model2.MLP


CLASS_ORDER = [-3, -2, -1, 0, 1, 2, 3]

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
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parent
    PROC = (ROOT / "../datasets/processed").resolve()
    MODELS = (ROOT / "../models").resolve()

    X_train, y_train, X_test, y_test, meta = load_splits(PROC / args.name)
    # after loading meta
    num_classes = len(meta.get("class_order", [-3, -2, -1, 0, 1, 2, 3]))
    CLASS_ORDER = meta.get("class_order", [-3, -2, -1, 0, 1, 2, 3])
    input_dim = meta["input_dim"]


    ckpt = torch.load(MODELS / f"{args.name}_mlp.pth", map_location="cpu")
    model = MLP(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).float())
        preds = torch.argmax(logits, dim=1).numpy()

    acc = accuracy_score(y_test, preds)
    f1m = f1_score(y_test, preds, average="macro")
    cm  = confusion_matrix(y_test, preds, labels=list(range(num_classes)))

    print(f"Test accuracy: {acc:.3f}")
    print(f"Macro F1:     {f1m:.3f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nIndex→class mapping:", {i:c for i,c in enumerate(CLASS_ORDER)})
    print(classification_report(y_test, preds, labels=list(range(num_classes)), zero_division=0))

if __name__ == "__main__":
    main()
