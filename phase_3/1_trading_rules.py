import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from importlib.machinery import SourceFileLoader
import torch.nn.functional as F


# Paths and loaders

THIS   = Path(__file__).resolve()
ROOT   = THIS.parent.parent                 # project root
PROC   = ROOT / "datasets" / "processed"
MODELS = ROOT / "models"
MODEL_PY = ROOT / "phase_2" / "2_model.py"  # provided by professor

# Load MLP from phase_2/2_model.py without renaming file
MLP = SourceFileLoader("model2", str(MODEL_PY)).load_module().MLP

def load_meta(prefix: Path):
    with open(prefix.with_suffix(".json")) as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name",
                    choices=["vector_dtm","vector_tfidf","vector_curated"],
                    default="vector_dtm",
                    help="Which processed feature set to use")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Optional explicit path to model .pth; defaults to models/{name}_mlp.pth")
    args = ap.parse_args()

    prefix = PROC / args.name
    meta = load_meta(prefix)
    input_dim   = int(meta["input_dim"])
    class_order = meta.get("class_order", [-1, 0, 1])  # ternary for your runs
    num_classes = len(class_order)

    
    # Load features (test split)
    
    X_test = np.load(prefix.with_suffix(".X_test.npy"))
    idx_csv = prefix.with_suffix(".index_test.csv")
    idx_df = None
    if idx_csv.exists():
        idx_df = pd.read_csv(idx_csv)
        # Normalize index columns if present
        if "date" in idx_df.columns:
            idx_df["date"] = pd.to_datetime(idx_df["date"], errors="coerce")

    
    # Build model + load weights
    
    model = MLP(input_dim=input_dim, num_classes=num_classes)
    ckpt_path = Path(args.ckpt) if args.ckpt else (MODELS / f"{args.name}_mlp.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    
    # Predict
    
    X_test_t = torch.from_numpy(X_test)
    with torch.no_grad():
        logits = model(X_test_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        pred_idx = probs.argmax(axis=1)

    # confidence gate: if max prob < TH, force neutral (0)
    TH = 0.65  # try 0.60–0.75
    conf = probs.max(axis=1)
    pred_class = np.array(class_order, dtype=int)[pred_idx]
    pred_class = np.where(conf >= TH, pred_class, 0)

    
    # Assemble output
    
    if idx_df is not None and not idx_df.empty:
        out = idx_df.copy()
        out["pred_idx"] = pred_idx
        out["pred_class"] = pred_class

        # Clean duplicates and ordering if (date,symbol) exist
        if {"date","symbol"}.issubset(out.columns):
            out = (
                out.dropna(subset=["date"])
                   .sort_values(["symbol","date","pred_idx"], kind="mergesort")
                   .drop_duplicates(["symbol","date"], keep="last")
                   .reset_index(drop=True)
            )
    else:
        # Fallback if we don't have an index file
        out = pd.DataFrame({
            "row_id": np.arange(len(pred_idx)),
            "pred_idx": pred_idx,
            "pred_class": pred_class
        })

    
    # Save where phase_3 step 2 expects it
    
    out_path = ROOT / "phase_3" / "predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("[OK] wrote", out_path)

if __name__ == "__main__":
    main()
