#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3 — Baseline submission helper
- Reads a GOLD file (CSV with columns: text,label)
- Runs Zero-shot (optional) and Rules-only baselines
- Saves metrics (Accuracy, Macro-F1 + full classification_report JSON)
- Saves confusion matrix PNGs
- Saves predictions_*.csv
- Exports Top-20 frequent errors per baseline

Usage examples:
  # Run both (need internet + transformers for zero-shot)
  python step3_submit_baselines.py --gold data/processed/gold_test.csv --outdir outputs_step3 --zs_model joeddav/xlm-roberta-large-xnli

  # Offline: only rules baseline
  python step3_submit_baselines.py --gold data/processed/gold_test.csv --outdir outputs_step3 --skip_zero_shot
"""
import argparse, os, json, sys
import pandas as pd
import numpy as np
from typing import List
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

LABELS = ["KIS","How-to","Music","News","Sports","Review","Entertainment","Other"]

def load_gold(path: str, labels: List[str]) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Gold CSV must have columns: 'text' and 'label'")
    df = df.dropna(subset=["text","label"]).copy()
    # keep only rows with valid labels
    df = df[df["label"].isin(labels)].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows with valid labels after filtering. Check your taxonomy/labels.")
    return df

def predict_zero_shot(texts: List[str], labels: List[str], model_name: str) -> List[str]:
    try:
        from transformers import pipeline
    except Exception as e:
        print("[!] transformers not available. Install 'transformers' to run zero-shot.", file=sys.stderr)
        raise
    clf = pipeline("zero-shot-classification", model=model_name)
    preds = []
    for t in texts:
        r = clf(str(t), candidate_labels=labels, multi_label=False)
        preds.append(r["labels"][0])
    return preds

def predict_rules_only(texts: List[str]) -> List[str]:
    try:
        from lfs.keyword_lfs_8labels import predict_rules_only as pr
        return [pr(str(t), include_other=True) for t in texts]
    except Exception:
        # Fallback: very light rule (minimal)
        import re
        RX = {
            "Music": re.compile(r"\blyrics?\b|karaoke|\bmv\b|official\s+audio|\blofi\b|\bnhạc\b", re.I),
            "How-to": re.compile(r"\bhow\s*to\b|tutorial|hướng\s*dẫn|\bcách\b|\bfix\b|\bsửa\b|install|setup|cài\s*đặt|khắc\s*phục", re.I),
            "Sports": re.compile(r"\b(highlight|highlights)\b|\bfull\s*match\b|\bvs\b|\btrận\b|world\s*cup|premier\s*league|\blive\b", re.I),
            "News": re.compile(r"\bnews\b|breaking|thời\s*sự|bản\s*tin|tin\s*tức", re.I),
            "Review": re.compile(r"\breview\b|đánh\s*giá|so\s*sánh|\bunboxing\b|\bgiá\b", re.I),
            "KIS": re.compile(r"trailer|official\s*trailer|episode|\bep\.?\b|season|\btập\s*\d+\b|vietsub|full\s*movie", re.I),
            "Entertainment": re.compile(r"\bvlog\b|reaction|pranks?|challenge|funny|memes?|skit|talk\s*show|podcast|live\s*stream|livestream|streaming|asmr|parody", re.I),
        }
        out = []
        for t in texts:
            t = str(t)
            hits = [lab for lab, rx in RX.items() if rx.search(t)]
            out.append(hits[0] if hits else "Other")
        return out

def save_metrics_and_cm(y_true: List[str], y_pred: List[str], labels: List[str], title: str, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0)

    with open(os.path.join(outdir, f"metrics_{title}.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "macro_f1": macro_f1, "report": report}, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix — {title}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    # counts overlay
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"confusion_matrix_{title}.png"), dpi=200)
    plt.close(fig)

def export_top_errors(df_pred: pd.DataFrame, title: str, outdir: str, topk: int = 20):
    """df_pred has columns: text, y_true, y_pred"""
    err = df_pred[df_pred["y_true"] != df_pred["y_pred"]].copy()
    if len(err) == 0:
        # still export empty files for consistency
        err.to_csv(os.path.join(outdir, f"errors_{title}.csv"), index=False, encoding="utf-8")
        pd.DataFrame(columns=["y_true","y_pred","count"]).to_csv(
            os.path.join(outdir, f"errors_pairs_{title}.csv"), index=False, encoding="utf-8"
        )
        return

    # Save all errors
    err.to_csv(os.path.join(outdir, f"errors_{title}.csv"), index=False, encoding="utf-8")

    # Top-K frequent confusion pairs
    pair_counts = err.groupby(["y_true","y_pred"]).size().reset_index(name="count")
    pair_counts = pair_counts.sort_values("count", ascending=False).head(topk)
    pair_counts.to_csv(os.path.join(outdir, f"errors_pairs_{title}.csv"), index=False, encoding="utf-8")

    # Create a small sample set with up to 3 examples per top pair
    rows = []
    for _, row in pair_counts.iterrows():
        yt, yp = row["y_true"], row["y_pred"]
        ex = err[(err["y_true"]==yt) & (err["y_pred"]==yp)].head(3)
        for _, e in ex.iterrows():
            rows.append({"y_true": yt, "y_pred": yp, "text": e["text"]})
    pd.DataFrame(rows).to_csv(os.path.join(outdir, f"errors_pairs_examples_{title}.csv"), index=False, encoding="utf-8")

def run_baseline(df: pd.DataFrame, labels: List[str], baseline_name: str, predictor, outdir: str):
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].astype(str).tolist()
    y_pred = predictor(texts)
    # Save predictions
    pred_df = pd.DataFrame({"text": texts, "y_true": y_true, "y_pred": y_pred})
    pred_df.to_csv(os.path.join(outdir, f"predictions_{baseline_name}.csv"), index=False, encoding="utf-8")
    # Metrics + CM
    save_metrics_and_cm(y_true, y_pred, labels, baseline_name, outdir)
    # Top errors
    export_top_errors(pred_df, baseline_name, outdir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="CSV with columns: text,label")
    ap.add_argument("--outdir", default="outputs_step3")
    ap.add_argument("--zs_model", default="joeddav/xlm-roberta-large-xnli", help="HuggingFace zero-shot model")
    ap.add_argument("--skip_zero_shot", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_gold(args.gold, LABELS)

    # Rules-only
    print("[1/2] Rules-only baseline ...")
    run_baseline(df, LABELS, "rules_only", predict_rules_only, args.outdir)

    # Zero-shot (optional)
    if not args.skip_zero_shot:
        try:
            print("[2/2] Zero-shot baseline ...", args.zs_model)
            predictor = lambda texts: predict_zero_shot(texts, LABELS, args.zs_model)
            run_baseline(df, LABELS, "zero_shot", predictor, args.outdir)
        except Exception as e:
            print("[!] Zero-shot failed or unavailable:", e)
            print("    -> Continue with Rules-only outputs only.")

    print("[✓] Done. Outputs in:", args.outdir)

if __name__ == "__main__":
    main()
