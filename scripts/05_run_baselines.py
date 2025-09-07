#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run two baselines on a GOLD set (manually labeled): 
  (1) Zero-shot with facebook/bart-large-mnli
  (2) Rules-only with keyword LFs
Outputs: per-sample predictions, metrics JSON, confusion matrix PNG.
"""
import argparse, os, json
import pandas as pd
from typing import List
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

LABELS = ["KIS", "How-to", "Music", "News", "Sports", "Review"]

def load_gold(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Gold file must have columns: 'text', 'label'")
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    return df

def predict_zero_shot(texts: List[str], labels: List[str]) -> List[str]:
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    preds = []
    for t in texts:
        res = clf(str(t), candidate_labels=labels, multi_label=False)
        preds.append(res["labels"][0])
    return preds

def predict_rules_only(texts: List[str]) -> List[str]:
    from lfs.keyword_lfs import predict_rules_only
    out = []
    for t in texts:
        lab = predict_rules_only(str(t))
        out.append(lab if lab is not None else "UNK")
    return out

def metrics_and_cm(y_true: List[str], y_pred: List[str], title: str, outdir: str, labels: List[str]):
    import pandas as pd, json, os
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df.to_csv(os.path.join(outdir, f"predictions_{title}.csv"), index=False, encoding="utf-8")

    mask = df["y_pred"].isin(labels)
    acc = accuracy_score(df["y_true"][mask], df["y_pred"][mask]) if mask.any() else 0.0
    macro_f1 = f1_score(df["y_true"][mask], df["y_pred"][mask], average="macro") if mask.any() else 0.0
    report = classification_report(df["y_true"][mask], df["y_pred"][mask], labels=labels, target_names=labels, output_dict=True)
    with open(os.path.join(outdir, f"metrics_{title}.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "macro_f1": macro_f1, "report": report}, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(df["y_true"][mask], df["y_pred"][mask], labels=labels)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix — {title}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f"confusion_matrix_{title}.png"), dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gold_test.csv", help="CSV with columns: text,label")
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    df = load_gold(args.gold)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].astype(str).tolist()

    print("[1/2] Running zero-shot (facebook/bart-large-mnli)...")
    zs_pred = predict_zero_shot(texts, LABELS)
    metrics_and_cm(y_true, zs_pred, "zero_shot_bart_mnli", args.outdir, LABELS)

    print("[2/2] Running rules-only (keyword LFs)...")
    rules_pred = predict_rules_only(texts)
    metrics_and_cm(y_true, rules_pred, "rules_only", args.outdir, LABELS)

    print("[✓] Done. See outputs directory.")

if __name__ == "__main__":
    main()
