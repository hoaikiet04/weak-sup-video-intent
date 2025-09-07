#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run baselines on GOLD set for 8 labels:
  (1) Zero-shot (model configurable via --zs_model)
  (2) Rules-only (keyword_lfs_8labels)
"""
import argparse, os, json
import pandas as pd
from typing import List
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

LABELS = ["KIS","How-to","Music","News","Sports","Review","Entertainment","Other"]

def load_gold(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Gold file must have columns: 'text', 'label'")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    # allow "Other" now
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    return df

def predict_zero_shot(texts: List[str], labels: List[str], model_name: str) -> List[str]:
    from transformers import pipeline
    clf = pipeline("zero-shot-classification", model=model_name)
    preds = []
    for t in texts:
        r = clf(str(t), candidate_labels=labels, multi_label=False)
        preds.append(r["labels"][0])
    return preds

def predict_rules_only(texts: List[str]) -> List[str]:
    from lfs.keyword_lfs_8labels import predict_rules_only
    out = []
    for t in texts:
        out.append(predict_rules_only(str(t), include_other=True))
    return out

def metrics_and_cm(y_true: List[str], y_pred: List[str], title: str, outdir: str, labels: List[str]):
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df.to_csv(os.path.join(outdir, f"predictions_{title}.csv"), index=False, encoding="utf-8")

    acc = accuracy_score(df["y_true"], df["y_pred"])
    macro_f1 = f1_score(df["y_true"], df["y_pred"], average="macro", labels=labels)
    report = classification_report(df["y_true"], df["y_pred"], labels=labels, target_names=labels, output_dict=True, zero_division=0)
    with open(os.path.join(outdir, f"metrics_{title}.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "macro_f1": macro_f1, "report": report}, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(df["y_true"], df["y_pred"], labels=labels)
    fig = plt.figure(figsize=(7.5,6.5))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix — {title}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"confusion_matrix_{title}.png"), dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/processed/gold_test.csv", help="CSV with columns: text,label")
    ap.add_argument("--outdir", default="outputs_8labels")
    ap.add_argument("--zs_model", default="joeddav/xlm-roberta-large-xnli", help="Zero-shot model (multi-lingual recommended)")
    args = ap.parse_args()

    df = load_gold(args.gold)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].astype(str).tolist()

    print("[1/2] Zero-shot ...", args.zs_model)
    zs_pred = predict_zero_shot(texts, LABELS, args.zs_model)
    metrics_and_cm(y_true, zs_pred, "zero_shot", args.outdir, LABELS)

    print("[2/2] Rules-only ...")
    rules_pred = predict_rules_only(texts)
    metrics_and_cm(y_true, rules_pred, "rules_only", args.outdir, LABELS)

    print("[✓] Done. See folder:", args.outdir)

if __name__ == "__main__":
    main()
