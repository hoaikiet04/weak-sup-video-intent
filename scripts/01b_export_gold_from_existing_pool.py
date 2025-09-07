#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a gold label template from an existing unlabeled_pool.csv with configurable size (default 600).
"""
import argparse, pandas as pd, os, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/unlabeled_pool.csv")
    ap.add_argument("--out", default="data/gold_label_template.csv")
    ap.add_argument("--gold_size", type=int, default=600)
    args = ap.parse_args()

    df = pd.read_csv(args.pool)
    texts = df["text"].dropna().astype(str).tolist()
    n = min(len(texts), args.gold_size)
    random.seed(42)
    gold = random.sample(texts, n)
    out = pd.DataFrame({"text": gold, "label": ""})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[✓] Exported {n} rows to {args.out}.")

if __name__ == "__main__":
    main()
