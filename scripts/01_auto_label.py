#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-label gold template using keyword LFs.
"""

import pandas as pd
import sys, os

# đảm bảo import được lfs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lfs.keyword_lfs import predict_rules_only

def main():
    in_file = "data/processed/gold_label_template.csv"
    out_file = "data/processed/gold_test_autolabeled.csv"

    df = pd.read_csv(in_file)
    df["auto_label"] = df["text"].apply(predict_rules_only)
    df.to_csv(out_file, index=False, encoding="utf-8")

    print(f"[✓] Saved auto-labeled file to {out_file}")

if __name__ == "__main__":
    main()
