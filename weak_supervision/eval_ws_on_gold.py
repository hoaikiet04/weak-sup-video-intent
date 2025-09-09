# eval_ws_on_gold.py
import pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from snorkel_setup import LABELS

gold = pd.read_csv("data/processed/gold_test.csv")             # text,label
pred = pd.read_csv("outputs_ws/weak_labels_all.csv")            # text, ws_label, ws_conf

df = gold.merge(pred[["text","ws_label"]], on="text", how="inner")
acc = accuracy_score(df["label"], df["ws_label"])
macro = f1_score(df["label"], df["ws_label"], average="macro", labels=LABELS)
print("WS Acc:", acc, "Macro-F1:", macro)
print(classification_report(df["label"], df["ws_label"], labels=LABELS, zero_division=0))
