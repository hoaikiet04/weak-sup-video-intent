# llm_labeler_hf.py
import random
import pandas as pd
from snorkel_setup import ABSTAIN, LABELS, L2I
from transformers import pipeline

def hf_zero_shot_votes(texts, label_names=LABELS, model="joeddav/xlm-roberta-large-xnli", top1_threshold=0.65, max_n=None, seed=42):
    clf = pipeline("zero-shot-classification", model=model)
    idxs = list(range(len(texts)))
    random.Random(seed).shuffle(idxs)
    if max_n:
        idxs = idxs[:max_n]
    votes = {i: ABSTAIN for i in range(len(texts))}
    for i in idxs:
        t = str(texts[i])
        r = clf(t, candidate_labels=label_names, multi_label=False)
        lab, p = r["labels"][0], float(r["scores"][0])
        votes[i] = L2I[lab] if p >= top1_threshold else ABSTAIN
    return votes  # dict: row_index -> label_id (or ABSTAIN)
