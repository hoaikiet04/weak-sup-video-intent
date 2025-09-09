# run_label_model.py
import os, pandas as pd, numpy as np
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel
from snorkel_setup import ABSTAIN, LABELS, L2I, I2L
from lfs_text import LFS
from llm_labeler_hf import hf_zero_shot_votes

UNLAB = "data/unlabeled_pool.csv"
OUTDIR = "outputs_ws"
os.makedirs(OUTDIR, exist_ok=True)

# 4.1) Load data
df = pd.read_csv(UNLAB)
df = df.dropna(subset=["text"]).reset_index(drop=True)

# 4.2) Apply LFs
applier = PandasLFApplier(LFS)
L = applier.apply(df=df)   # shape: [N, num_LFs], values in {ABSTAIN, 0..K-1}

# 4.3) Add LLM-labeler votes for ~20%
votes = hf_zero_shot_votes(df["text"].tolist(), max_n=int(0.2*len(df)))
# convert to a column vector
import numpy as np
LLM_col = np.array([votes.get(i, ABSTAIN) for i in range(len(df))]).reshape(-1,1)
# stack: [LFs ... , LLM]
L_all = np.hstack([L, LLM_col])

# 4.4) Train LabelModel
label_model = LabelModel(cardinality=len(LABELS), verbose=True)
label_model.fit(L_all, n_epochs=500, log_freq=50, seed=42, lr=1e-2)

# 4.5) Get probabilistic labels & hard labels
Y_prob = label_model.predict_proba(L_all)         # [N, K], each row sums to 1
Y_hat  = Y_prob.argmax(axis=1)                    # hard labels (argmax)
conf   = Y_prob.max(axis=1)                       # confidence

out = df.copy()
out["ws_label_id"] = Y_hat
out["ws_label"]    = [I2L[i] for i in Y_hat]
out["ws_conf"]     = conf.round(4)
out.to_csv(f"{OUTDIR}/weak_labels_all.csv", index=False, encoding="utf-8")

# 4.6) Chọn subset tin cậy để train baseline discriminative model
MASK = conf >= 0.75
subset = out[MASK][["text","ws_label"]].rename(columns={"ws_label":"label"})
subset.to_csv(f"{OUTDIR}/weak_train_0p75.csv", index=False, encoding="utf-8")

# 4.7) Kiểm tra phân phối lớp
dist = subset["label"].value_counts().reindex(LABELS, fill_value=0)
print("Class distribution (weak_train, >=0.75):\n", dist)
dist.to_csv(f"{OUTDIR}/class_dist_weak_train.csv")
