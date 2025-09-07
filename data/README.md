# Baseline Scaffold — Video Query Intent (Weak Supervision)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 1) Build unlabeled pool + export gold template

```bash
python scripts/01_build_pool_and_export_gold_template.py   --output_unlabeled data/unlabeled_pool.csv   --export_gold_template data/gold_label_template.csv   --synthetic 600 --sample 5000
```

> Fill labels in `data/gold_label_template.csv` and save as `data/gold_test.csv`.

## 2) Run baselines

```bash
python scripts/02_run_baselines.py --gold data/gold_test.csv --outdir outputs
```

Artifacts: metrics*\*.json, predictions*_.csv, confusion*matrix*_.png
