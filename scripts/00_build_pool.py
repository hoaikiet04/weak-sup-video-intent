#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an unlabeled text pool from MSR-VTT captions (text-only),
optionally add simple synthetic queries via templates, then export a CSV for manual gold labeling.
Requires: datasets, pandas
"""

import argparse, os, random
from pathlib import Path
import pandas as pd

def load_msrvtt(config="train_9k"):
    from datasets import load_dataset
    print(f"  -> using MSR-VTT config = {config}")
    ds = load_dataset("friedrichor/MSR-VTT", config, split="train")
    
    texts = []
    for r in ds:
        cap = r.get("caption", "")
        if isinstance(cap, list):   # nếu caption là list thì nối từng phần tử
            texts.extend([c for c in cap if isinstance(c, str)])
        elif isinstance(cap, str):
            texts.append(cap)
    return texts

def gen_synthetic(n=600):
    # Simple template-based synthetic queries (EN + VN)
    music = ["lyrics {song}", "{song} official mv", "karaoke {song}", "lofi chill", "official audio {song}"]
    howto = ["how to fix {thing}", "tutorial {thing}", "hướng dẫn {thing}", "cách {thing}"]
    sports = ["{team1} vs {team2} highlights", "{team1} vs {team2} full match", "world cup highlights", "premier league live"]
    news = ["breaking news {topic}", "thời sự 19h hôm nay", "bản tin {topic}"]
    kis = ["{series} tập {ep} vietsub", "{series} trailer", "{series} season {season} episode {ep}"]
    review = ["review {product}", "unboxing {product}", "so sánh {product} vs {product2}", "giá {product}"]

    SONGS = ["Hãy Trao Cho Anh", "Drivers License", "Shape of You", "Đom Đóm"]
    THINGS = ["wifi windows 11", "máy in không nhận lệnh", "latte art", "react build error"]
    TEAMS = [("MU","Liverpool"), ("Lakers","Celtics"), ("Vietnam","Thailand")]
    TOPICS = ["bão số 5", "wildfires", "lạm phát"]
    SERIES = ["Conan", "One Piece", "Friends"]
    PRODS = ["iPhone 15", "Galaxy S23", "PS5", "air purifier"]

    out = []
    for _ in range(max(1, n//6)):
        out += [random.choice(music).format(song=random.choice(SONGS))]
        out += [random.choice(howto).format(thing=random.choice(THINGS))]
        t1,t2 = random.choice(TEAMS)
        out += [random.choice(sports).format(team1=t1, team2=t2)]
        out += [random.choice(news).format(topic=random.choice(TOPICS))]
        out += [random.choice(kis).format(series=random.choice(SERIES), ep=random.randint(1,200), season=random.randint(1,10))]
        out += [random.choice(review).format(product=random.choice(PRODS), product2=random.choice(PRODS))]
    # Dedup
    out = list(dict.fromkeys(out))
    return out[:n]

def basic_filter(texts, min_len=3, max_len=80):
    cleaned = []
    for t in texts:
        t = (t or "").strip()
        if not t: 
            continue
        L = len(t.split())
        if L < min_len or L > max_len:
            continue
        cleaned.append(t)
    return cleaned

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_unlabeled", default="data/processed/unlabeled_pool.csv")
    ap.add_argument("--export_gold_template", default="data/processed/gold_label_template.csv")
    ap.add_argument("--synthetic", type=int, default=600, help="how many synthetic queries to add")
    ap.add_argument("--sample", type=int, default=5000, help="max pool size")
    ap.add_argument("--msrvtt_config", type=str, default="train_9k",
                    choices=["train_9k","train_7k","test_1k"],
                    help="Which MSR-VTT config to use")
    args = ap.parse_args()

    os.makedirs(Path(args.output_unlabeled).parent, exist_ok=True)

    print("[1/3] Loading MSR-VTT captions (text-only)...")
    msr_texts = load_msrvtt(args.msrvtt_config)
    print(f"  -> got {len(msr_texts)} captions")

    print("[2/3] Adding simple synthetic queries...")
    syn = gen_synthetic(args.synthetic)
    print(f"  -> added {len(syn)} synthetic")

    pool = list(dict.fromkeys(msr_texts + syn))  # dedupe, preserve order
    pool = basic_filter(pool, 3, 80)
    if len(pool) > args.sample:
        random.seed(42)
        pool = random.sample(pool, args.sample)

    pd.DataFrame({"text": pool}).to_csv(args.output_unlabeled, index=False, encoding="utf-8")
    print(f"[✓] Saved unlabeled pool to {args.output_unlabeled} (n={len(pool)})")

    gold = random.sample(pool, min(300, len(pool)))
    pd.DataFrame({"text": gold, "label": ""}).to_csv(args.export_gold_template, index=False, encoding="utf-8")
    print(f"[✓] Exported gold template to {args.export_gold_template} (fill 'label' with one of: KIS, How-to, Music, News, Sports, Review)")

if __name__ == "__main__":
    main()
