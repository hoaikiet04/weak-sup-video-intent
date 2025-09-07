#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an unlabeled text pool (MSR-VTT captions + synthetic and/or TVR queries),
then export a GOLD template with configurable size (default 600).

Examples:
  ONLINE:
    python scripts/01_build_pool_and_export_gold_template.py \
      --output_unlabeled data/unlabeled_pool.csv \
      --export_gold_template data/gold_label_template.csv \
      --synthetic 800 --sample 8000 --gold_size 600

  OFFLINE (synthetic only):
    python scripts/01_build_pool_and_export_gold_template.py \
      --offline_only --synthetic 4000 --sample 8000 --gold_size 600
"""
import argparse, os, random, json
import pandas as pd

def load_msrvtt():
    from datasets import load_dataset
    ds = load_dataset("friedrichor/MSR-VTT", split="train")
    return [r.get("caption","") for r in ds if r.get("caption")]

def load_tvr_queries(json_path: str):
    texts = []
    if json_path.endswith(".jsonl"):
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "query" in obj and obj["query"]:
                        texts.append(str(obj["query"]))
                except Exception:
                    continue
    else:
        data = json.load(open(json_path, "r", encoding="utf-8"))
        if isinstance(data, list):
            for obj in data:
                q = obj.get("query")
                if q:
                    texts.append(str(q))
    return texts

def gen_synthetic(n=800):
    import random
    music = ["lyrics {song}", "{song} official mv", "karaoke {song}", "lofi chill", "official audio {song}"]
    howto = ["how to fix {thing}", "tutorial {thing}", "hướng dẫn {thing}", "cách {thing}"]
    sports = ["{team1} vs {team2} highlights", "{team1} vs {team2} full match", "world cup highlights", "premier league live"]
    news = ["breaking news {topic}", "thời sự 19h hôm nay", "bản tin {topic}"]
    kis = ["{series} tập {ep} vietsub", "{series} trailer", "{series} season {season} episode {ep}"]
    review = ["review {product}", "unboxing {product}", "so sánh {product} vs {product2}", "giá {product}"]

    SONGS = ["Hãy Trao Cho Anh", "Drivers License", "Shape of You", "Đom Đóm", "Haru Haru", "Nơi Này Có Anh"]
    THINGS = ["wifi windows 11", "máy in không nhận lệnh", "latte art", "react build error", "node not found"]
    TEAMS = [("MU","Liverpool"), ("Lakers","Celtics"), ("Vietnam","Thailand"), ("Real Madrid","Barcelona")]
    TOPICS = ["bão số 5", "wildfires", "lạm phát", "earthquake"]
    SERIES = ["Conan", "One Piece", "Friends", "Stranger Things"]
    PRODS = ["iPhone 15", "Galaxy S23", "PS5", "air purifier", "Kindle Paperwhite"]

    out = []
    for _ in range(max(1, n//6)):
        out += [random.choice(music).format(song=random.choice(SONGS))]
        out += [random.choice(howto).format(thing=random.choice(THINGS))]
        t1,t2 = random.choice(TEAMS)
        out += [random.choice(sports).format(team1=t1, team2=t2)]
        out += [random.choice(news).format(topic=random.choice(TOPICS))]
        out += [random.choice(kis).format(series=random.choice(SERIES), ep=random.randint(1,200), season=random.randint(1,10))]
        out += [random.choice(review).format(product=random.choice(PRODS), product2=random.choice(PRODS))]
    out = list(dict.fromkeys(out))
    return out[:n]

def basic_filter(texts, min_len=3, max_len=80):
    cleaned = []
    for t in texts:
        t = (t or "").strip()
        if not t: continue
        L = len(t.split())
        if L < min_len or L > max_len: continue
        cleaned.append(t)
    return cleaned

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_unlabeled", default="data/unlabeled_pool.csv")
    ap.add_argument("--export_gold_template", default="data/gold_label_template.csv")
    ap.add_argument("--synthetic", type=int, default=800, help="how many synthetic queries to add")
    ap.add_argument("--sample", type=int, default=8000, help="max pool size after dedupe/filter")
    ap.add_argument("--offline_only", action="store_true", help="skip dataset download, use synthetic only")
    ap.add_argument("--tvr_json", type=str, default="", help="optional: path to TVR JSON/JSONL containing 'query'")
    ap.add_argument("--gold_size", type=int, default=600, help="size of gold template to export")
    args = ap.parse_args()

    os.makedirs("data", exist_ok=True)
    pool = []

    if not args.offline_only:
        print("[1] Loading MSR-VTT captions (text-only) via datasets...")
        try:
            msr_texts = load_msrvtt()
            print(f"    -> {len(msr_texts)} captions")
            pool += msr_texts
        except Exception as e:
            print("    !! MSR-VTT loading failed, continue without it.\n       Reason:", e)

    if args.tvr_json:
        print("[2] Loading TVR queries from:", args.tvr_json)
        try:
            tv_texts = load_tvr_queries(args.tvr_json)
            print(f"    -> {len(tv_texts)} queries")
            pool += tv_texts
        except Exception as e:
            print("    !! TVR loading failed, continue without it.\n       Reason:", e)

    print("[3] Adding synthetic queries...")
    syn = gen_synthetic(args.synthetic)
    print(f"    -> {len(syn)} synthetic")
    pool += syn

    pool = list(dict.fromkeys(pool))
    pool = basic_filter(pool, 3, 80)
    if len(pool) > args.sample:
        import random; random.seed(42)
        pool = random.sample(pool, args.sample)

    import pandas as pd, random
    pd.DataFrame({"text": pool}).to_csv(args.output_unlabeled, index=False, encoding="utf-8")
    print(f"[✓] Saved unlabeled pool to {args.output_unlabeled} (n={len(pool)})")

    gold_n = min(args.gold_size, len(pool))
    gold = random.sample(pool, gold_n)
    pd.DataFrame({"text": gold, "label": ""}).to_csv(args.export_gold_template, index=False, encoding="utf-8")
    print(f"[✓] Exported gold template ({gold_n} rows) to {args.export_gold_template}\n    Fill 'label' with one of: KIS, How-to, Music, News, Sports, Review")

if __name__ == "__main__":
    main()
