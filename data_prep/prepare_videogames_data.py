"""
Build Video Games dataset for GRLM in_domain pipeline.
Creates sequential_data.txt and videogames.item.json from memory_dev processed data.
"""
import numpy as np
import json
import csv
import os
from collections import defaultdict

MEMORY_DEV = "/workspace/jiangzhuosong/memory_dev/data"
INFO_DIR = f"{MEMORY_DEV}/info"
PREFIX = "Video_Games_5_2012-10-2018-11"
OUTPUT_DIR = "/workspace/jiangzhuosong/GRLM_0/in_domain/videogames/raw_data"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("/workspace/jiangzhuosong/GRLM_0/in_domain/videogames/sum_data", exist_ok=True)

    # Load maps
    print("Loading maps...")
    item_map = np.load(f"{INFO_DIR}/{PREFIX}_item_map.npy", allow_pickle=True).item()
    title_map = np.load(f"{INFO_DIR}/{PREFIX}_title_map.npy", allow_pickle=True).item()
    desc_map = np.load(f"{INFO_DIR}/{PREFIX}_description_map.npy", allow_pickle=True).item()
    cat_map = np.load(f"{INFO_DIR}/{PREFIX}_category_map.npy", allow_pickle=True).item()

    id2item = item_map['id2item']
    n_items = len(id2item)
    print(f"Total items: {n_items}")

    # Build videogames.item.json (0-based -> 1-based)
    print("Building videogames.item.json...")
    item_json = {}
    for rec_id in range(n_items):
        title = title_map['recid2title'].get(rec_id, "")
        desc = desc_map['recid2description'].get(rec_id, "")
        cats = cat_map['recid2category'].get(rec_id, [])
        if isinstance(cats, list):
            cats_str = " > ".join(cats)
        else:
            cats_str = str(cats)
        item_json[str(rec_id + 1)] = {
            "title": title,
            "description": desc,
            "categories": cats_str
        }

    with open(f"{OUTPUT_DIR}/videogames.item.json", "w", encoding="utf-8") as f:
        json.dump(item_json, f, ensure_ascii=False, indent=2)
    print(f"  Written {len(item_json)} items to videogames.item.json")

    # Build sequential_data.txt from all D* CSVs
    print("Aggregating user interactions from CSVs...")
    users = defaultdict(list)
    for d in range(5):
        csv_path = f"{MEMORY_DEV}/D{d}/{PREFIX}.csv"
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping")
            continue
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                user_id = row[0]
                item_id = int(row[4])  # 0-based from 0_process.py
                timestamp = int(row[10])
                users[user_id].append((timestamp, item_id + 1))  # +1 for 1-based

    print(f"  Total users from CSVs: {len(users)}")

    # Sort by timestamp, deduplicate, write sequential_data.txt
    print("Writing sequential_data.txt...")
    user_id_counter = 0
    with open(f"{OUTPUT_DIR}/sequential_data.txt", "w") as f:
        for user_asin in sorted(users.keys()):
            interactions = users[user_asin]
            interactions.sort(key=lambda x: x[0])
            # Deduplicate consecutive same items (keep order)
            item_seq = []
            for _, item_id in interactions:
                if not item_seq or item_seq[-1] != item_id:
                    item_seq.append(item_id)
            if len(item_seq) >= 3:  # need at least 3 items for train/valid/test
                user_id_counter += 1
                line = f"{user_id_counter} " + " ".join(str(x) for x in item_seq)
                f.write(line + "\n")

    print(f"  Written {user_id_counter} users to sequential_data.txt")

    # Stats
    print("\nHistory length stats:")
    lengths = []
    with open(f"{OUTPUT_DIR}/sequential_data.txt") as f:
        for line in f:
            parts = line.strip().split()
            lengths.append(len(parts) - 1)
    lengths.sort()
    n = len(lengths)
    print(f"  Users: {n}")
    print(f"  Mean: {sum(lengths)/n:.1f}")
    print(f"  Median: {lengths[n//2]}")
    print(f"  p90: {lengths[int(0.9*n)]}")
    print(f"  p95: {lengths[int(0.95*n)]}")
    print(f"  Max: {lengths[-1]}")
    for t in [5, 10, 20, 30, 40, 50]:
        cnt = sum(1 for l in lengths if l >= t)
        print(f"  >= {t} items: {cnt} ({cnt/n*100:.1f}%)")

if __name__ == "__main__":
    main()
