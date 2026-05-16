"""
Generate h30 and h40 truncated training data from existing full training data.
"""
import json
import re
import os

OUTPUT_DIR = "/workspace/jiangzhuosong/GRLM_0/LlamaFactory/data/grlm_in_domain"
NEW_CAPS = [30, 40]

def truncate_sample(sample, max_hist):
    input_text = sample["input"]
    pattern = r"Item text ID: \[.*?\] Title: .*?\.\n"
    items = re.findall(pattern, input_text)
    if len(items) <= max_hist:
        return sample
    truncated_items = items[-max_hist:]
    new_sample = sample.copy()
    new_sample["input"] = "".join(truncated_items)
    return new_sample

for period in range(4):
    full_path = f"{OUTPUT_DIR}/amazon_books_cl_D{period}_train.json"
    print(f"Loading {full_path}...")
    with open(full_path, 'r') as f:
        data = json.load(f)
    print(f"  {len(data)} samples")

    for cap in NEW_CAPS:
        truncated = [truncate_sample(s, cap) for s in data]
        out_path = f"{OUTPUT_DIR}/amazon_books_cl_D{period}_train_h{cap}.json"
        with open(out_path, 'w') as f:
            json.dump(truncated, f, ensure_ascii=False, indent=2)
        print(f"  Saved h{cap}: {out_path}")

print("Done!")
