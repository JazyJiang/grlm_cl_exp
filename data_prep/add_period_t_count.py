"""
Add period_t_count field to existing eval JSONs.
For eval D{t}->D{t+1}, period_t_count = user's interaction count in D{t}.
"""
import json
import csv
from collections import defaultdict

MEMORY_DEV = "/workspace/jiangzhuosong/memory_dev/data"
PREFIX = "Books_5_2016-10-2018-11"
EVAL_DIR = "/workspace/jiangzhuosong/GRLM_0/LlamaFactory/data/grlm_in_domain"

# Load per-period user interaction counts
period_user_counts = [defaultdict(int) for _ in range(5)]
for d in range(5):
    csv_path = f"{MEMORY_DEV}/D{d}/{PREFIX}.csv"
    print(f"Loading {csv_path}...")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        seen = set()
        for row in reader:
            user_id = row[0]
            item_id = int(row[4])
            key = (user_id, item_id)
            if key not in seen:
                seen.add(key)
                period_user_counts[d][user_id] += 1
    print(f"  D{d}: {len(period_user_counts[d])} users")

# Update eval JSONs
for t in range(4):
    eval_path = f"{EVAL_DIR}/amazon_books_cl_D{t}_eval.json"
    print(f"\nUpdating {eval_path}...")
    with open(eval_path, 'r') as f:
        eval_data = json.load(f)

    updated = 0
    for sample in eval_data:
        user_id = sample["user_id"]
        sample["period_t_count"] = period_user_counts[t].get(user_id, 0)
        updated += 1

    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, ensure_ascii=False)
    print(f"  Updated {updated} samples (mean period_t_count={sum(s['period_t_count'] for s in eval_data)/len(eval_data):.1f})")

print("\nDone!")
