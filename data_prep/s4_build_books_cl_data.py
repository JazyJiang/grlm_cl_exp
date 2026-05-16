"""
s4_build_books_cl_data.py
Generate Continual Learning SFT data for Books dataset.

Protocol:
- D0: train from scratch, eval on D1
- D1: fine-tune from D0 ckpt, eval on D2
- D2: fine-tune from D1 ckpt, eval on D3
- D3: fine-tune from D2 ckpt, eval on D4

Training format:
- D0: input = first item in D0 sequence, output = rest of D0 sequence
- D1+: input = accumulated history (D0..D(t-1)), output = Dt items

Eval format:
- Users in both Dt and Dt+1
- prompt = accumulated history (D0..Dt)
- targets = all Dt+1 items
"""
import os
import json
import csv
import re
from collections import defaultdict
from tqdm import tqdm

MEMORY_DEV = "/workspace/jiangzhuosong/memory_dev/data"
PREFIX = "Books_5_2016-10-2018-11"
ID2META_FILE = "/workspace/jiangzhuosong/GRLM_0/in_domain/books/sum_data/books_id2meta.json"
OUTPUT_DIR = "/workspace/jiangzhuosong/GRLM_0/LlamaFactory/data/grlm_in_domain"

INSTRUCTION = (
    "Based on the user's historical product interaction sequence, predict the next product's "
    "characteristic words. \nEach product is represented by exactly 5 characteristic words "
    "enclosed in square brackets []. The historical sequence shows the user's interaction pattern.\n"
)

HISTORY_CAPS = [2, 5, 10, 20]


def load_id2meta():
    print(f"Loading id2meta from {ID2META_FILE}...")
    with open(ID2META_FILE, 'r', encoding='utf-8') as f:
        id2meta = json.load(f)
    print(f"  Loaded {len(id2meta)} items")
    return id2meta


def load_period_data():
    """Load all D0-D4 CSVs, return per-user per-period interactions."""
    # period_users[period][user_id] = [(item_id_1based, timestamp), ...]
    period_users = [defaultdict(list) for _ in range(5)]

    for d in range(5):
        csv_path = f"{MEMORY_DEV}/D{d}/{PREFIX}.csv"
        print(f"  Loading {csv_path}...")
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                user_id = row[0]
                item_id = int(row[4])  # 0-based
                timestamp = int(row[10])
                period_users[d][user_id].append((item_id + 1, timestamp))  # 1-based

    # Deduplicate and sort by timestamp within each period
    for d in range(5):
        for user_id in period_users[d]:
            interactions = period_users[d][user_id]
            interactions.sort(key=lambda x: x[0])  # sort by item_id for stable dedup
            interactions.sort(key=lambda x: x[1])  # then sort by timestamp
            # Deduplicate consecutive same items
            deduped = []
            for item_id, ts in interactions:
                if not deduped or deduped[-1][0] != item_id:
                    deduped.append((item_id, ts))
            period_users[d][user_id] = deduped

    for d in range(5):
        print(f"  D{d}: {len(period_users[d])} users")

    return period_users


def format_item(item_id_str, id2meta):
    """Format a single item as 'Item text ID: [w1, w2, w3, w4, w5] Title: xxx.\n'"""
    meta = id2meta.get(item_id_str)
    if meta is None:
        return None
    summary_words = meta.get('summary_words', [])
    if not summary_words or len(summary_words) < 5:
        return None
    if "" in summary_words:
        return None
    words = [w.replace("[", "").replace("]", "") for w in summary_words[:5]]
    if any(not w.strip() for w in words):
        return None
    title = meta.get('title', 'None')
    if not title:
        title = 'None'
    return f"Item text ID: [{', '.join(words)}] Title: {title}.\n"


def build_train_d0(period_users, id2meta):
    """Build training data for D0: sequence within D0, input=first item, output=rest."""
    samples = []
    skipped = 0

    users_d0 = period_users[0]
    for user_id, interactions in tqdm(users_d0.items(), desc="D0 train"):
        if len(interactions) < 2:
            skipped += 1
            continue

        item_ids = [str(iid) for iid, _ in interactions]

        # Format all items
        formatted = []
        valid = True
        for iid_str in item_ids:
            fmt = format_item(iid_str, id2meta)
            if fmt is None:
                valid = False
                break
            formatted.append(fmt)

        if not valid or len(formatted) < 2:
            skipped += 1
            continue

        sample = {
            "instruction": INSTRUCTION,
            "input": formatted[0],
            "output": "".join(formatted[1:]),
            "metadata": {
                "user_id": user_id,
                "period": 0,
                "history_len": 1,
                "target_len": len(formatted) - 1,
            }
        }
        samples.append(sample)

    print(f"  D0 train: {len(samples)} samples, {skipped} skipped")
    return samples


def build_train_dt(period_users, id2meta, period):
    """Build training data for period t (t>=1): input=D0..D(t-1), output=Dt items."""
    samples = []
    skipped = 0

    users_dt = period_users[period]
    for user_id, interactions_dt in tqdm(users_dt.items(), desc=f"D{period} train"):
        if len(interactions_dt) < 1:
            skipped += 1
            continue

        # Accumulate history from D0..D(t-1)
        history_items = []
        for prev_d in range(period):
            if user_id in period_users[prev_d]:
                for iid, _ in period_users[prev_d][user_id]:
                    history_items.append(str(iid))

        # Target items in Dt
        target_items = [str(iid) for iid, _ in interactions_dt]

        # Users with no prior history but >=2 items in Dt: use D0 format
        if len(history_items) == 0:
            if len(target_items) < 2:
                skipped += 1
                continue
            formatted = []
            valid = True
            for iid_str in target_items:
                fmt = format_item(iid_str, id2meta)
                if fmt is None:
                    valid = False
                    break
                formatted.append(fmt)
            if not valid or len(formatted) < 2:
                skipped += 1
                continue
            sample = {
                "instruction": INSTRUCTION,
                "input": formatted[0],
                "output": "".join(formatted[1:]),
                "metadata": {
                    "user_id": user_id,
                    "period": period,
                    "history_len": 1,
                    "target_len": len(formatted) - 1,
                }
            }
            samples.append(sample)
            continue

        # Format history (input) and targets (output)
        hist_formatted = []
        valid = True
        for iid_str in history_items:
            fmt = format_item(iid_str, id2meta)
            if fmt is None:
                valid = False
                break
            hist_formatted.append(fmt)

        if not valid or len(hist_formatted) == 0:
            skipped += 1
            continue

        tgt_formatted = []
        for iid_str in target_items:
            fmt = format_item(iid_str, id2meta)
            if fmt is None:
                valid = False
                break
            tgt_formatted.append(fmt)

        if not valid or len(tgt_formatted) == 0:
            skipped += 1
            continue

        sample = {
            "instruction": INSTRUCTION,
            "input": "".join(hist_formatted),
            "output": "".join(tgt_formatted),
            "metadata": {
                "user_id": user_id,
                "period": period,
                "history_len": len(hist_formatted),
                "target_len": len(tgt_formatted),
            }
        }
        samples.append(sample)

    print(f"  D{period} train: {len(samples)} samples, {skipped} skipped")
    return samples


def build_eval_dt(period_users, id2meta, period):
    """Build eval data: users in both Dt and D(t+1), prompt=D0..Dt, targets=D(t+1) items."""
    eval_data = []
    skipped = 0

    users_dt = set(period_users[period].keys())
    users_dt1 = set(period_users[period + 1].keys())
    overlap_users = users_dt & users_dt1

    for user_id in tqdm(overlap_users, desc=f"D{period}->D{period+1} eval"):
        # Accumulate history D0..Dt
        history_items = []
        for d in range(period + 1):
            if user_id in period_users[d]:
                for iid, _ in period_users[d][user_id]:
                    history_items.append(str(iid))

        if len(history_items) == 0:
            skipped += 1
            continue

        # Format prompt
        prompt_parts = []
        valid = True
        for iid_str in history_items:
            fmt = format_item(iid_str, id2meta)
            if fmt is None:
                valid = False
                break
            prompt_parts.append(fmt)

        if not valid or len(prompt_parts) == 0:
            skipped += 1
            continue

        # Targets in D(t+1)
        target_interactions = period_users[period + 1][user_id]
        target_item_ids = [str(iid) for iid, _ in target_interactions]

        # Get target TIDs
        target_tids = []
        valid_targets = []
        for iid_str in target_item_ids:
            meta = id2meta.get(iid_str)
            if meta and meta.get('summary_words') and len(meta['summary_words']) >= 5:
                words = [w.replace("[", "").replace("]", "") for w in meta['summary_words'][:5]]
                if all(w.strip() for w in words):
                    target_tids.append(words)
                    valid_targets.append(iid_str)

        if len(valid_targets) == 0:
            skipped += 1
            continue

        eval_sample = {
            "user_id": user_id,
            "prompt": "".join(prompt_parts),
            "target_item_ids": valid_targets,
            "target_tids": target_tids,
            "history_len": len(prompt_parts),
        }
        eval_data.append(eval_sample)

    print(f"  Eval D{period}->D{period+1}: {len(eval_data)} users, {skipped} skipped")
    return eval_data


def truncate_train_sample(sample, max_hist):
    """Truncate input history to last max_hist items."""
    input_text = sample["input"]
    pattern = r"Item text ID: \[.*?\] Title: .*?\.\n"
    items = re.findall(pattern, input_text)

    if len(items) <= max_hist:
        return sample

    truncated_items = items[-max_hist:]
    new_sample = sample.copy()
    new_sample["input"] = "".join(truncated_items)
    new_sample["metadata"] = sample["metadata"].copy()
    new_sample["metadata"]["history_len"] = max_hist
    new_sample["metadata"]["truncated_from"] = len(items)
    return new_sample


def truncate_eval_sample(sample, max_hist):
    """Truncate eval prompt to last max_hist items."""
    prompt = sample["prompt"]
    pattern = r"Item text ID: \[.*?\] Title: .*?\.\n"
    items = re.findall(pattern, prompt)

    if len(items) <= max_hist:
        return sample

    truncated_items = items[-max_hist:]
    new_sample = sample.copy()
    new_sample["prompt"] = "".join(truncated_items)
    new_sample["history_len"] = max_hist
    return new_sample


def save_json(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(data)} entries to {filepath}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    id2meta = load_id2meta()
    print("Loading period data from CSVs...")
    period_users = load_period_data()

    # === Training data ===
    print("\n=== Building Training Data ===")
    train_data = {}

    # D0
    train_data[0] = build_train_d0(period_users, id2meta)

    # D1-D3
    for t in range(1, 4):
        train_data[t] = build_train_dt(period_users, id2meta, t)

    # Save full training data
    for t in range(4):
        # Full version (for LlamaFactory: instruction, input, output)
        sft_samples = [{
            "instruction": s["instruction"],
            "input": s["input"],
            "output": s["output"]
        } for s in train_data[t]]
        save_json(sft_samples, f"{OUTPUT_DIR}/amazon_books_cl_D{t}_train.json")

        # Truncated versions
        for cap in HISTORY_CAPS:
            truncated = [truncate_train_sample(s, cap) for s in train_data[t]]
            sft_truncated = [{
                "instruction": s["instruction"],
                "input": s["input"],
                "output": s["output"]
            } for s in truncated]
            save_json(sft_truncated, f"{OUTPUT_DIR}/amazon_books_cl_D{t}_train_h{cap}.json")

    # === Eval data ===
    print("\n=== Building Eval Data ===")
    for t in range(4):
        eval_data = build_eval_dt(period_users, id2meta, t)
        save_json(eval_data, f"{OUTPUT_DIR}/amazon_books_cl_D{t}_eval.json")

        # Truncated eval versions
        for cap in HISTORY_CAPS:
            truncated_eval = [truncate_eval_sample(s, cap) for s in eval_data]
            save_json(truncated_eval, f"{OUTPUT_DIR}/amazon_books_cl_D{t}_eval_h{cap}.json")

    # === Summary ===
    print("\n=== Summary ===")
    for t in range(4):
        print(f"D{t} train: {len(train_data[t])} samples")
    for t in range(4):
        eval_file = f"{OUTPUT_DIR}/amazon_books_cl_D{t}_eval.json"
        if os.path.exists(eval_file):
            with open(eval_file) as f:
                eval_d = json.load(f)
            print(f"D{t}->D{t+1} eval: {len(eval_d)} users")


if __name__ == "__main__":
    main()
