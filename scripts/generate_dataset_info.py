"""
Generate dataset_info.json for LlamaFactory.
Creates entries for all CL periods x history caps, for any amazon_<dataset>_cl_*_train*.json files found.

Usage:
    python scripts/generate_dataset_info.py \
        --data_dir data/cl_sft \
        --output LlamaFactory/data/dataset_info.json
"""
import os
import re
import json
import argparse
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing CL SFT JSON files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for dataset_info.json')
    args = parser.parse_args()

    dataset_info = {}

    # Match any amazon_<dataset>_cl_*_train*.json
    pattern = re.compile(r'^amazon_(?P<dataset>[a-z0-9]+)_cl_(?P<period>D\d+)_train(?P<suffix>_h\d+)?\.json$')

    train_files = sorted(glob(os.path.join(args.data_dir, 'amazon_*_cl_*_train*.json')))
    for fpath in train_files:
        fname = os.path.basename(fpath)
        m = pattern.match(fname)
        if not m:
            continue
        ds = m.group('dataset')        # books / videogames / cds / toys ...
        period = m.group('period')     # D0 / D1 / ...
        suffix = m.group('suffix') or ''  # ''  / '_h10' / ...
        dataset_key = f"grlm_indomain_{ds}_cl_{period}{suffix}"
        dataset_info[dataset_key] = {
            "file_name": os.path.join("grlm_in_domain", fname),
            "columns": {"prompt": "prompt", "response": "response"},
        }

    # Merge with existing dataset_info.json if present
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            existing = json.load(f)
        existing.update(dataset_info)
        dataset_info = existing

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(dataset_info, f, indent=2)

    cl_entries = [k for k in dataset_info if 'cl_D' in k]
    print(f"Generated {len(dataset_info)} dataset entries -> {args.output}")
    print(f"CL entries: {len(cl_entries)}")
    for k in sorted(cl_entries):
        print(f"  {k}: {dataset_info[k]['file_name']}")


if __name__ == "__main__":
    main()
