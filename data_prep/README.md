# Data Prep: SID Pipeline for GRLM-CL

This directory contains the full **Semantic-ID (SID) pipeline** that converts raw Amazon Reviews into the CL SFT JSON files consumed by training. Use this if you want to build a dataset from scratch (e.g. for a new Amazon category) instead of downloading a pre-built dump.

> If you only want to **run baselines on Books**, you can skip this directory and download the pre-built data via `setup.sh`. For VG (or any new Amazon category), follow the steps below — everything starts from the public McAuley Lab Amazon Reviews v2 dump and is fully reproducible.

---

## Pipeline overview

```
Amazon raw JSON   --[memory_dev/scripts/setup_data.sh]-->   *_item_map.npy + *_sequential.npy
                                                                       |
                                                                       v
                                       prepare_<dataset>_data.py    ==> raw_data/{<dataset>.item.json, sequential_data.txt}
                                                                       |
                                                                       v
                                          s0_run_<dataset>.py        ==> sum_data/<dataset>_similarities.json   (embeddings + top-k similar items)
                                                                       |
                                                                       v
                                          s1_run_<dataset>.py        ==> sum_data/<dataset>_summaries_with_similarity.jsonl  (Qwen3-4B keyword summaries)
                                                                       |
                                                                       v
                                          s2_build_id2meta.py        ==> sum_data/<dataset>_id2meta.json   (item_id -> {title, keywords})
                                                                       |
                                                                       v
                                          s3_build_meta2tid_sft_data.py ==> sum_data/item_id2tid/<dataset>_tid2item_id.json   (TID assignment)
                                                                       |
                                                                       v
                                          s4_build_<dataset>_cl_data.py ==> ../data/cl_sft/amazon_<dataset>_cl_D{0..3}_{train,eval}[_h{cap}].json
```

---

## External dependencies

The SID pipeline starts from **npy maps** produced by the [memory_dev](https://github.com/JazyJiang/memory_dev) repo (Amazon raw JSON download + temporal split + K-core filtering). You **do not need to write your own preprocessing** — `memory_dev` does that step end-to-end from the public McAuley Lab Amazon Reviews v2 dump.

| Dep | What | Where |
|-----|------|-------|
| [`memory_dev`](https://github.com/JazyJiang/memory_dev) | Downloads Amazon raw JSON from McAuley Lab + runs K-core filtering + temporal split → `*_item_map.npy` / `*_title_map.npy` / `*_description_map.npy` / `*_category_map.npy` + per-period sequential data under `data/D{0..3}/`. One command: `bash scripts/setup_data.sh <dataset>` | clone separately, see Step 0 below |
| `sentence-transformers` model | item-text embedding for s0 | downloaded automatically by `s0_init_emb_st.py` |
| **Qwen3-4B-Instruct** | LLM keyword summaries for s1 | edit `model_name` in `s1_init_sum.py` |

---

## Per-script setup (edit these constants before running)

Each file has a constants block at the top. Most reference `/workspace/jiangzhuosong/...` absolute paths that **you must edit** to match your local layout. Quick reference:

| File | Constants to edit |
|------|-------------------|
| `prepare_videogames_data.py` | `MEMORY_DEV`, `OUTPUT_DIR` |
| `prepare_books_data.py` | `MEMORY_DEV`, `OUTPUT_DIR` |
| `s0_init_emb_st.py` | model path (sentence-transformers download cache) |
| `s1_init_sum.py` | `model_name` (Qwen3-4B-Instruct local path) |
| `s4_build_videogames_cl_data.py` | `MEMORY_DEV`, `ID2META_FILE`, `OUTPUT_DIR` |
| `s4_build_books_cl_data.py` | same |
| `gen_h30_h40.py` | `OUTPUT_DIR` |
| `add_period_t_count.py` | `MEMORY_DEV`, `EVAL_DIR` |

After s4 finishes, the CL JSON files land in `LlamaFactory/data/grlm_in_domain/` (or `OUTPUT_DIR` as configured). Move/symlink them into `<repo_root>/data/cl_sft/` so the train scripts pick them up.

---

## Run order (VG example, end-to-end)

```bash
# 0. Preprocess Amazon raw -> npy maps (via memory_dev — fully public, no internal data)
#    Supported datasets: Toys_and_Games, Video_Games, CDs_and_Vinyl, Books
git clone git@github.com:JazyJiang/memory_dev.git
cd memory_dev
bash scripts/setup_data.sh Video_Games           # downloads + k-core + temporal split
#   -> produces memory_dev/data/info/Video_Games_5_<time>_*.npy
#   -> and memory_dev/data/D{0..3}/ sequential splits
cd ../grlm_cl_exp
# Then edit MEMORY_DEV constant in data_prep/prepare_videogames_data.py + s4_build_videogames_cl_data.py
#   to point at ../memory_dev/data  (or wherever you cloned it)

# 1. Build raw_data/ for VG
python data_prep/prepare_videogames_data.py

# 2. Embeddings + similarities (uses GPU)
python data_prep/s0_run_videogames.py

# 3. Keyword summaries with Qwen3-4B-Instruct (heavy, multi-GPU)
python data_prep/s1_run_videogames.py

# 4. Build id2meta JSON
python data_prep/s2_build_id2meta.py videogames

# 5. Assign TIDs
python data_prep/s3_build_meta2tid_sft_data.py videogames

# 6. Build CL train/eval JSON (D0..D3)
python data_prep/s4_build_videogames_cl_data.py

# 7. Generate extra history caps (h30/h40)
python data_prep/gen_h30_h40.py

# 8. Symlink outputs into <repo>/data/
mkdir -p data/cl_sft
ln -sf <abs_path_to>/videogames/sum_data/videogames_id2meta.json data/
ln -sf <abs_path_to>/videogames/sum_data/item_id2tid/videogames_tid2item_id.json data/
ln -sf <abs_path_to>/LlamaFactory/data/grlm_in_domain/amazon_videogames_cl_*.json data/cl_sft/

# 9. Regenerate dataset_info.json
python scripts/generate_dataset_info.py --data_dir data/cl_sft --output LlamaFactory/data/dataset_info.json
```

For Books, replace `videogames` with `books` throughout.

---

## Notes / caveats

- The pipeline was authored incrementally; constants are not yet centralized. PRs welcome.
- `s1_init_sum.py` is the slowest stage (LLM generation over all items). Plan for multi-GPU.
- D0 train file is the same regardless of history cap (no prior history), so `s4` writes once and `setup.sh` (for Books) symlinks `D0_train_h{cap}.json -> D0_train.json` for every cap.
