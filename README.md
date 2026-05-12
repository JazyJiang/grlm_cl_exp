# GRLM-CL: Continual Learning for Generative Recommendation with LLMs

Qwen3-based generative sequential recommender with **Sliding-Window Routing** and **Auxiliary Loss** for continual learning on Amazon Books.

## Quick Start

```bash
# 1. Clone and setup (downloads models + data + LlamaFactory)
git clone git@github.com:JazyJiang/grlm_cl_exp.git
cd grlm_cl_exp
bash setup.sh

# 2. Run a single baseline experiment
bash run_books_cl_v2.sh 06b h10 0

# 3. Run a routing experiment
bash run_books_cl_routed.sh 06b full 0 512 0.1
```

---

## 1. Environment Setup

### 1.1 Requirements

- NVIDIA GPU with 80GB VRAM (H100 recommended)
- Python 3.10+, PyTorch 2.x, CUDA 12+
- `huggingface-cli` (`pip install huggingface_hub`)
- Disk: ~50GB models + ~10GB data + ~100GB checkpoints (auto-cleaned)

### 1.2 `setup.sh` Pipeline

| Step | 说明 |
|------|------|
| 1 | Clone [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 并 `pip install` |
| 2 | 从 HuggingFace 下载 Qwen3-0.6B / 1.7B / 4B → `models/` |
| 3 | 从 HuggingFace 下载实验数据 (`JazySong/grlm-books-cl-data`) → `data/` |
| 4 | 解压 `cl_sft.tar.gz`（各 period × history cap 的 train/eval JSON） |
| 5 | 生成 LlamaFactory 的 `dataset_info.json` |

所有步骤有幂等检查，重复运行安全。

### 1.3 Models

| Model | Params | HuggingFace | 用途 |
|-------|--------|-------------|------|
| `Qwen3-0.6B` | 0.6B | `Qwen/Qwen3-0.6B` | 快速实验 |
| `Qwen3-1.7B` | 1.7B | `Qwen/Qwen3-1.7B` | 中等规模 |
| `Qwen3-4B` | 4B | `Qwen/Qwen3-4B` | 大模型，需 2×GPU + DeepSpeed Zero-2 |

### 1.4 Data

**Dataset:** Amazon Books 2016-10 ~ 2018-11, ~300K users, ~142K items

数据文件（由 `setup.sh` 自动下载）：

| 文件 | 说明 |
|------|------|
| `data/books_id2meta.json` | Item metadata (142K items, keywords + title) |
| `data/books_tid2item_id.json` | TID → item_id 映射 (eval 时将生成的 TID 转为 item_id) |
| `data/cl_sft/amazon_books_cl_D{0-3}_train[_h{cap}].json` | 各 period × history cap 的 SFT 训练数据 |
| `data/cl_sft/amazon_books_cl_D{0-3}_eval.json` | 各 period 的评估数据 |

History cap 说明：`h10` 表示训练时 user history 截断到最近 10 条，`full` 表示不截断。D0 训练数据对所有 cap 相同（无先验 history）。

---

## 2. Baseline 实验

### 2.1 运行方式

```bash
bash run_books_cl_v2.sh <model_size> <cap> <gpu_ids>
```

| 参数 | 取值 | 说明 |
|------|------|------|
| `model_size` | `06b`, `17b`, `4b` | 模型大小 |
| `cap` | `h2`, `h5`, `h10`, `h20`, `h30`, `h40`, `full` | History 截断长度 |
| `gpu_ids` | `0`, `1`, `4,5` | GPU 编号（4B 需 2 卡） |

### 2.2 训练参数

| 参数 | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|------|-----------|-----------|---------|
| D0 Learning Rate | 7e-5 | 5e-5 | 1e-4 |
| D1+ Learning Rate | 3e-5 | 2e-5 | 5e-5 |
| D0 Epochs | 5 | 5 | 5 |
| D1+ Epochs | 3 | 3 | 3 |
| Per-device Batch Size | 4 | 4 | 4 |
| Gradient Accumulation | 16 | 16 | 8 |
| Effective Batch Size | 64 | 64 | 64 (2 GPUs) |
| Optimizer | AdamW | AdamW | AdamW |
| LR Scheduler | cosine | cosine | cosine |
| Precision | bf16 | bf16 | bf16 |
| DeepSpeed | - | - | Zero-2 |

### 2.3 批量运行

```bash
# 方式 A: 7 GPU 并行跑所有 cap (推荐)
bash dispatch_all.sh

# 方式 B: 单 GPU 顺序跑
bash dispatch_sequential.sh

# 方式 C: 多 GPU 手动并行
bash run_books_cl_v2.sh 06b h2 0 &
bash run_books_cl_v2.sh 06b h10 1 &
bash run_books_cl_v2.sh 06b full 2 &
wait
```

**预估时间 (7×H100):**
- 0.6B 全部 cap: ~4-6h (并行)
- 1.7B 全部 cap: ~6-10h (并行)
- 4B 全部 cap: ~10-16h (3-4 组并行，每组 2 卡)

---

## 3. Routing 实验

### 3.1 Sliding-Window Routing

在 Qwen3 中配置 per-layer attention 类型：
- **大多数层**: sliding window attention（只看最近 `sliding_window` 个 token）
- **指定 `full_layer`**: full attention（看全部 history）

这让模型在长 history 场景中，既能通过 full attention 层保留全局信息，又通过 sliding window 层聚焦近期行为。

```yaml
# 配置参数
sliding_window: 512    # token 数 (~24 items，每 item ≈ 21 tokens)
full_layer: null       # 默认: num_hidden_layers // 2 (中间层)
```

### 3.2 Auxiliary Prediction Loss

在 `full_layer` 输出处挂一个辅助预测 head（与 `lm_head` 权重共享，零额外参数），让中间层也学到 item 预测信号：

```yaml
aux_loss_weight: 0.1   # aux loss 权重 (0 = 关闭)
```

`total_loss = lm_loss + aux_loss_weight * aux_loss`

### 3.3 运行方式

```bash
bash run_books_cl_routed.sh <model_size> <cap> <gpu_ids> [sliding_window] [aux_weight]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sliding_window` | 512 | Sliding window token 数 |
| `aux_weight` | 0.1 | Aux loss 权重 |

示例:

```bash
# 0.6B, full history, sw=512, aux=0.1 (默认)
bash run_books_cl_routed.sh 06b full 0 512 0.1

# 4B, full history, sw=512, aux=0.1, 双卡
bash run_books_cl_routed.sh 4b full 4,5 512 0.1

# 只开 routing 不开 aux loss
bash run_books_cl_routed.sh 06b full 0 512 0.0
```

### 3.4 实现细节

| 文件 | 说明 |
|------|------|
| `routing/config_patch.py` | Patch Qwen3Config：设置 `layer_types` (full/sliding per layer) 和 `sliding_window` |
| `routing/aux_head.py` | Aux prediction head：hook full_layer 输出，共享 lm_head 权重 |
| `routing/train_with_routing.py` | Monkey-patch LlamaFactory trainer：注入 config patch + aux loss 到 `compute_loss` |

保存 checkpoint 时临时摘掉 aux_head 避免 safetensors 权重共享报错；加载时自动重建。

---

## 4. Evaluation

### 4.1 Evaluation Protocol

**Tiger-style sequential per-target prediction:**

对每个用户在 D_{t+1} 的 target items（按时间排序），逐条预测：
1. 用当前 history 构建 prompt（可选 `--max_hist` 截断）
2. Beam search 生成 top-20 候选 item TID
3. 将 TID 转为 item_id，判断是否命中 target
4. 将 target 加入 history，预测下一个

每个 (user, target) pair 独立评估。

### 4.2 Evaluation 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num_beams` | 20 | Beam search width |
| `--num_return_sequences` | 20 | 返回 top-K 候选 |
| `--max_new_tokens` | 30 | 生成 token 数上限 |
| `--max_users` | 5000 | 最多评估用户数 |
| `--batch_size` | 视模型/cap 自动调整 | Eval batch size |
| `--num_gpus` | 1 (0.6B/1.7B), 2 (4B) | 多 GPU 并行评估 |
| `--max_hist` | cap 值 / 不设=full | 评估时 history 截断 |

### 4.3 Metrics

| Metric | 说明 |
|--------|------|
| Recall@{1,5,10,20} | target 出现在 top-K 候选中的比例 |
| NDCG@{5,10,20} | `1/log2(hit_pos+1)` 的均值，hit_pos 为 1-indexed |

### 4.4 User Grouping

用户按其在 **D_{t-1}** 的活跃度分为 5 组 (G1–G5)：
- **G1**: 历史最长（最容易受 history noise 影响）
- **G5**: 历史最短

每组独立计算 Recall 和 NDCG。

### 4.5 Re-compute Recall

已有 `seq_results_*.jsonl` 时，无需重新推理，直接重算 metrics：

```bash
python eval/recompute_cl_recall.py \
    --results_dir results/cl_results_seq/06b_h10 \
    --eval_dir data/cl_sft
```

---

## 5. CL Protocol

每个实验自动跑完整 continual learning 链：

```
D0 train (from pretrained) → eval on D0→D1
D1 finetune (from D0 ckpt) → eval on D1→D2
D2 finetune (from D1 ckpt) → eval on D2→D3
D3 finetune (from D2 ckpt) → eval on D3→D4
```

- D0 用更高 lr + 更多 epochs（from scratch）
- D1+ 用更低 lr + 更少 epochs（减少遗忘）
- 每 period 训练完后自动删除上一 period 的 checkpoint（节省磁盘）

---

## 6. Project Structure

```
grlm_cl_exp/
├── setup.sh                        # 一键安装 (models + data + LlamaFactory)
├── run_books_cl_v2.sh              # Baseline CL 链 (train + eval, 4 periods)
├── run_books_cl_routed.sh          # Routing CL 链 (sliding window + aux loss)
├── dispatch_all.sh                 # 7 GPU 并行跑所有 cap
├── dispatch_sequential.sh          # 单 GPU 顺序跑
├── routing/
│   ├── config_patch.py             # Qwen3 per-layer attention 配置
│   ├── aux_head.py                 # Auxiliary prediction head
│   └── train_with_routing.py       # Monkey-patch LlamaFactory trainer
├── eval/
│   ├── s5_books_cl_eval_seq.py     # Sequential eval (Recall + NDCG, per-group)
│   └── recompute_cl_recall.py      # 从已有结果重算 metrics
├── scripts/
│   └── generate_dataset_info.py    # 生成 LlamaFactory dataset_info.json
├── LlamaFactory/                   # setup.sh 自动 clone (不在 git 中)
├── models/                         # setup.sh 自动下载 (不在 git 中)
│   ├── Qwen3-0.6B/
│   ├── Qwen3-1.7B/
│   └── Qwen3-4B/
├── data/                           # setup.sh 自动下载 (不在 git 中)
│   ├── books_id2meta.json          # 142K items metadata
│   ├── books_tid2item_id.json      # TID → item_id 映射
│   └── cl_sft/                     # Train + eval JSON per period × cap
├── checkpoints/                    # 训练 checkpoint (自动清理)
├── logs/                           # 训练 + eval 日志
└── results/
    └── cl_results_seq/             # 评估结果
        ├── 06b_h10/
        │   ├── seq_recall_h10_D0.json    # Recall + NDCG (overall + per-group)
        │   └── seq_results_h10_D0.jsonl  # Per-target 命中详情
        └── ...
```

## 7. Output Format

每个 period 的 eval 输出：

- `seq_recall_{tag}_D{t}.json`: 整体 + per-group 的 Recall@{1,5,10,20} 和 NDCG@{5,10,20}
- `seq_results_{tag}_D{t}.jsonl`: 每条 (user, target) 的命中位置，用于后续分析

```json
{
  "overall": {"recall@5": 0.089, "recall@10": 0.134, "NDCG@10": 0.072, ...},
  "group_1": {"recall@5": 0.065, ...},
  ...
}
```

## 8. Monitoring

```bash
# 查看某条链的日志
tail -f results/cl_results_seq/06b_h10/eval_D0.log

# 查看已完成的 eval 结果
ls results/cl_results_seq/*/seq_recall_*.json

# GPU 利用率
nvidia-smi
```
