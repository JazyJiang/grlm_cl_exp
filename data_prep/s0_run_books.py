"""Run embedding + GPU-accelerated similarity for books dataset."""
import json
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import time

from s0_init_emb import load_data, generate_embeddings_multi_gpu

def compute_similarities_gpu(embeddings, item_ids, k=20, device='cuda:0', chunk_size=2000):
    """GPU-accelerated top-k similarity computation."""
    n = len(embeddings)
    print(f"Computing top-{k} similarities for {n} items on GPU...")
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_norm = embeddings / norms
    
    emb_tensor = torch.tensor(emb_norm, device=device, dtype=torch.float16)
    
    results = {}
    start = time.time()
    
    for i in tqdm(range(0, n, chunk_size), desc="GPU similarity"):
        end = min(i + chunk_size, n)
        chunk = emb_tensor[i:end]  # (chunk_size, dim)
        sims = chunk @ emb_tensor.T  # (chunk_size, n)
        
        # zero out self-similarity
        for j in range(i, end):
            sims[j - i, j] = -1.0
        
        topk_vals, topk_idx = torch.topk(sims, k=k, dim=1)
        topk_vals = topk_vals.cpu().numpy()
        topk_idx = topk_idx.cpu().numpy()
        
        for local_j in range(end - i):
            global_j = i + local_j
            similar_items = []
            for rank in range(k):
                similar_items.append({
                    "item_id": item_ids[topk_idx[local_j, rank]],
                    "similarity": float(topk_vals[local_j, rank])
                })
            results[item_ids[global_j]] = similar_items
    
    elapsed = time.time() - start
    print(f"Similarity computation done in {elapsed:.1f}s")
    return results

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    dataset = 'books'
    model_name = '/workspace/jiangzhuosong/GRLM_0/hf_qwen3_emb_8b'
    num_gpus = 4
    
    data_file = f'./{dataset}/raw_data/{dataset}.item.json'
    print(f'Loading data: {data_file}')
    data = load_data(data_file)
    print(f'Loaded {len(data)} items')
    
    print("Generating embeddings on 4 GPUs...")
    results_with_embeddings, embeddings = generate_embeddings_multi_gpu(
        data, model_name, num_gpus=num_gpus, batch_size=16
    )
    print(f'Embeddings shape: {embeddings.shape}')
    
    item_ids = [item['id'] for item in results_with_embeddings]
    
    # GPU similarity
    similarity_results = compute_similarities_gpu(embeddings, item_ids, k=20, device='cuda:0')
    
    output_file = f'./{dataset}/sum_data/{dataset}_similarities.json'
    print(f'Saving to {output_file}')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(similarity_results, f, ensure_ascii=False, indent=2)
    print(f'Done! {len(similarity_results)} items.')
