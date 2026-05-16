"""
Generate item embeddings and similarities using sentence-transformers.
Replacement for s0_init_emb.py that uses a downloadable model.
"""
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result_list = []
    for key, value in data.items():
        new_item = {"id": key}
        new_item.update(value)
        result_list.append(new_item)
    return result_list

def prepare_text(item):
    title = item.get('title', '')
    description = item.get('description', '')
    categories = item.get('categories', '')
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if description:
        desc = description[:200] if len(description) > 200 else description
        parts.append(f"Description: {desc}")
    if categories:
        parts.append(f"Categories: {categories}")
    return " | ".join(parts) if parts else "unknown"

def compute_similarities(embeddings, item_ids, k=20):
    n = len(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_norm = embeddings / norms
    
    results = {}
    batch_size = 1000
    
    for start in tqdm(range(0, n, batch_size), desc="Computing similarities"):
        end = min(start + batch_size, n)
        sims = np.dot(emb_norm[start:end], emb_norm.T)
        
        for i in range(start, end):
            local_i = i - start
            sim_row = sims[local_i].copy()
            sim_row[i] = -1  # exclude self
            top_k_idx = np.argpartition(sim_row, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(sim_row[top_k_idx])[::-1]]
            
            similar_items = []
            for idx in top_k_idx:
                similar_items.append({
                    "item_id": item_ids[idx],
                    "similarity": float(sim_row[idx])
                })
            results[item_ids[i]] = similar_items
    
    return results

def main():
    dataset = "books"
    data_file = f"./{dataset}/raw_data/{dataset}.item.json"
    
    print(f"Loading data: {data_file}")
    data = load_data(data_file)
    print(f"Loaded {len(data)} items")
    
    texts = [prepare_text(item) for item in data]
    item_ids = [item['id'] for item in data]
    
    print("Loading sentence-transformers model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Encoding {len(texts)} items...")
    start = time.time()
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, 
                              normalize_embeddings=False)
    elapsed = time.time() - start
    print(f"Encoding done in {elapsed:.1f}s, shape: {embeddings.shape}")
    
    print("Computing top-20 similarities...")
    similarity_results = compute_similarities(embeddings, item_ids, k=20)
    
    output_file = f"./{dataset}/sum_data/{dataset}_similarities.json"
    print(f"Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(similarity_results, f, ensure_ascii=False, indent=2)
    
    print(f"Done! {len(similarity_results)} items with similarities.")

if __name__ == "__main__":
    main()
