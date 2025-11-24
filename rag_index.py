import json
import faiss
import faiss.contrib.torch_utils as faiss_torch
import numpy as np
from sentence_transformers import SentenceTransformer

docs = [json.loads(l) for l in open("combined_docs.jsonl", "r", encoding="utf-8")]
texts = [(d["instruction"] + "\n" + d["output"]) for d in docs]

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = embedder.encode(texts, convert_to_numpy=True, batch_size=128).astype("float32")
faiss.normalize_L2(emb)

dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(emb)
faiss.write_index(index, "faiss.index")

# save docs mapping
with open("docs.jsonl", "w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print("FAISS index written (faiss.index) and docs.jsonl saved.")
