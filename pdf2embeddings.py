#!/usr/bin/env python3
"""
pdf_to_index.py

Extract text from a LinkedIn PDF (or any PDF), chunk it, compute OpenAI embeddings
(text-embedding-3-small) for each chunk, and save the index at:
  index/embeddings.npy       -- float32 numpy array shape (n_chunks, dim)
  index/chunks.json          -- list of {"id": str, "text": str, "source": str, "start_word": int, "end_word": int}

Usage:
  pip install -r requirements.txt
  export OPENAI_API_KEY="sk-..."
  python pdf_to_index.py linkedin.pdf

Config at top (CHUNK_SIZE, OVERLAP, BATCH_SIZE).
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
from PyPDF2 import PdfReader

# ----------------- CONFIG -----------------
EMB_MODEL = "text-embedding-3-small"   # OpenAI embedding model
CHUNK_SIZE = 200      # words per chunk (tune as needed)
OVERLAP = 50          # words overlap between chunks
BATCH_SIZE = 16       # number of chunks per embedding batch
SLEEP_BETWEEN_BATCHES = 0.2  # seconds between batches
INDEX_DIR = Path("index")
EMBS_PATH = INDEX_DIR / "embeddings.npy"
META_PATH = INDEX_DIR / "chunks.json"
# ------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages_text = []
    for p in reader.pages:
        try:
            text = p.extract_text() or ""
        except Exception:
            text = ""
        pages_text.append(text)
    full_text = "\n\n".join(pages_text)
    return full_text

def chunk_text_by_words(text: str, chunk_size=CHUNK_SIZE, overlap=OVERLAP) -> List[dict]:
    words = text.strip().split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size
    for i in range(0, len(words), step):
        start = i
        end = min(i + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append({"start_word": start, "end_word": end, "text": chunk_text})
        if end == len(words):
            break
    return chunks

def embed_batch_openai(texts):
    if not client.api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    vecs = [emb.embedding for emb in resp.data]
    return vecs

def ensure_index_dir():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

def save_index(embeddings_np: np.ndarray, meta: List[dict]):
    ensure_index_dir()
    np.save(EMBS_PATH, embeddings_np.astype("float32"))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved embeddings -> {EMBS_PATH} (shape={embeddings_np.shape})")
    print(f"Saved metadata -> {META_PATH} (count={len(meta)})")

def main(pdf_file: str):
    pdf_path = Path(pdf_file)
    if not pdf_path.exists():
        print("PDF file not found:", pdf_file)
        sys.exit(1)

    print("Extracting text from PDF:", pdf_file)
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("No text extracted from PDF. Exiting.")
        sys.exit(1)

    print("Chunking text...")
    chunks = chunk_text_by_words(text, CHUNK_SIZE, OVERLAP)
    print(f"Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={OVERLAP})")

    meta = []
    all_vecs = []
    total = len(chunks)
    if total == 0:
        print("No chunks to embed. Exiting.")
        return

    print("Embedding chunks in batches...")
    for i in tqdm(range(0, total, BATCH_SIZE)):
        batch = chunks[i:i+BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vecs = embed_batch_openai(texts)
        for j, c in enumerate(batch):
            idx = i + j
            chunk_id = f"linkedin_chunk_{idx:06d}"
            meta.append({
                "id": chunk_id,
                "text": c["text"],
                "source": str(pdf_path.name),
                "start_word": c["start_word"],
                "end_word": c["end_word"],
            })
            all_vecs.append(vecs[j])
        time.sleep(SLEEP_BETWEEN_BATCHES)

    emb_np = np.vstack([np.array(v, dtype="float32") for v in all_vecs])
    # normalize vectors for cosine similarity
    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_np = emb_np / norms

    save_index(emb_np, meta)
    print("Indexing complete. You can now run the RAG app using these files.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_index.py linkedin.pdf")
        sys.exit(1)
    main(sys.argv[1])
