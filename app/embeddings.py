# app/embeddings.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

MODEL_NAME = "allenai-specter"  # Best for academic texts

def load_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ File not found: {csv_path}")
    return pd.read_csv(csv_path)

def embed_abstracts(df, model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    texts = df["summary"].fillna("").tolist()
    print(f"🔍 Encoding {len(texts)} abstracts...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings)

def save_embeddings(embeddings, out_path):
    np.save(out_path, embeddings)
    print(f"✅ Saved embeddings to: {out_path}")

if __name__ == "__main__":
    csv_path = input("Enter path to your CSV file (e.g., data/arxiv_xyz.csv): ").strip()
    df = load_csv(csv_path)

    embs = embed_abstracts(df)
    
    out_path = csv_path.replace(".csv", "_embeddings.npy")
    save_embeddings(embs, out_path)
