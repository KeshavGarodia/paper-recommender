import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "allenai-specter"

def load_data(csv_path, embedding_path):
    df = pd.read_csv(csv_path)
    embeddings = np.load(embedding_path)
    return df, embeddings

def get_user_embedding(text, model_name=MODEL_NAME):
    model = SentenceTransformer(model_name)
    return model.encode([text])[0]

def recommend_papers(user_vector, paper_vectors, df, top_k=5):
    sims = cosine_similarity([user_vector], paper_vectors)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    print(f"\nTop {top_k} Similar Papers:\n")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {df.iloc[idx]['title']}")
        print(f"   📅 Published: {df.iloc[idx]['published']}")
        print(f"   🔗 Link     : {df.iloc[idx]['link']}")
        print(f"   📚 Abstract : {df.iloc[idx]['summary'][:300]}...\n")

if __name__ == "__main__":
    csv_path = input("Enter path to CSV file: ").strip()
    emb_path = csv_path.replace(".csv", "_embeddings.npy")

    print("\nPaste your abstract or paragraph describing your idea:\n")
    user_input = input("> ").strip()

    df, paper_embeddings = load_data(csv_path, emb_path)
    user_vec = get_user_embedding(user_input)
    recommend_papers(user_vec, paper_embeddings, df)
