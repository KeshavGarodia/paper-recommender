import streamlit as st
import pandas as pd
import numpy as np
import feedparser
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "allenai-specter"

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

def fetch_arxiv_papers(query, max_results=50):
    base_url = "http://export.arxiv.org/api/query?"
    query = query.replace(" ", "+")
    url = f"{base_url}search_query=all:{query}&start=0&max_results={max_results}"
    feed = feedparser.parse(url)

    papers = []
    for entry in feed.entries:
        papers.append({
            "title": entry.title.replace('\n', ' ').strip(),
            "summary": entry.summary.replace('\n', ' ').strip(),
            "authors": ', '.join(author.name for author in entry.authors),
            "published": entry.published,
            "link": entry.link
        })

    return pd.DataFrame(papers)

def embed_texts(texts, model):
    return model.encode(texts, show_progress_bar=True)

def recommend(user_input, paper_df, paper_embeddings, model, top_k=5):
    user_vec = model.encode([user_input])[0]
    sims = cosine_similarity([user_vec], paper_embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return paper_df.iloc[top_indices].copy(), sims[top_indices]

def run():
    st.set_page_config(page_title="AI Paper Recommender", layout="centered")
    st.title("📚 Academic Paper Recommender")

    topic = st.text_input("Enter a topic (e.g. 'graph neural networks'):")
    max_results = st.slider("Number of papers to fetch:", 10, 100, 50)
    abstract = st.text_area("Paste your research idea or abstract here:", height=200)

    if st.button("Find Similar Papers"):
        if not topic or not abstract:
            st.error("Please fill both the topic and abstract fields.")
            return

        st.info("Fetching papers from arXiv...")
        papers = fetch_arxiv_papers(topic, max_results)

        if papers.empty:
            st.warning("No papers found for that topic. Try something broader.")
            return

        st.success(f"Fetched {len(papers)} papers. Embedding...")

        model = load_model()
        paper_embeddings = embed_texts(papers["summary"].fillna("").tolist(), model)

        st.info("Finding similar papers...")
        results, scores = recommend(abstract, papers, paper_embeddings, model)

        for i, row in results.iterrows():
            st.markdown(f"### {row['title']}")
            st.markdown(f"- 📅 *{row['published']}*")
            st.markdown(f"- 🔗 [Link to arXiv]({row['link']})")
            st.markdown(f"> {row['summary'][:500]}...")
            st.markdown("---")

if __name__ == "__main__":
    run()
