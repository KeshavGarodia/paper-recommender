# 🔍 Academic Paper Recommender

An AI-powered Streamlit app that recommends the most relevant arXiv papers based on your research abstract or idea.

## 🚀 What It Does

- Takes a research **topic**
- Fetches the latest **N arXiv papers** on that topic
- Lets you paste an **abstract or idea**
- Returns the **top 5 most similar papers** using semantic similarity

Built with:
- ✅ `allenai-specter` transformer embeddings
- ✅ Real-time arXiv scraping
- ✅ Cosine similarity
- ✅ Streamlit frontend

---

## 🧪 Example Use Case

> Topic: `graph neural networks`  
> Abstract: “We propose a novel GNN for molecule generation…”  
> → App recommends 5 recent arXiv papers closest in meaning

---

## 🔧 How to Run It

### 1. Clone this repo

```bash
git clone https://github.com/KeshavGarodia/paper-recommender.git
cd paper-recommender
