import feedparser
import pandas as pd
import urllib.parse
from datetime import datetime

def fetch_arxiv_papers(query="machine learning", max_results=50):
    base_url = "http://export.arxiv.org/api/query?"
    query_encoded = urllib.parse.quote(query)
    search_query = f"search_query=all:{query_encoded}&start=0&max_results={max_results}"
    url = base_url + search_query

    feed = feedparser.parse(url)
    entries = feed.entries

    if not entries:
        print("No papers found. Try a different query.")
        return pd.DataFrame()

    papers = []
    for entry in entries:
        papers.append({
            "title": entry.title.replace('\n', ' ').strip(),
            "summary": entry.summary.replace('\n', ' ').strip(),
            "authors": ', '.join(author.name for author in entry.authors),
            "published": entry.published,
            "link": entry.link
        })

    df = pd.DataFrame(papers)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"data/arxiv_{query.replace(' ', '_')}_{timestamp}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n✅ Fetched {len(df)} papers on '{query}' and saved to '{out_path}'\n")
    return df

if __name__ == "__main__":
    query = input("Enter a topic to search on arXiv: ").strip()
    max_results = input("Enter number of papers to fetch (default=50): ").strip()
    max_results = int(max_results) if max_results.isdigit() else 50

    df = fetch_arxiv_papers(query, max_results)
    print(df[["title", "published", "link"]].head())
