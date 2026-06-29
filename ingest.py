import requests
import os
from bs4 import BeautifulSoup


def ingest_url(url, output_path):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    main = soup.find("main", id="main-content")
    content_tags = main.find_all(["h1", "h2", "h3", "p", "li"])
    parts = [tag.get_text(strip=True) for tag in content_tags]
    text = "\n".join(parts)
    os.makedirs("data/processed", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved {len(text)} characters to {output_path}")
    return text


if __name__ == "__main__":
    ingest_url(
        "https://www.nimh.nih.gov/health/publications/generalized-anxiety-disorder-gad",
        "data/processed/gad.txt",
    )
    ingest_url(
        "https://www.nimh.nih.gov/health/publications/depression",
        "data/processed/depression.txt",
    )
