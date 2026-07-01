import requests
import os
import json
from bs4 import BeautifulSoup


def ingest_url(url, output_path, source, date, authority_tier):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    main = soup.find("main", id="main-content")
    content_tags = main.find_all(["h1", "h2", "h3", "p", "li"])
    parts = [tag.get_text(strip=True) for tag in content_tags]
    text = "\n".join(parts)

    document = {
        "text": text,
        "source": source,
        "source_url": url,
        "date": date,
        "authority_tier": authority_tier,
    }

    os.makedirs("data/processed", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)

    print(f"Saved document from {source} to {output_path}")
    return document


if __name__ == "__main__":
    ingest_url(
        "https://www.nimh.nih.gov/health/publications/generalized-anxiety-disorder-gad",
        "data/processed/gad.json",
        source="NIMH: Generalized Anxiety Disorder",
        date="2025",
        authority_tier=1,
    )
    ingest_url(
        "https://www.nimh.nih.gov/health/publications/depression",
        "data/processed/depression.json",
        source="NIMH: Depression",
        date="2025",
        authority_tier=1,
    )
