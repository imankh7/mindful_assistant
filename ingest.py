import requests
import os
from bs4 import BeautifulSoup

url = "https://www.nimh.nih.gov/health/publications/generalized-anxiety-disorder-gad"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
main = soup.find("main", id="main-content")
content_tags = main.find_all(["h1","h2","h3","p"])
parts = [tag.get_text(strip=True)for tag in content_tags]
text = "\n".join(parts)
os.makedirs("data/processed", exist_ok=True)
with open ("data/processed/gad.txt", "w", encoding= "utf-8") as f:
    f.write(text)
print(f"Saved {len(text)} characters to data/processed/gad.txt")