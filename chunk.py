from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import json


def chunk_document(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    text_chunks = splitter.split_text(doc["text"])
    documents = []
    for i, chunk_text in enumerate(text_chunks):
        metadata = {
            "source": doc["source"],
            "source_url": doc["source_url"],
            "date": doc["date"],
            "authority_tier": doc["authority_tier"],
            "chunk_index": i,
        }
        documents.append(Document(page_content=chunk_text, metadata=metadata))

    print(f"Split {doc['source']} into {len(documents)} chunks")
    return documents


def load_corpus(folder="data/processed"):
    # Sorted so the corpus order (and so retrieval/eval results) is reproducible.
    paths = sorted(Path(folder).glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No .json documents found in {folder}")

    documents = []
    for path in paths:
        documents.extend(chunk_document(path))

    print(f"Loaded {len(documents)} chunks from {len(paths)} documents")
    return documents


if __name__ == "__main__":
    chunks = load_corpus()

    first = chunks[0]
    print("-" * 60)
    print("First chunk text:")
    print(first.page_content[:200])
    print("-" * 60)
    print("First chunk metadata:")
    print(first.metadata)
