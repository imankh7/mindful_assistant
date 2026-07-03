from sentence_transformers import SentenceTransformer
from chunk import chunk_document


def embed_chunks(documents, model):
    texts = [doc.page_content for doc in documents]
    vectors = model.encode(texts)
    return vectors


if __name__ == "__main__":
    print("Loading model ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    documents = chunk_document("data/processed/gad.json")
    vectors = embed_chunks(documents, model)
    print(f"Number of chunks: {len(documents)}")
    print(f"Number of vectors: {len(vectors)}")
    print(f"Each vector has {len(vectors[0])} dimensions")
    print(f"\nFirst chunk text (preview): {documents[0].page_content[:80]}")
    print(f"Its vector (first 5 numbers): {vectors[0][:5]}")
