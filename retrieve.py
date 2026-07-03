from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from chunk import chunk_document

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = chunk_document("data/processed/gad.json")
print("Building FAISS index...")
store = FAISS.from_documents(documents, embeddings)
print(f"Index built with {store.index.ntotal} vectors.")
query = "What is the problem when people feel restless?"
results = store.similarity_search(query, k=7)
print(f"\nQuery: {query}")
print("=" * 60)
for i, doc in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(f"Source: {doc.metadata['source']}")
    print(doc.page_content[:300])
