from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from chunk import chunk_document
from gold_set import gold_set

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = chunk_document("data/processed/gad.json")
print("Building index...")
store = FAISS.from_documents(documents, embeddings)
K = 3
hits = 0

print(f"\nEvaluating {len(gold_set)} questions at k={K}\n")
print("=" * 70)

for item in gold_set:
    question = item["question"]
    expected = item["expected_phrase"]

    results = store.similarity_search(question, k=K)

    retrieved_text = " ".join(doc.page_content for doc in results)
    hit = expected in retrieved_text

    if hit:
        hits += 1

    status = "HIT " if hit else "MISS"
    print(f"[{status}] {question}")

score = hits / len(gold_set)
print("=" * 70)
print(f"\nHit-rate at k={K}: {hits}/{len(gold_set)} = {score:.1%}")
