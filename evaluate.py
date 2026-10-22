from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from chunk import load_corpus
from gold_set import gold_set
from metrics import in_scope, source_hit, summarize, print_summary

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = load_corpus()
print("Building index...")
store = FAISS.from_documents(documents, embeddings)
K = 3
METRICS = [f"hit@{K}", f"source_hit@{K}"]
items = in_scope(gold_set)
rows = []

print(f"\nEvaluating {len(items)} in-scope questions at k={K}\n")
print("=" * 70)

for item in items:
    question = item["question"]
    expected = item["expected_phrase"]

    results = store.similarity_search(question, k=K)

    retrieved_text = " ".join(doc.page_content for doc in results)
    hit = expected in retrieved_text

    rows.append(
        {
            "category": item["category"],
            METRICS[0]: 1 if hit else 0,
            METRICS[1]: 1 if source_hit(results, item) else 0,
        }
    )

    status = "HIT " if hit else "MISS"
    print(f"[{status}] {question}")

print("=" * 70)
print_summary(summarize(rows, METRICS), METRICS)
