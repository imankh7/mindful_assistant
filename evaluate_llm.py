from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from chunk import load_corpus
from gold_set import gold_set
from judge import judge_relevance, cache_stats
from metrics import in_scope, source_hit, summarize, print_summary

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = load_corpus()

print("Building index ...")
store = FAISS.from_documents(documents, embeddings)

K = 3
METRICS = [f"precision@{K}", f"source_hit@{K}"]
items = in_scope(gold_set)
rows = []

print(f"\nEvaluating {len(items)} in-scope questions at k={K} with LLM judge\n")
print("=" * 70)

for item in items:
    question = item["question"]

    # Retrieve top-K chunks
    results = store.similarity_search(question, k=K)

    # Judge each retrieved chunk
    relevant_count = 0
    for doc in results:
        if judge_relevance(question, doc.page_content):
            relevant_count += 1

    # Precision for THIS question = relevant retrieved / total retrieved
    question_precision = relevant_count / K
    rows.append(
        {
            "category": item["category"],
            METRICS[0]: question_precision,
            METRICS[1]: 1 if source_hit(results, item) else 0,
        }
    )

    print(f"{question_precision:.2f}  {question}")


print("=" * 70)
print_summary(summarize(rows, METRICS), METRICS)
print(f"\nJudge cache: {cache_stats['hits']} hits, {cache_stats['api_calls']} API calls")
