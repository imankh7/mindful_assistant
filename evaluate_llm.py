from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from chunk import chunk_document
from gold_set import gold_set
from judge import judge_relevance

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = chunk_document("data/processed/gad.json")

print("Building index ...")
store = FAISS.from_documents(documents, embeddings)

K = 3
precisions = []

print(f"\nEvaluating {len(gold_set)} questions at k={K} with LLM judge\n")
print("=" * 70)

for item in gold_set:
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
    precisions.append(question_precision)

    print(f"{question_precision:.2f}  {question}")


avg_precision = sum(precisions) / len(precisions)
print("=" * 70)
print(f"\nMean Precision@{K}: {avg_precision:.1%}")
