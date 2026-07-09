from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from chunk import chunk_document
from gold_set import gold_set
from judge import judge_relevance

K = 3


def evaluate_retriever(retriever, label):
    precisions = []
    print(f"\n--- {label} ---")

    for item in gold_set:
        question = item["question"]

        results = retriever.invoke(question)

        relevant_count = 0
        for doc in results:
            if judge_relevance(question, doc.page_content):
                relevant_count += 1

        question_precision = relevant_count / K
        precisions.append(question_precision)

    avg = sum(precisions) / len(precisions)
    print(f"Mean Precision@{K}: {avg:.1%}")
    return avg


if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = chunk_document("data/processed/gad.json")

    # 1: Dense retriever(semantic)
    store = FAISS.from_documents(documents, embeddings)
    dense = store.as_retriever(search_kwargs={"k": K})

    # 2: Sparse retriever(BM25)
    sparse = BM25Retriever.from_documents(documents)
    sparse.k = K

    # 3: Hybrid retriever
    hybrid = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[0.5, 0.5],
    )

    # Run all three through the same evaluation
    evaluate_retriever(dense, "DENSE (semantic)")
    evaluate_retriever(sparse, "SPARSE (BM25 keyword)")
    evaluate_retriever(hybrid, "HYBRID (dense + sparse, RRF)")
