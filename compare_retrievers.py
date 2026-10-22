from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from chunk import load_corpus
from gold_set import gold_set
from judge import judge_relevance, cache_stats
from metrics import in_scope, source_hit, source_coverage, summarize, print_summary

K = 3
METRICS = [f"precision@{K}", f"source_hit@{K}", f"coverage@{K}"]


def evaluate_retriever(retriever, label):
    rows = []
    print(f"\n--- {label} ---")

    for item in in_scope(gold_set):
        question = item["question"]

        results = retriever.invoke(question)[:K]

        relevant_count = 0
        for doc in results:
            if judge_relevance(question, doc.page_content):
                relevant_count += 1

        rows.append(
            {
                "category": item["category"],
                METRICS[0]: relevant_count / len(results) if results else 0,
                METRICS[1]: 1 if source_hit(results, item) else 0,
                METRICS[2]: 1 if source_coverage(results, item) else 0,
            }
        )

    summary = summarize(rows, METRICS)
    print_summary(summary, METRICS)
    return summary


if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    documents = load_corpus()

    # 1: Dense retriever(semantic)
    store = FAISS.from_documents(documents, embeddings)
    dense = store.as_retriever(search_kwargs={"k": K})

    # 2: Sparse retriever(BM25)
    sparse = BM25Retriever.from_documents(documents)
    sparse.k = K

    sparse_broad = BM25Retriever.from_documents(documents)
    sparse_broad.k = 15

    # 3: Hybrid retriever
    hybrid = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[0.5, 0.5],
    )

    # 4. HYBRID + RERANK
    hybrid_broad = EnsembleRetriever(
        retrievers=[
            store.as_retriever(search_kwargs={"k": 15}),
            sparse_broad,
        ],
        weights=[0.5, 0.5],
    )

    cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=K)

    reranked = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_broad,
    )

    # Run all four through the same evaluation
    results = {
        "dense": evaluate_retriever(dense, "DENSE (semantic)"),
        "sparse": evaluate_retriever(sparse, "SPARSE (BM25 keyword)"),
        "hybrid": evaluate_retriever(hybrid, "HYBRID (dense + sparse, RRF)"),
        "rerank": evaluate_retriever(reranked, "HYBRID + RERANK (cross-encoder)"),
    }

    print(f"\n=== Overall ({results['dense']['overall']['n']} in-scope questions) ===")
    print(f"{'retriever':<10}" + "".join(f"{name:>16}" for name in METRICS))
    for name, summary in results.items():
        overall = summary["overall"]
        print(f"{name:<10}" + "".join(f"{overall[m]:>16.1%}" for m in METRICS))

    print(f"\nJudge cache: {cache_stats['hits']} hits, {cache_stats['api_calls']} API calls")
