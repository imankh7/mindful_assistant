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

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = load_corpus()

store = FAISS.from_documents(documents, embeddings)
sparse = BM25Retriever.from_documents(documents)
sparse.k = 3

hybrid = EnsembleRetriever(
    retrievers=[store.as_retriever(search_kwargs={"k": 3}), sparse],
    weights=[0.5, 0.5],
)

sparse_broad = BM25Retriever.from_documents(documents)
sparse_broad.k = 15
hybrid_broad = EnsembleRetriever(
    retrievers=[store.as_retriever(search_kwargs={"k": 15}), sparse_broad],
    weights=[0.5, 0.5],
)
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
reranked = ContextualCompressionRetriever(
    base_compressor=reranker, base_retriever=hybrid_broad
)

# A few questions to inspect
questions = [
    "What are the symptoms of generalized anxiety disorder?",
    "how do I bring up my worries with someone I trust?",
    "are benzodiazepines addictive?",
]

for q in questions:
    print("\n" + "=" * 70)
    print(f"QUESTION: {q}")

    print("\n--- HYBRID top 3 ---")
    for i, doc in enumerate(hybrid.invoke(q)):
        print(f"[{i}] {doc.page_content[:90]}")

    print("\n--- RERANKED top 3 ---")
    for i, doc in enumerate(reranked.invoke(q)):
        print(f"[{i}] {doc.page_content[:90]}")
