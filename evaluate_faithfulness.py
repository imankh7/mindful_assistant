from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from chunk import chunk_document
from gold_set import gold_set
from generate import generate_answer
from faithfulness import judge_faithfulness

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = chunk_document("data/processed/gad.json")
store = FAISS.from_documents(documents, embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})

faithful_count = 0
results_log = []

print(f"\nEvaluating faithfulness on {len(gold_set)} questions\n")
print("=" * 70)

for item in gold_set:
    question = item["question"]

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    answer = generate_answer(question, retriever)

    is_faithful = judge_faithfulness(answer, context)

    if is_faithful:
        faithful_count += 1

    status = "GROUNDED " if is_faithful else "UNGROUNDED"
    print(f"[{status}] {question}")
    results_log.append((question, answer, is_faithful))

rate = faithful_count / len(gold_set)
print("=" * 70)
print(f"\nFaithfulness rate: {faithful_count}/{len(gold_set)} = {rate:.1%}")

for q, a, ok in results_log:
    if not ok:
        print("\n--- Example UNGROUNDED answer ---")
        print(f"Q: {q}")
        print(f"A: {a[:400]}")
        break
