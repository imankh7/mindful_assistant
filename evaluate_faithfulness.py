from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from chunk import load_corpus
from gold_set import gold_set
from generate import generate_answer
from faithfulness import judge_faithfulness
from metrics import in_scope

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = load_corpus()
store = FAISS.from_documents(documents, embeddings)
retriever = store.as_retriever(search_kwargs={"k": 3})

items = in_scope(gold_set)
faithful_count = 0
results_log = []

print(f"\nEvaluating faithfulness on {len(items)} in-scope questions\n")
print("=" * 70)

for item in items:
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

rate = faithful_count / len(items)
print("=" * 70)
print(f"\nFaithfulness rate: {faithful_count}/{len(items)} = {rate:.1%}")

for q, a, ok in results_log:
    if not ok:
        print("\n--- Example UNGROUNDED answer ---")
        print(f"Q: {q}")
        print(f"A: {a[:400]}")
        break
