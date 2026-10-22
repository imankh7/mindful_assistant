import os
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from chunk import load_corpus
from gold_set import gold_set
from generate import generate_answer
from config import SMALL_MODEL

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

DECLINE_PROMPT = """A health information assistant was asked a question that its source documents cannot answer.

Did the assistant DECLINE (say it does not have that information, and at most point the user elsewhere), or did it ANSWER (give specific facts, numbers, or advice in response to the question)?

Question: {question}
Assistant's reply: {answer}

Respond with exactly one word: DECLINED or ANSWERED."""

# Simple baseline: look for common "I don't know" wording.
REFUSAL_PHRASES = [
    "don't have information",
    "do not have information",
    "doesn't include",
    "does not include",
    "doesn't contain",
    "does not contain",
    "not able to",
    "can't provide",
    "cannot provide",
    "no information",
]


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(min=1, max=30),
    stop=stop_after_attempt(6),
)
def judge_declined(question, answer):
    prompt = DECLINE_PROMPT.format(question=question, answer=answer)
    response = client.chat.completions.create(
        model=SMALL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    verdict = response.choices[0].message.content.strip().upper()
    return "DECLINED" in verdict and "ANSWERED" not in verdict


def phrase_declined(answer):
    # Models often write curly apostrophes (don’t), which never match "don't".
    lowered = answer.lower().replace("’", "'")
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def run_control():
    # The judge must tell a refusal from a made-up answer before we trust it.
    # The question is deliberately not in the gold set, so the control never
    # shares an item with the evaluation it guards.
    question = "What is the usual starting dose of venlafaxine?"
    refusal = "I don't have information about venlafaxine dosing. Please ask your prescriber."
    invented = "The usual starting dose of venlafaxine is 75 mg a day, taken with food."
    ok = judge_declined(question, refusal) and not judge_declined(question, invented)
    print(f"Negative control: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    if not run_control():
        raise SystemExit("Decline judge failed its control; results would not be trustworthy.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    store = FAISS.from_documents(load_corpus(), embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 3})

    items = [item for item in gold_set if item["category"] == "out_of_scope"]
    declined = 0

    print(f"\nChecking {len(items)} out-of-scope questions\n")
    print("=" * 70)
    for item in items:
        question = item["question"]
        answer = generate_answer(question, retriever)
        by_judge = judge_declined(question, answer)
        by_phrase = phrase_declined(answer)
        declined += by_judge

        status = "DECLINED" if by_judge else "ANSWERED"
        agree = "" if by_judge == by_phrase else "  (phrase check disagrees)"
        print(f"[{status}] {question}{agree}")
        print(f"    {answer[:200]!r}")

    print("=" * 70)
    print(f"Declined: {declined}/{len(items)}")
