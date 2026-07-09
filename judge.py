import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

JUDGE_PROMPT = """ You are evaluating wether a piece of text is relevant to a question.

A text is RELEVANT if it contains information that helps to answer the question.
A text is NOT_RELEVANT if it does not help answer the question.

Question: {question}
Text : {chunk}

Respond exactly with one word: RELEVANT or NOT_RELEVANT.
Do not explain or add any other words."""


def judge_relevance(question, chunk):
    prompt = JUDGE_PROMPT.format(question=question, chunk=chunk)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    verdict = response.choices[0].message.content.strip()

    return "RELEVANT" in verdict and "NOT_RELEVANT" not in verdict


if __name__ == "__main__":
    q = "What are the symptoms of anxiety?"
    good_chunk = "People with GAD may feel restless, have trouble concentrating, and experience muscle tension."
    bad_chunk = "GAD may run in families, but the exact cause is unknown."

    print(f"Good chunk -> {judge_relevance(q, good_chunk)}")
    print(f"Bad chunk  -> {judge_relevance(q, bad_chunk)}")
