import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

FAITHFULNESS_PROMPT = """ You are checking whether an answer is faithful to the source text.

An answer is GROUNDED if every claim it makes is supported by the source text.
An answer is NOT_GROUNDED if it includes any claim which is not supported by the source text, even if it's plausible.

Source text: 
{source}

Answer:
{answer}

Respond exactly with one word, GROUNDED or NOT_GROUNDED."""


def judge_faithfulness(answer, source):
    prompt = FAITHFULNESS_PROMPT.format(source=source, answer=answer)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    verdict = response.choices[0].message.content.strip()
    return "GROUNDED" in verdict and "NOT_GROUNDED" not in verdict


if __name__ == "__main__":
    source = "GAD may run in families, but the exact cause of the disorder is unknown."

    faithful_answer = (
        "The exact cause of GAD is unknown, though it may run in families."
    )
    unfaithful_answer = "GAD is caused by a specific gene discovered in 2019 that affects 80% of patients."

    print(
        f"Faithful answer   -> {judge_faithfulness(faithful_answer, source)}"
    )  # expect True
    print(
        f"Unfaithful answer -> {judge_faithfulness(unfaithful_answer, source)}"
    )  # expect False
