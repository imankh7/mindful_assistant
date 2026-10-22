import os
from dotenv import load_dotenv
from groq import Groq, RateLimitError
from config import SMALL_MODEL
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential,
    stop_after_attempt,
)
import hashlib
import json
from pathlib import Path

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

JUDGE_PROMPT = """ You are evaluating whether a piece of text is relevant to a question.

A text is RELEVANT if it contains information that helps to answer the question.
A text is NOT_RELEVANT if it does not help answer the question.

Question: {question}
Text : {chunk}

Respond exactly with one word: RELEVANT or NOT_RELEVANT.
Do not explain or add any other words."""


CACHE_PATH = Path(".cache/judge_relevance.json")
cache_stats = {"hits": 0, "api_calls": 0}


def _load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


_cache = _load_cache()


def _save_cache():
    CACHE_PATH.parent.mkdir(exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_cache, f)
    tmp.replace(CACHE_PATH)  # atomic, so a crash never leaves a half-written cache


def _cache_key(question, chunk):
    # The model and prompt are part of the key, so changing either never
    # reuses verdicts from the old judge.
    raw = "\x00".join([SMALL_MODEL, JUDGE_PROMPT, question, chunk])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def judge_relevance(question, chunk):
    key = _cache_key(question, chunk)
    if key in _cache:
        cache_stats["hits"] += 1
        return _cache[key]

    verdict = _ask_judge(question, chunk)
    cache_stats["api_calls"] += 1
    _cache[key] = verdict
    _save_cache()
    return verdict


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(min=1, max=30),
    stop=stop_after_attempt(6),
)
def _ask_judge(question, chunk):

    prompt = JUDGE_PROMPT.format(question=question, chunk=chunk)

    response = client.chat.completions.create(
        model=SMALL_MODEL,
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
