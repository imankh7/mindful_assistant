import json
import sys
from collections import Counter
from pathlib import Path
from chunk import load_corpus
from gold_set import gold_set

CATEGORIES = {"gad", "depression", "cross_topic", "out_of_scope"}


def load_texts(folder="data/processed"):
    texts = {}
    for path in sorted(Path(folder).glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        texts[doc["source"]] = doc["text"]
    return texts


def validate(gold_set, texts, chunks):
    failures = []
    seen = set()

    for item in gold_set:
        question = item["question"]
        category = item.get("category")
        sources = item.get("expected_source")
        phrase = item.get("expected_phrase")

        if question in seen:
            failures.append((question, "duplicate question"))
        seen.add(question)

        if category not in CATEGORIES:
            failures.append((question, f"unknown category {category!r}"))
            continue
        if not isinstance(sources, list):
            failures.append((question, "expected_source must be a list"))
            continue

        if category == "out_of_scope":
            if sources:
                failures.append((question, "out_of_scope must have an empty expected_source"))
            continue

        if not sources:
            failures.append((question, "in-scope item has an empty expected_source"))
            continue

        unknown = [s for s in sources if s not in texts]
        if unknown:
            failures.append((question, f"unknown source(s) {unknown}"))
            continue

        # The phrase only has to appear in one expected source (cross_topic
        # items have two). It must also fit in a single chunk, since the
        # hit-rate check searches chunk text, not whole documents.
        if not any(phrase in texts[s] for s in sources):
            failures.append((question, f"phrase not found in {sources}: {phrase!r}"))
        elif not any(
            phrase in c.page_content for c in chunks if c.metadata["source"] in sources
        ):
            failures.append((question, f"phrase spans a chunk boundary: {phrase!r}"))

    return failures


if __name__ == "__main__":
    texts = load_texts()
    chunks = load_corpus()
    failures = validate(gold_set, texts, chunks)

    counts = Counter(item.get("category") for item in gold_set)
    print(f"\n{len(gold_set)} items: {dict(counts)}")

    if failures:
        print(f"{len(failures)} failure(s):")
        for question, reason in failures:
            print(f"  - {question}\n      {reason}")
        sys.exit(1)

    print("All items valid.")
