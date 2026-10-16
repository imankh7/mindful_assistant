# Mindful Assistant

A safety-aware, agentic Retrieval-Augmented Generation (RAG) assistant that answers questions about anxiety using authoritative public-health content from the U.S. National Institute of Mental Health (NIMH). It is built to demonstrate production-minded RAG engineering: retrieval chosen by measurement rather than default, grounded and cited generation, hallucination checking with a validated LLM judge, layered crisis-safety handling, and a LangGraph agent that routes each message to the right response path.

> **This is an educational information tool, not a diagnostic tool, not therapy, and not a substitute for professional care.** If you or someone you know is in crisis, contact the 988 Suicide & Crisis Lifeline (call or text 988) or emergency services (911).

---

## Overview

Each user message goes through a stateful agent that first scores emotional risk and classifies intent, then routes the message down one of five paths:

- **Psychoeducation**: grounded question-answering from a vector knowledge base, with a cited source.
- **Support**: a warm, scoped acknowledgement when the user shares stress or a hard time, encouraging human and professional support.
- **Clinical advice**: a fixed, safe redirect with no diagnosis and no medication guidance.
- **Crisis**: a fixed, compassionate response with hotline resources; no retrieval or generation involved.
- **Greeting**: a scoped welcome that discloses what the assistant is and isn't.

Crisis handling is enforced **structurally**: crisis messages never reach the generation path, rather than relying on prompt instructions alone. The agent keeps conversation memory across turns and accumulates a risk score, so a gradual decline across several messages can trigger escalation even when no single message would.

Each major component was chosen by measuring alternatives against an evaluation set, and every decision is recorded in [`DECISIONS.md`](./DECISIONS.md).

---

## Architecture

Each message first gets a risk score, which accumulates across the conversation, and then an intent label. If the cumulative risk reaches the threshold or the intent is CRISIS, the message goes to the fixed crisis reply. Otherwise it goes to the clinical redirect, the RAG path, the support reply, or the greeting.

**Pipeline stages**

1. **Ingestion** (`ingest.py`): fetches NIMH pages, extracts the main content (including list items), and stores the text with provenance metadata (source, URL, date, authority tier) as JSON.
2. **Chunking** (`chunk.py`): splits documents with LangChain's `RecursiveCharacterTextSplitter` (size 1000, overlap 200), keeping metadata on every chunk.
3. **Embeddings** (`embed.py`): encodes chunks locally with `all-MiniLM-L6-v2` (384 dimensions). Local by design, so sensitive text never leaves the environment.
4. **Vector store & retrieval** (`retrieve.py`): builds a FAISS index and retrieves by semantic similarity.
5. **Generation** (`generate.py`): composes a grounded, cited answer from retrieved context under strict safety and grounding rules (sent as a system prompt).
6. **Agent** (`agent.py`): a LangGraph state machine that scores risk, classifies intent, routes to the right handler, and persists conversation state.
7. **Serving** (`app.py`): a FastAPI `/chat` endpoint, containerized with Docker.

Model names live in one place (`config.py`), so switching models is a one-line change.

---

## Key Features

- **Measured retrieval**: dense, sparse (BM25), hybrid (RRF fusion), and cross-encoder reranking compared on a labeled gold set; the simplest option within noise of the best was chosen.
- **LLM-as-judge evaluation**: automated Precision@k with an LLM labelling relevance, plus a separate faithfulness judge for hallucination detection.
- **Validated evaluators**: the faithfulness judge is checked with a negative-control set (known-good and known-bad answers) to confirm it actually discriminates.
- **Grounded, cited generation**: answers use only retrieved context and attribute their source.
- **Layered crisis safety**: an intent classifier for crisis signals, plus a cumulative risk score that escalates gradual decline across turns.
- **Fail-safe defaults**: when unsure between SUPPORT and CRISIS the classifier chooses CRISIS, and unparseable classifier output falls back to the fixed clinical reply, never to generation.
- **Structural guardrails**: crisis and clinical responses are fixed text, not model-generated, so they cannot hallucinate or drift.
- **Conversation memory**: multi-turn state via a LangGraph checkpointer, keyed by thread ID.
- **Resilient evaluation**: the relevance judge retries with exponential backoff on rate limits and caches verdicts per (question, chunk).
- **Deployable**: served over HTTP with FastAPI and packaged as a Docker container, with secrets injected at runtime.

---

## Tech Stack

| Tool | Role |
|---|---|
| **LangChain / LangGraph** | Pipeline components and the agent state machine |
| **FAISS** | Local vector store for semantic search |
| **sentence-transformers** (`all-MiniLM-L6-v2`) | Local embeddings (privacy-preserving) |
| **Groq** (`gpt-oss-20b`, `gpt-oss-120b`) | LLMs for classification, judging, and generation, configured in `config.py` |
| **FastAPI + Uvicorn** | HTTP serving layer |
| **Docker** | Containerized, reproducible deployment |
| **tenacity** | Retry with exponential backoff for rate-limited API calls |

Models are right-sized to the task: the larger model for answer generation, the smaller, faster model for classification, risk scoring, and judging.

---

## Evaluation & Results

Evaluation uses a hand-built **gold set** of 23 questions (`gold_set.py`), deliberately including paraphrased, casual, and specific-detail phrasings, so it stresses retrieval rather than flattering it.

**Retrieval comparison** (Precision@3 on the 23-question gold set, judged by `gpt-oss-20b`):

| Retriever | Precision@3 |
|---|---|
| Dense (semantic) | 56.5% |
| Sparse (BM25) | 47.8% |
| Hybrid (dense + sparse, RRF) | 53.6% |
| Hybrid + cross-encoder rerank | 59.4% |

**Chosen: dense retrieval.** Hybrid + rerank scored marginally higher, but the gap is two chunk judgements out of 69, which is within noise for a gold set this size. Dense is simpler and faster, so it is the better trade-off here. BM25 trails by about 9 points: on a small, clean, single-topic corpus, semantic matching already captures what keyword matching would. Hybrid search and reranking are expected to pay off on a larger, multi-topic corpus, so the decision is documented to be revisited at scale.

Absolute numbers depend on the judge model. When the project migrated from Llama 3.1 8B to `gpt-oss-20b` as the judge, all scores dropped because the new judge is stricter. With only 16 chunks, many questions have a single relevant chunk, which caps Precision@3 at 33% for those questions. The comparison *between* retrievers under the same judge is what the decision rests on.

**Catching a measurement bug.** An early comparison was distorted because the ensemble (hybrid) retriever returns a variable number of chunks, so precision computed against a fixed *k* was wrong. Inspecting the actual retrieved chunks, not just the metric, revealed it. The fix: every retriever's results are truncated to the top *k* before judging, so all retrievers are scored on exactly *k* chunks.

**Faithfulness.** Generated answers are judged for groundedness against their source context. The judge is validated with a negative-control test (`test_faithfulness_control.py`) covering invented statistics, overstated claims, invented treatment advice, and subtle factual contradictions. It classifies all 5 correctly.

**Intent classification.** The classifier is checked against 11 labelled messages (`python classify.py`), including boundary cases such as passive crisis statements and "I'm stressed, is that a disorder?". It passes 11/11.

---

## Safety Design

Safety is treated as an engineering problem, not a disclaimer:

- **Crisis and clinical replies are fixed text**, not model-generated, so they cannot hallucinate or be prompted into unsafe output.
- **Crisis messages bypass retrieval and generation entirely.** The agent routes them straight to a compassionate response with hotline resources (988 / 911), so a safe reply never depends on what the vector search returned.
- **Two-layer crisis detection:** the intent classifier catches crisis signals immediately, including passive ones ("I can't keep going"); the cumulative risk score is a backstop for gradual decline that no single message would trip.
- **Fail toward safety:** when unsure between SUPPORT and CRISIS, the classifier chooses CRISIS; unparseable output defaults to the fixed clinical reply.
- **Scope boundaries:** the assistant never diagnoses, never matches a user's described symptoms to diagnostic criteria, never advises on medication (start, stop, or change), never confirms or denies a user's beliefs about their own condition, discloses that it is an AI tool, and consistently directs users toward qualified professionals and human support.
- **Defense in depth:** the generation prompt enforces the same boundaries, so even a message the classifier misroutes is still answered safely (`test_safety.py` checks this layer on its own).

---

## Setup & Running

**Prerequisites:** Python 3.11, a [Groq API key](https://console.groq.com), and (optionally) Docker.

**1. Install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure secrets**
Copy the example file and add your Groq API key to it:
```bash
cp .env.example .env
```

**3. Run locally**
```bash
uvicorn app:app --reload
```
Then open the interactive docs at `http://127.0.0.1:8000/docs` and try the `/chat` endpoint.

**4. Or run in Docker**
```bash
docker build -t mindful-assistant .
docker run -p 8000:8000 --env-file .env mindful-assistant
```

**Example request**
```json
POST /chat
{
  "message": "What are the symptoms of generalized anxiety disorder?",
  "thread_id": "user-123"
}
```

**Run the evaluations**
```bash
python classify.py                  # intent classifier test cases
python test_faithfulness_control.py # faithfulness judge negative control
python compare_retrievers.py        # retrieval comparison (Precision@3)
python test_safety.py               # adversarial safety cases for generation
```

**Data:** content from the U.S. National Institute of Mental Health (NIMH), which is in the public domain. Processed copies are in `data/processed/`.

---

## Design Decisions

Major architectural decisions and their reasoning are recorded in [`DECISIONS.md`](./DECISIONS.md): the chunking strategy, local embeddings for privacy, the all-LangChain framework, the measured retrieval decision, the LLM provider and the model migration, the safety architecture, the SUPPORT intent, and rate-limit handling. Each entry notes the alternatives considered and what would be reconsidered at scale.

---

## Limitations & Future Work

This is a portfolio project built to demonstrate engineering rigor, and it is honest about its scope:

- **Single-topic corpus.** The knowledge base currently covers generalized anxiety disorder only. A depression document is ingested but not yet indexed; a multi-topic corpus is the natural next step and would likely shift the retrieval trade-offs toward hybrid search and reranking.
- **Small evaluation set.** 23 questions mean small metric differences are noisy; results are read as directional, not exact.
- **Risk threshold needs re-calibration.** The cumulative risk threshold was tuned on a previous model and needs re-calibration against a labelled set of risk messages.
- **Crisis resources over-appear.** The generation model sometimes adds crisis resources to non-crisis answers; the prompt's safety override needs tuning for the new model.
- **Crisis handling is a conservative baseline, not clinically validated.** It errs toward caution, but real deployment would require clinical review and far more robust, tested detection.
- **In-memory conversation state.** The checkpointer keeps history in RAM; production would use a persistent, database-backed store.
- **Retrieval does not yet use conversation history.** Memory is used for risk tracking, but follow-up questions are retrieved on the latest message only.

**Planned next steps:** index the depression corpus and more NIMH topics, grow the gold set to ~80-100 questions with an `expected_source` field and out-of-scope questions, add Recall and a framework-based evaluation (e.g. RAGAS), re-calibrate the risk threshold, feed conversation history into retrieval, add a persistent checkpointer, and migrate the FAISS import from `langchain-community` to the standalone `langchain-faiss` package.

---

*Built as a hands-on MVP of RAG and agent engineering, with an emphasis on measuring decisions, grounding outputs, and treating safety as architecture rather than an afterthought.*

