# Mindful Assistant

A safety-aware, agentic Retrieval-Augmented Generation (RAG) assistant that answers questions about anxiety using authoritative public-health sources (the U.S. National Institute of Mental Health). It is built to demonstrate production-minded RAG engineering: retrieval chosen by measurement rather than default, grounded generation, hallucination checking with an LLM judge, layered crisis-safety handling, and a LangGraph agent that routes each message to the appropriate response path.

> **This is an educational information tool — not a diagnostic tool, not therapy, and not a substitute for professional care.** If you or someone you know is in crisis, contact the 988 Suicide & Crisis Lifeline (call or text 988) or emergency services (911).

---

## Overview

Mindful Assistant takes a user's message and runs it through a stateful agent that first scores emotional risk and classifies intent, then routes the message down one of four paths:

**Psychoeducation** grounded question-answering from a vector knowledge base
**Clinical advice** a safe redirect (no diagnosis, no medication guidance)
**Crisis** a compassionate response surfacing hotline resources, with no retrieval or generation involved
**Greeting**  a scoped welcome that discloses what the assistant is and isn't

Every factual answer is generated *only* from retrieved source text and cites its source. Crisis handling is enforced **structurally** crisis messages never reach the generation path, rather than relying on prompt instructions alone. The agent maintains conversation memory across turns and accumulates a risk score, so a gradual decline in a user's messages can trigger escalation even when no single message would.

The project was deliberately built the way a real system would be: each major component was chosen by measuring alternatives against an evaluation set.

--

## Architecture

The system is a modular pipeline feeding a LangGraph agent:

**Pipeline stages**

1. **Ingestion** (`ingest.py`) fetches NIMH pages, extracts the main content (including list items), and stores text plus provenance metadata (source, URL, date, authority tier) as JSON.
2. **Chunking** (`chunk.py`) splits documents with LangChain's `RecursiveCharacterTextSplitter` (size 1000, overlap 200), preserving metadata per chunk.
3. **Embeddings** (`embed.py`) encodes chunks locally with `all-MiniLM-L6-v2` (384-dim). Local by design, so sensitive text never leaves the environment.
4. **Vector store & retrieval** (`retrieve.py`) — builds a FAISS index and retrieves by semantic similarity.
5. **Generation** (`generate.py`) composes a grounded, cited answer from retrieved context, under strict safety and grounding rules.
6. **Agent** (`agent.py`) a LangGraph state machine that scores risk, classifies intent, routes to the correct handler, and persists conversation state.
7. **Serving** (`app.py`) a FastAPI `/chat` endpoint, containerized with Docker.

--

## Key Features

- **Measured retrieval** — dense, sparse (BM25), hybrid (RRF fusion), and cross-encoder reranking were compared on a labeled gold set; the simplest option that performed best was chosen.
- **LLM-as-judge evaluation** — automated Precision@k using an LLM to label relevance, plus a separate faithfulness judge for hallucination detection.
- **Validated evaluators** — the faithfulness judge is checked with a negative-control test (known-good and known-bad answers) to confirm it actually discriminates.
- **Grounded, cited generation** — answers use only retrieved context and attribute their source.
- **Layered crisis safety** — an intent classifier for explicit crisis signals *plus* a cumulative risk score that escalates gradual decline across turns.
- **Structural guardrails** — crisis and clinical responses are fixed text, not model-generated, so they cannot hallucinate or drift.
- **Conversation memory** — multi-turn state via a LangGraph checkpointer, keyed by thread ID.
- **Deployable** — served over HTTP with FastAPI and packaged as a Docker container, with secrets injected at runtime.

---

## Tech Stack

| Tool | Role |

| **LangChain / LangGraph** | Pipeline components and the agent state machine |
| **FAISS** | Local vector store for semantic search |
| **sentence-transformers** (`all-MiniLM-L6-v2`) | Local embeddings (privacy-preserving) |
| **Groq** (Llama models) | LLM for generation, classification, and judging |
| **FastAPI + Uvicorn** | HTTP serving layer |
| **Docker** | Containerized, reproducible deployment |

LLMs are right-sized to the task: a large model for answer generation, small fast models for classification and judging.

---

## Evaluation & Results

Evaluation is done against a hand-built **gold set** of 23 questions (`gold_set.py`), deliberately including paraphrased, casual, and specific-detail phrasings so it stresses retrieval rather than flattering it.

**Retrieval comparison** (Precision@3, LLM-judged):

| Retriever | Precision@3 |

| Dense (semantic) | ~87% |
| Sparse (BM25) | ~77% |
| Hybrid (dense + sparse, RRF) | ~78% |
| Hybrid + cross-encoder rerank | ~87% |

**Chosen: dense retrieval.** On this small, clean, single-topic corpus, semantic search is already near-optimal; hybrid added noise rather than signal, and reranking did not beat plain dense. Hybrid and reranking are expected to pay off on larger, noisier, or jargon-heavy corpora — the decision is documented to be revisited at scale.

A notable part of this work was catching a **measurement bug**: an early comparison distorted results because the ensemble retriever returned a variable number of chunks, so precision computed against a fixed *k* was wrong. Inspecting the actual retrieved chunks (not just the metric) revealed it; the fix was to divide by the true number of results returned.

**Faithfulness.** Generated answers are judged for groundedness against their source context. The judge is validated with a negative-control test (`test_faithfulness_control.py`) covering invented statistics, overstated claims, invented treatment advice, and subtle factual contradictions — all correctly caught.

--

## Safety Design

Safety is treated as an engineering problem, not a disclaimer:

**Crisis and clinical replies are fixed text**, not model-generated — they cannot hallucinate or be prompted into unsafe output.
**Crisis messages bypass retrieval and generation entirely** — the agent routes them straight to a compassionate response with hotline resources (988 / 911), so a safe reply never depends on what the vector search happened to return.
**Two-layer crisis detection:** an intent classifier catches explicit signals immediately; a cumulative risk score is a backstop for gradual decline that no single message would trip.
**Scope boundaries:** the assistant never diagnoses, never advises on medication (start, stop, or change), never confirms or denies a user's beliefs about their own condition, discloses that it is an AI tool, and consistently directs users toward qualified professionals and human support.

--

## Setup & Running

**Prerequisites:** Python 3.11, a [Groq API key](https://console.groq.com), and (optionally) Docker.

**1. Install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure secrets** create a `.env` file

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

---

## Design Decisions

Major architectural decisions and their reasoning are recorded in [`DECISIONS.md`](./DECISIONS.md), including the chunking strategy, the choice of local embeddings (privacy), the all-LangChain framework, the measured retrieval decision, and the safety architecture. Each entry notes the alternatives considered and what would be reconsidered at scale.

---

## Limitations & Future Work

This is a portfolio project built to demonstrate engineering rigor, and its scope is honest about what it is:

**Single-topic corpus.** The knowledge base currently covers generalized anxiety disorder only. A depression document is ingested but not yet wired into the live index; adding it (a multi-topic corpus) is the natural next step and would likely change the retrieval trade-offs in favor of hybrid search.
**Small evaluation set.** 23 questions and a lightweight LLM judge mean small metric differences are noisy; results are read as directional, not exact.
**Crisis handling is a conservative baseline, not clinically validated.** It is designed to err toward caution, but real deployment would require clinical review and far more robust, tested detection.
**In-memory conversation state.** The checkpointer keeps history in RAM; a production system would use a persistent, database-backed store.
**Retrieval does not yet use conversation history.** Follow-up questions currently work partly because the corpus is single-topic; multi-topic use would require feeding history into retrieval.

**Planned next steps:** wire in the depression corpus, grow and diversify the gold set, add Recall and a framework-based evaluation (e.g. RAGAS), feed conversation history into retrieval, and add a persistent checkpointer.

---

*Built as a hands-on MVP RAG and agent engineering with an emphasis on measuring decisions, grounding outputs, and treating safety as architecture rather than an afterthought.*