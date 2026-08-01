# Decision Log

Architecture and design decisions for the mindful assistant project.
Each entry: the decision, the alternatives considered, the reasoning, and what I'd reconsider at scale.

## 1. Chunking strategy: recursive character splitting

**Decision:** Use LangChain's `RecursiveCharacterTextSplitter` (chunk_size 1000, overlap 200).

**Alternatives considered:** Fixed size character chunking (simplest, but ignores meaning and cuts mid-sentence); semantic chunking (splits on meaning shifts, higher quality but slow and complex); document-structure-aware splitting (splits on the page's own headings).

**Reasoning:** Recursive splitting respects natural boundaries (paragraphs, then sentences) while keeping size tunable — the sweet spot of coherence vs. simplicity. Fixed size was too crude for structured health content; semantic was over engineering for a first build.

**Reconsider at scale / to improve:** Test structure-aware splitting (using the NIMH `<h2>` sections) and semantic chunking in Phase 3, and let retrieval metrics decide. Chunk size/overlap are tunable variables, not final.

---

## 2. Framework: all-LangChain (over mixing LlamaIndex)

**Decision:** Use the LangChain/LangGraph ecosystem end to end, including chunking.

**Alternatives considered:** LlamaIndex for retrieval mixed with LangGraph for the agent; writing splitting/retrieval by hand.

**Reasoning:** The project's differentiator is the agentic layer, which is LangGraph's strength. Using one coherent framework avoids two-framework glue code and version-juggling, and concentrates the deep learning where it matters most (the agent). LlamaIndex is a fair choice if the project were retrieval-only.

**Reconsider at scale:** If retrieval needed LlamaIndex-specific features (sentence-window, hierarchical parsing), consider integrating it behind an interface.

## 3. Embedding model: local sentence-transformers (all-MiniLM-L6-v2)

**Decision:** Run embeddings locally using the open-source `all-MiniLM-L6-v2` model via sentence-transformers.

**Alternatives considered:** (a) API embeddings — OpenAI, Cohere, Voyage — higher quality, zero infra, but text leaves our environment. (b) Larger local models — mpnet, BGE — better quality, but slower and heavier.

**Reasoning:** Two decisions. *Local over API:* user messages are sensitive mental-health data, so processing embeddings entirely in-house — with text never transmitted to a third party — is a privacy and responsible-design requirement, not a preference. *MiniLM specifically:* it's small, fast, and well-understood, which makes it a solid default to build the whole pipeline against; and since the embedding model is a variable I'll tune by evaluation in Phase 3, I deliberately started with a fast default rather than over-optimizing the choice now.

**Reconsider / to improve:** Test larger local models (mpnet, BGE) in Phase 3 and let retrieval metrics decide. At scale, weigh the cost/latency/ops burden of self-hosting — but the privacy imperative doesn't diminish with scale, so I'd only consider an API if privacy requirements could still be met. Separately, add a swappable *free* API embedding option in Phase 5 to demonstrate modularity and API integration.     

## 4. Retrieval strategy: dense retrieval (after comparing dense / sparse / hybrid / reranking)

**Decision:** Use plain dense (semantic) retrieval for this corpus.

**Alternatives tested — measured on a 23-question gold set with an LLM judge (Precision@3):**
- Dense (semantic): ~87%
- Sparse (BM25 keyword): ~77%
- Hybrid (dense + sparse, RRF fusion): ~78%
- Hybrid + cross-encoder reranking (bge-reranker-base): ~87%

**Reasoning:** On this small (~16 chunks), clean, single-topic corpus, dense retrieval is already near-optimal. Hybrid did not help and slightly hurt — semantic matching is strong here, so BM25 keyword fusion mostly added noise rather than signal. Reranking recovered the noisier hybrid pool back to dense's level but did not beat plain dense. So the simplest option (dense) is also the best here.

**Key process note:** An early comparison wrongly showed hybrid at 87% and reranking hurting — this was a *measurement bug*: EnsembleRetriever returns a variable number of chunks (not capped at k), so precision computed as relevant/k was distorted. Fixed by dividing by the actual number of chunks returned. Lesson: inspect the retrieved chunks, not just the metric.

**Reconsider at scale:** Hybrid and reranking are expected to pull ahead on larger, noisier, or jargon-heavy corpora, and with exact-term queries (e.g., drug names, codes). Revisit both when the knowledge base grows. Also test a stronger reranker (bge-reranker-v2-m3) at that point.

**Caveats:** Small gold set (23 Qs) and a lightweight judge (8B) mean small gaps are noisy; treat differences under ~10 points as suggestive, not conclusive.

---

## 5. LLM provider: Groq free tier

**Decision:** Use Groq's free API tier (Llama models) for generation, classification, and judging.

**Alternatives considered:** OpenAI / Anthropic APIs (higher quality, paid); Google Gemini free tier (generous but trains on prompts); running a local LLM (fully private, but slow on consumer hardware).

**Reasoning:** Free, no credit card, generous daily limits, fast inference, and OpenAI-compatible so the integration pattern is portable. Critically, the privacy calculus differs by use: evaluation only ever sends public NIMH content and synthetic test questions, so a training-on-prompts free tier is acceptable there. Real user messages in production would not be routed through a free tier.

**Reconsider at scale:** Move to a paid tier or self-hosted model if privacy requirements tighten or rate limits bind. Models are also right-sized per task — 70B for generation, 8B for classification and judging — which cut token usage substantially after hitting a daily rate limit.

---

## 6. Generation safety: prompt guardrails with a hard crisis override

**Decision:** Enforce safety through explicit prompt rules, with the crisis instruction positioned first and last and stated to override the grounding rule, plus few-shot examples.

**Alternatives considered:** Grounding rules alone; a post-generation safety filter; fine-tuning for safe behavior.

**Reasoning:** Adversarial testing exposed a real failure: a message expressing hopelessness was answered with "I don't have enough information," because the grounding rule overrode the crisis rule. The fix was structural — crisis instructions at both edges of the prompt (models attend most to edges), explicit override language, and hardcoded crisis resources so a safe response never depends on retrieval. Few-shot examples were then added to fix over-triggered refusals, calibrating *when* to refuse versus answer.

**Key lesson:** Writing a safety rule is not the same as it working. Only adversarial testing revealed the conflict.

**Reconsider:** Prompt-level guardrails are soft. The agent's routing layer (decision 8) makes crisis handling structural rather than instruction-dependent.

---

## 7. Faithfulness evaluation: LLM judge with negative control

**Decision:** Measure generation groundedness with a dedicated LLM judge (`GROUNDED` / `NOT_GROUNDED`), validated by a negative-control test.

**Alternatives considered:** Manual review; RAGAS or a similar framework; string-overlap heuristics.

**Reasoning:** String matching cannot detect paraphrased hallucination. A separate judge — with different inputs from the relevance judge (answer vs. source, not question vs. chunk) — handles this. Crucially, an all-GROUNDED result is meaningless unless the judge is proven to discriminate, so a negative-control set of deliberately unfaithful answers (invented statistic, overstatement, invented treatment advice, a 3-vs-6-month factual contradiction) was used to confirm it catches subtle hallucination.

**Reconsider:** Adopt RAGAS for standardized metrics and add Recall alongside Precision as the corpus and gold set grow.

---

## 8. Architecture: LangGraph agent with intent routing

**Decision:** Replace the fixed retrieve→generate pipeline with a LangGraph state machine that classifies each message and routes it to one of four handlers (psychoeducation, clinical, crisis, greeting).

**Alternatives considered:** Keeping the linear pipeline with prompt-based branching; hand-rolled if/else routing.

**Reasoning:** Prompt-based safety is a soft guardrail, the model *chooses* to comply. Routing makes it structural: crisis messages never reach retrieval or generation at all. Crisis and clinical handlers return fixed text with no LLM call, so those paths cannot hallucinate. The greeting handler does use an LLM (tightly constrained) so responses feel natural while still disclosing scope — flexibility where stakes are low, hardcoding where they are high.

**Reconsider:** Add human-in-the-loop escalation and a therapist-handoff path.

---

## 9. Cross-turn risk tracking

**Decision:** Accumulate a per-message risk score (0–3) in agent state; if the cumulative total crosses a threshold, override normal routing and escalate to the crisis path.

**Alternatives considered:** Per-message classification only; sentiment analysis; no cross-turn tracking.

**Reasoning:** Risk often builds gradually, no single message trips a crisis classifier, but the trajectory is concerning. Conversation memory makes accumulation possible. The two mechanisms are complementary: the classifier catches explicit signals immediately, the risk score is a backstop for gradual decline. In testing the classifier usually escalates first, which is the safe outcome; the risk tracker covers cases that stay individually sub-threshold.

**Reconsider:** Threshold and scoring are tuned conservatively (a false escalation is far less harmful than a missed one) but are not clinically validated. Real deployment would require clinical review and testing against real conversation data.

---

## 10. Deployment: FastAPI + Docker, secrets at runtime

**Decision:** Serve the agent via a FastAPI `/chat` endpoint, containerized with Docker, with `.env` excluded from the image and injected at runtime.

**Reasoning:** FastAPI gives typed request/response validation and auto-generated interactive docs. Docker makes the environment reproducible anywhere and is the foundation for cloud deployment. Secrets are never baked into the image — anyone with the image could read them — so they are passed in at `docker run`.

**Reconsider at scale:** Swap the in-memory checkpointer for a persistent store, mount the corpus as a volume rather than baking it into the image (so data updates don't require a rebuild), and add rate-limit handling with exponential backoff plus caching of judge results.