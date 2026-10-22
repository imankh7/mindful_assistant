# Decision Log

Architecture and design decisions for the Mindful Assistant project.
Each entry records the decision, the alternatives considered, the reasoning, and what I'd reconsider at scale.

---

## 1. Chunking strategy: recursive character splitting

**Decision:** Use LangChain's `RecursiveCharacterTextSplitter` (chunk_size 1000, overlap 200).

**Alternatives considered:** Fixed-size character chunking (simplest, but ignores meaning and cuts mid-sentence); semantic chunking (splits on meaning shifts, higher quality but slow and complex); document-structure-aware splitting (splits on the page's own headings).

**Reasoning:** Recursive splitting respects natural boundaries (paragraphs, then sentences) while keeping size tunable: the sweet spot of coherence vs. simplicity. Fixed-size was too crude for structured health content; semantic chunking was over-engineering for a first build.

**Reconsider / to improve:** Test structure-aware splitting (using the NIMH `<h2>` sections) and semantic chunking once the corpus grows, and let retrieval metrics decide. Chunk size and overlap are tunable variables, not final.

---

## 2. Framework: all-LangChain (over mixing in LlamaIndex)

**Decision:** Use the LangChain/LangGraph ecosystem end to end, including chunking.

**Alternatives considered:** LlamaIndex for retrieval mixed with LangGraph for the agent; writing splitting and retrieval by hand.

**Reasoning:** The project's differentiator is the agentic layer, which is LangGraph's strength. One coherent framework avoids two-framework glue code and version juggling, and concentrates the deep learning where it matters most (the agent). LlamaIndex would be a fair choice for a retrieval-only project.

**Reconsider at scale:** If retrieval needed LlamaIndex-specific features (sentence-window retrieval, hierarchical parsing), integrate it behind an interface. Separately, `langchain-community` is being sunset, so the FAISS import should move to the standalone `langchain-faiss` package.

---

## 3. Embedding model: local sentence-transformers (all-MiniLM-L6-v2)

**Decision:** Run embeddings locally with the open-source `all-MiniLM-L6-v2` model via sentence-transformers.

**Alternatives considered:** (a) API embeddings (OpenAI, Cohere, Voyage): higher quality and zero infrastructure, but text leaves our environment. (b) Larger local models (mpnet, BGE): better quality, but slower and heavier.

**Reasoning:** Two decisions. *Local over API:* user messages are sensitive mental-health data, so embedding them entirely in-house, with text never sent to a third party, is a privacy and responsible-design requirement, not a preference. *MiniLM specifically:* it's small, fast, and well understood, which makes it a solid default to build the whole pipeline against. Since the embedding model is a variable to tune by evaluation, I deliberately started with a fast default rather than over-optimizing the choice up front.

**Reconsider / to improve:** Test larger local models (mpnet, BGE) and let retrieval metrics decide. At scale, weigh the cost, latency, and operational burden of self-hosting, but the privacy requirement doesn't shrink with scale, so an API would only be considered if privacy could still be guaranteed.

---

## 4. Retrieval strategy: dense retrieval (after comparing dense / sparse / hybrid / reranking)

**Decision:** Use plain dense (semantic) retrieval for this corpus.

**Alternatives tested (Precision@3 on a 23-question gold set, judged by `gpt-oss-20b`):**
- Dense (semantic): 56.5%
- Sparse (BM25 keyword): 47.8%
- Hybrid (dense + sparse, RRF fusion): 53.6%
- Hybrid + cross-encoder reranking (`bge-reranker-base`): 59.4%

**Reasoning:** Hybrid + reranking scored marginally higher, but the gap is two chunk judgements out of 69, which is within noise for a gold set this size. Dense is simpler and faster, so it is the better trade-off here. BM25 trails by about 9 points: on a small (16 chunks), clean, single-topic corpus, semantic matching already captures what keyword matching would, so fusing BM25 in mostly adds noise.

**Key process note (a measurement bug):** An early comparison wrongly favoured hybrid. `EnsembleRetriever` returns a variable number of chunks (not capped at *k*), so hybrid was effectively scored on more chunks than dense. A first fix divided by the number of chunks actually returned, but that still compared retrievers on different numbers of chunks. The final fix truncates every retriever's results to the top *k* before judging, so all are scored on exactly *k* chunks. Lesson: inspect the retrieved chunks, not just the metric.

**Judge dependence:** Absolute scores dropped when the judge changed from Llama 3.1 8B to `gpt-oss-20b` (see decision 5), because the new judge is stricter. With only 16 chunks, many questions have a single relevant chunk, which caps Precision@3 at 33% for them. The decision rests on the comparison between retrievers under the same judge, not on absolute numbers.

**Update:** These numbers are from the GAD-only corpus. Decision 14 re-runs the comparison after the depression page joined the corpus.

**Reconsider at scale:** Hybrid search and reranking are expected to pull ahead on larger, noisier, or jargon-heavy corpora, and on exact-term queries (drug names, codes). Revisit both when the knowledge base grows, and test a stronger reranker (`bge-reranker-v2-m3`) then. Treat differences under ~10 points as suggestive, not conclusive, until the gold set is larger.

---

## 5. LLM provider: Groq free tier, with model names centralized in `config.py`

**Decision:** Use Groq's free API tier for generation, classification, risk scoring, and judging, currently `openai/gpt-oss-120b` for generation and `openai/gpt-oss-20b` for everything else. Model names are defined once in `config.py`.

**Alternatives considered:** OpenAI / Anthropic APIs (higher quality, paid); Google Gemini free tier (generous, but trains on prompts); a local LLM (fully private, but slow on consumer hardware).

**Reasoning:** Free, no credit card, fast inference, and OpenAI-compatible, so the integration pattern is portable. The privacy calculus differs by use: evaluation only ever sends public NIMH content and synthetic test questions, so a free tier is acceptable there. Real user messages in production would not be routed through a free tier. Models are right-sized per task (the larger model for generation, the smaller one for classification and judging), which keeps token usage within free-tier limits.

**Model migration:** Groq retired the original models (`llama-3.3-70b-versatile` and `llama-3.1-8b-instant`) in August 2026. The project migrated to `gpt-oss-120b` / `gpt-oss-20b` and moved the model names, previously hard-coded in five files, into `config.py`, so the next change is one line. The migration also exposed behaviour changes covered in decisions 9 and 11: thresholds and intent categories tuned on one model don't automatically hold on another, so every evaluation was re-run after the switch.

**Reconsider at scale:** Move to a paid tier or self-hosted model if privacy requirements tighten or rate limits bind.

---

## 6. Generation safety: prompt guardrails with a hard crisis override

**Decision:** Enforce safety through explicit rules in a system prompt, with the crisis instruction positioned first and last and stated to override the grounding rule, plus few-shot examples.

**Alternatives considered:** Grounding rules alone; a post-generation safety filter; fine-tuning for safe behaviour.

**Reasoning:** Adversarial testing exposed a real failure: a message expressing hopelessness was answered with "I don't have enough information," because the grounding rule overrode the crisis rule. The fix was structural: crisis instructions at both edges of the prompt (models attend most to the edges), explicit override language, and hard-coded crisis resources, so a safe response never depends on retrieval. Few-shot examples were then added to fix over-triggered refusals, calibrating *when* to refuse versus answer.

After the model migration, adversarial testing (`test_safety.py`) caught a new failure: for "Do I have GAD?", the model listed the diagnostic criteria and then matched the user's own symptoms to them, which is effectively a diagnosis. A rule now forbids comparing the user's described symptoms to diagnostic criteria, with a concrete example of the phrasing to avoid. The instructions were also moved from a `user` message to a `system` message, where models treat them as rules rather than input.

**Key lesson:** Writing a safety rule is not the same as it working, and a rule that works on one model may not hold on the next. Only adversarial testing reveals these conflicts.

**Reconsider:** Prompt-level guardrails are soft; the agent's routing layer (decision 8) makes crisis handling structural. The current model sometimes adds crisis resources to non-crisis answers. That is harmless, but it is at odds with the "non-alarming" rule, so the override wording needs tuning for this model.

---

## 7. Faithfulness evaluation: LLM judge with negative control

**Decision:** Measure generation groundedness with a dedicated LLM judge (`GROUNDED` / `NOT_GROUNDED`), validated by a negative-control test.

**Alternatives considered:** Manual review; RAGAS or a similar framework; string-overlap heuristics.

**Reasoning:** String matching cannot detect paraphrased hallucination. A separate judge, with different inputs from the relevance judge (answer vs. source, not question vs. chunk), handles this. Crucially, an all-GROUNDED result is meaningless unless the judge is proven to discriminate, so a negative-control set of deliberately unfaithful answers (invented statistic, overstatement, invented treatment advice, a 3-vs-6-month factual contradiction) confirms it catches subtle hallucination. The control set was re-run after the model migration and still scores 5/5.

**Reconsider:** Adopt RAGAS for standardized metrics, and add Recall alongside Precision as the corpus and gold set grow.

---

## 8. Architecture: LangGraph agent with intent routing

**Decision:** Replace the fixed retrieve-then-generate pipeline with a LangGraph state machine that scores risk, classifies each message, and routes it to one of five handlers: psychoeducation, support, clinical, crisis, greeting.

**Alternatives considered:** Keeping the linear pipeline with prompt-based branching; hand-rolled if/else routing.

**Reasoning:** Prompt-based safety is a soft guardrail: the model *chooses* to comply. Routing makes it structural: crisis messages never reach retrieval or generation at all. Crisis and clinical handlers return fixed text with no LLM call, so those paths cannot hallucinate. The greeting and support handlers use the small model with a tightly constrained prompt, so replies feel natural while staying in scope: flexibility where stakes are low, hard-coding where they are high.

**Reconsider:** Add human-in-the-loop escalation and a therapist-handoff path.

---

## 9. Cross-turn risk tracking

**Decision:** Score each message's emotional risk (0-3) and accumulate it in agent state, with older scores decaying (decision 13); if the cumulative score reaches a threshold (currently 4), override normal routing and escalate to the crisis path.

**Alternatives considered:** Per-message classification only; sentiment analysis; no cross-turn tracking.

**Reasoning:** Risk often builds gradually: no single message trips a crisis classifier, but the trajectory is concerning. Conversation memory makes accumulation possible. The two mechanisms are complementary: the classifier catches crisis signals immediately, and the risk score is a backstop for gradual decline. In testing the classifier usually escalates first, which is the safe outcome; the risk tracker covers cases that stay individually below the classifier's bar.

**Known issues after the model migration:** The same four-message test conversation now totals 3, below the threshold of 4, because the new model rates distress more conservatively than the one the threshold was tuned on. The score also never decayed: once a thread crossed the threshold, every later message in it routed to crisis. Decision 13 fixes this with a decay.

**Reconsider:** Re-calibrate the threshold against a labelled set of risk messages. Threshold and scoring are deliberately conservative (a false escalation is far less harmful than a missed one) but are not clinically validated; real deployment would require clinical review and testing on real conversation data.

---

## 10. Deployment: FastAPI + Docker, secrets at runtime

**Decision:** Serve the agent through a FastAPI `/chat` endpoint, containerized with Docker, with `.env` excluded from the image and injected at runtime. An `.env.example` documents the required variables.

**Reasoning:** FastAPI gives typed request/response validation and auto-generated interactive docs. Docker makes the environment reproducible anywhere and is the foundation for cloud deployment. Secrets are never baked into the image (anyone with the image could read them), so they are passed in at `docker run`.

**Reconsider at scale:** Swap the in-memory checkpointer for a persistent store, mount the corpus as a volume rather than baking it into the image (so data updates don't require a rebuild), install CPU-only PyTorch to keep the image small, and add a health-check endpoint for cloud deployment.

---

## 11. Intent design: a SUPPORT path, and failing toward safety

**Decision:** Add a fifth intent, SUPPORT, for users who share stress, tiredness, or a hard time without asking a question or showing danger. It routes to a warm, constrained reply that acknowledges the feeling, encourages talking to someone they trust or a professional, and offers general information.

**Alternatives considered:** Keeping four intents and folding venting into GREETING or CLINICAL_ADVICE; a fixed-text support reply.

**Reasoning:** After the model migration, venting messages ("I've been really stressed lately", "work has been overwhelming and I'm exhausted") were classified as GREETING or CLINICAL_ADVICE, and the user received a cold medication disclaimer. The categories had a gap: nothing fit someone simply sharing how they feel. Two safety rules came with the new intent:
- **Fail toward CRISIS:** the classifier is told to choose CRISIS when unsure between SUPPORT and CRISIS, and the CRISIS definition now explicitly includes passive statements ("I can't keep going"). A false crisis reply offers a hotline that wasn't needed; a missed one is far worse.
- **Safe fallback:** unparseable classifier output now defaults to the fixed CLINICAL_ADVICE reply instead of PSYCHOEDUCATION, which would have generated an answer.

A bug was fixed at the same time: the prompt asked for labels spelled differently from the ones the parser searched for (`CLINICAL ADVICE` vs. `CLINICAL_ADVICE`), so a model that followed the instructions exactly would have fallen through to the default. The classifier test set grew to 11 cases, including boundary cases, and passes 11/11.

**Reconsider:** Grow the classifier test set, and measure SUPPORT/CRISIS confusion specifically, since that boundary matters most.

---

## 12. Evaluation resilience: retries with backoff, and caching judge verdicts

**Decision:** The relevance judge retries with exponential backoff (via `tenacity`) on rate-limit errors, and caches verdicts per (question, chunk) with `functools.lru_cache`.

**Alternatives considered:** Sleeping a fixed time between calls; upgrading to a paid tier; running evaluations in smaller batches.

**Reasoning:** A full retrieval comparison makes roughly 276 judge calls, and the free tier's limit of 8,000 tokens per minute crashed the run partway through. Exponential backoff (1s, 2s, 4s... up to 30s, six attempts) absorbs short bursts without slowing every call. Caching cuts repeated calls, since retrievers often return the same chunks, and it also makes the comparison fairer: a chunk receives the same verdict whichever retriever found it.

**Update:** The in-memory `lru_cache` was later replaced by a cache on disk (decision 15), so repeated runs reuse verdicts.

**Reconsider:** Apply the same retry policy to the other judges and to the agent's API calls.

---

## 13. Risk score decay: exponential, factor 0.7

**Decision:** Each turn, the cumulative risk score becomes `old × RISK_DECAY + new`, with `RISK_DECAY = 0.7`. The decay factor and the threshold (4) are defined together in `config.py`.

**Alternatives considered:** No decay (the original behaviour); a sliding window that sums only the last N turns; a decay factor of 0.5.

**Reasoning:** Without decay, a thread that crossed the threshold stayed in crisis routing for good, and a long conversation of mild messages would eventually trigger it. A decay keeps recent messages weighted most while still remembering earlier ones, and needs only one number of state, where a sliding window has to store the last N scores.

The factor was chosen with arithmetic, not by feel. If a user keeps sending messages rated `s`, the score levels off at `s / (1 − d)`. For sustained moderate distress (2s) to trigger, that ceiling must exceed 4, so `d > 0.5`. For sustained mild distress (1s) not to trigger, it must stay below 4, so `d < 0.75`. The first idea, 0.5, sits exactly on the lower edge: repeated 2s approach 4 but never reach it, so the score could never escalate gradual decline, which is the job it exists for. At 0.7, repeated 2s trigger on the third message (2, 3.4, 4.38), repeated 1s level off near 3.33, and two 3s trigger on the second. A single message (maximum 3) still cannot trigger by itself; the classifier's CRISIS intent covers that case.

**Implementation note:** The decay is applied by the LangGraph reducer on `risk_score`, which runs each time a node writes the field. Only `risk_node` may return `risk_score`; a second writer would decay the score twice per turn.

**Results:** The original four-message test conversation was rated 0, 1, 1, 1 and scored 0.00, 1.00, 1.70, 2.19, staying on normal routing (SUPPORT), as it should for mild venting. A second test conversation (`agent.py`, thread `conversation-2`) sends three messages that are each rated 2 but classified SUPPORT, so only the risk score can escalate. It scored 2.00, 3.40, 4.38 and routed to crisis on the third message while the classifier still said SUPPORT, which confirms the decay catches gradual decline on its own. An earlier version of that test used "I feel hopeless and can't cope", but the classifier routed it to CRISIS on the first message, which hid the risk score's effect. The test messages now have to be rated 2 by the risk model while the classifier still says SUPPORT.

**Reconsider:** The 0.5-0.75 range assumes the current threshold of 4 and the current model's ratings. Re-derive it if either changes. After the score crosses the threshold, one calm message (rated 0) multiplies it by 0.7 and can drop it back below 4 (4.38 becomes 3.07), so consider whether a sticky crisis mode that holds for a few turns would be safer. As with decision 9, these values are not clinically validated.


---

## 14. Corpus loading: every document in `data/processed/`

**Decision:** `load_corpus()` in `chunk.py` chunks every `.json` file in `data/processed/`, in sorted order, and returns one combined list. All 11 places that hard-coded `data/processed/gad.json` now call it, so adding a document no longer means editing every script. The first effect is that `depression.json`, which was already ingested but never loaded, joins the corpus: 37 chunks instead of 16.

**Alternatives considered:** Listing the files explicitly in `config.py`; keeping per-script paths.

**Reasoning:** A folder scan makes the corpus whatever has been ingested, with one place to change if that rule changes. Files are sorted because `glob` order depends on the operating system, and the order of documents in FAISS affects retrieval ties and evaluation runs. Only `*.json` is matched, since the folder also holds the `.txt` copies. An empty folder raises a clear error instead of letting FAISS fail later. Nothing in the code reads `chunk_index` (which now restarts at 0 for each file); `source` + `chunk_index` stays unique, and the relevance judge caches by chunk text.

**Retrieval comparison on the larger corpus (same 23 GAD gold questions, same judge, Precision@3):**
- Dense: 56.5% → 58.0% (+1 of 69 judgements)
- Sparse (BM25): 47.8% → 39.1% (−6)
- Hybrid: 53.6% → 53.6% (unchanged)
- Hybrid + rerank: 59.4% → 59.4% (unchanged)

Counting retrieved chunks by source explains the pattern. Of the 69 chunks each retriever returned for the GAD questions, depression chunks made up 8 for dense, 19 for hybrid, 21 for rerank and 29 for BM25. BM25 dropped the most because the two NIMH pages share generic wording ("treatment", "medication", "help"), so keyword matching pulls in the wrong page. Hybrid and rerank also retrieved many depression chunks without losing precision: the pages overlap in content (SSRIs, psychotherapy, finding help, co-occurring anxiety and depression), and the judge scores whether a chunk helps answer the question, not which page it came from. Dense's +1 is within noise.

**Conclusion:** Decision 4 stands. Dense remains the simple choice, and its gap to hybrid + rerank shrank from two judgements to one. The expectation that hybrid gains on a larger corpus did not hold here, but the corpus grew with a closely related page rather than a noisier or more jargon-heavy one.

**Reconsider:** Precision@3 cannot see one risk: a depression chunk judged relevant to an anxiety question (for example on medication) could lead the generator to present depression guidance as anxiety guidance. Measure faithfulness on mixed-source answers, and consider filtering or boosting by `source` when the question names a condition. The gold set is GAD-only, so depression retrieval is not measured at all; add depression questions, and some that span both pages, before adding more documents.

---

## 15. Gold set expansion: 55 questions, per-category results, and Source Hit@3

**Decision:** Grow the gold set from 23 GAD questions to 55: 23 GAD, 20 depression, 6 cross-topic (answerable from both pages) and 6 out-of-scope (the corpus cannot answer them). Every item now has `expected_source` (a list of source names, empty for out-of-scope) and `category`. Retrieval metrics use only the 49 in-scope items and are reported per category. A new metric, Source Hit@3, records whether at least one of the top 3 chunks came from an expected page.

**Alternatives considered:** Keeping a GAD-only gold set; separate gold set files per topic; measuring out-of-scope questions with retrieval metrics.

**Reasoning:** After decision 14, depression retrieval was not measured at all, and Precision@3 could not tell whether a relevant-looking chunk came from the right page. Source Hit@3 answers that directly and needs no judge calls. Out-of-scope questions have no right chunk, so they are excluded from retrieval metrics and checked on generation instead. One list with a `category` field keeps every evaluation script working with a one-line filter.

**Quality checks on the gold set itself:** `validate_gold_set.py` checks every item: a known category, `expected_source` as a list naming a real document, the phrase present in an expected source, and the phrase inside a single chunk (the hit-rate check searches chunk text, so a phrase split across a chunk boundary could never count as a hit). A negative control with one broken item per check confirmed each one fires.

**Retrieval results (49 in-scope questions, Precision@3 / Source Hit@3):**

| Retriever | GAD (23) | Depression (20) | Cross-topic (6) | Overall (49) |
|---|---|---|---|---|
| Dense | 58.0% / 100% | 61.7% / 100% | 55.6% / 100% | 59.2% / 100% |
| Sparse (BM25) | 39.1% / 95.7% | 36.7% / 100% | 44.4% / 100% | 38.8% / 98.0% |
| Hybrid | 53.6% / 100% | 55.0% / 100% | 50.0% / 100% | 53.7% / 100% |
| Hybrid + rerank | 59.4% / 95.7% | 63.3% / 100% | 38.9% / 100% | 58.5% / 98.0% |

**Interpretation:** Neither hybrid nor rerank beats dense. Overall, dense and rerank differ by one judgement out of 147 (87 vs. 86), which is noise; rerank is one judgement ahead on depression and three behind on cross-topic, where there are only 18 judgements. Hybrid trails dense in every category, and BM25 trails by about 20 points. Decision 4 stands. The GAD numbers match decision 14 exactly, which shows the judge is stable at temperature 0. Both Source Hit misses are single questions: BM25 answered "how many symptoms do you need to be diagnosed with GAD?" with three depression chunks (a real miss, from shared words like "symptoms" and "diagnosed"), and rerank did the same for "can you have anxiety and depression at the same time?", which is really a cross-topic question labelled GAD-only.

**Out-of-scope check (`evaluate_out_of_scope.py`):** Generation declined all 6 out-of-scope questions (two dosage questions, OCD, schizophrenia, therapy cost, a sprained ankle). Two methods were compared. An LLM judge (DECLINED or ANSWERED) handles varied wording but costs a call per answer, so it is run only after passing a negative control (one refusal, one invented dose). A refusal-phrase check is free but brittle: it first missed 5 of 6 refusals because the model writes curly apostrophes ("don’t"); after normalising apostrophes it agreed with the judge on all 6.

**Judge cost:** Relevance verdicts are now cached on disk (`.cache/judge_relevance.json`, not committed), keyed by a hash of the judge model, the judge prompt, the question and the chunk, so changing the judge or its prompt never reuses old verdicts. The first full comparison needed 296 API calls for 588 judgements (the four retrievers often return the same chunks); a repeat run needs none.

**Reconsider:** The depression questions were written while reading the page, so they may echo its wording more closely than the older GAD questions; phrase hit-rate is 90% for depression against 61% for GAD, and part of that gap may be the questions, not retrieval. Relabel "can you have anxiety and depression at the same time?" as cross-topic. Six cross-topic and six out-of-scope questions are too few to draw conclusions per category; grow both, and add out-of-scope questions that are close to the corpus (for example a GAD medication the page does not name).
