# Decision Log

Architecture and design decisions for the mindful-assistant project.
Each entry: the decision, the alternatives considered, the reasoning, and what I'd reconsider at scale.

## 1. Chunking strategy: recursive character splitting

**Decision:** Use LangChain's `RecursiveCharacterTextSplitter` (chunk_size 1000, overlap 200).

**Alternatives considered:** Fixed-size character chunking (simplest, but ignores meaning and cuts mid-sentence); semantic chunking (splits on meaning shifts, higher quality but slow and complex); document-structure-aware splitting (splits on the page's own headings).

**Reasoning:** Recursive splitting respects natural boundaries (paragraphs, then sentences) while keeping size tunable — the sweet spot of coherence vs. simplicity. Fixed-size was too crude for structured health content; semantic was over-engineering for a first build.

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