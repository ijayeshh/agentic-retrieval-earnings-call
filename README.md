# Agentic Earnings-Call Intelligence Engine — Build Spec

**What this is:** An agentic RAG system that ingests tech-sector earnings call transcripts, autonomously extracts financial topics, organizes them into a self-evolving hierarchical ontology, and surfaces trend insights plus cited analyst Q&A. Method reproduces the Bloomberg FinIR 2025 paper *"Agentic Retrieval of Topics and Insights from Earnings Calls"* (arXiv 2507.07906).

**Stack:** TypeScript/React frontend · FastAPI backend · LangChain + LangGraph · Groq (`openai/gpt-oss-20b`) for LLM roles · local HuggingFace `sentence-transformers` for embeddings · SQLite + ChromaDB (all local, no Docker) · LangSmith for tracing + eval + monitoring · one MCP server.

---

## 1. Scope

| Dimension | Value |
|---|---|
| Sector | Technology only |
| Companies | 2 (e.g. MSFT + GOOGL, or NVDA + AMD — overlapping AI/cloud narratives) |
| Time span | 4 quarters each |
| Corpus | 8 transcripts (~1,000 paragraphs), placed manually in `data/raw/` |
| Granularity | Paragraph-level |

Ingestion reads transcripts already present in `data/raw/`. There is no fetch/scrape/API step — the user provides the 8 transcript files.

---

## 2. Method (from the paper)

Three components + one insight analysis:

1. **Topic Retriever** — an LLM reads each paragraph and extracts financially-relevant topics, each with a short grounded excerpt. Topics span general (guidance, capex, dividends, buybacks, M&A, supply chain, margins) to specific (named products/initiatives). One concept per topic — never concatenate two concepts in one topic name. Base strictly on the text; no speculation.

2. **Topics Ontology** — a tree; each node is a topic with properties: `topic_id (UUID)`, `name (string)`, `aliases (list[string])`, `created_on (datetime)`, `updated_on (datetime)`.

3. **Ontologist Agent** — maintains the tree via two operations:
   - **Topic Existence:** decide whether a candidate topic is a true bidirectional semantic equivalent of an existing node/alias (same scope and specificity — NOT a parent/superset, NOT a child/subset). On match, append the candidate name as an alias and bump `updated_on`.
   - **Topic Insertion:** if no equivalent exists, find the most specific valid existing parent by top-down traversal and attach as its child; if none fits, create a new root node. Never invent categories, never skip levels.

The ontology is **seeded** with fixed top-level topics before ingestion (see `seed_topics.json`). Each ingested paragraph adds a topic, adds an alias, or attaches to a parent, and is stored enriched with the matched topic IDs + excerpts.

**Trend analysis:** a topic is trending if its quarter-over-quarter mention frequency changes monotonically; significance via **Kendall's tau**. Exclude pure product-name topics (their curve only tracks launch timing). This is deterministic Python, not an agent.

---

## 3. Architecture

```
┌───────────────────────────────────────────────────────────────┐
│  TypeScript / React frontend (single page)                     │
│  • query box + streamed cited answer                           │
│  • ontology tree explorer   • one trend chart                  │
└───────────────▲───────────────────────────────────────────────┘
                │ REST (actions) + SSE/WebSocket (streaming)
┌───────────────┴───────────────────────────────────────────────┐
│  FastAPI backend (Python)                                      │
│  • /ingest  /query  /trends  /health  /metrics                 │
│  • hosts LangGraph runtime; sequential resumable ingest        │
│  • /trends → plain-Python Kendall's tau (NOT an agent)         │
└───────┬───────────────────────────────────────────────────────┘
        │
┌───────┴─────────────────────  LangGraph  ──────────────────────┐
│  Supervisor (router)                                           │
│   ├── ingestion path:  Topic Retriever → Ontologist            │
│   └── Q&A path:        RAG/Retrieval → Judge → answer          │
└───────┬───────────────────────────────────────────────────────┘
        │
┌───────┴────────┐   ┌─────────────────────┐   ┌─────────────────┐
│ ChromaDB       │   │ SQLite              │   │ 1 MCP server    │
│ (paragraph     │   │ • ontology tree     │   │ (ontology/      │
│  chunks +      │   │   (adjacency list)  │   │  trends:        │
│  local embeds) │   │ • topic occurrences │   │  query_ontology,│
│                │   │ • session checkpoint│   │  topic_trend)   │
│                │   │ • ingest checkpoint │   │                 │
└────────────────┘   └─────────────────────┘   └─────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│ Cross-cutting: LangSmith (tracing + dashboards/alerts) •          │
│ model broker (Groq gpt-oss-20b + local HF embeddings)             │
└───────────────────────────────────────────────────────────────────┘
```

---

## 4. Data layer

### 4.1 Ingestion pipeline (sequential, resumable, local-first)
Process one transcript at a time; a single writer updates the ontology (prevents duplicate-node races). Checkpoint after each paragraph batch so a run can stop and resume.
1. **Read** transcript from `data/raw/`.
2. **Clean** — strip boilerplate/operator lines; keep speaker + section (prepared remarks vs Q&A).
3. **Segment** into paragraphs (unit of analysis).
4. **Metadata** per paragraph: `company_ticker, quarter, call_date, speaker, section, paragraph_index`.
5. **Embed** paragraphs **locally** (sentence-transformers) → upsert to vector store.
6. **Topic extraction** — batch ~12 paragraphs per Groq call (fixed/cached system prompt).
7. **Ontology update** — for each candidate topic: run a **local embedding existence pre-filter** (cosine vs existing node embeddings; above threshold → treat as alias with no LLM call; borderline → LLM existence check). Collect genuinely-new topics and **batch-insert** them (~10 per LLM call).
8. **Persist** — store the paragraph enriched with topic IDs + excerpts; write occurrence rows; update the ingest checkpoint.

### 4.2 Storage schema (SQLite + ChromaDB — all local, no Docker)
- **ChromaDB** (in-process, stores to a local folder): `{id, text, embedding, company, quarter, date, speaker, section, topic_ids[]}`.
- **SQLite — ontology** (adjacency list): `topic(topic_id TEXT PK, name TEXT, aliases JSON, parent_id TEXT NULL FK, created_on TEXT, updated_on TEXT)`. `parent_id NULL` = root. Node embeddings stored as JSON-serialized lists (or in a separate small table) to power the existence pre-filter.
- **SQLite — occurrences**: `topic_occurrence(topic_id, company, quarter, paragraph_id, excerpt)` — powers trend math.
- **SQLite — ingest checkpoint**: `ingest_progress(transcript_id, last_paragraph_index, status)` — enables resume.

All in one `data/earnings_intel.db` file. Zero setup, zero config, fully portable.

---

## 5. Agentic system (LangGraph)

### 5.1 Graph state
```
GraphState = {
  request_type,            # ingest | qa
  input_doc | question,
  retrieved_paragraphs,    # for QA
  candidate_topics,        # [{topic_name, excerpts[]}] from retriever
  ontology_ops,            # existence/insertion decisions + reasoning
  draft_answer,
  citations,               # [{claim, paragraph_id, span}]
  judge_verdict,           # pass | retry + reason
  retries,
  final_answer,
  messages                 # session history
}
```
Trend requests do not enter the graph — they hit the `/trends` Python module directly.

### 5.2 Nodes
- **(a) Supervisor / Router** — classify request into `ingest | qa`; route via conditional edges.
- **(b) Topic Retriever** *(ingestion)* — extract topics + grounded excerpts from a batch of ~12 paragraphs; JSON-only output (Groq JSON mode).
- **(c) Ontologist** *(ingestion)* — Topic Existence (local pre-filter → LLM for borderline; alias on true equivalence) and Topic Insertion (most-specific parent, else root; batched).
- **(d) RAG / Retrieval** *(Q&A)* — hybrid retrieval (local dense + optional BM25) over paragraphs, filterable by company/quarter/topic; returns paragraphs with IDs; drafts a cited answer.
- **(e) Judge (LLM-as-judge)** *(Q&A)* — verify every claim is grounded in retrieved spans and citations are correct; on failure emit a bounded `retry` (max 2).

### 5.3 Flows
- **Ingestion (sequential):** per transcript → per paragraph-batch → Retriever → Ontologist (existence pre-filter → LLM if borderline → batched insertion) → persist + checkpoint.
- **Query:** question → Supervisor → RAG → Judge (retry if needed) → stream cited answer → update session memory.
- **Trend:** `/trends` → Python (frequency series from occurrences → Kendall's tau → up/down/flat).

### 5.4 Seed ontology (`seed_topics.json`)
Curated tech roots (e.g. *AI & Machine Learning, Cloud & Data Center, Semiconductors & Hardware, Advertising, Consumer Devices, Software & Subscriptions, Capital Allocation, Macro & Regulatory, Supply Chain, Guidance*) plus a second level under each. Loaded into the `topic` table before ingestion.

Prompts for nodes (b)–(e) live in `backend/prompts.py`.

---

## 6. Models

Groq handles LLM roles; **HF embeddings run locally** (sentence-transformers — no API, no rate limit). All behind a model-agnostic broker so models swap via config.

| Role | Provider | Model (config-driven; verify live in Groq console) |
|---|---|---|
| Extraction, routing, Q&A synthesis, Judge | Groq | `openai/gpt-oss-20b` |
| Ontology insertion | Groq | `openai/gpt-oss-20b` (step up to `qwen/qwen3.6-27b` only if quality is weak) |
| Retrieval + existence-prefilter embeddings | local HF | a small BGE / GTE / E5 (`sentence-transformers`) |
| Coherence-metric embeddings | local HF | `all-MiniLM-L6-v2` |

Free-tier discipline: ~30 req/min, ~6,000 tokens/min, plus a daily token cap. Reuse fixed system prompts so cached tokens don't count toward limits; throttle to the RPM and back off on HTTP 429 using the `retry-after` header. Batch extraction and insertion to minimize calls.

---

## 7. Memory
- **Session:** LangGraph checkpointer (SQLite) holds thread state for follow-up questions.
- **Persistent:** SQLite ontology + occurrences + ChromaDB vector store. The ontology is the long-term memory; it grows and reorganizes as transcripts are ingested.

---

## 8. Trend insights (plain Python)
Build per-(company, topic) quarterly frequency series from `topic_occurrence` → Kendall's tau → classify up/down/flat with a significance threshold → exclude pure product-name topics. Exposed via `/trends`. Not an agent.

---

## 9. MCP
One MCP server exposing your engine with tools `query_ontology(topic|company)` and `topic_trend(company, topic)`, so external MCP clients (e.g. Claude Desktop) can query it. Two-way protocol demonstration.

---

## 10. Evaluation
Two things evaluated; only the RAG side uses LLM-as-judge.

**Ontology / topic system (local, no API):**
- **Ontology coherence** — `all-MiniLM-L6-v2`; mean parent↔child cosine similarity vs a random-parent baseline. Target: clearly beat random.
- **Topic-evolution accuracy** — small hand-labeled gold set: same topic recognized across quarters; genuinely new topics flagged new.

**RAG answers:**
- **Retrieval:** precision/recall/hit-rate@k, MRR, nDCG vs a small Q&A set with known-relevant paragraphs (programmatic).
- **Faithfulness / citation accuracy:** RAGAS + Judge on a ~15-question set (LLM-as-judge).

All datasets + evaluators registered in **LangSmith** and run as a regression suite on prompt/model changes. Gold sets are hand-built (small).

---

## 11. Observability & monitoring (LangSmith)
- **Observability:** tracing on every agent step — inputs/outputs, tool calls, tokens, latency per node, retry reasons.
- **Monitoring:** LangSmith dashboards + alerts — cost/run, latency, error rate, retrieval hit-rate, faithfulness-failure rate, ingestion throughput.

---

## 12. Frontend (TypeScript / React)
Single page:
- **Query interface** — streamed answers (SSE/WebSocket) + inline citation highlighting (click a claim → source paragraph).
- **Ontology explorer** — interactive tree; expand nodes, view aliases, per-topic timeline.
- **One trend chart** — quarter-over-quarter topic frequency for a selected company/topic (Recharts).

## 13. Backend (FastAPI)
Endpoints: `/ingest` (async, sequential, resumable), `/query` (stream), `/trends` (Python Kendall's tau), `/health`, `/metrics`. Hosts the LangGraph runtime; background worker for ingestion; auto OpenAPI docs.

---

## 14. Repo structure
```
earnings-intel/
├── CLAUDE.md
├── earnings-call-intelligence-project-plan.md
├── backend/
│   ├── app/ (FastAPI: routes, schemas, deps)
│   ├── agents/ (supervisor, retriever, ontologist, rag, judge)
│   ├── graph/ (LangGraph state, nodes, edges, checkpointer)
│   ├── ingestion/ (read, clean, segment, embed, enrich, resume)
│   ├── ontology/ (node model, sqlite tree ops, existence_prefilter.py, seed_topics.json)
│   ├── insights/ (trend_kendall.py)
│   ├── models/ (broker: groq + local hf embeddings, config.yaml)
│   ├── mcp/ (ontology_server)
│   ├── prompts.py
│   └── eval/ (coherence.py, ragas_suite.py, gold/)
├── frontend/ (React + TS: query, ontology explorer, one trend chart)
├── data/ (raw/  ← 8 transcripts provided by user, processed/)
├── notebooks/ (EDA, coherence analysis)
├── .env.example  .gitignore
└── README.md
```

---

## 15. Build phases

| Phase | Deliverable |
|---|---|
| 0. Skeleton | Scaffold repo; model broker + smoke test (Groq call + local embedding); FastAPI + React skeleton; basic cited RAG Q&A (RAG + Judge) reading from `data/raw/`. |
| 1. Core ontology | Topic Retriever + Ontologist (local existence pre-filter + insertion) + seed ontology; SQLite tree + occurrences; sequential resumable ingest. |
| 2. Trend insights | Kendall's tau module + `/trends`; one trend chart + ontology explorer in the UI. |
| 3. Orchestration + eval | Supervisor routing; Judge + retry; ontology coherence vs random + RAGAS (15Q) + retrieval metrics; LangSmith regression suite. |
| 4. MCP + observability + polish | One MCP server; LangSmith tracing + dashboards; README with the results table. |

Build thin end-to-end first (Phase 0) before deepening the ontology (Phase 1).

---

## 16. Correctness rules (must hold)
- **Sequential single-writer ingestion** — never run parallel Ontologist writes, or duplicate root nodes appear.
- **Existence pre-filter is a fast path, not a replacement** — borderline candidates still go to the LLM equivalence check.
- **One concept per topic** — the Retriever must not concatenate concepts.
- **Insertion picks the most specific valid parent, else root** — no invented or skipped levels.
- **Every RAG claim carries a citation** — the Judge fails answers with uncited factual claims.
- **Model names are config-driven** — never hardcoded (Groq rotates its lineup).

---

## 17. Requirements checklist
1. TypeScript — React+TS single page. 2. FastAPI — `/ingest /query /trends /health /metrics`. 3. LangChain + LangGraph — Supervisor → (Retriever → Ontologist) | (RAG → Judge). 4. Models/embeddings/HF — Groq `gpt-oss-20b` + local HF embeddings + broker. 5. Memory (both) — checkpointer (session) + SQLite/ChromaDB (persistent). 6. Evaluation — ontology coherence vs random + RAGAS + retrieval metrics in LangSmith. 7. Observability — LangSmith tracing. 8. Monitoring — LangSmith dashboards/alerts. 9. MCP — one ontology/trends server.
