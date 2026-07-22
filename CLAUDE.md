# CLAUDE.md

## Current Phase
**Phase 3 - Orchestration + eval** (next)
Supervisor routing; Judge + retry in-graph; ontology coherence vs random + RAGAS (15Q) + retrieval metrics; LangSmith regression suite.

**Phase 2 — Trend insights: ✅ complete (code).** `backend/insights/trend_kendall.py` (per-company/topic quarterly frequency → Kendall's tau → up/down/flat, product-name topics excluded); `GET /trends` and `GET /ontology` endpoints; frontend tabs — Recharts trend chart + interactive ontology explorer. Verified against AMD Q1+Q2 data.

**Phase 1 — Core Ontology: ✅ complete (code); ingestion PAUSED.** Topic Retriever (batched) + Ontologist (local existence pre-filter → batched LLM existence/insertion) + expanded seed ontology (13 roots); SQLite tree/occurrences/embeddings/checkpoint (`data/earnings_intel.db`). Groq gpt-oss-20b tuned for free-tier (reasoning_effort=low, max_tokens, groq_rpm throttle + 429 backoff). **Ingest paused at Groq daily caps:** amd-q1 done, amd-q2 partial (para 53), rest pending. Resume any day via `ingest_all()` (skips done, resumes partial) — **do NOT wipe `data/`**.

**Phase 0 — Skeleton: ✅ complete.** Repo scaffolded; config-driven model broker + smoke test (Groq + local embedding); ingestion pipeline (dual AMD/NVIDIA transcript formats → ChromaDB, resumable checkpoint); RAG + Judge cited Q&A; FastAPI (`/query` SSE stream, `/health`); React+TS single-page frontend verified end-to-end through the Vite proxy.

## What this is
An agentic RAG system that ingests tech-sector earnings call transcripts, autonomously extracts financial topics, organizes them into a self-evolving hierarchical ontology, and surfaces trend insights plus cited analyst Q&A. It reproduces the Bloomberg FinIR 2025 paper *"Agentic Retrieval of Topics and Insights from Earnings Calls"* (arXiv 2507.07906). Scope: technology sector, 2 companies × 4 quarters = 8 transcripts placed manually in `data/raw/`, paragraph-level granularity.

**Current corpus:** 7 transcripts present (AMD Q1/Q2/Q4 + NVDA Q1–Q4; **AMD Q3 2025 missing**). Ingested into the ontology so far: AMD Q1 (full) + AMD Q2 (partial, para 53) — see paused-ingest note above.

## Tech stack
- **LLM roles:** Groq `openai/gpt-oss-20b` (extraction, routing, Q&A synthesis, Judge, ontology insertion)
- **Embeddings:** local HuggingFace `sentence-transformers` (retrieval + existence pre-filter; `all-MiniLM-L6-v2` for coherence metric) — no API, no rate limit
- **Orchestration:** LangChain + LangGraph
- **Backend:** FastAPI (Python, async)
- **Frontend:** TypeScript + React (single page, Recharts)
- **Storage:** SQLite (ontology tree, occurrences, session + ingest checkpoints) + ChromaDB (paragraph chunks + local embeddings) — all local, one `data/earnings_intel.db`, no Docker
- **Tracing/eval/monitoring:** LangSmith
- **Interop:** one MCP server (`query_ontology`, `topic_trend`)

> Note: the project plan specifies **SQLite + ChromaDB** (local-first, no Docker), not Postgres. Follow the plan.

## Requirements checklist (Section 17)
1. **TypeScript** — React + TS single page.
2. **FastAPI** — `/ingest /query /trends /health /metrics`.
3. **LangChain + LangGraph** — Supervisor → (Retriever → Ontologist) | (RAG → Judge).
4. **Models/embeddings/HF** — Groq `gpt-oss-20b` + local HF embeddings + broker.
5. **Memory (both)** — checkpointer (session) + SQLite/ChromaDB (persistent).
6. **Evaluation** — ontology coherence vs random + RAGAS + retrieval metrics in LangSmith.
7. **Observability** — LangSmith tracing.
8. **Monitoring** — LangSmith dashboards/alerts.
9. **MCP** — one ontology/trends server.

## Correctness rules (Section 16 — must hold)
1. **Sequential single-writer ingestion** — never run parallel Ontologist writes, or duplicate root nodes appear.
2. **Existence pre-filter is a fast path, not a replacement** — borderline candidates still go to the LLM equivalence check.
3. **One concept per topic** — the Retriever must not concatenate concepts.
4. **Insertion picks the most specific valid parent, else root** — no invented or skipped levels.
5. **Every RAG claim carries a citation** — the Judge fails answers with uncited factual claims.
6. **Model names are config-driven** — never hardcoded (Groq rotates its lineup).

## Repo structure (Section 14)
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

## Coding conventions
- **Python 3.11+**, type hints on all functions.
- **Async FastAPI** endpoints.
- **Pydantic models** for all schemas (requests, responses, graph state, LLM JSON I/O).
- Model names are **config-driven** (`backend/models/config.yaml`), never hardcoded.
- Groq calls reuse fixed/cached system prompts; batch extraction (~12 paragraphs) and insertion (~10 topics); throttle to RPM and back off on HTTP 429 via `retry-after`.
