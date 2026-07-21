"""
prompts.py — Prompt templates for the Agentic Earnings-Call Intelligence Engine

Covers every LLM node in the LangGraph (plan Section 5.2):
  1. SUPERVISOR_ROUTER      -> classify request: ingest | qa
  2. TOPIC_RETRIEVER        -> extract topics + grounded excerpts from paragraphs
  3. TOPIC_EXISTENCE        -> semantic-equivalence check vs shortlisted existing nodes
  4. TOPIC_INSERTION        -> pick the most specific valid parent (else root)
  5. RAG_SYNTHESIS          -> grounded, citation-linked answer for analyst Q&A
  6. JUDGE_FAITHFULNESS     -> LLM-as-judge: is every claim grounded + citations correct?

Design notes
------------
- Placeholders use string.Template (${var}) so the many literal JSON braces in the
  prompts DON'T need escaping (unlike str.format / f-strings).
      from string import Template
      msg = Template(TOPIC_RETRIEVER_USER).safe_substitute(paragraph_text=para)
- Every prompt that must return JSON says "JSON only, no preamble". Call these on Groq
  with JSON mode / structured output enabled and temperature 0 (0.2 max) for determinism.
- Model routing (from plan Section 6):
    all nodes (retriever, ontologist, judge, synthesis, routing) -> openai/gpt-oss-20b
    step up to qwen/qwen3.6-27b for insertion/judge ONLY if quality is weak
  Names are config-driven; Groq rotates its lineup.
- TOPIC_EXISTENCE receives a SHORTLIST of candidate nodes (top-k by embedding similarity
  to the query topic), NOT the whole ontology — keeps the prompt small and cheap.
- TOPIC_INSERTION receives a compact view of the tree: {super_parent: [child, child, ...]}.
- Prompts are adapted from and aligned with the methodology in the Bloomberg FinIR 2025
  paper "Agentic Retrieval of Topics and Insights from Earnings Calls" (arXiv 2507.07906),
  with structural improvements for Groq JSON-mode compatibility, batched ingestion,
  and our tech-sector / 2-company scope.
"""

# ---------------------------------------------------------------------------
# 1. SUPERVISOR / ROUTER
# ---------------------------------------------------------------------------
SUPERVISOR_SYSTEM = """You are the router for a financial earnings-call analysis system.
Classify the incoming request into exactly one route. Return JSON only, no preamble."""

SUPERVISOR_USER = """Classify the request into one route:

- "ingest": the input is an earnings-call transcript (or paragraph) to be processed
  into topics and the ontology.
- "qa": the input is an analyst question to be answered from already-ingested calls
  (e.g. "What did NVDA say about data-center demand in 2024?").

Note: trend/frequency questions (e.g. "is AI capex rising?") are handled by a separate
deterministic endpoint, NOT by you — only route "qa" for questions answerable from
retrieved transcript text.

Request:
${request_text}

Respond with JSON only:
{"route": "ingest" | "qa", "reason": "<one short sentence>"}"""


# ---------------------------------------------------------------------------
# 2. TOPIC RETRIEVER  (ingestion path)
# ---------------------------------------------------------------------------
TOPIC_RETRIEVER_SYSTEM = """You are a financial analyst at a buy-side investment firm.
Your goal is to extract insights on different topics from earnings-call text. These
topics will later be organized into a hierarchical ontology and used for trend detection
and cross-company analysis. You are precise, grounded, and never speculate.
Return JSON only, with no text before or after the JSON."""

TOPIC_RETRIEVER_USER = """From the earnings-call paragraph(s) below, extract the
financially relevant topics an equity analyst would track, each with short grounded
excerpts.

Rules:
1. Tag multiple topics when present — both general and specific.
   a. General examples: guidance, capex, dividends, buybacks, M&A, macro environment,
      labor, supply chain, pricing, consumer demand, margins.
   b. Specific examples: data center demand, AI inference, on-device AI, high bandwidth
      memory, cloud migration, advertising revenue, foundry capacity, Blackwell, EPYC.
   c. If no relevant financial topic is present, return an empty list.

2. When identifying topics, consider (not limited to):
   a. Key industry terms, processes, or entities mentioned
      (e.g. "pharmacy benefits managers", "NVLink", "Instinct GPU").
   b. Specific initiatives, products, or strategies discussed.
   c. Regulatory or market forces impacting the business.

3. Each topic is ONE concept. Never concatenate two concepts in a topic name
   (write "AI" and "Autopilot" separately, never "Autopilot and AI").

4. For each topic, write a concise excerpt (you may lightly reword for clarity) that
   captures how the topic was discussed. Do NOT speculate or add opinions. Base all
   observations strictly on the text.

5. The topics you extract here will later be assigned parent topics in a hierarchical
   ontology — so extract at the natural granularity of the text, neither too broad
   nor artificially narrow.

Context (for your reference only, do not extract topics about these metadata fields):
  Company: ${company}   Quarter: ${quarter}   Speaker: ${speaker}

Paragraph(s):
${paragraph_text}

Respond with JSON only, directly parseable, in exactly this shape:
[{"topic_name": "Data Center Demand", "excerpts": ["excerpt 1", "excerpt 2"]}]"""


# ---------------------------------------------------------------------------
# 3. ONTOLOGIST — TOPIC EXISTENCE  (semantic-equivalence check)
# ---------------------------------------------------------------------------
TOPIC_EXISTENCE_SYSTEM = """You are a semantic topic matcher for a financial ontology.
Your task is to determine if a given topic is semantically equivalent to any topics in a
candidate list. Matches must be true semantic equivalents — not broader parent categories,
not narrower child subsets. Return JSON only, no preamble."""

TOPIC_EXISTENCE_USER = """Determine if the QUERY topic is semantically equivalent to any
CANDIDATE topic. Follow these steps:

Step 1 — Analyze the core meaning, key concepts, and scope of the QUERY topic.

Step 2 — For each CANDIDATE, evaluate:
  - Core concepts and subject matter
  - Scope and specificity level
  - Whether a parent/subset relationship exists (this DISQUALIFIES a match)

Step 3 — For potential matches, verify and rate:
  - Confirm it is NOT a parent category or superset of the query
  - Confirm it is NOT a child or subset of the query
  - Rate similarity on a scale of 0–100
  - The relationship must be bidirectional: each topic should be substitutable for the other

Matching rules:
- Equivalent = same scope AND same specificity. Bidirectional substitution must hold.
- A candidate that is a PARENT/superset of the query is NOT a match.
  (e.g. "Data Center Demand" is NOT equivalent to "Cloud" — cloud is broader)
- A candidate that is a CHILD/subset of the query is NOT a match.
  (e.g. "Generative AI" is NOT equivalent to "AI & Machine Learning" — that is a parent)
- Aliases count: if the query means the same as a candidate's known alias, that is a match.

Valid match examples:
  "M&A" = "Mergers & Acquisitions"       (same concept, same scope)
  "AI inference" = "Inference workloads" (same activity, different words)
  "Customer acquisition" = "User growth" (same concept, same specificity)
Invalid match examples:
  "Data center demand"  != "Cloud"                 (cloud is broader)
  "Generative AI"       != "AI & Machine Learning" (parent category)
  "Market expansion"    != "Business Strategy"     (strategy is a superset)

QUERY topic:
${query_topic}

CANDIDATE topics (id | name | aliases):
${candidate_topics}

Respond with JSON only:
{
  "query_topic": "${query_topic}",
  "best_match": {
    "is_match": true | false,
    "matched_topic_id": "<uuid or null>",
    "matched_topic_name": "<name or null>",
    "similarity": <0-100>,
    "reasoning": "<why this is or isn't a same-scope equivalent>",
    "parent_subset_check": "<explicit statement that neither topic is a parent/subset of the other, or why it is>"
  },
  "other_candidates_considered": [
    {
      "topic_name": "<name>",
      "similarity": <0-100>,
      "rejected_reason": "<why not a match — e.g. parent category, too narrow, different domain>"
    }
  ]
}"""


# ---------------------------------------------------------------------------
# 4. ONTOLOGIST — TOPIC INSERTION  (find most specific parent, else root)
# ---------------------------------------------------------------------------
TOPIC_INSERTION_SYSTEM = """You are an expert ontologist agent maintaining a financial topic
tree. You place a new topic under the MOST SPECIFIC valid existing parent. You never invent
new categories, never skip levels, and always verify your answer exists in the tree.
Return JSON only, no preamble."""

TOPIC_INSERTION_USER = """Assign the NEW topic to the most appropriate parent already in
the ontology tree.

Planning steps (follow in order):
1. Start by finding the most appropriate SUPER-PARENT topic for the new topic.
2. Within that super-parent's children, identify the most SPECIFIC child that can serve
   as the parent. Drill down: prefer the most granular valid parent.
3. If no child is more specific than the super-parent itself, assign the super-parent
   as the parent.
4. If NO existing node in the entire tree is a sensible parent, assign "__ROOT__"
   (the topic becomes a new top-level node).

Rules:
- The parent MUST be an existing node in the tree below. Do not invent nodes.
- Do not skip levels.
- Do not rename anything.
- Maximum specificity: prefer "Machine Learning" over "Artificial Intelligence"
  when both could work.

NEW topic:
${new_topic}

Ontology tree (super-parent -> its child topics):
${ontology_tree}

Example 1:
  NEW topic: "Roboadvisor"
  Tree: {"Financial Technology": ["Digital Payments", "Fintech"], "Technology": ["5G", "Automation"]}
  Answer: {"parent": "Fintech", "reasoning": "Roboadvisor falls under super-parent Financial Technology, and within that Fintech is the most granular parent."}

Example 2:
  NEW topic: "Robotics"
  Tree: {"Technology": ["5G", "Automation"], "Environmental Issues": ["Air Quality"]}
  Answer: {"parent": "Technology", "reasoning": "Robotics falls under super-parent Technology, but no child is more specific, so Technology itself is the parent."}

Now assign the NEW topic above. Verify that the parent you choose exists in the tree.

Respond with JSON only:
{
  "parent": "<exact existing topic name, or __ROOT__>",
  "reasoning": "<which super-parent you chose, then the most granular child found (or why none), and confirmation the parent exists in the tree>"
}"""


# ---------------------------------------------------------------------------
# 5. RAG SYNTHESIS  (grounded, citation-linked answer)
# ---------------------------------------------------------------------------
RAG_SYNTHESIS_SYSTEM = """You are an equity-research assistant. You answer ONLY from the
retrieved earnings-call excerpts provided. Every factual claim must be traceable to a
specific excerpt id. If the excerpts do not support an answer, say so. Return JSON only."""

RAG_SYNTHESIS_USER = """Answer the analyst's question using ONLY the retrieved excerpts.

Requirements:
1. Ground every claim in the excerpts. Do not use outside knowledge or speculate.
2. Attach the supporting excerpt id(s) to each claim via the "citations" field.
3. If the excerpts are insufficient or unrelated, set "answer" to a brief statement that
   the available calls don't cover this, and return an empty "claims" list.
4. Be concise and analyst-appropriate; no filler.

Question:
${question}

Retrieved excerpts (id | company | quarter | text):
${retrieved_excerpts}

Respond with JSON only:
{
  "answer": "<concise prose answer grounded in the excerpts>",
  "claims": [
    {"claim": "<one factual statement>", "citations": ["<excerpt_id>", "..."]}
  ]
}"""


# ---------------------------------------------------------------------------
# 6. JUDGE  (LLM-as-judge: faithfulness + citation correctness)
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """You are a strict evaluator of grounded financial answers. You verify that
each claim is fully supported by its cited excerpt(s) and that nothing is fabricated.
You are conservative: if support is partial or absent, you fail the claim. Return JSON only."""

JUDGE_USER = """Evaluate whether the answer is faithful to the cited excerpts.

For each claim, assign a rating:
- "supported": the cited excerpt(s) clearly state or directly imply the claim.
- "partial": only part of the claim is supported, or the citation is weak.
- "unsupported": the excerpt(s) do not support the claim, or no valid citation is given.

Also flag any statement in the answer that has NO citation but asserts a fact.

Verdict rules:
- If ANY claim is "unsupported", verdict = "retry".
- If more than one claim is "partial", verdict = "retry".
- Otherwise, verdict = "pass".

Question:
${question}

Answer under evaluation (with its claims + citations):
${answer_json}

Excerpts referenced (id | text):
${referenced_excerpts}

Respond with JSON only:
{
  "verdict": "pass" | "retry",
  "claim_evaluations": [
    {"claim": "<claim>", "rating": "supported" | "partial" | "unsupported", "note": "<short reason>"}
  ],
  "uncited_assertions": ["<any factual sentence lacking a citation>"],
  "retry_reason": "<if verdict=retry, what to fix; else empty string>"
}"""


# ---------------------------------------------------------------------------
# Convenience registry + tiny usage helper
# ---------------------------------------------------------------------------
PROMPTS = {
    "supervisor":       {"system": SUPERVISOR_SYSTEM,       "user": SUPERVISOR_USER},
    "topic_retriever":  {"system": TOPIC_RETRIEVER_SYSTEM,  "user": TOPIC_RETRIEVER_USER},
    "topic_existence":  {"system": TOPIC_EXISTENCE_SYSTEM,  "user": TOPIC_EXISTENCE_USER},
    "topic_insertion":  {"system": TOPIC_INSERTION_SYSTEM,  "user": TOPIC_INSERTION_USER},
    "rag_synthesis":    {"system": RAG_SYNTHESIS_SYSTEM,    "user": RAG_SYNTHESIS_USER},
    "judge":            {"system": JUDGE_SYSTEM,            "user": JUDGE_USER},
}


def build_messages(name: str, **vars) -> list[dict]:
    """Return chat messages for a given prompt, with ${placeholders} filled.

    Example:
        msgs = build_messages("topic_retriever",
                              company="NVDA", quarter="2024Q3",
                              speaker="CFO", paragraph_text=para)
        # -> pass msgs to your Groq chat client with JSON mode + temperature 0
    """
    from string import Template
    p = PROMPTS[name]
    return [
        {"role": "system", "content": p["system"]},
        {"role": "user",   "content": Template(p["user"]).safe_substitute(**vars)},
    ]
