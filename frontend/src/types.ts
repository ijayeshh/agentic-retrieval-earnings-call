// Shapes mirroring the backend /query SSE payloads (backend/app/main.py).

export interface Claim {
  claim: string;
  citations: string[];
}

export interface CitationSource {
  id: string;
  text: string;
}

export interface Citation {
  claim: string;
  sources: CitationSource[];
}

export type Verdict = "pass" | "retry" | null;

/** One decoded SSE frame. `event` discriminates the payload. */
export interface QueryEvent {
  event: "status" | "retrieved" | "draft" | "verdict" | "final";
  stage?: string;
  count?: number;
  ids?: string[];
  attempt?: number;
  answer?: string;
  claims?: Claim[];
  citations?: Citation[];
  verdict?: Verdict;
  retry_reason?: string;
  attempts?: number;
}

export interface QueryRequest {
  question: string;
  company?: string | null;
  quarter?: string | null;
  k?: number;
}

// --- Trends (/trends) ---
export interface TrendRow {
  company: string;
  topic_id: string;
  topic_name: string;
  aliases: string[];
  series: Record<string, number>;
  tau: number | null;
  p_value: number | null;
  direction: "up" | "down" | "flat" | "insufficient";
  significant: boolean;
}

// --- Ontology (/ontology) ---
export interface OntologyNode {
  topic_id: string;
  name: string;
  aliases: string[];
  occurrences: number;
  children: OntologyNode[];
}
