import { useEffect, useRef, useState } from "react";
import { fetchHealth, streamQuery } from "./api";
import type { Citation, Claim, QueryEvent, Verdict } from "./types";
import "./App.css";

const COMPANIES = ["", "AMD", "NVDA"];
const QUARTERS = ["", "Q1", "Q2", "Q3", "Q4"];

export default function App() {
  const [question, setQuestion] = useState("");
  const [company, setCompany] = useState("");
  const [quarter, setQuarter] = useState("");

  const [statusLog, setStatusLog] = useState<string[]>([]);
  const [answer, setAnswer] = useState("");
  const [claims, setClaims] = useState<Claim[]>([]);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [verdict, setVerdict] = useState<Verdict>(null);
  const [attempts, setAttempts] = useState(0);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<string>("checking…");

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    fetchHealth()
      .then((h) => setHealth(`${h.status} · ${h.ingested_paragraphs} paragraphs`))
      .catch(() => setHealth("backend unreachable"));
  }, []);

  const reset = () => {
    setStatusLog([]);
    setAnswer("");
    setClaims([]);
    setCitations([]);
    setVerdict(null);
    setAttempts(0);
    setError(null);
  };

  const handleEvent = (evt: QueryEvent) => {
    switch (evt.event) {
      case "status":
        setStatusLog((l) => [...l, `Retrieving…`]);
        break;
      case "retrieved":
        setStatusLog((l) => [...l, `Retrieved ${evt.count} excerpts`]);
        break;
      case "draft":
        setStatusLog((l) => [...l, `Draft answer (attempt ${evt.attempt})`]);
        setAnswer(evt.answer ?? "");
        setClaims(evt.claims ?? []);
        break;
      case "verdict":
        setStatusLog((l) => [
          ...l,
          `Judge verdict: ${evt.verdict}${evt.retry_reason ? ` — ${evt.retry_reason}` : ""}`,
        ]);
        break;
      case "final":
        setAnswer(evt.answer ?? "");
        setClaims(evt.claims ?? []);
        setCitations(evt.citations ?? []);
        setVerdict(evt.verdict ?? null);
        setAttempts(evt.attempts ?? 0);
        break;
    }
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || loading) return;
    reset();
    setLoading(true);
    abortRef.current = new AbortController();
    try {
      await streamQuery(
        {
          question: question.trim(),
          company: company || null,
          quarter: quarter || null,
        },
        handleEvent,
        abortRef.current.signal,
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>Earnings-Call Intelligence Engine</h1>
        <span className={`health ${health.startsWith("ok") ? "ok" : "bad"}`}>{health}</span>
      </header>

      <form onSubmit={onSubmit} className="query-form">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about the ingested earnings calls, e.g. 'How is NVIDIA describing Blackwell demand?'"
          rows={3}
        />
        <div className="controls">
          <label>
            Company
            <select value={company} onChange={(e) => setCompany(e.target.value)}>
              {COMPANIES.map((c) => (
                <option key={c} value={c}>{c || "Any"}</option>
              ))}
            </select>
          </label>
          <label>
            Quarter
            <select value={quarter} onChange={(e) => setQuarter(e.target.value)}>
              {QUARTERS.map((q) => (
                <option key={q} value={q}>{q || "Any"}</option>
              ))}
            </select>
          </label>
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? "Thinking…" : "Ask"}
          </button>
        </div>
      </form>

      {error && <div className="error">{error}</div>}

      {statusLog.length > 0 && (
        <ul className="status-log">
          {statusLog.map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      )}

      {answer && (
        <section className="answer">
          <div className="answer-header">
            <h2>Answer</h2>
            {verdict && (
              <span className={`verdict ${verdict}`}>
                {verdict} · {attempts} attempt{attempts === 1 ? "" : "s"}
              </span>
            )}
          </div>
          <p>{answer}</p>

          {claims.length > 0 && (
            <>
              <h3>Claims</h3>
              <ul className="claims">
                {claims.map((c, i) => (
                  <li key={i}>
                    {c.claim}{" "}
                    {c.citations.map((cid) => (
                      <code key={cid} className="cite">{cid}</code>
                    ))}
                  </li>
                ))}
              </ul>
            </>
          )}

          {citations.length > 0 && (
            <>
              <h3>Sources</h3>
              {citations.map((c, i) => (
                <details key={i} className="source">
                  <summary>{c.claim}</summary>
                  {c.sources.map((s) => (
                    <blockquote key={s.id}>
                      <code className="cite">{s.id}</code>
                      <span>{s.text}</span>
                    </blockquote>
                  ))}
                </details>
              ))}
            </>
          )}
        </section>
      )}
    </div>
  );
}
