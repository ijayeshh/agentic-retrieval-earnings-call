import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fetchTrends } from "./api";
import type { TrendRow } from "./types";

const QUARTERS = ["Q1", "Q2", "Q3", "Q4"];
const COMPANIES = ["AMD", "NVDA"];

function seriesToData(row: TrendRow) {
  return QUARTERS.filter((q) => q in row.series).map((q) => ({
    quarter: q,
    mentions: row.series[q],
  }));
}

const DIRECTION_COLOR: Record<string, string> = {
  up: "#1b7a44",
  down: "#b02a2a",
  flat: "#7a7a8c",
  insufficient: "#aaa",
};

export default function TrendChart() {
  const [company, setCompany] = useState("AMD");
  const [rows, setRows] = useState<TrendRow[]>([]);
  const [selected, setSelected] = useState<TrendRow | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTrends(company)
      .then((d) => {
        setRows(d.trends);
        setSelected(d.trends[0] ?? null);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [company]);

  return (
    <div className="trends">
      <div className="controls">
        <label>
          Company
          <select value={company} onChange={(e) => setCompany(e.target.value)}>
            {COMPANIES.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </label>
      </div>

      {error && <div className="error">{error}</div>}
      {!error && rows.length === 0 && (
        <p className="muted">No topics with ≥2 quarters yet — ingest more transcripts.</p>
      )}

      {selected && (
        <>
          <h3>
            {selected.topic_name}{" "}
            <span className={`verdict ${selected.direction}`}>
              {selected.direction}
              {selected.tau !== null ? ` · τ=${selected.tau}` : ""}
            </span>
          </h3>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={seriesToData(selected)} margin={{ top: 8, right: 16, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e4e6ef" />
              <XAxis dataKey="quarter" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="mentions"
                stroke={DIRECTION_COLOR[selected.direction] ?? "#4b56d2"}
                strokeWidth={2}
                dot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}

      {rows.length > 0 && (
        <table className="trend-table">
          <thead>
            <tr><th>Topic</th><th>Series</th><th>τ</th><th>Trend</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr
                key={r.topic_id}
                className={selected?.topic_id === r.topic_id ? "active" : ""}
                onClick={() => setSelected(r)}
              >
                <td>{r.topic_name}</td>
                <td className="muted">
                  {QUARTERS.filter((q) => q in r.series).map((q) => `${q}:${r.series[q]}`).join(" ")}
                </td>
                <td>{r.tau ?? "—"}</td>
                <td><span className={`verdict ${r.direction}`}>{r.direction}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
