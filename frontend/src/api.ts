import type { QueryEvent, QueryRequest } from "./types";

/**
 * POST to /query and parse the Server-Sent Events response incrementally.
 *
 * EventSource only supports GET, and our endpoint is POST-with-SSE, so we read
 * the response body stream manually and split on the SSE frame delimiter.
 */
export async function streamQuery(
  request: QueryRequest,
  onEvent: (event: QueryEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`Query failed: ${response.status} ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      const json = line.slice("data:".length).trim();
      if (!json) continue;
      try {
        onEvent(JSON.parse(json) as QueryEvent);
      } catch {
        // Ignore partial/non-JSON frames.
      }
    }
  }
}

export async function fetchHealth(): Promise<{ status: string; ingested_paragraphs: number }> {
  const response = await fetch("/health");
  if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
  return response.json();
}
