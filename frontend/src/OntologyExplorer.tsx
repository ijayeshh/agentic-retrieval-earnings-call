import { useEffect, useState } from "react";
import { fetchOntology } from "./api";
import type { OntologyNode } from "./types";

function TreeNode({ node, depth }: { node: OntologyNode; depth: number }) {
  const [open, setOpen] = useState(depth === 0);
  const hasChildren = node.children.length > 0;

  return (
    <li>
      <div className="tree-row" onClick={() => hasChildren && setOpen(!open)}>
        <span className="twist">{hasChildren ? (open ? "▾" : "▸") : "•"}</span>
        <span className="tree-name">{node.name}</span>
        {node.occurrences > 0 && <span className="badge">{node.occurrences}</span>}
        {node.aliases.length > 0 && (
          <span className="aliases" title={node.aliases.join(", ")}>
            ≈ {node.aliases.slice(0, 3).join(", ")}
            {node.aliases.length > 3 ? "…" : ""}
          </span>
        )}
      </div>
      {open && hasChildren && (
        <ul>
          {node.children.map((c) => (
            <TreeNode key={c.topic_id} node={c} depth={depth + 1} />
          ))}
        </ul>
      )}
    </li>
  );
}

export default function OntologyExplorer() {
  const [roots, setRoots] = useState<OntologyNode[]>([]);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchOntology()
      .then((d) => {
        setRoots(d.roots);
        setTotal(d.total);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  return (
    <div className="ontology">
      {error && <div className="error">{error}</div>}
      {!error && <p className="muted">{total} topics · click a node to expand</p>}
      <ul className="tree">
        {roots.map((r) => (
          <TreeNode key={r.topic_id} node={r} depth={0} />
        ))}
      </ul>
    </div>
  );
}
