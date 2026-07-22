"""Phase 3 eval runner (plan Section 10) — regression harness.

Runs the local evals (ontology coherence, retrieval metrics) and, optionally, the
LLM-bound faithfulness eval on a subset. Prints a summary table and, when LangSmith
is configured, logs the run (metrics as outputs) via ``@traceable``.

Usage:
    python -m backend.eval.run_eval                 # coherence + retrieval
    python -m backend.eval.run_eval --faithfulness 5  # + faithfulness on 5 Qs
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from backend import observability
from backend.eval.coherence import evaluate_coherence
from backend.eval.retrieval_metrics import evaluate_retrieval

try:  # best-effort LangSmith logging; no-op if unavailable
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(*args: Any, **kwargs: Any):  # type: ignore
        def deco(fn):
            return fn
        return deco(args[0]) if args and callable(args[0]) else deco


@traceable(name="phase3-eval", run_type="chain")
def run(faithfulness_limit: int | None = None, k: int = 10) -> dict[str, Any]:
    """Run the eval suite and return a results dict (also the traced outputs)."""
    results: dict[str, Any] = {
        "coherence": evaluate_coherence(),
        "retrieval": evaluate_retrieval(k=k),
    }
    if faithfulness_limit is not None:
        from backend.eval.faithfulness import evaluate_faithfulness
        results["faithfulness"] = evaluate_faithfulness(limit=faithfulness_limit)
    return results


def _print_summary(results: dict[str, Any]) -> None:
    coh = results["coherence"]
    ret = results["retrieval"]["aggregate"]
    print("\n===== Phase 3 Eval Summary =====")
    print(f"Ontology coherence : {coh['coherence']} vs random {coh['random_baseline']} "
          f"(delta {coh['delta']}, beats_random={coh['beats_random']})")
    print(f"Retrieval @k={results['retrieval']['k']} : hit={ret['hit']} "
          f"recall={ret['recall']} mrr={ret['mrr']} ndcg={ret['ndcg']} precision={ret['precision']}")
    if "faithfulness" in results:
        f = results["faithfulness"]["aggregate"]
        print(f"Faithfulness (n={results['faithfulness']['n']}) : faithfulness={f['faithfulness']} "
              f"citation_accuracy={f['citation_accuracy']} pass_rate={f['pass_rate']} "
              f"uncited={f['uncited']}")
    print(f"\nLangSmith tracing: {'on' if observability.TRACING_ENABLED else 'off'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 eval harness")
    parser.add_argument("--k", type=int, default=10, help="retrieval cutoff k")
    parser.add_argument("--faithfulness", type=int, default=None,
                        help="run faithfulness eval on the first N gold questions (LLM calls)")
    parser.add_argument("--json", action="store_true", help="print full JSON results")
    args = parser.parse_args()

    results = run(faithfulness_limit=args.faithfulness, k=args.k)
    _print_summary(results)
    if args.json:
        print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
