import json
from pathlib import Path

ORDER = [
    "recall_at_k", "ndcg_at_k", "mrr", "context_precision", "context_recall",
    "faithfulness", "answer_relevance", "citation_validity", "abstention_accuracy",
]


def render(payload: dict, path: Path) -> None:
    m = payload["metrics"]
    lines = [
        f"# Relatório — {payload['label']}",
        "",
        f"Execução: `{payload['timestamp']}`  ",
        f"Itens: {m['n']}",
        "",
        "| Métrica | Valor |",
        "|---|---|",
    ]
    lines += [f"| {name} | {m[name]:.4f} |" for name in ORDER if name in m]

    failures = [
        r for r in payload["results"]
        if r["answerable"] and r.get("faithfulness", 1) < 0.6
    ][:10]

    lines += ["", "## Piores casos (faithfulness < 0.6)", ""]
    for r in failures:
        lines.append(f"- `{r['id']}` ({r['category']}) faith={r['faithfulness']:.2f}")
        if r.get("answer"):
            lines.append(f"  > {r['answer'][:240]}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare(a_path: Path, b_path: Path) -> str:
    a = json.loads(a_path.read_text())["metrics"]
    b = json.loads(b_path.read_text())["metrics"]
    rows = ["| Métrica | A | B | Δ |", "|---|---|---|---|"]
    for name in ORDER:
        if name in a and name in b:
            rows.append(f"| {name} | {a[name]:.3f} | {b[name]:.3f} | {b[name] - a[name]:+.3f} |")
    return "\n".join(rows)
