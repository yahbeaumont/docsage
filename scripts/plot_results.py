"""Gera os gráficos do README a partir dos JSONs reais de eval.

Cada figura lê um arquivo produzido pela própria suíte (eval-out/*.json,
configs/calibrated.json). Nada é inventado: se a métrica for ruim, o gráfico
mostra a métrica ruim.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

ORDER = [
    ("recall_at_k", "Recall@k"),
    ("ndcg_at_k", "nDCG@k"),
    ("mrr", "MRR"),
    ("context_precision", "Precisão ctx"),
    ("context_recall", "Recall ctx"),
    ("faithfulness", "Faithfulness"),
    ("answer_relevance", "Relevância"),
    ("citation_validity", "Citação válida"),
    ("abstention_accuracy", "Abstenção"),
]
COLORS = {"baseline": "#d29922", "docsage": "#1f6feb"}


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=100)
    plt.close(fig)
    print(f"  {name}")


def plot_metrics(baseline: dict, current: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    width = 0.38
    for i, (key, _label) in enumerate(ORDER):
        for j, (arm, payload) in enumerate([("baseline", baseline), ("docsage", current)]):
            value = payload["metrics"].get(key, 0.0)
            x = i + (j - 0.5) * width
            ax.bar(x, value, width, color=COLORS[arm],
                   label=arm.capitalize() if i == 0 else None)
            ax.text(x, value + 0.015, f"{value:.2f}", ha="center", fontsize=11)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([label for _, label in ORDER], rotation=20, ha="right", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Valor da métrica (0–1)", fontsize=14)
    ax.set_title("Baseline vetorial vs. híbrido com rerank — 60 itens do golden set", fontsize=17)
    ax.legend(fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out, "metrics-comparison.png")


def plot_calibration(calibrated: dict, out: Path) -> None:
    curve = calibrated["curve"]
    thresholds = [p["threshold"] for p in curve]
    fig, ax = plt.subplots(figsize=(16, 9))
    for key, label, color in [("precision", "Precisão", "#1f6feb"),
                              ("recall", "Recall", "#d29922"),
                              ("f1", "F1", "#2ea043")]:
        ax.plot(thresholds, [p[key] for p in curve], label=label, color=color, linewidth=2)
    chosen = calibrated["chosen_threshold"]
    ax.axvline(chosen, color="#cf222e", linestyle="--", linewidth=2,
               label=f"limiar escolhido = {chosen:.4f}")
    ax.set_xlabel("Limiar de abstenção (cosseno do melhor chunk)", fontsize=14)
    ax.set_ylabel("Valor", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Curva de calibração — 5 folds, σ entre folds = {calibrated['fold_spread_std']:.4f}",
        fontsize=17,
    )
    ax.legend(fontsize=13)
    ax.grid(alpha=0.3)
    _save(fig, out, "calibration-curve.png")


def plot_forest(ab: dict, out: Path) -> None:
    metrics = [m for m in ab["metrics"] if m["metric"] in {k for k, _ in ORDER}]
    labels = dict(ORDER)
    fig, ax = plt.subplots(figsize=(16, 9))
    tolerance = ab["tolerance"]
    ax.axvspan(-tolerance, tolerance, color="#2ea043", alpha=0.12,
               label=f"zona de tolerância (±{tolerance})")
    ax.axvline(0, color="black", linewidth=1)
    for i, m in enumerate(metrics):
        color = {"pass": "#1f6feb", "fail": "#cf222e", "inconclusive": "#d29922"}[m["decision"]]
        ax.plot([m["ci_low"], m["ci_high"]], [i, i], color=color, linewidth=3)
        ax.plot(m["delta"], i, "o", color=color, markersize=10)
        ax.text(m["ci_high"], i + 0.22,
                f" Δ={m['delta']:+.3f} p={m['p_adjusted']:.3f} {m['decision']}",
                fontsize=11, color=color, va="center")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([labels.get(m["metric"], m["metric"]) for m in metrics], fontsize=13)
    ax.set_xlabel("Δ da métrica (híbrido − baseline), IC95 bootstrap pareado", fontsize=14)
    ax.set_title(f"A/B pareado com correção de Benjamini-Hochberg "
                 f"({ab['n_pairs']} itens respondíveis)", fontsize=17)
    ax.legend(fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, out, "ab-forest.png")


def plot_latency(current: dict, baseline: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for arm, payload in [("baseline", baseline), ("docsage", current)]:
        lat = sorted(r["latency_ms"] / 1000 for r in payload["results"])
        axes[0].plot(range(1, len(lat) + 1), lat, label=arm, color=COLORS[arm], linewidth=2)
    axes[0].set_xlabel("Itens ordenados por latência", fontsize=13)
    axes[0].set_ylabel("Latência ponta a ponta (s)", fontsize=13)
    axes[0].set_title("Latência por item", fontsize=15)
    axes[0].legend(fontsize=12)
    axes[0].grid(alpha=0.3)

    arms = ["baseline", "docsage"]
    payloads = {"baseline": baseline, "docsage": current}
    p50 = [payloads[a]["metrics"]["latency_p50_ms"] / 1000 for a in arms]
    p95 = [payloads[a]["metrics"]["latency_p95_ms"] / 1000 for a in arms]
    cost = [payloads[a]["metrics"]["cost_usd_per_item"] * 1000 for a in arms]
    x = range(len(arms))
    axes[1].bar([i - 0.2 for i in x], p50, 0.4, label="p50 (s)", color="#1f6feb")
    axes[1].bar([i + 0.2 for i in x], p95, 0.4, label="p95 (s)", color="#8250df")
    for i, (a, b, c) in enumerate(zip(p50, p95, cost, strict=True)):
        axes[1].text(i - 0.2, a, f"{a:.1f}s", ha="center", va="bottom", fontsize=12)
        axes[1].text(i + 0.2, b, f"{b:.1f}s", ha="center", va="bottom", fontsize=12)
        axes[1].text(i, -0.9, f"US$ {c:.2f}/item", ha="center", fontsize=12)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([a.capitalize() for a in arms], fontsize=13)
    axes[1].set_ylabel("Segundos", fontsize=13)
    axes[1].set_title("Cauda de latência e custo por item", fontsize=15)
    axes[1].legend(fontsize=12)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].margins(y=0.15)
    _save(fig, out, "latency-and-cost.png")


def main(inp: Path = typer.Option(Path("eval-out"), "--in"),
         calibrated: Path = typer.Option(Path("configs/calibrated.json"), "--calibrated"),
         out: Path = typer.Option(Path("docs/img"), "--out")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((inp / "baseline.json").read_text(encoding="utf-8"))
    current = json.loads((inp / "current.json").read_text(encoding="utf-8"))
    print(f"gerando gráficos em {out}")
    plot_metrics(baseline, current, out)
    plot_latency(current, baseline, out)
    if calibrated.exists():
        plot_calibration(json.loads(calibrated.read_text(encoding="utf-8")), out)
    ab_path = inp / "ab.json"
    if ab_path.exists():
        plot_forest(json.loads(ab_path.read_text(encoding="utf-8")), out)


if __name__ == "__main__":
    typer.run(main)
