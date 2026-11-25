"""Gate de regressão: falha se qualquer métrica cair além da tolerância."""

import json
from pathlib import Path

import typer

from docsage.logging import configure, get_logger

configure()
log = get_logger(__name__)
app = typer.Typer(add_completion=False)

DEFAULT_TOLERANCE = 0.03
SIGMA_MULTIPLIER = 2.0


def allowed_drop(metric: str, tolerance: float, noise_std: dict[str, float]) -> float:
    """Tolerância base, ampliada para métricas com ruído medido."""
    return max(tolerance, SIGMA_MULTIPLIER * noise_std.get(metric, 0.0))


def check(baseline: dict, current: dict, tolerance: float) -> list[dict]:
    failures: list[dict] = []
    noise = baseline.get("noise_std", {})

    for metric, base_value in baseline["metrics"].items():
        # métricas não numéricas (error_ids, por exemplo) não têm ordem para regredir
        if isinstance(base_value, bool) or not isinstance(base_value, (int, float)):
            continue
        if metric not in current["metrics"]:
            failures.append({"metric": metric, "reason": "ausente na execução atual",
                             "base": base_value, "current": None, "delta": None})
            continue

        value = current["metrics"][metric]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        delta = value - base_value
        limit = allowed_drop(metric, tolerance, noise)
        if delta < -limit:
            failures.append({"metric": metric, "reason": "regressão", "base": base_value,
                             "current": value, "delta": delta, "allowed_drop": limit})
    return failures


def check_verdict(data: dict) -> tuple[list[dict], str]:
    """Interpreta o A/B pareado: só reprova o que o bootstrap declarou como regressão."""
    failures = [m for m in data["metrics"] if m["decision"] == "fail"]
    return failures, data["overall"]


@app.command()
def main(
    current: Path = typer.Option(Path("eval-out/current.json")),
    baseline: Path = typer.Option(Path("eval/baseline.json")),
    verdict: Path | None = typer.Option(None),
    tolerance: float = typer.Option(DEFAULT_TOLERANCE),
    update_baseline: bool = typer.Option(False, "--update-baseline"),
) -> None:
    if verdict is not None:
        data = json.loads(verdict.read_text(encoding="utf-8"))
        failures, overall = check_verdict(data)
        for m in failures:
            typer.secho(f"  REGRESSÃO {m['metric']}: Δ={m['delta']:+.4f} "
                        f"IC95=[{m['ci_low']:+.4f},{m['ci_high']:+.4f}]", fg="red", err=True)
        if failures:
            raise typer.Exit(1)
        if overall == "inconclusive":
            typer.secho("inconclusivo — sem regressão estatisticamente detectada, "
                        "mas resolução insuficiente para liberar", fg="yellow")
            raise typer.Exit(2)
        typer.secho("sem regressões (bootstrap)", fg="green")
        return

    run = json.loads(current.read_text(encoding="utf-8"))

    if update_baseline:
        base = json.loads(baseline.read_text(encoding="utf-8")) if baseline.exists() else {}
        meta = dict(run.get("settings", {}))
        meta.update({"date": run["timestamp"], "promoted_from": str(current)})
        baseline.write_text(
            json.dumps({"metrics": run["metrics"], "_meta": meta,
                        "noise_std": base.get("noise_std", {})}, indent=2),
            encoding="utf-8",
        )
        typer.secho(f"baseline atualizado a partir de {current}", fg="green")
        return

    base = json.loads(baseline.read_text(encoding="utf-8"))
    failures = check(base, run, tolerance)

    for metric, value in run["metrics"].items():
        base_value = base["metrics"].get(metric)
        if base_value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if isinstance(base_value, bool) or not isinstance(base_value, (int, float)):
            continue
        delta = value - base_value
        color = "green" if delta >= 0 else ("yellow" if delta >= -tolerance else "red")
        typer.secho(f"{metric:24s} {base_value:.4f} -> {value:.4f} ({delta:+.4f})", fg=color)

    if not failures:
        typer.secho("\nsem regressões", fg="green")
        raise typer.Exit(0)

    typer.secho(f"\n{len(failures)} regressão(ões):", fg="red", err=True)
    for f in failures:
        typer.secho(f"  {f['metric']}: {f['base']:.4f} -> {f['current']:.4f} "
                    f"(limite -{f['allowed_drop']:.4f})", fg="red", err=True)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
