"""Clona o corpus em um commit fixado, para reprodutibilidade."""

import subprocess
from pathlib import Path

from docsage.logging import get_logger

log = get_logger(__name__)

DEFAULT_REPO = "https://github.com/fastapi/fastapi.git"
# Release 0.165.0 — commit completo, porque `git fetch` por sha curto não é resolvido
# pelo servidor remoto. Fixar o sha é o que torna a tabela do RESULTS.md reproduzível.
DEFAULT_COMMIT = "05e6df7e906c3decba377c37548019973f40bf15"
INCLUDE_GLOBS = ("docs/en/**/*.md", "docs/en/**/*.mdx")


def _run(args: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} falhou: {result.stderr.strip()[:400]}")
    return result.stdout


def fetch(repo: str, commit: str, out: Path) -> Path:
    out = out.resolve()
    if (out / ".git").exists():
        log.info("corpus.present", path=str(out))
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", repo, str(out)])
    # busca o objeto exato; funciona para sha completo, tag ou branch
    _run(["git", "-C", str(out), "fetch", "--depth", "1", "origin", commit])
    _run(["git", "-C", str(out), "checkout", "--force", "FETCH_HEAD"])
    head = _run(["git", "-C", str(out), "rev-parse", "HEAD"]).strip()
    if len(commit) == 40 and head != commit:
        raise RuntimeError(f"HEAD {head} difere do commit fixado {commit}")
    log.info("corpus.ready", commit=head)
    return out


def list_markdown(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in INCLUDE_GLOBS:
        files.extend(sorted(root.glob(pattern)))
    return [p for p in files if p.is_file()]
