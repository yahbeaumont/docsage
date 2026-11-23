import subprocess

import pytest

from docsage.corpus import fetch as fetch_module


def test_list_markdown_filters_extensions(tmp_path):
    (tmp_path / "docs/en/tutorial").mkdir(parents=True)
    (tmp_path / "docs/en/tutorial/index.md").write_text("# a")
    (tmp_path / "docs/en/tutorial/page.mdx").write_text("# b")
    (tmp_path / "docs/en/tutorial/image.png").write_bytes(b"x")
    found = fetch_module.list_markdown(tmp_path)
    assert [p.name for p in found] == ["index.md", "page.mdx"]


def test_fetch_raises_with_command_output(monkeypatch, tmp_path):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 1, "", "fatal: not found")

    monkeypatch.setattr(fetch_module.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="fatal: not found"):
        fetch_module.fetch("https://example.invalid/x.git", "deadbeef", tmp_path / "c")


def test_fetch_rejects_sha_mismatch(monkeypatch, tmp_path):
    def fake_run(args, **kwargs):
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, "0" * 40 + "\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(fetch_module.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="difere do commit fixado"):
        fetch_module.fetch("repo", "1" * 40, tmp_path / "c")


def test_default_commit_is_full_sha():
    assert len(fetch_module.DEFAULT_COMMIT) == 40
    int(fetch_module.DEFAULT_COMMIT, 16)
