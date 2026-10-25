"""Automated test validating all executable Python snippets in README.md."""

import re
from pathlib import Path
from unittest.mock import patch

import pytest


def extract_python_blocks(markdown_text: str):
    # Regex to find python blocks: ```python ... ```
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    blocks = pattern.findall(markdown_text)
    return blocks


@patch("matplotlib.pyplot.show")
def test_all_readme_snippets(mock_show, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    assert readme_path.exists(), "README.md not found"

    content = readme_path.read_text(encoding="utf-8")
    blocks = extract_python_blocks(content)
    assert len(blocks) > 0, "No Python blocks found in README.md"

    # Context dictionary to persist state between sequential blocks if needed
    exec_globals = {}

    for idx, block in enumerate(blocks):
        # Skip purely conceptual or incomplete blocks if any
        try:
            exec(block, exec_globals)
        except Exception as exc:
            pytest.fail(
                f"README.md Python snippet #{idx + 1} failed:\n```python\n{block}```\nError: {exc}"
            )
