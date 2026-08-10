from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_static_core_architecture_asset_is_valid_and_local() -> None:
    svg = ROOT / "docs/assets/trustcxr-core-architecture.svg"
    root = ET.fromstring(svg.read_bytes())
    assert root.tag.endswith("svg")
    text = svg.read_text(encoding="utf-8").lower()
    assert "<script" not in text
    assert "https://" not in text
    assert text.count("http://") == 1


def test_readme_uses_static_architecture_without_mermaid() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/assets/trustcxr-core-architecture.svg" in readme
    assert "```mermaid" not in readme
    assert "flowchart TD" not in readme
