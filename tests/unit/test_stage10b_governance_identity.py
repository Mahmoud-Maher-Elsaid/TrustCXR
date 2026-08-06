from __future__ import annotations

import json
from pathlib import Path

from scripts.localization.run_stage10b_governance_identity import stable_hash

ROOT = Path(__file__).resolve().parents[2]


def test_stage10b_prohibits_training_and_test_access() -> None:
    config = json.loads(
        (ROOT / "configs/localization/stage10b_governance_identity.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["training_permitted"] is False
    assert config["test_access_permitted"] is False
    assert all(item["license_decision"] is None for item in config["datasets"])


def test_identity_hash_is_namespaced_and_deterministic() -> None:
    assert stable_hash("dataset:patient", "abc") == stable_hash("dataset:patient", "abc")
    assert stable_hash("dataset:patient", "abc") != stable_hash("other:patient", "abc")
    assert "abc" not in stable_hash("dataset:patient", "abc")
