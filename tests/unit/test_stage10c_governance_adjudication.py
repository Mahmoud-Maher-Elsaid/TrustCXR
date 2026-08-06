from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.localization.run_stage10c_governance_adjudication import adjudicate

ROOT = Path(__file__).resolve().parents[2]


def test_stage10c_contract_and_conservative_identity_decisions() -> None:
    config = json.loads(
        (ROOT / "configs/localization/stage10c_governance_adjudication.json").read_text()
    )
    assert config["training_permitted"] is False
    assert config["test_access_permitted"] is False
    accepted = [
        row["name"]
        for row in config["datasets"]
        if row["identity_decision"] == "ACCEPT_STAGE10B_PATIENT_TRACKING"
    ]
    assert accepted == ["RSNA_Pneumonia"]
    decisions = {row["name"]: row["license_decision"] for row in config["datasets"]}
    assert decisions == {
        "VinBigData": "APPROVED_FOR_RESEARCH",
        "RSNA_Pneumonia": "APPROVED_FOR_RESEARCH",
        "SIIM_Pneumothorax": "APPROVED_FOR_RESEARCH",
        "TBX11K": "APPROVED_FOR_RESEARCH",
        "CRD_Masks": "PENDING_MANUAL_REVIEW",
    }


def test_stage10c_rejects_identity_claim_without_stage10b_evidence() -> None:
    config = {
        "datasets": [
            {
                "name": "example",
                "identity_decision": "ACCEPT_STAGE10B_PATIENT_TRACKING",
                "license_decision": "PENDING_MANUAL_REVIEW",
                "license_source": "https://example.invalid",
            }
        ]
    }
    with pytest.raises(RuntimeError, match="contradicts"):
        adjudicate(config, {"example": {"identity_status": "PATIENT_TRACKING_UNRESOLVED"}})
