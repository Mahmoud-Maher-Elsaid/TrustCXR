from __future__ import annotations

import json
from pathlib import Path

from scripts.reliability.run_stage16c_validation_reliability_preparation import partition


def test_stage16c_preserves_frozen_validation_only_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/reliability/stage16c_validation_reliability_preparation.json").read_text()
    )
    assert (
        config["required_contract_fingerprint"]
        == "3bb34f4edd5d4ed1f27120e028efacf92eb7e0e5217dbc6a1d29c59521b9af9e"
    )
    assert config["partition_salt"] == "trustcxr-stage16-reliability-v1"
    assert config["stage9"]["inference_permitted"] is False
    assert (
        config["stage13"]["checkpoint_sha256"]
        == "09a8db4e83861fcc41172d64f29215d6e9d24cd41a8d6b51f825ca6d00fb1c77"
    )
    assert config["stage13"]["split"] == "validation"
    assert config["stage13"]["frontal_image_only"] is True
    assert config["calibration_fitting_permitted"] is False
    assert config["abstention_threshold_selection_permitted"] is False
    assert config["locked_test_access_permitted"] is False
    assert config["reliability_metrics_permitted"] is False
    assert config["ood_status"] == "WITHHELD_NO_GOVERNED_OOD_COHORT"


def test_stage16c_patient_partition_is_deterministic_and_disjoint() -> None:
    salt = "trustcxr-stage16-reliability-v1"
    assignments = [partition(f"patient-{index}", salt) for index in range(100)]
    assert assignments == [partition(f"patient-{index}", salt) for index in range(100)]
    assert set(assignments) == {0, 1, 2}
