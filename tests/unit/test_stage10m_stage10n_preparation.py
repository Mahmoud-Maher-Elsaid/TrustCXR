from __future__ import annotations

import json
from pathlib import Path

from scripts.localization.run_stage10n_localization_acceptance_decision import (
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[2]


def test_stage10n_contract_keeps_test_locked_and_claims_limited() -> None:
    config = json.loads(
        (ROOT / "configs/localization/stage10n_localization_acceptance_decision.json").read_text()
    )
    validate_contract(config)
    assert config["training_permitted"] is False
    assert config["final_test_split_locked"] is True
    assert config["final_test_images_accessed"] == 0
    assert config["test_predictions_permitted"] is False
    assert config["required_anatomical_claim"] == (
        "IMAGE_GEOMETRY_AND_THORACIC_LOCATION_PROXY_ONLY"
    )


def test_stage10n_prevents_localization_absence_from_negating_classifier() -> None:
    config = json.loads(
        (ROOT / "configs/localization/stage10n_localization_acceptance_decision.json").read_text()
    )
    assert config["decision_policy"]["absence_of_localization_may_contradict_classifier"] is False
    assert config["decision_policy"]["allow_final_test_evaluation"] is False
    assert config["decision_policy"]["allow_clinical_localization_claim"] is False
