import json
from pathlib import Path


ROOT = Path("configs/research_extensions/ext4i")
REPORT = Path("reports/research_extensions/ext4i/EXT4I0_SEMANTIC_FAILURE_ROOT_CAUSE_AUDIT.json")


def test_ext4i_inventory_separates_resolved_and_unresolved():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    assert report["inventory"] == {"resolved_pass_units": 21, "resolved_model_failure_units": 16, "unresolved_context_insufficiency_units": 43, "model_failure_cases": 12}
    assert report["root_cause_layers"]["LLM_SURFACE_REALIZATION_DRIFT"] == 16
    assert report["root_cause_layers"]["REVIEW_CONTEXT_PACKAGING_DEFECT"] == 43


def test_ext4i_state_constraints_are_machine_readable():
    contract = json.loads((ROOT / "ext4i_semantic_atom_contract_v1.json").read_text(encoding="utf-8"))
    states = contract["state_constraints"]
    assert set(states) == {"SUPPORTED", "PARTIALLY_SUPPORTED", "WITHHELD", "CONTRADICTED", "NOT_AVAILABLE", "NOT_APPLICABLE", "DEFER"}
    assert "unavailable" in states["WITHHELD"]["forbidden"]
    assert "negative finding" in states["NOT_AVAILABLE"]["forbidden"]
    assert "reconciliation" in states["CONTRADICTED"]["forbidden"]
    assert "decision or action is deferred" in states["DEFER"]["must_preserve"]


def test_ext4i_regression_excludes_unresolved_units():
    regression = json.loads((ROOT / "ext4i_h5_resolved_failure_regression_v1.json").read_text(encoding="utf-8"))
    assert regression["regression_id"] == "EXT4I_H5_RESOLVED_FAILURE_REGRESSION_V1"
    assert regression["resolved_failure_count"] == 16
    assert regression["unresolved_units_excluded"] == 43
    assert all(item["primary_failure_layer"] == "LLM_SURFACE_REALIZATION_DRIFT" for item in regression["cases"])
