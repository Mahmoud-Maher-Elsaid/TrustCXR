import json
from pathlib import Path


def test_ext4i1_report_is_zero_model_and_regression_complete():
    report = json.loads(Path("reports/research_extensions/ext4i/EXT4I1_SEMANTICALLY_BOUNDED_REALIZATION_REPORT.json").read_text(encoding="utf-8"))
    assert report["historical_regression"] == {"suite_id": "EXT4I_H5_RESOLVED_FAILURE_REGRESSION_V1", "rejected": 16, "total": 16, "replacement_pass": 16, "total_replacements": 16}
    assert report["mutation_testing"]["unexpected_accepts"] == 0
    assert report["model_load_calls"] == report["forward_calls"] == report["generate_calls"] == 0
