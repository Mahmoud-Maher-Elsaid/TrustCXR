from __future__ import annotations

import hashlib

from scripts.multiview.run_stage13g_locked_test_pair_readiness import patient_split, trusted_view


def test_stage13g_patient_split_matches_frozen_stage5_contract() -> None:
    config = {
        "patient_split_salt": "trustcxr-stage5",
        "train_fraction": 0.8,
        "validation_fraction": 0.1,
    }
    for patient in ("patient00001", "patient12345", "patient99999"):
        fraction = int.from_bytes(
            hashlib.sha256(f"trustcxr-stage5:{patient}".encode()).digest()[:8], "big"
        ) / float(2**64)
        expected = "train" if fraction < 0.8 else "validation" if fraction < 0.9 else "test"
        assert patient_split(patient, config) == expected


def test_stage13g_trusted_views_do_not_invent_unknown_or_other() -> None:
    assert trusted_view({"Frontal/Lateral": "Frontal", "AP/PA": "AP"}) == "AP"
    assert trusted_view({"Frontal/Lateral": "Lateral", "AP/PA": ""}) == "LATERAL"
    assert trusted_view({"Frontal/Lateral": "Frontal", "AP/PA": "LL"}) is None
    assert trusted_view({"Frontal/Lateral": "", "AP/PA": ""}) is None
