from __future__ import annotations

import json
from pathlib import Path

import pytest

from trustcxr.grounded_llm.ext4g1_identity import (
    EXT4G1_BACKEND_VERSION,
    EXT4G1_BENCHMARK_SHA256,
    EXT4G1_REPOSITORY,
    EXT4G1_REVISION,
    EXT4G1_SCHEMA_SHA256,
    Ext4g1IdentityError,
    verify_revision,
)


def test_ext4g1_identity_and_frozen_boundaries():
    assert EXT4G1_REPOSITORY == "google/gemma-3-4b-it"
    assert EXT4G1_REVISION == "093f9f388b31de276ce2de164bdc2081324b9767"
    assert EXT4G1_BACKEND_VERSION == "1.8.0"
    assert EXT4G1_SCHEMA_SHA256 == (
        "99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1"
    )
    benchmark = json.loads(
        Path("configs/research_extensions/ext4f/ext4f_development_benchmark_v1.json").read_text()
    )
    assert benchmark["benchmark_sha256"] == EXT4G1_BENCHMARK_SHA256


def test_floating_or_wrong_revision_fails_closed():
    with pytest.raises(Ext4g1IdentityError):
        verify_revision(EXT4G1_REPOSITORY, "main")
    with pytest.raises(Ext4g1IdentityError):
        verify_revision("google/other-model", EXT4G1_REVISION)
