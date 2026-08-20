"""Deterministic EXT-4I semantically bounded realization.

No model, network, or free-text authority is used here.  Inputs are plain
deterministic authority dictionaries so the historical failure suite can run
without loading the research runtime.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

ATOM_CONTRACT_ID = "EXT4I_SEMANTIC_ATOM_CONTRACT_V1"
BOUNDARY_ID = "EXT4I_PROPOSITION_BOUNDARY_V1"
SKELETON_ID = "EXT4I_PHRASE_SKELETON_V1"
LEXICAL_ID = "EXT4I_BOUNDED_LEXICAL_CHOICE_V1"
VALIDATOR_ID = "EXT4I_SEMANTIC_VALIDATOR_V2"
EVIDENCE_STATES = ("SUPPORTED", "PARTIALLY_SUPPORTED", "WITHHELD", "CONTRADICTED", "NOT_AVAILABLE", "NOT_APPLICABLE")
CLINICAL_FORBIDDEN_PATTERNS = (r"\bdiagnos\w*\b", r"\btreat\w*\b", r"\bmanagement\b", r"\burgent\w*\b", r"\bsevere\w*\b", r"\bleft\b", r"\bright\b", r"\blocaliz(?:ed|ation)\b", r"\bcaus(?:al|ality|ation)?\b", r"\bmeasurement\w*\b")
LEXICAL_CHOICES = {"connector": ("and", "while")}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def build_ext4i_semantic_atoms(authority: dict[str, Any]) -> dict[str, Any]:
    state = authority["evidence_state"]
    if state not in EVIDENCE_STATES:
        raise ValueError("EXT4I_UNKNOWN_EVIDENCE_STATE")
    topic = authority.get("topic", "the specified topic")
    atoms: dict[str, Any] = {
        "contract_id": ATOM_CONTRACT_ID,
        "required_atoms": [], "optional_atoms": [], "forbidden_atoms": [], "forbidden_implications": [],
        "required_polarity": authority.get("polarity", "UNSPECIFIED"),
        "evidence_state": state, "uncertainty_semantics": authority.get("uncertainty", "PRESERVE_PLAN_STATE"),
        "provenance_semantics": authority.get("provenance", "PLAN_BOUND_ONLY"),
        "reference_bindings": list(authority.get("references", [])),
        "defer_semantics": authority.get("defer_semantics", "INACTIVE"),
        "contradiction_semantics": authority.get("contradiction", "NOT_APPLICABLE"),
        "topic_boundary": topic,
        "forbidden_clinical_inference_classes": ["diagnosis", "treatment", "management", "urgency", "severity", "laterality", "localization", "causality", "measurement"],
    }
    state_atoms = {
        "SUPPORTED": (f"evidence supports {topic}", ["unsupported", "contradicted"]),
        "PARTIALLY_SUPPORTED": (f"evidence partially supports {topic}", ["fully supported", "strongly supports"]),
        "WITHHELD": (f"evidence for {topic} is withheld", ["not available", "unavailable", "absent", "negative", "normal", "no evidence"]),
        "CONTRADICTED": (f"evidence for {topic} contains unresolved conflict", ["reconciled", "resolved", "consistent", "supersedes"]),
        "NOT_AVAILABLE": (f"evidence for {topic} is not available", ["negative", "absent", "normal", "does not exist"]),
        "NOT_APPLICABLE": (f"{topic} is not applicable", ["absent", "negative", "normal", "unavailable"]),
    }
    required, forbidden = state_atoms[state]
    atoms["required_atoms"].append(required); atoms["forbidden_atoms"].extend(forbidden)
    if authority.get("defer_semantics") == "ACTIVE":
        atoms["required_atoms"].append(f"{authority.get('deferred_object', 'DECISION')} is deferred")
        atoms["forbidden_implications"].extend(["explanation is active", "decision completed", "decision overridden"])
    if authority.get("contradiction") == "UNRESOLVED_CONFLICT":
        atoms["required_atoms"].append("both contradiction sides remain unresolved")
        atoms["forbidden_implications"].append("RECONCILIATION")
    atoms["authority_identity"] = authority.get("authority_identity", "DETERMINISTIC_AUTHORITY")
    atoms["atom_sha256"] = _sha(atoms)
    return atoms


def build_ext4i_proposition_boundary(atoms: dict[str, Any]) -> dict[str, Any]:
    boundary = {"boundary_id": BOUNDARY_ID, "required_propositions": list(atoms["required_atoms"]), "optional_propositions": list(atoms["optional_atoms"]), "forbidden_propositions": list(atoms["forbidden_atoms"]), "forbidden_implications": list(atoms["forbidden_implications"]), "atom_sha256": atoms["atom_sha256"]}
    boundary["boundary_sha256"] = _sha(boundary)
    return boundary


def build_ext4i_phrase_skeleton(atoms: dict[str, Any]) -> dict[str, Any]:
    state, topic = atoms["evidence_state"], atoms["topic_boundary"]
    if state == "WITHHELD": template = f"Evidence for {topic} is withheld from this conclusion."
    elif state == "NOT_AVAILABLE": template = f"Evidence for {topic} is not available in the authorized evidence."
    elif state == "NOT_APPLICABLE": template = f"{topic} is not applicable to this review context."
    elif state == "PARTIALLY_SUPPORTED": template = f"The available evidence partially supports {topic}."
    elif state == "CONTRADICTED": template = f"The authorized evidence contains an unresolved conflict about {topic}."
    else: template = f"The available evidence supports {topic}."
    if atoms.get("defer_semantics") == "ACTIVE": template = f"The {atoms.get('deferred_object', 'DECISION').lower()} is deferred pending the authorized review condition."
    skeleton = {"skeleton_id": SKELETON_ID, "template": template, "atom_sha256": atoms["atom_sha256"]}
    skeleton["skeleton_sha256"] = _sha(skeleton)
    return skeleton


def realize_ext4i_slot(atoms: dict[str, Any], boundary: dict[str, Any], skeleton: dict[str, Any], lexical_choices: dict[str, str] | None = None) -> dict[str, Any]:
    choices = lexical_choices or {"connector": "and"}
    if any(k not in LEXICAL_CHOICES or v not in LEXICAL_CHOICES[k] for k, v in choices.items()):
        raise ValueError("EXT4I_LEXICAL_CHOICE_INVALID")
    if boundary["atom_sha256"] != atoms["atom_sha256"] or skeleton["atom_sha256"] != atoms["atom_sha256"]:
        raise ValueError("EXT4I_AUTHORITY_IDENTITY_MISMATCH")
    result = {"realized_text": skeleton["template"], "atom_sha256": atoms["atom_sha256"], "boundary_sha256": boundary["boundary_sha256"], "skeleton_sha256": skeleton["skeleton_sha256"], "lexical_choice_id": _sha({"contract": LEXICAL_ID, "choices": choices}), "lexical_choices": choices}
    result["semantic_realization_sha256"] = _sha(result)
    return result


def validate_ext4i_realization(atoms: dict[str, Any], boundary: dict[str, Any], skeleton: dict[str, Any], realized_text: str) -> dict[str, Any]:
    failures: list[str] = []
    if boundary.get("atom_sha256") != atoms.get("atom_sha256"): failures.append("EXT4I_AUTHORITY_IDENTITY_MISMATCH")
    text = realized_text.lower()
    state = atoms["evidence_state"]
    # Remove the explicitly authorized topic before clinical-boundary checks;
    # e.g. a WITHHELD localization topic is allowed to name localization.
    clinical_text = text.replace(str(atoms.get("topic_boundary", "")).lower(), " ")
    if state == "WITHHELD" and any(x in text for x in ("not available", "unavailable", "absent", "negative", "no evidence")): failures.append("EXT4I_WITHHELD_TO_NOT_AVAILABLE")
    if state in {"NOT_AVAILABLE", "NOT_APPLICABLE"} and any(x in text for x in ("negative", "absent", "normal", "does not exist")): failures.append("EXT4I_STATE_TO_NEGATIVE_OR_ABSENCE")
    if state == "PARTIALLY_SUPPORTED" and any(x in text for x in ("fully supported", "strongly supports")): failures.append("EXT4I_PARTIAL_SUPPORT_INFLATION")
    if state == "SUPPORTED" and any(x in text for x in ("unsupported", "not supported", "contradicted")): failures.append("EXT4I_SUPPORTED_STATE_SUBSTITUTION")
    if state == "CONTRADICTED" and any(x in text for x in ("reconciled", "resolved", "consistent", "supersedes")): failures.append("EXT4I_CONTRADICTION_RECONCILIATION")
    if state == "CONTRADICTED" and "uncertainty about" in text: failures.append("EXT4I_UNSUPPORTED_PROPOSITION")
    if state == "CONTRADICTED" and "authoritative" in text: failures.append("EXT4I_PROVENANCE_INVENTION")
    if atoms.get("defer_semantics") == "ACTIVE" and "explanation" in text and "active" in text: failures.append("EXT4I_DEFER_OBJECT_MISMATCH")
    if any(re.search(pattern, clinical_text) for pattern in CLINICAL_FORBIDDEN_PATTERNS): failures.append("EXT4I_FORBIDDEN_CLINICAL_INFERENCE")
    if atoms.get("provenance_semantics") == "PLAN_BOUND_ONLY" and any(x in text for x in ("stage 16", "authoritative source")): failures.append("EXT4I_PROVENANCE_INVENTION")
    return {"validator_id": VALIDATOR_ID, "status": "FAIL" if failures else "PASS", "failure_codes": failures}


def build_ext4i_review_context(atoms: dict[str, Any], realized_text: str) -> dict[str, Any]:
    return {"context_id": "EXT4I_BLINDED_REVIEW_CONTEXT_V2", "evidence_state": atoms["evidence_state"], "required_atoms": atoms["required_atoms"], "forbidden_atoms": atoms["forbidden_atoms"], "polarity": atoms["required_polarity"], "uncertainty": atoms["uncertainty_semantics"], "provenance": atoms["provenance_semantics"], "references": atoms["reference_bindings"], "defer": atoms["defer_semantics"], "contradiction": atoms["contradiction_semantics"], "topic_boundary": atoms["topic_boundary"], "realized_text": realized_text, "candidate_identity": None, "model_identity": None, "expected_prose": None}
