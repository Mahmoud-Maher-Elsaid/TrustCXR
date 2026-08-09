from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from trustcxr.serving.schemas import ComponentId

STAGE21B_FINGERPRINT = "6d92a8ddab32c2669d9e41b0b3f98bcac7485188f3b0ead628a7c2f87ae15c5f"
STAGE21B_SUMMARY_SHA256 = "649a046c31499bb1a280bff09793045683cb4f0fb760cfb72b56672c1c7d3d82"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FrozenComponent:
    component_id: ComponentId
    server_model_version: str
    config_sha256: str
    checkpoint_sha256: str | None
    config_path: str
    checkpoint_path: str | None
    structured_input: str
    structured_output: str
    limitations: tuple[str, ...]
    compute: str

    def public_metadata(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id.value,
            "server_model_version": self.server_model_version,
            "config_sha256": self.config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "structured_input": self.structured_input,
            "structured_output": self.structured_output,
            "limitations": self.limitations,
            "compute": self.compute,
        }


class FrozenComponentRegistry:
    def __init__(self, components: Mapping[ComponentId, FrozenComponent]) -> None:
        self._components = MappingProxyType(dict(components))

    @classmethod
    def from_stage21b(cls, root: Path) -> FrozenComponentRegistry:
        summary_path = root / "reports/stage21/stage21b_backend_api_worker_contract_summary.json"
        if sha256(summary_path) != STAGE21B_SUMMARY_SHA256:
            raise RuntimeError("Stage 21B registry evidence hash mismatch.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["contract_fingerprint"] != STAGE21B_FINGERPRINT:
            raise RuntimeError("Stage 21B registry fingerprint mismatch.")
        components: dict[ComponentId, FrozenComponent] = {}
        for item in summary["eligible_components"]:
            component_id = ComponentId(item["id"])
            version_payload = f"{item['id']}:{item['config_sha256']}:{item['artifact_sha256']}"
            server_version = "frozen-" + hashlib.sha256(version_payload.encode()).hexdigest()[:24]
            components[component_id] = FrozenComponent(
                component_id=component_id,
                server_model_version=server_version,
                config_sha256=item["config_sha256"],
                checkpoint_sha256=item["artifact_sha256"],
                config_path=item["config_path"],
                checkpoint_path=item["artifact_relative_path"],
                structured_input=item["structured_input"],
                structured_output=item["structured_output"],
                limitations=tuple(item["limitations"]),
                compute=item["compute"],
            )
        if set(components) != set(ComponentId):
            raise RuntimeError("Frozen component registry is incomplete.")
        return cls(components)

    def resolve(self, component_id: ComponentId) -> FrozenComponent:
        return self._components[component_id]

    def public_registry(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._components[key].public_metadata() for key in sorted(self._components))

    def __len__(self) -> int:
        return len(self._components)
