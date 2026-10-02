"""Formal Sprint 12 go-live eligibility review. This module never places orders."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from research.artifact_registry import ArtifactRegistry


REQUIRED_ARTIFACT_TYPES = (
    "validated-alpha",
    "runtime-package",
    "shadow-paper-proof",
)


@dataclass(frozen=True)
class GoLiveGate:
    name: str
    passed: bool
    details: str


@dataclass(frozen=True)
class GoLiveReview:
    eligible_for_tiny_live_experiment: bool
    live_capital_authorized: bool
    gates: tuple[GoLiveGate, ...]
    artifacts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_for_tiny_live_experiment": self.eligible_for_tiny_live_experiment,
            "live_capital_authorized": self.live_capital_authorized,
            "gates": [asdict(gate) for gate in self.gates],
            "artifacts": list(self.artifacts),
        }


def review_go_live(
    registry: ArtifactRegistry,
    artifact_ids: list[str],
    *,
    secret_hygiene_passed: bool,
    recovery_test_passed: bool,
    supervisor_test_passed: bool,
    rbac_enabled: bool,
    backup_restore_test_passed: bool,
) -> GoLiveReview:
    manifests = [registry.read(item) for item in artifact_ids]
    types = {str(item.get("artifact_type")) for item in manifests}
    gates: list[GoLiveGate] = []
    for artifact_type in REQUIRED_ARTIFACT_TYPES:
        gates.append(GoLiveGate(f"artifact:{artifact_type}", artifact_type in types, f"required artifact type {artifact_type}"))
    for manifest in manifests:
        artifact_id = str(manifest.get("artifact_id"))
        failed = [g.get("name") for g in manifest.get("gates", []) if not g.get("passed", False)]
        gates.append(GoLiveGate(f"artifact-gates:{artifact_id}", not failed, "all recorded artifact gates must pass" if not failed else f"failed: {failed}"))
    controls = {
        "secret_hygiene": secret_hygiene_passed,
        "recovery_restore": recovery_test_passed,
        "supervisor_fault_recovery": supervisor_test_passed,
        "argus_rbac": rbac_enabled,
        "backup_restore": backup_restore_test_passed,
    }
    gates.extend(GoLiveGate(name, passed, "mandatory production hardening control") for name, passed in controls.items())
    eligible = all(item.passed for item in gates)
    return GoLiveReview(
        eligible_for_tiny_live_experiment=eligible,
        # Eligibility only permits a separately reviewed experiment. This codebase
        # still contains no live-capital broker adapter, so authorization remains false.
        live_capital_authorized=False,
        gates=tuple(gates),
        artifacts=tuple(artifact_ids),
    )
