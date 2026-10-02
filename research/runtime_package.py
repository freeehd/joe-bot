"""Immutable Victory Sprint 8 runtime-package factory."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from research.artifact_registry import ArtifactRegistry, PromotionGate, sha256_file


@dataclass(frozen=True)
class RuntimeComponent:
    name: str
    path: str | Path
    required: bool = True


class RuntimePackageBuilder:
    def __init__(self, *, registry_root: str | Path = "data/registry", package_root: str | Path = "data/runtime_packages") -> None:
        self.registry = ArtifactRegistry(registry_root)
        self.package_root = Path(package_root)

    def build(
        self,
        *,
        package_id: str,
        parent_artifact_id: str,
        components: Iterable[RuntimeComponent],
        runtime_config: dict[str, Any],
        drift_baseline_path: str | Path,
        repo_root: str | Path = ".",
    ) -> dict:
        parent = self.registry.read(parent_artifact_id)
        if not self.registry.verify(parent_artifact_id, repo_root=repo_root)["valid"]:
            raise RuntimeError("parent artifact failed integrity verification")
        failed_parent_gates = [g["name"] for g in parent.get("gates", []) if not g.get("passed", False)]
        if failed_parent_gates:
            raise RuntimeError(f"parent artifact has failed gates: {failed_parent_gates}")

        target = self.package_root / package_id
        if target.exists():
            raise FileExistsError(f"runtime package {package_id!r} already exists")
        target.mkdir(parents=True)
        copied = []
        for component in components:
            src = Path(component.path)
            if not src.is_file():
                if component.required:
                    raise FileNotFoundError(src)
                continue
            dst = target / component.name
            shutil.copy2(src, dst)
            copied.append({"name": component.name, "path": str(dst), "sha256": sha256_file(dst), "size_bytes": dst.stat().st_size})
        drift_src = Path(drift_baseline_path)
        if not drift_src.is_file():
            raise FileNotFoundError(drift_src)
        drift_dst = target / "drift_baseline.json"
        shutil.copy2(drift_src, drift_dst)
        copied.append({"name": "drift_baseline.json", "path": str(drift_dst), "sha256": sha256_file(drift_dst), "size_bytes": drift_dst.stat().st_size})

        config_path = target / "runtime_config.json"
        config_path.write_text(json.dumps(runtime_config, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        copied.append({"name": "runtime_config.json", "path": str(config_path), "sha256": sha256_file(config_path), "size_bytes": config_path.stat().st_size})
        manifest_path = target / "package_manifest.json"
        package_manifest = {
            "package_id": package_id,
            "parent_artifact_id": parent_artifact_id,
            "parent_manifest_sha256": parent["manifest_sha256"],
            "components": sorted(copied, key=lambda item: item["name"]),
            "runtime_config": runtime_config,
            "live_capital_authorized": False,
        }
        manifest_path.write_text(json.dumps(package_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

        registry_artifact = self.registry.register(
            artifact_id=package_id,
            artifact_type="runtime-package",
            files=[manifest_path, config_path, drift_dst, *[Path(item["path"]) for item in copied if item["name"] not in {"runtime_config.json", "drift_baseline.json"}]],
            source_datasets=parent.get("source_datasets", []),
            parents=[parent_artifact_id],
            config=runtime_config,
            metrics={"package_component_count": len(copied)},
            gates=[PromotionGate("parent_artifact_all_gates_pass", True), PromotionGate("live_capital_authorized", False, False, "must remain false until tiny-live gate")],
            status="candidate",
            repo_root=repo_root,
            notes="Runtime package is eligible for shadow packaging only; live-capital flag is intentionally false.",
        )
        return {"package_dir": str(target), "manifest": package_manifest, "artifact_manifest_sha256": registry_artifact.digest}
