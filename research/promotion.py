"""Explicit runtime-package promotion CLI with channel-specific safety rules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.artifact_registry import ArtifactRegistry


def promote_runtime_package(registry: ArtifactRegistry, artifact_id: str, *, channel: str, repo_root: str | Path = ".") -> Path:
    artifact = registry.read(artifact_id)
    if artifact.get("artifact_type") != "runtime-package":
        raise ValueError("only runtime-package artifacts can use the Sprint 8 promotion CLI")
    if channel not in {"shadow", "paper"}:
        raise ValueError("Sprint 8 promotion supports only shadow or paper; tiny-live is a later gate")
    # Runtime packages intentionally contain a failed live-capital authorization
    # gate. Ignore only that sentinel for shadow/paper, never integrity failures.
    verification = registry.verify(artifact_id, repo_root=repo_root)
    if not verification["valid"]:
        raise RuntimeError("runtime package failed integrity verification")
    failed = [g for g in artifact.get("gates", []) if not g.get("passed", False) and g.get("name") != "live_capital_authorized"]
    if failed:
        raise RuntimeError(f"runtime package has failed gates: {[g['name'] for g in failed]}")
    registry.promotions_dir.mkdir(parents=True, exist_ok=True)
    path = registry.promotions_dir / f"{channel}.json"
    record = {
        "artifact_id": artifact_id,
        "artifact_manifest_sha256": artifact["manifest_sha256"],
        "channel": channel,
        "live_capital_authorized": False,
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a governed runtime package to shadow or paper")
    parser.add_argument("artifact_id")
    parser.add_argument("--channel", choices=["shadow", "paper"], required=True)
    parser.add_argument("--registry-root", default="data/registry")
    args = parser.parse_args()
    path = promote_runtime_package(ArtifactRegistry(args.registry_root), args.artifact_id, channel=args.channel)
    print(path)


if __name__ == "__main__":
    main()
