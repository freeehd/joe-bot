"""Governed immutable artifact registry for Joe Bot research/runtime promotion."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

UTC = timezone.utc


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit(cwd: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class PromotionGate:
    name: str
    passed: bool
    value: Any = None
    threshold: Any = None
    details: str | None = None


@dataclass(frozen=True)
class ArtifactManifest:
    artifact_id: str
    artifact_type: str
    created_at_utc: str
    status: str
    code_commit: str | None
    files: tuple[ArtifactFile, ...]
    source_datasets: tuple[dict[str, Any], ...] = ()
    parents: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    gates: tuple[PromotionGate, ...] = ()
    environment: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "created_at_utc": self.created_at_utc,
            "status": self.status,
            "code_commit": self.code_commit,
            "files": [asdict(item) for item in self.files],
            "source_datasets": list(self.source_datasets),
            "parents": list(self.parents),
            "config": self.config,
            "metrics": self.metrics,
            "gates": [asdict(item) for item in self.gates],
            "environment": self.environment,
            "notes": self.notes,
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))


class ArtifactRegistry:
    """Filesystem registry with immutable candidates and explicit promotion records."""

    def __init__(self, root: str | Path = "data/registry") -> None:
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.promotions_dir = self.root / "promotions"

    def artifact_dir(self, artifact_id: str) -> Path:
        return self.artifacts_dir / artifact_id

    def manifest_path(self, artifact_id: str) -> Path:
        return self.artifact_dir(artifact_id) / "manifest.json"

    def register(
        self,
        *,
        artifact_id: str,
        artifact_type: str,
        files: Iterable[str | Path],
        source_datasets: Iterable[dict[str, Any]] = (),
        parents: Iterable[str] = (),
        config: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        gates: Iterable[PromotionGate] = (),
        status: str = "candidate",
        notes: str | None = None,
        repo_root: str | Path | None = None,
    ) -> ArtifactManifest:
        if status not in {"candidate", "rejected", "promoted"}:
            raise ValueError("status must be candidate, rejected, or promoted")
        path = self.manifest_path(artifact_id)
        if path.exists():
            raise FileExistsError(f"Artifact {artifact_id!r} already exists; registry entries are immutable")

        file_records: list[ArtifactFile] = []
        base = Path(repo_root).resolve() if repo_root is not None else None
        for item in files:
            file_path = Path(item)
            if not file_path.is_file():
                raise FileNotFoundError(file_path)
            resolved = file_path.resolve()
            display = str(resolved.relative_to(base)) if base is not None and resolved.is_relative_to(base) else str(file_path)
            file_records.append(ArtifactFile(display, sha256_file(file_path), file_path.stat().st_size))
        if not file_records:
            raise ValueError("an artifact must contain at least one file")

        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            created_at_utc=datetime.now(tz=UTC).isoformat(),
            status=status,
            code_commit=current_git_commit(repo_root),
            files=tuple(sorted(file_records, key=lambda item: item.path)),
            source_datasets=tuple(source_datasets),
            parents=tuple(parents),
            config=config or {},
            metrics=metrics or {},
            gates=tuple(gates),
            environment={"python": platform.python_version(), "platform": platform.platform()},
            notes=notes,
        )
        path.parent.mkdir(parents=True, exist_ok=False)
        payload = manifest.to_dict()
        payload["manifest_sha256"] = manifest.digest
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return manifest

    def read(self, artifact_id: str) -> dict[str, Any]:
        return json.loads(self.manifest_path(artifact_id).read_text(encoding="utf-8"))

    def verify(self, artifact_id: str, *, repo_root: str | Path | None = None) -> dict[str, Any]:
        manifest = self.read(artifact_id)
        root = Path(repo_root) if repo_root is not None else None
        results = []
        for record in manifest["files"]:
            path = Path(record["path"])
            if root is not None and not path.is_absolute():
                path = root / path
            exists = path.is_file()
            actual = sha256_file(path) if exists else None
            results.append({
                "path": record["path"],
                "exists": exists,
                "expected_sha256": record["sha256"],
                "actual_sha256": actual,
                "matches": bool(exists and actual == record["sha256"]),
            })
        return {"artifact_id": artifact_id, "valid": all(x["matches"] for x in results), "files": results}

    def promote(self, artifact_id: str, *, channel: str = "shadow", repo_root: str | Path | None = None) -> Path:
        if channel not in {"shadow", "paper", "tiny-live"}:
            raise ValueError("unsupported promotion channel")
        manifest = self.read(artifact_id)
        verification = self.verify(artifact_id, repo_root=repo_root)
        if not verification["valid"]:
            raise RuntimeError(f"artifact {artifact_id!r} failed file-integrity verification")
        failed = [gate for gate in manifest.get("gates", []) if not gate.get("passed", False)]
        if failed:
            raise RuntimeError(f"artifact {artifact_id!r} has failed promotion gates: {[g['name'] for g in failed]}")
        self.promotions_dir.mkdir(parents=True, exist_ok=True)
        path = self.promotions_dir / f"{channel}.json"
        record = {
            "artifact_id": artifact_id,
            "artifact_manifest_sha256": manifest["manifest_sha256"],
            "channel": channel,
            "promoted_at_utc": datetime.now(tz=UTC).isoformat(),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        return path
