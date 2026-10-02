"""Configuration, secret-hygiene, and RBAC policy primitives."""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable


class OperatorRole(str, Enum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


ROLE_PERMISSIONS = {
    OperatorRole.VIEWER: frozenset({"read"}),
    OperatorRole.OPERATOR: frozenset({"read", "paper-risk"}),
    OperatorRole.ADMIN: frozenset({"read", "paper-risk", "configure"}),
}


def role_allows(role: OperatorRole | str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS[OperatorRole(role)]


@dataclass(frozen=True)
class SecretAuditResult:
    passed: bool
    findings: tuple[str, ...]


_SECRET_ASSIGNMENT = re.compile(r"(?im)^\s*(?:ALPACA_API_KEY|ALPACA_SECRET_KEY|API_KEY|SECRET_KEY|TOKEN|PASSWORD)\s*=\s*(?!\s*$).+")


def audit_repository_secrets(root: str | Path, *, allowed_names: Iterable[str] = (".env.example",)) -> SecretAuditResult:
    root = Path(root).resolve()
    allowed = set(allowed_names)
    findings: list[str] = []
    try:
        tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"], stderr=subprocess.DEVNULL)
        candidates = [root / item.decode("utf-8") for item in tracked.split(b"\0") if item]
    except Exception:
        candidates = list(root.rglob("*"))
    for path in candidates:
        if not path.is_file() or ".git" in path.parts or path.name in allowed:
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".env", ".json", ".yaml", ".yml", ""}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _SECRET_ASSIGNMENT.search(text):
            findings.append(str(path.relative_to(root)))
    return SecretAuditResult(not findings, tuple(sorted(findings)))


def required_env(names: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        value = os.getenv(name)
        if not value:
            missing.append(name)
        else:
            result[name] = value
    if missing:
        raise RuntimeError(f"missing required environment variables: {missing}")
    return result
