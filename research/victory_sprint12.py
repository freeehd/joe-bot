"""Victory Sprint 12 production-hardening review. Does not enable live trading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ops.security import audit_repository_secrets
from research.artifact_registry import ArtifactRegistry
from research.go_live_review import review_go_live


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Sprint 12 formal production-hardening review")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--registry-root", default="data/registry")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--recovery-test-passed", action="store_true")
    parser.add_argument("--supervisor-test-passed", action="store_true")
    parser.add_argument("--rbac-enabled", action="store_true")
    parser.add_argument("--backup-restore-test-passed", action="store_true")
    parser.add_argument("--output", default="data/experiments/sprint12_go_live_review.json")
    args = parser.parse_args()

    secrets = audit_repository_secrets(args.repo_root)
    review = review_go_live(
        ArtifactRegistry(args.registry_root),
        args.artifact,
        secret_hygiene_passed=secrets.passed,
        recovery_test_passed=args.recovery_test_passed,
        supervisor_test_passed=args.supervisor_test_passed,
        rbac_enabled=args.rbac_enabled,
        backup_restore_test_passed=args.backup_restore_test_passed,
    )
    payload = review.to_dict()
    payload["secret_findings"] = list(secrets.findings)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
