"""CLI for registering, verifying, and promoting immutable Joe Bot artifacts."""
from __future__ import annotations

import argparse
import json

from research.artifact_registry import ArtifactRegistry, PromotionGate


def main() -> None:
    parser = argparse.ArgumentParser(description="Joe Bot artifact registry")
    parser.add_argument("--registry-root", default="data/registry")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register")
    register.add_argument("artifact_id")
    register.add_argument("artifact_type")
    register.add_argument("files", nargs="+")
    register.add_argument("--dataset-version")
    register.add_argument("--dataset-fingerprint")
    register.add_argument("--repo-root", default=".")

    verify = sub.add_parser("verify")
    verify.add_argument("artifact_id")
    verify.add_argument("--repo-root", default=".")

    promote = sub.add_parser("promote")
    promote.add_argument("artifact_id")
    promote.add_argument("--channel", choices=["shadow", "paper", "tiny-live"], default="shadow")
    promote.add_argument("--repo-root", default=".")

    args = parser.parse_args()
    registry = ArtifactRegistry(args.registry_root)
    if args.command == "register":
        source = []
        if args.dataset_version or args.dataset_fingerprint:
            if not args.dataset_version or not args.dataset_fingerprint:
                raise ValueError("dataset version and fingerprint must be supplied together")
            source.append({"dataset_version": args.dataset_version, "dataset_fingerprint": args.dataset_fingerprint})
        manifest = registry.register(
            artifact_id=args.artifact_id,
            artifact_type=args.artifact_type,
            files=args.files,
            source_datasets=source,
            repo_root=args.repo_root,
            gates=[PromotionGate("registered_files_hashed", True)],
        )
        print(json.dumps({"artifact_id": manifest.artifact_id, "manifest_sha256": manifest.digest}, indent=2))
    elif args.command == "verify":
        print(json.dumps(registry.verify(args.artifact_id, repo_root=args.repo_root), indent=2))
    else:
        path = registry.promote(args.artifact_id, channel=args.channel, repo_root=args.repo_root)
        print(path)


if __name__ == "__main__":
    main()
