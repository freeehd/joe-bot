"""FastAPI REST/WebSocket backend for the ARGUS Golden GUI."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from argus_api.controls import OperatorControls, ReadOnlyControls
from argus_api.state import OperatorStateStore
from database.db import AuditStore
from research.artifact_registry import ArtifactRegistry


class ControlRequest(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


def create_app(
    *,
    state: OperatorStateStore | None = None,
    controls: OperatorControls | None = None,
    audit_db: str | Path | None = None,
    registry_root: str | Path = "data/registry",
) -> FastAPI:
    state = state or OperatorStateStore()
    controls = controls or ReadOnlyControls()
    app = FastAPI(title="ARGUS Operator API", version="1.0.0")

    @app.get("/api/command-center")
    def command_center() -> dict[str, Any]:
        snap = state.snapshot()
        return {key: snap[key] for key in ("mode", "account", "performance", "regime", "scanner", "positions", "portfolio", "risk", "alerts", "updated_at", "version")}

    @app.get("/api/scanner")
    def scanner() -> list[dict[str, Any]]:
        return state.snapshot()["scanner"]

    @app.get("/api/portfolio")
    def portfolio() -> dict[str, Any]:
        snap = state.snapshot()
        return {"portfolio": snap["portfolio"], "positions": snap["positions"], "account": snap["account"]}

    @app.get("/api/positions")
    def positions() -> list[dict[str, Any]]:
        return state.snapshot()["positions"]

    @app.get("/api/risk")
    def risk() -> dict[str, Any]:
        return state.snapshot()["risk"]

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        snap = state.snapshot()
        return {"services": snap["system"], "alerts": snap["alerts"], "updated_at": snap["updated_at"]}

    @app.get("/api/models")
    def models() -> list[dict[str, Any]]:
        return state.snapshot()["models"]

    @app.get("/api/laya")
    def laya() -> dict[str, Any]:
        models = state.snapshot()["models"]
        item = next((model for model in models if str(model.get("name", "")).lower() == "laya"), None)
        return item or {"name": "Laya", "status": "NOT CONFIGURED", "calibration": "UNKNOWN", "recent_decisions": []}

    @app.get("/api/execution")
    def execution() -> dict[str, Any]:
        orders = state.snapshot()["orders"]
        return {"orders": orders, "order_count": len(orders), "live_capital_authorized": False}

    @app.get("/api/backtests")
    def backtests() -> list[dict[str, Any]]:
        root = Path("data/experiments")
        if not root.exists():
            return []
        return [
            {"name": path.name, "path": str(path), "size_bytes": path.stat().st_size}
            for path in sorted(root.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)[:100]
        ]

    @app.get("/api/research/artifacts")
    def research_artifacts() -> list[dict[str, Any]]:
        root = ArtifactRegistry(registry_root).artifacts_dir
        if not root.exists():
            return []
        rows = []
        for manifest_path in sorted(root.glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append({
                "artifact_id": manifest.get("artifact_id"), "artifact_type": manifest.get("artifact_type"),
                "status": manifest.get("status"), "created_at_utc": manifest.get("created_at_utc"),
                "gates": manifest.get("gates", []),
            })
        return rows

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        snap = state.snapshot()
        return {"mode": snap["mode"], "controls": "read-only" if isinstance(controls, ReadOnlyControls) else "paper-risk", "live_capital_authorized": False}

    @app.get("/api/replay/{decision_id}")
    def replay(decision_id: str) -> dict[str, Any]:
        if audit_db is None:
            raise HTTPException(status_code=503, detail="audit database is not configured")
        with AuditStore(audit_db) as audit:
            result = audit.replay(decision_id)
        if result["event_count"] == 0:
            raise HTTPException(status_code=404, detail="decision not found")
        return result

    @app.get("/api/artifacts/{artifact_id}")
    def artifact(artifact_id: str) -> dict[str, Any]:
        try:
            return ArtifactRegistry(registry_root).read(artifact_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="artifact not found") from exc

    @app.post("/api/risk/disable-entries")
    def disable_entries(request: ControlRequest) -> dict[str, Any]:
        result = controls.disable_entries(request.reason)
        if not result["accepted"]:
            raise HTTPException(status_code=403, detail=result["reason"])
        return result

    @app.post("/api/risk/enable-entries")
    def enable_entries(request: ControlRequest) -> dict[str, Any]:
        result = controls.enable_entries(request.reason)
        if not result["accepted"]:
            raise HTTPException(status_code=403, detail=result["reason"])
        return result

    channel_map = {
        "scanner": "scanner",
        "positions": "positions",
        "orders": "orders",
        "risk": "risk",
        "system": "system",
    }

    @app.websocket("/ws/live/{channel}")
    async def live_channel(websocket: WebSocket, channel: str) -> None:
        if channel not in channel_map:
            await websocket.close(code=1008, reason="unknown channel")
            return
        await websocket.accept()
        last_version = -1
        try:
            while True:
                snap = state.snapshot()
                if snap["version"] != last_version:
                    await websocket.send_json({"channel": channel, "version": snap["version"], "data": snap[channel_map[channel]], "updated_at": snap["updated_at"]})
                    last_version = snap["version"]
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            return

    return app


app = create_app()
