"""FastAPI application. Localhost-only; serves the built frontend, the JSON
API, and repo-local images (with strict path validation).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .paths import RepoPaths
from .service import ReviewService


class ReviewerPayload(BaseModel):
    reviewer: str = Field(min_length=1, max_length=32)


class GuidePreferencePayload(BaseModel):
    do_not_autoshow: bool


class DraftPayload(BaseModel):
    draft: Dict[str, Any]


class FinalizePayload(BaseModel):
    decision: str
    review: Dict[str, Any]


class CursorPayload(BaseModel):
    position: int = Field(ge=0)


def create_app(service: Optional[ReviewService] = None) -> FastAPI:
    svc = service or ReviewService()
    app = FastAPI(title="MuLAn Human Review", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.service = svc

    def guard(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    # -- meta ------------------------------------------------------------
    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"ok": True}

    @app.get("/api/bootstrap")
    def bootstrap() -> Dict[str, Any]:
        return guard(svc.bootstrap)

    @app.post("/api/index/rebuild")
    def rebuild_index() -> Dict[str, Any]:
        return guard(svc.rebuild_index)

    @app.post("/api/reviewer")
    def set_reviewer(payload: ReviewerPayload) -> Dict[str, Any]:
        return guard(svc.set_active_reviewer, payload.reviewer)

    @app.post("/api/reviewer/switch")
    def switch_reviewer() -> Dict[str, Any]:
        svc.clear_active_reviewer()
        return {"reviewer": None}

    @app.post("/api/guide-preference")
    def set_guide_preference(payload: GuidePreferencePayload) -> Dict[str, Any]:
        if svc.store is None:
            raise HTTPException(status_code=409, detail="No reviewer selected.")
        svc.store.set_guide_preference(payload.do_not_autoshow)
        return {"guide": svc.store.state.get("guide")}

    # -- queue / scenes --------------------------------------------------
    @app.get("/api/queue/current")
    def queue_current() -> Dict[str, Any]:
        scene_id = guard(svc.current_scene_id)
        return {"scene_id": scene_id}

    @app.post("/api/queue/cursor")
    def queue_cursor(payload: CursorPayload) -> Dict[str, Any]:
        guard(svc.ensure_ready)
        svc.store.set_cursor(payload.position)
        return {"cursor": svc.store.state.get("cursor"), "scene_id": guard(svc.current_scene_id)}

    @app.get("/api/deferred")
    def deferred() -> Any:
        return guard(svc.list_deferred)

    @app.get("/api/scene/{scene_id}")
    def get_scene(scene_id: str, check_files: bool = Query(default=True)) -> Dict[str, Any]:
        return guard(svc.get_scene, scene_id, check_files)

    @app.post("/api/scene/{scene_id}/draft")
    def save_draft(scene_id: str, payload: DraftPayload) -> Dict[str, Any]:
        return guard(svc.save_draft, scene_id, payload.draft)

    @app.post("/api/scene/{scene_id}/defer")
    def defer_scene(scene_id: str) -> Dict[str, Any]:
        return guard(svc.defer_scene, scene_id)

    @app.post("/api/scene/{scene_id}/undefer")
    def undefer_scene(scene_id: str) -> Dict[str, Any]:
        guard(svc.ensure_ready)
        svc.store.undefer(scene_id)
        return {"deferred": False}

    @app.post("/api/scene/{scene_id}/finalize")
    def finalize(scene_id: str, payload: FinalizePayload) -> Any:
        result = guard(svc.finalize, scene_id, payload.review, payload.decision)
        status = 200 if result.get("ok") else 422
        return JSONResponse(result, status_code=status)

    @app.post("/api/undo")
    def undo() -> Dict[str, Any]:
        return guard(svc.undo_last)

    # -- stats / maintenance --------------------------------------------
    @app.get("/api/progress")
    def get_progress() -> Dict[str, Any]:
        return guard(svc.get_progress)

    @app.get("/api/dashboard")
    def get_dashboard() -> Dict[str, Any]:
        return guard(svc.get_dashboard)

    @app.post("/api/export")
    def export_now() -> Dict[str, Any]:
        return guard(svc.export_now)

    @app.post("/api/backup")
    def backup_now() -> Dict[str, Any]:
        return guard(svc.backup_now)

    # -- images ----------------------------------------------------------
    @app.get("/api/image/{rel_path:path}")
    def serve_image(rel_path: str) -> FileResponse:
        try:
            resolved = svc.paths.resolve_servable(rel_path)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not resolved.exists():
            raise HTTPException(status_code=404, detail=f"image not found: {rel_path}")
        return FileResponse(
            resolved,
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    # -- static frontend -------------------------------------------------
    dist = svc.paths.frontend_dist
    if dist.exists() and (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    else:  # pragma: no cover - depends on whether the frontend was built
        @app.get("/")
        def not_built() -> JSONResponse:
            return JSONResponse(
                {
                    "detail": "Frontend not built. Run setup.ps1 (npm run build) first; "
                    "the API is available under /api/."
                },
                status_code=503,
            )

    return app
