from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rs_agent.agent.state import AssetRecord
from rs_agent.factory import build_runtime


app = FastAPI(title="Remote Sensing Agent MVP", version="0.1.0")
runtime = build_runtime()
LEGACY_STATIC_DIR = Path(__file__).resolve().parent / "static"
FRONTEND_DIST_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"

if FRONTEND_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_DIR), name="frontend-assets")
app.mount("/app", StaticFiles(directory=LEGACY_STATIC_DIR), name="legacy-app")


class CreateTaskRequest(BaseModel):
    user_goal: str
    image_t1_uri: Optional[str] = None
    image_t2_uri: Optional[str] = None
    asset_t1_id: Optional[str] = None
    asset_t2_id: Optional[str] = None
    user_id: str = "local_user"
    project_id: str = "default_project"
    auto_confirm: bool = False


class FeedbackRequest(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: str = ""
    accepted: Optional[bool] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegisterAssetRequest(BaseModel):
    uri: str
    sensor: Optional[str] = None
    acquired_at: Optional[str] = None
    cloud_cover: Optional[float] = None
    crs: Optional[str] = None
    resolution: Optional[float] = None
    bbox: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def workspace() -> FileResponse:
    index_path = FRONTEND_DIST_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return FileResponse(LEGACY_STATIC_DIR / "index.html")


@app.post("/api/tasks")
def create_task(request: CreateTaskRequest) -> Dict[str, Any]:
    try:
        image_t1, image_t2 = _resolve_inputs(request)
        state = runtime.create_task(
            user_goal=request.user_goal,
            image_t1=image_t1,
            image_t2=image_t2,
            user_id=request.user_id,
            project_id=request.project_id,
            auto_confirm=request.auto_confirm,
        )
        return state.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/tasks")
def list_tasks() -> List[Dict[str, Any]]:
    return [task.model_dump(mode="json") for task in runtime.store.list_tasks()]


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> Dict[str, Any]:
    try:
        return runtime.store.load_task(task_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/tasks/{task_id}/resume")
def resume_task(task_id: str) -> Dict[str, Any]:
    try:
        return runtime.run_task(task_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/tasks/{task_id}/events")
def get_events(task_id: str) -> List[Dict[str, Any]]:
    return [event.model_dump(mode="json") for event in runtime.list_events(task_id)]


@app.get("/api/tasks/{task_id}/artifacts")
def get_artifacts(task_id: str) -> List[Dict[str, Any]]:
    try:
        task = runtime.store.load_task(task_id)
        return [artifact.model_dump(mode="json") for artifact in task.artifacts.values()]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/tasks/{task_id}/artifacts/{artifact_id}/file")
def get_artifact_file(task_id: str, artifact_id: str) -> FileResponse:
    try:
        task = runtime.store.load_task(task_id)
        artifact = task.artifacts[artifact_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    path = Path(artifact.uri)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Artifact file not found: {artifact_id}")
    return FileResponse(path, filename=path.name)


@app.get("/api/tasks/{task_id}/interrupts")
def get_interrupts(task_id: str) -> List[Dict[str, Any]]:
    try:
        task = runtime.store.load_task(task_id)
        return [interrupt.model_dump(mode="json") for interrupt in task.interrupts]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/tasks/{task_id}/interrupts/{interrupt_id}/approve")
def approve_interrupt(task_id: str, interrupt_id: str) -> Dict[str, Any]:
    try:
        return runtime.approve_interrupt(task_id, interrupt_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/tasks/{task_id}/feedback")
def submit_feedback(task_id: str, request: FeedbackRequest) -> Dict[str, Any]:
    try:
        state = runtime.submit_feedback(
            task_id=task_id,
            rating=request.rating,
            comment=request.comment,
            accepted=request.accepted,
            metadata=request.metadata,
        )
        return state.model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/tools")
def list_tools() -> List[Dict[str, Any]]:
    return [spec.model_dump(mode="json") for spec in runtime.executor.registry.specs()]


@app.post("/api/assets/register")
def register_asset(request: RegisterAssetRequest) -> Dict[str, Any]:
    asset = AssetRecord(**request.model_dump())
    runtime.store.register_asset(asset)
    return asset.model_dump(mode="json")


@app.post("/api/assets/upload")
def upload_asset(file: UploadFile = File(...)) -> Dict[str, Any]:
    safe_name = Path(file.filename or "upload.bin").name
    path = runtime.store.upload_path(safe_name)
    with path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    asset = AssetRecord(uri=str(path.resolve()), metadata={"original_filename": file.filename})
    runtime.store.register_asset(asset)
    return asset.model_dump(mode="json")


@app.post("/api/assets/search")
def search_assets(request: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    sensor = (request or {}).get("sensor")
    return [asset.model_dump(mode="json") for asset in runtime.store.list_assets(sensor=sensor)]


@app.get("/api/assets/{asset_id}")
def get_asset(asset_id: str) -> Dict[str, Any]:
    try:
        return runtime.store.load_asset(asset_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/memories")
def list_memories(user_id: Optional[str] = None, project_id: Optional[str] = None) -> List[Dict[str, Any]]:
    return [
        memory.model_dump(mode="json")
        for memory in runtime.store.list_memories(user_id=user_id, project_id=project_id)
    ]


@app.post("/api/knowledge/search")
def search_knowledge(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    query = request.get("query", "")
    limit = int(request.get("limit", 5))
    return [chunk.model_dump(mode="json") for chunk in runtime.rag.search(query, limit=limit)]


def _resolve_inputs(request: CreateTaskRequest) -> tuple[str, str]:
    image_t1 = request.image_t1_uri
    image_t2 = request.image_t2_uri
    if request.asset_t1_id:
        image_t1 = runtime.store.load_asset(request.asset_t1_id).uri
    if request.asset_t2_id:
        image_t2 = runtime.store.load_asset(request.asset_t2_id).uri
    if not image_t1 or not image_t2:
        raise ValueError("Provide image_t1_uri/image_t2_uri or asset_t1_id/asset_t2_id.")
    return image_t1, image_t2
