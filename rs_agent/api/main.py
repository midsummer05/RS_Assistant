from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

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
    agent_mode: str = Field(default="workflow", pattern="^(workflow|agent)$")
    execution_budget: Optional[int] = Field(default=None, ge=1, le=100)


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


class KnowledgeSearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=50)


class MemorySearchRequest(BaseModel):
    query: str
    user_id: str = "local_user"
    project_id: Optional[str] = "default_project"
    tags: List[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=50)


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
        if request.agent_mode == "agent" and not runtime.supports_agent_mode():
            raise ValueError(
                "Agent mode requires RS_AGENT_LLM_API_KEY and RS_AGENT_LLM_MODEL."
            )
        image_t1, image_t2 = _resolve_inputs(request)
        state = runtime.create_task(
            user_goal=request.user_goal,
            image_t1=image_t1,
            image_t2=image_t2,
            user_id=request.user_id,
            project_id=request.project_id,
            auto_confirm=request.auto_confirm,
            agent_mode=request.agent_mode,
            execution_budget=request.execution_budget,
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


@app.post("/api/tasks/{task_id}/retry")
def retry_task(task_id: str) -> Dict[str, Any]:
    try:
        return runtime.retry_task(task_id).model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
def list_memories(
    user_id: str = "local_user",
    project_id: Optional[str] = "default_project",
) -> List[Dict[str, Any]]:
    return [
        memory.model_dump(mode="json")
        for memory in runtime.store.list_memories(user_id=user_id, project_id=project_id)
    ]


@app.post("/api/memories/search")
def search_memories(request: MemorySearchRequest) -> List[Dict[str, Any]]:
    return [
        memory.model_dump(mode="json")
        for memory in runtime.store.search_memories(
            query=request.query,
            user_id=request.user_id,
            project_id=request.project_id,
            tags=request.tags,
            limit=request.limit,
        )
    ]


@app.delete("/api/memories/{memory_id}")
def archive_memory(memory_id: str) -> Dict[str, str]:
    try:
        runtime.store.archive_memory(memory_id)
        return {"status": "archived", "memory_id": memory_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/memories/purge-expired")
def purge_expired_memories() -> Dict[str, int]:
    return {"purged": runtime.store.purge_expired_memories()}


@app.post("/api/knowledge/search")
def search_knowledge(request: KnowledgeSearchRequest) -> List[Dict[str, Any]]:
    return [
        chunk.model_dump(mode="json")
        for chunk in runtime.rag.search(request.query, limit=request.limit)
    ]


@app.get("/api/knowledge/documents")
def list_knowledge_documents() -> List[Dict[str, Any]]:
    return runtime.store.knowledge_memory.list_documents()


@app.get("/api/knowledge/stats")
def knowledge_memory_stats() -> Dict[str, Any]:
    return runtime.store.knowledge_memory.stats()


@app.post("/api/knowledge/documents/upload")
def upload_knowledge_document(
    file: UploadFile = File(...),
    source_type: str = "document",
    version: str = "1",
    task_tags: str = "change_detection",
) -> Dict[str, Any]:
    safe_name = Path(file.filename or "knowledge.txt").name
    source_dir = runtime.store.root / "knowledge_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / f"{uuid4().hex}_{safe_name}"
    written = 0
    maximum_size = 10 * 1024 * 1024
    with path.open("wb") as handle:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > maximum_size:
                handle.close()
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Knowledge document exceeds 10 MB.")
            handle.write(chunk)
    try:
        result = runtime.store.knowledge_memory.ingest_document(
            path,
            source_type=source_type,
            version=version,
            task_tags=[tag.strip() for tag in task_tags.split(",") if tag.strip()],
            metadata={"original_filename": file.filename},
        )
        if result.get("duplicate"):
            path.unlink(missing_ok=True)
        return result
    except ValueError as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/knowledge/documents/{document_id}")
def delete_knowledge_document(document_id: str) -> Dict[str, str]:
    try:
        runtime.store.knowledge_memory.delete_document(document_id)
        return {"status": "deleted", "document_id": document_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
