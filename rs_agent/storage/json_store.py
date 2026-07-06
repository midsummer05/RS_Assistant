from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rs_agent.agent.state import AssetRecord, Event, MemoryRecord, TaskState, utc_now


class JsonFileStore:
    """Small local persistence layer used by the MVP.

    It mirrors the future database/object-store boundaries from the design
    document while staying runnable on a single developer machine.
    """

    def __init__(self, root: str | Path = ".rs_agent_data") -> None:
        self.root = Path(root)
        self.tasks_dir = self.root / "tasks"
        self.events_dir = self.root / "events"
        self.checkpoints_dir = self.root / "checkpoints"
        self.artifacts_dir = self.root / "artifacts"
        self.memories_dir = self.root / "memories"
        self.uploads_dir = self.root / "uploads"
        self.assets_dir = self.root / "assets"
        for directory in [
            self.tasks_dir,
            self.events_dir,
            self.checkpoints_dir,
            self.artifacts_dir,
            self.memories_dir,
            self.uploads_dir,
            self.assets_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def save_task(self, state: TaskState) -> None:
        state.touch()
        self._atomic_write_json(
            self.tasks_dir / f"{state.task_id}.json",
            state.model_dump(mode="json"),
        )

    def load_task(self, task_id: str) -> TaskState:
        path = self.tasks_dir / f"{task_id}.json"
        if not path.exists():
            raise KeyError(f"Task not found: {task_id}")
        return TaskState.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def list_tasks(self) -> List[TaskState]:
        states: List[TaskState] = []
        for path in sorted(self.tasks_dir.glob("*.json")):
            states.append(TaskState.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        return states

    def append_event(self, event: Event) -> None:
        path = self.events_dir / f"{event.task_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def list_events(self, task_id: str) -> List[Event]:
        path = self.events_dir / f"{task_id}.jsonl"
        if not path.exists():
            return []
        return [
            Event.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def save_checkpoint(self, state: TaskState, label: str) -> None:
        payload = {
            "checkpoint_id": f"{utc_now().strftime('%Y%m%d%H%M%S%f')}_{label}",
            "task_id": state.task_id,
            "label": label,
            "created_at": utc_now().isoformat(),
            "state": state.model_dump(mode="json"),
        }
        path = self.checkpoints_dir / f"{state.task_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._atomic_write_json(self.checkpoints_dir / f"{state.task_id}.latest.json", payload)

    def load_latest_checkpoint(self, task_id: str) -> TaskState:
        path = self.checkpoints_dir / f"{task_id}.latest.json"
        if not path.exists():
            return self.load_task(task_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return TaskState.model_validate(payload["state"])

    def checkpoint_count(self, task_id: str) -> int:
        path = self.checkpoints_dir / f"{task_id}.jsonl"
        if not path.exists():
            return 0
        return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])

    def artifact_path(self, task_id: str, *parts: str) -> Path:
        path = self.artifacts_dir / task_id
        for part in parts:
            path = path / part
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def upload_path(self, filename: str) -> Path:
        path = self.uploads_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_memory(self, memory: MemoryRecord) -> None:
        path = self.memories_dir / "memories.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(memory.model_dump_json() + "\n")

    def list_memories(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> List[MemoryRecord]:
        path = self.memories_dir / "memories.jsonl"
        if not path.exists():
            return []
        wanted_tags = set(tags or [])
        records: List[MemoryRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = MemoryRecord.model_validate(json.loads(line))
            if user_id and record.user_id != user_id:
                continue
            if project_id and record.project_id not in {project_id, None}:
                continue
            if wanted_tags and not wanted_tags.intersection(record.tags):
                continue
            records.append(record)
        return records

    def _asset_index_path(self) -> Path:
        return self.assets_dir / "assets.json"

    def _read_assets(self) -> Dict[str, Any]:
        path = self._asset_index_path()
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def register_asset(self, asset: AssetRecord) -> AssetRecord:
        assets = self._read_assets()
        assets[asset.asset_id] = asset.model_dump(mode="json")
        self._atomic_write_json(self._asset_index_path(), assets)
        return asset

    def load_asset(self, asset_id: str) -> AssetRecord:
        assets = self._read_assets()
        if asset_id not in assets:
            raise KeyError(f"Asset not found: {asset_id}")
        return AssetRecord.model_validate(assets[asset_id])

    def list_assets(self, sensor: Optional[str] = None) -> List[AssetRecord]:
        assets = [
            AssetRecord.model_validate(payload)
            for payload in self._read_assets().values()
        ]
        if sensor:
            assets = [asset for asset in assets if asset.sensor == sensor]
        return assets

