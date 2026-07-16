from pathlib import Path

import numpy as np

from rs_agent.agent.planner import DeterministicPlanner
from rs_agent.agent.state import TaskState
from rs_agent.models.BIT_CD.inference import _positions, _prepare_tile, _tile_windows
from rs_agent.rag.retriever import LocalRagRetriever
from rs_agent.storage.json_store import JsonFileStore
from rs_agent.tools.ml import bit_change_detection as bit_tool
from rs_agent.tools.schemas import ToolContext


def test_bit_tiling_covers_image_and_normalizes():
    windows = list(_tile_windows(300, 410, tile_size=256, overlap=32))
    coverage = np.zeros((300, 410), dtype=np.uint8)
    for row, col, height, width in windows:
        coverage[row : row + height, col : col + width] = 1
    assert coverage.all()
    assert _positions(410, 256, 224)[-1] == 154

    tile = np.ones((10, 12, 3), dtype=np.float32)
    prepared = _prepare_tile(tile, 256)
    assert prepared.shape == (3, 256, 256)
    assert prepared[:, :10, :12].max() == 1.0
    assert prepared[:, 10:, :].min() == -1.0


def test_bit_tool_preserves_raster_metadata(monkeypatch, tmp_path):
    class FakeDetector:
        def __init__(self, config):
            self.config = config

        def predict(self, image_t1, image_t2):
            result = np.zeros(image_t1.shape[:2], dtype=np.uint8)
            result[10:20, 10:20] = 1
            return result

    monkeypatch.setattr(bit_tool, "BITChangeDetector", FakeDetector)
    store = JsonFileStore(tmp_path)
    state = TaskState(user_id="tester", user_goal="BIT change detection")
    result = bit_tool.bit_change_detection(
        ToolContext(task_state=state, store=store),
        {
            "raster_t1": "demo://image_t1",
            "raster_t2": "demo://image_t2",
            "output_alias": "bit_change",
        },
    )

    artifact = result.artifacts[0]
    assert artifact.alias == "bit_change"
    assert artifact.crs == "EPSG:32650"
    assert artifact.metadata["model_id"] == "BIT_LEVIR"
    assert Path(artifact.uri).exists()


def test_planner_selects_bit_only_when_requested():
    state = TaskState(
        user_id="tester",
        user_goal="请使用 BIT Transformer 深度模型执行变化检测",
        constraints={"inputs": {"image_t1": "a", "image_t2": "b"}},
    )
    planner = DeterministicPlanner()
    context = LocalRagRetriever().retrieve_for_planning(state)
    plan = planner.generate_plan(state, context, [])
    assert next(step for step in plan.steps if step.step_id == "s07_detect_change").tool_name == "ml.bit_change_detection"
