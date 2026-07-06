from pathlib import Path

from rs_agent.factory import build_runtime


def test_change_detection_minimum_loop_with_plan_review(tmp_path):
    runtime = build_runtime(tmp_path)

    waiting = runtime.create_task(
        user_goal="请对两期 Sentinel-2 影像做建设用地扩张变化检测，并输出图斑、面积统计和报告。",
        image_t1="demo://image_t1",
        image_t2="demo://image_t2",
        user_id="tester",
        project_id="project_demo",
        auto_confirm=False,
    )

    assert waiting.status == "waiting_human"
    assert waiting.task_type == "change_detection"
    assert waiting.plan[0].tool_name == "raster.inspect_metadata"
    assert any(chunk.source_type == "model_card" for chunk in waiting.retrieved_context)
    assert waiting.interrupts and waiting.interrupts[-1].type == "plan_review"

    final = runtime.approve_interrupt(waiting.task_id, waiting.interrupts[-1].interrupt_id)

    assert final.status == "succeeded"
    assert {"change_preview", "change_vector", "area_statistics", "markdown_report"}.issubset(final.artifact_refs)
    assert final.working_memory["metadata_t1"]["crs"] == "EPSG:32650"
    assert final.working_memory["metadata_t2"]["acquired_at"] == "2025-07-01"
    assert final.working_memory["area_statistics"]["summary"]["area_m2"] > 0
    assert final.working_memory["quality"]["passed"] is True

    report_path = Path(final.artifact_by_alias("markdown_report").uri)
    assert report_path.exists()
    assert "遥感变化检测报告" in report_path.read_text(encoding="utf-8")

    event_types = [event.event_type for event in runtime.list_events(final.task_id)]
    assert "ContextRetrieved" in event_types
    assert "PlanGenerated" in event_types
    assert "ToolCallSucceeded" in event_types
    assert "MemoryWritten" in event_types
    assert runtime.store.checkpoint_count(final.task_id) >= 10

    with_feedback = runtime.submit_feedback(
        final.task_id,
        rating=4,
        comment="图斑整体可用，后续希望增加人工抽查。",
        accepted=True,
    )
    assert with_feedback.user_feedback[-1]["rating"] == 4
    memories = runtime.store.list_memories(user_id="tester", project_id="project_demo", tags=["feedback"])
    assert memories
    assert "人工抽查" in memories[-1].content

