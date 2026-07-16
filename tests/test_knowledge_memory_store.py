from datetime import timedelta

from rs_agent.agent.state import MemoryRecord, utc_now
from rs_agent.knowledge.embeddings import HashingEmbedder
from rs_agent.storage.knowledge_memory_store import KnowledgeMemoryStore
from rs_agent.storage.json_store import JsonFileStore


def test_document_ingestion_versioning_dedup_and_hybrid_search(tmp_path):
    store = KnowledgeMemoryStore(tmp_path / "knowledge.sqlite3", HashingEmbedder())
    document = tmp_path / "bit_model_card.md"
    document.write_text(
        "# BIT 模型卡\n\nBIT Transformer 用于双时相建筑物变化检测。"
        "输入必须精确配准，使用 RGB 波段和 256 像素切片。",
        encoding="utf-8",
    )

    first = store.ingest_document(
        document,
        source_type="model_card",
        task_tags=["change_detection", "bit"],
        version="1",
    )
    second = store.ingest_document(
        document,
        source_type="model_card",
        task_tags=["change_detection", "bit"],
        version="1",
    )
    results = store.search_knowledge("BIT Transformer 建筑物变化检测", limit=3)

    assert first["chunk_count"] >= 1
    assert second["unchanged"] is True
    assert results
    assert results[0].source_type == "model_card"
    assert "精确配准" in results[0].content
    assert store.list_documents()[0]["chunk_count"] >= 1


def test_memory_dedup_ranking_access_and_expiration(tmp_path):
    store = KnowledgeMemoryStore(tmp_path / "memory.sqlite3", HashingEmbedder())
    memory = MemoryRecord(
        user_id="user",
        project_id="project",
        memory_type="result_feedback",
        title="BIT 参数反馈",
        content="BIT 在该区域应使用 32 像素重叠并进行人工抽查。",
        tags=["change_detection", "feedback", "bit"],
        confidence=0.8,
        importance=0.9,
    )
    first = store.save_memory(memory)
    reinforced = store.save_memory(memory.model_copy(update={"memory_id": "another"}))
    expired = store.save_memory(
        MemoryRecord(
            user_id="user",
            project_id="project",
            memory_type="temporary",
            title="临时记忆",
            content="该记录已经过期。",
            tags=["change_detection"],
            expires_at=utc_now() - timedelta(days=1),
        )
    )

    results = store.search_memories(
        query="BIT 重叠 人工抽查",
        user_id="user",
        project_id="project",
        tags=["change_detection"],
        limit=3,
    )

    assert first.memory_id == reinforced.memory_id
    assert reinforced.confidence > first.confidence
    assert results[0].memory_id == first.memory_id
    assert results[0].access_count == 1
    assert store.purge_expired_memories() == 1
    assert all(item.memory_id != expired.memory_id for item in store.list_memories("user", "project"))


def test_memory_archive_removes_it_from_active_retrieval(tmp_path):
    store = KnowledgeMemoryStore(tmp_path / "archive.sqlite3", HashingEmbedder())
    memory = store.save_memory(
        MemoryRecord(
            user_id="user",
            project_id="project",
            memory_type="preference",
            title="模型偏好",
            content="优先使用 BIT 模型。",
            tags=["change_detection"],
        )
    )
    store.archive_memory(memory.memory_id)

    assert store.list_memories("user", "project") == []


def test_legacy_jsonl_memories_are_migrated_once(tmp_path):
    memories_dir = tmp_path / "memories"
    memories_dir.mkdir(parents=True)
    legacy = MemoryRecord(
        user_id="legacy-user",
        project_id="legacy-project",
        memory_type="feedback",
        title="旧反馈",
        content="需要人工抽查变化图斑。",
        tags=["change_detection", "feedback"],
    )
    (memories_dir / "memories.jsonl").write_text(
        legacy.model_dump_json() + "\n",
        encoding="utf-8",
    )

    first = JsonFileStore(tmp_path)
    second = JsonFileStore(tmp_path)

    migrated = first.list_memories("legacy-user", "legacy-project")
    assert len(migrated) == 1
    assert migrated[0].content == legacy.content
    assert second.list_memories("legacy-user", "legacy-project")[0].memory_id == migrated[0].memory_id
    assert (memories_dir / ".sqlite_migrated").exists()
