from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from rs_agent.agent.state import KnowledgeChunk, MemoryRecord, utc_now
from rs_agent.knowledge.chunking import chunk_document, parse_document
from rs_agent.knowledge.embeddings import Embedder, cosine_similarity


class KnowledgeMemoryStore:
    def __init__(self, path: str | Path, embedder: Embedder) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    document_id TEXT PRIMARY KEY,
                    source_uri TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(source_uri, version)
                );
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES knowledge_documents(document_id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    task_tags_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    UNIQUE(document_id, ordinal)
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
                    chunk_id UNINDEXED, title, content, tags, tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_id TEXT,
                    memory_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_task_id TEXT,
                    tags_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    status TEXT NOT NULL,
                    importance REAL NOT NULL,
                    expires_at TEXT,
                    last_accessed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    fingerprint TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_scope
                    ON memories(user_id, project_id, status, memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_fingerprint
                    ON memories(user_id, project_id, fingerprint, status);
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    memory_id UNINDEXED, title, content, tags, tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS system_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
        self._ensure_embedding_index()

    def _ensure_embedding_index(self) -> None:
        signature = f"{self.embedder.name}:{self.embedder.dimensions}"
        with self._connect() as db:
            row = db.execute(
                "SELECT value FROM system_metadata WHERE key='embedding_signature'"
            ).fetchone()
            if row and row["value"] == signature:
                return
            knowledge_rows = db.execute(
                "SELECT chunk_id, content FROM knowledge_chunks ORDER BY chunk_id"
            ).fetchall()
            memory_rows = db.execute(
                "SELECT memory_id, title, content FROM memories ORDER BY memory_id"
            ).fetchall()
            if knowledge_rows:
                vectors = self.embedder.embed([item["content"] for item in knowledge_rows])
                db.executemany(
                    "UPDATE knowledge_chunks SET embedding_json=? WHERE chunk_id=?",
                    [
                        (json.dumps(vector), row_item["chunk_id"])
                        for row_item, vector in zip(knowledge_rows, vectors)
                    ],
                )
            if memory_rows:
                vectors = self.embedder.embed(
                    [f"{item['title']}\n{item['content']}" for item in memory_rows]
                )
                db.executemany(
                    "UPDATE memories SET embedding_json=? WHERE memory_id=?",
                    [
                        (json.dumps(vector), row_item["memory_id"])
                        for row_item, vector in zip(memory_rows, vectors)
                    ],
                )
            db.execute(
                """INSERT INTO system_metadata(key, value) VALUES('embedding_signature', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (signature,),
            )

    def ingest_document(
        self,
        path: str | Path,
        source_type: str = "document",
        task_tags: Optional[List[str]] = None,
        version: str = "1",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        path = Path(path).resolve()
        parsed = parse_document(path)
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        document_id = hashlib.sha256(f"{path}|{version}".encode()).hexdigest()[:24]
        chunks = chunk_document(parsed.text)
        embeddings = self.embedder.embed(chunks)
        now = utc_now().isoformat()
        combined_metadata = {**parsed.metadata, **(metadata or {})}
        with self._connect() as db:
            duplicate = db.execute(
                """SELECT document_id FROM knowledge_documents
                WHERE checksum=? AND source_type=? AND active=1 LIMIT 1""",
                (checksum, source_type),
            ).fetchone()
            if duplicate and duplicate["document_id"] != document_id:
                count = db.execute(
                    "SELECT count(*) AS count FROM knowledge_chunks WHERE document_id=?",
                    (duplicate["document_id"],),
                ).fetchone()["count"]
                return {
                    "document_id": duplicate["document_id"],
                    "chunk_count": count,
                    "unchanged": True,
                    "duplicate": True,
                }
            existing = db.execute(
                "SELECT checksum FROM knowledge_documents WHERE document_id=?",
                (document_id,),
            ).fetchone()
            if existing and existing["checksum"] == checksum:
                count = db.execute(
                    "SELECT count(*) AS count FROM knowledge_chunks WHERE document_id=?",
                    (document_id,),
                ).fetchone()["count"]
                return {
                    "document_id": document_id,
                    "chunk_count": count,
                    "unchanged": True,
                    "duplicate": False,
                }
            old_chunk_ids = [
                row["chunk_id"]
                for row in db.execute(
                    "SELECT chunk_id FROM knowledge_chunks WHERE document_id=?", (document_id,)
                )
            ]
            for chunk_id in old_chunk_ids:
                db.execute("DELETE FROM knowledge_fts WHERE chunk_id=?", (chunk_id,))
            db.execute("DELETE FROM knowledge_documents WHERE document_id=?", (document_id,))
            db.execute(
                """INSERT INTO knowledge_documents
                (document_id, source_uri, title, source_type, version, checksum, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    document_id,
                    str(path),
                    parsed.title,
                    source_type,
                    version,
                    checksum,
                    json.dumps(combined_metadata, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            for ordinal, (content, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{document_id}#{ordinal:04d}"
                chunk_checksum = hashlib.sha256(content.encode()).hexdigest()
                chunk_metadata = {**combined_metadata, "source_uri": str(path), "ordinal": ordinal}
                db.execute(
                    """INSERT INTO knowledge_chunks
                    (chunk_id, document_id, ordinal, title, content, task_tags_json, metadata_json, embedding_json, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk_id,
                        document_id,
                        ordinal,
                        parsed.title,
                        content,
                        json.dumps(task_tags or [], ensure_ascii=False),
                        json.dumps(chunk_metadata, ensure_ascii=False),
                        json.dumps(embedding),
                        chunk_checksum,
                    ),
                )
                db.execute(
                    "INSERT INTO knowledge_fts(chunk_id, title, content, tags) VALUES (?, ?, ?, ?)",
                    (chunk_id, parsed.title, content, " ".join(task_tags or [])),
                )
        return {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "unchanged": False,
            "duplicate": False,
        }

    def upsert_chunks(self, chunks: Iterable[KnowledgeChunk], source_uri: str = "builtin://default") -> None:
        chunks = list(chunks)
        if not chunks:
            return
        grouped: Dict[str, List[KnowledgeChunk]] = {}
        for chunk in chunks:
            document_id = chunk.document_id or chunk.chunk_id.split("#", 1)[0]
            grouped.setdefault(document_id, []).append(chunk)
        now = utc_now().isoformat()
        with self._connect() as db:
            for document_id, items in grouped.items():
                document_checksum = hashlib.sha256(
                    "\n".join(item.content for item in items).encode()
                ).hexdigest()
                db.execute(
                    """INSERT OR IGNORE INTO knowledge_documents
                    (document_id, source_uri, title, source_type, version, checksum, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        document_id,
                        f"{source_uri.rstrip('/')}/{document_id}",
                        items[0].title,
                        items[0].source_type,
                        items[0].version or "1",
                        document_checksum,
                        "{}",
                        now,
                        now,
                    ),
                )
                embeddings = self.embedder.embed([item.content for item in items])
                for ordinal, (item, embedding) in enumerate(zip(items, embeddings)):
                    exists = db.execute(
                        "SELECT 1 FROM knowledge_chunks WHERE chunk_id=?", (item.chunk_id,)
                    ).fetchone()
                    if exists:
                        continue
                    db.execute(
                        """INSERT INTO knowledge_chunks
                        (chunk_id, document_id, ordinal, title, content, task_tags_json, metadata_json, embedding_json, checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            item.chunk_id,
                            document_id,
                            ordinal,
                            item.title,
                            item.content,
                            json.dumps(item.task_tags, ensure_ascii=False),
                            json.dumps(item.metadata, ensure_ascii=False),
                            json.dumps(embedding),
                            hashlib.sha256(item.content.encode()).hexdigest(),
                        ),
                    )
                    db.execute(
                        "INSERT INTO knowledge_fts(chunk_id, title, content, tags) VALUES (?, ?, ?, ?)",
                        (item.chunk_id, item.title, item.content, " ".join(item.task_tags)),
                    )

    def search_knowledge(
        self,
        query: str,
        limit: int = 5,
        task_tags: Optional[List[str]] = None,
    ) -> List[KnowledgeChunk]:
        query_embedding = self.embedder.embed([query])[0]
        lexical = self._fts_scores("knowledge_fts", "chunk_id", query, limit * 5)
        with self._connect() as db:
            rows = db.execute(
                """SELECT c.*, d.source_type, d.version, d.active
                FROM knowledge_chunks c JOIN knowledge_documents d USING(document_id)
                WHERE d.active=1"""
            ).fetchall()
        wanted_tags = set(task_tags or [])
        scored = []
        for row in rows:
            tags = json.loads(row["task_tags_json"])
            if wanted_tags and not wanted_tags.intersection(tags):
                tag_bonus = 0.0
            else:
                tag_bonus = 0.15 if wanted_tags else 0.0
            vector_score = max(0.0, cosine_similarity(query_embedding, json.loads(row["embedding_json"])))
            lexical_score = lexical.get(row["chunk_id"], 0.0)
            score = 0.55 * lexical_score + 0.4 * vector_score + tag_bonus
            if score <= 0:
                continue
            scored.append(
                KnowledgeChunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    version=row["version"],
                    title=row["title"],
                    content=row["content"],
                    source_type=row["source_type"],
                    task_tags=tags,
                    metadata=json.loads(row["metadata_json"]),
                    score=round(score, 6),
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    def list_documents(self) -> List[Dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT d.*, count(c.chunk_id) AS chunk_count
                FROM knowledge_documents d LEFT JOIN knowledge_chunks c USING(document_id)
                GROUP BY d.document_id ORDER BY d.updated_at DESC"""
            ).fetchall()
        return [{**dict(row), "metadata": json.loads(row["metadata_json"])} for row in rows]

    def stats(self) -> Dict[str, Any]:
        with self._connect() as db:
            documents = db.execute(
                "SELECT count(*) AS count FROM knowledge_documents WHERE active=1"
            ).fetchone()["count"]
            chunks = db.execute("SELECT count(*) AS count FROM knowledge_chunks").fetchone()["count"]
            memories = db.execute(
                "SELECT count(*) AS count FROM memories WHERE status='active'"
            ).fetchone()["count"]
        return {
            "documents": documents,
            "chunks": chunks,
            "active_memories": memories,
            "embedder": self.embedder.name,
            "embedding_dimensions": self.embedder.dimensions,
            "database": str(self.path.resolve()),
        }

    def delete_document(self, document_id: str) -> None:
        with self._connect() as db:
            chunk_ids = [
                row["chunk_id"]
                for row in db.execute(
                    "SELECT chunk_id FROM knowledge_chunks WHERE document_id=?", (document_id,)
                )
            ]
            for chunk_id in chunk_ids:
                db.execute("DELETE FROM knowledge_fts WHERE chunk_id=?", (chunk_id,))
            result = db.execute("DELETE FROM knowledge_documents WHERE document_id=?", (document_id,))
            if result.rowcount == 0:
                raise KeyError(f"Knowledge document not found: {document_id}")

    def save_memory(self, memory: MemoryRecord) -> MemoryRecord:
        fingerprint = _fingerprint(memory)
        embedding = self.embedder.embed([f"{memory.title}\n{memory.content}"])[0]
        now = utc_now()
        with self._connect() as db:
            existing = db.execute(
                """SELECT * FROM memories WHERE user_id=? AND project_id IS ?
                AND fingerprint=? AND status='active' LIMIT 1""",
                (memory.user_id, memory.project_id, fingerprint),
            ).fetchone()
            if existing:
                metadata = json.loads(existing["metadata_json"])
                metadata["reinforcement_count"] = int(metadata.get("reinforcement_count", 1)) + 1
                confidence = min(1.0, max(float(existing["confidence"]), memory.confidence) + 0.03)
                db.execute(
                    """UPDATE memories SET confidence=?, importance=?, metadata_json=?,
                    updated_at=? WHERE memory_id=?""",
                    (
                        confidence,
                        max(float(existing["importance"]), memory.importance),
                        json.dumps(metadata, ensure_ascii=False),
                        now.isoformat(),
                        existing["memory_id"],
                    ),
                )
                db.commit()
                return self.get_memory(existing["memory_id"])
            payload = memory.model_copy(update={"updated_at": now})
            db.execute(
                """INSERT INTO memories
                (memory_id, user_id, project_id, memory_type, title, content, confidence,
                source_task_id, tags_json, metadata_json, scope, status, importance,
                expires_at, last_accessed_at, access_count, fingerprint, embedding_json,
                created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload.memory_id,
                    payload.user_id,
                    payload.project_id,
                    payload.memory_type,
                    payload.title,
                    payload.content,
                    payload.confidence,
                    payload.source_task_id,
                    json.dumps(payload.tags, ensure_ascii=False),
                    json.dumps(payload.metadata, ensure_ascii=False),
                    payload.scope,
                    payload.status,
                    payload.importance,
                    payload.expires_at.isoformat() if payload.expires_at else None,
                    None,
                    0,
                    fingerprint,
                    json.dumps(embedding),
                    payload.created_at.isoformat(),
                    payload.updated_at.isoformat(),
                ),
            )
            db.execute(
                "INSERT INTO memories_fts(memory_id, title, content, tags) VALUES (?, ?, ?, ?)",
                (payload.memory_id, payload.title, payload.content, " ".join(payload.tags)),
            )
        return payload

    def get_memory(self, memory_id: str) -> MemoryRecord:
        with self._connect() as db:
            row = db.execute("SELECT * FROM memories WHERE memory_id=?", (memory_id,)).fetchone()
        if not row:
            raise KeyError(f"Memory not found: {memory_id}")
        return _memory_from_row(row)

    def list_memories(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        status: str = "active",
    ) -> List[MemoryRecord]:
        clauses = ["status=?", "(expires_at IS NULL OR expires_at>?)"]
        values: List[Any] = [status, utc_now().isoformat()]
        if user_id:
            clauses.append("user_id=?")
            values.append(user_id)
        if project_id:
            clauses.append("(project_id=? OR project_id IS NULL)")
            values.append(project_id)
        with self._connect() as db:
            rows = db.execute(
                f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY updated_at",
                values,
            ).fetchall()
        wanted_tags = set(tags or [])
        records = [_memory_from_row(row) for row in rows]
        return [record for record in records if not wanted_tags or wanted_tags.intersection(record.tags)]

    def search_memories(
        self,
        query: str,
        user_id: str,
        project_id: Optional[str],
        tags: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[MemoryRecord]:
        query_embedding = self.embedder.embed([query])[0]
        lexical = self._fts_scores("memories_fts", "memory_id", query, limit * 5)
        candidates = self.list_memories(user_id=user_id, project_id=project_id, tags=tags)
        now = utc_now()
        scored = []
        with self._connect() as db:
            for memory in candidates:
                row = db.execute(
                    "SELECT embedding_json FROM memories WHERE memory_id=?", (memory.memory_id,)
                ).fetchone()
                vector_score = max(0.0, cosine_similarity(query_embedding, json.loads(row["embedding_json"])))
                lexical_score = lexical.get(memory.memory_id, 0.0)
                age_days = max(0.0, (now - memory.updated_at).total_seconds() / 86400)
                recency = math.exp(-age_days / 180)
                score = (
                    0.45 * lexical_score
                    + 0.3 * vector_score
                    + 0.1 * memory.confidence
                    + 0.1 * memory.importance
                    + 0.05 * recency
                )
                scored.append((score, memory))
            selected = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
            for _, memory in selected:
                db.execute(
                    """UPDATE memories SET access_count=access_count+1,
                    last_accessed_at=? WHERE memory_id=?""",
                    (now.isoformat(), memory.memory_id),
                )
        return [
            memory.model_copy(
                update={
                    "access_count": memory.access_count + 1,
                    "last_accessed_at": now,
                }
            )
            for _, memory in selected
        ]

    def archive_memory(self, memory_id: str) -> None:
        with self._connect() as db:
            result = db.execute(
                "UPDATE memories SET status='archived', updated_at=? WHERE memory_id=?",
                (utc_now().isoformat(), memory_id),
            )
            if result.rowcount == 0:
                raise KeyError(f"Memory not found: {memory_id}")

    def purge_expired_memories(self) -> int:
        now = utc_now().isoformat()
        with self._connect() as db:
            rows = db.execute(
                "SELECT memory_id FROM memories WHERE expires_at IS NOT NULL AND expires_at<=?",
                (now,),
            ).fetchall()
            for row in rows:
                db.execute("DELETE FROM memories_fts WHERE memory_id=?", (row["memory_id"],))
            db.executemany("DELETE FROM memories WHERE memory_id=?", [(row["memory_id"],) for row in rows])
        return len(rows)

    def _fts_scores(self, table: str, id_column: str, query: str, limit: int) -> Dict[str, float]:
        terms = [term.replace('"', "") for term in query.split() if term.strip()]
        if not terms:
            return {}
        match_query = " OR ".join(f'"{term}"' for term in terms[:32])
        try:
            with self._connect() as db:
                rows = db.execute(
                    f"SELECT {id_column}, bm25({table}) AS rank FROM {table} "
                    f"WHERE {table} MATCH ? ORDER BY rank LIMIT ?",
                    (match_query, limit),
                ).fetchall()
        except sqlite3.OperationalError:
            return {}
        if not rows:
            return {}
        raw = {row[id_column]: max(0.0, -float(row["rank"])) for row in rows}
        maximum = max(raw.values()) or 1.0
        return {key: value / maximum for key, value in raw.items()}


def _fingerprint(memory: MemoryRecord) -> str:
    normalized = " ".join(memory.content.lower().split())
    key = f"{memory.memory_type}|{normalized}|{'|'.join(sorted(memory.tags))}"
    return hashlib.sha256(key.encode()).hexdigest()


def _memory_from_row(row: sqlite3.Row) -> MemoryRecord:
    return MemoryRecord(
        memory_id=row["memory_id"],
        user_id=row["user_id"],
        project_id=row["project_id"],
        memory_type=row["memory_type"],
        title=row["title"],
        content=row["content"],
        confidence=float(row["confidence"]),
        source_task_id=row["source_task_id"],
        tags=json.loads(row["tags_json"]),
        metadata=json.loads(row["metadata_json"]),
        scope=row["scope"],
        status=row["status"],
        importance=float(row["importance"]),
        expires_at=_parse_datetime(row["expires_at"]),
        last_accessed_at=_parse_datetime(row["last_accessed_at"]),
        access_count=int(row["access_count"]),
        created_at=_parse_datetime(row["created_at"]) or utc_now(),
        updated_at=_parse_datetime(row["updated_at"]) or utc_now(),
    )


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None
