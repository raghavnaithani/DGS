from __future__ import annotations

import sqlite3


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        intent_id TEXT,
        title TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (intent_id) REFERENCES user_intents(id) ON DELETE SET NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at)",
    """
    CREATE TABLE IF NOT EXISTS user_intents (
        id TEXT PRIMARY KEY,
        original_prompt TEXT NOT NULL,
        domain TEXT NOT NULL,
        horizon_months INTEGER NOT NULL CHECK (horizon_months > 0),
        risk_tolerance INTEGER NOT NULL CHECK (risk_tolerance BETWEEN 0 AND 100),
        constraints_json TEXT NOT NULL DEFAULT '[]',
        personal_context TEXT NOT NULL,
        clarified_entities_json TEXT NOT NULL DEFAULT '[]',
        ambiguities_remaining_json TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_user_intents_domain ON user_intents(domain)",
    "CREATE INDEX IF NOT EXISTS idx_user_intents_created_at ON user_intents(created_at)",
    """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        job_type TEXT NOT NULL,
        request_json TEXT NOT NULL,
        status TEXT NOT NULL,
        progress INTEGER NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
        current_step TEXT NOT NULL DEFAULT 'queued',
        total_sources INTEGER NOT NULL DEFAULT 0 CHECK (total_sources >= 0),
        scraped_sources INTEGER NOT NULL DEFAULT 0 CHECK (scraped_sources >= 0),
        stored_chunks INTEGER NOT NULL DEFAULT 0 CHECK (stored_chunks >= 0),
        result_json TEXT,
        error_message TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)",
    """
    CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        description TEXT NOT NULL,
        time_step INTEGER NOT NULL CHECK (time_step >= 0),
        created_by_engine TEXT NOT NULL,
        alternatives_json TEXT NOT NULL DEFAULT '[]',
        risks_json TEXT NOT NULL DEFAULT '[]',
        source_citations_json TEXT NOT NULL DEFAULT '[]',
        confidence_score REAL NOT NULL CHECK (confidence_score BETWEEN 0.0 AND 1.0),
        speculative INTEGER NOT NULL CHECK (speculative IN (0, 1)),
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_nodes_session_id ON nodes(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_time_step ON nodes(time_step)",
    "CREATE INDEX IF NOT EXISTS idx_nodes_created_by_engine ON nodes(created_by_engine)",
    """
    CREATE TABLE IF NOT EXISTS edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        source_node_id TEXT NOT NULL,
        target_node_id TEXT NOT NULL,
        action_description TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
        FOREIGN KEY (source_node_id) REFERENCES nodes(id) ON DELETE CASCADE,
        FOREIGN KEY (target_node_id) REFERENCES nodes(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edges_session_id ON edges(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_source_node_id ON edges(source_node_id)",
    "CREATE INDEX IF NOT EXISTS idx_edges_target_node_id ON edges(target_node_id)",
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        content TEXT NOT NULL,
        source_url TEXT NOT NULL,
        source_title TEXT,
        chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
        embedding_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        ttl_days INTEGER NOT NULL DEFAULT 30 CHECK (ttl_days > 0),
        verification_status TEXT NOT NULL CHECK (verification_status IN ('verified', 'unverified', 'failed')),
        similarity_score REAL CHECK (similarity_score BETWEEN 0.0 AND 1.0),
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_session_id ON chunks(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_source_url ON chunks(source_url)",
)


def apply_schema(connection: sqlite3.Connection) -> None:
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)
