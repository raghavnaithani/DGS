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
        watchpoints_json TEXT NOT NULL DEFAULT '[]',
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
    CREATE TABLE IF NOT EXISTS graph_shares (
        public_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_graph_shares_session_id ON graph_shares(session_id)",
    # ---------------------------------------------------------------------------
    # v0.2 tables
    # ---------------------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS user_profiles (
        id                      TEXT PRIMARY KEY,
        email                   TEXT NOT NULL UNIQUE,
        display_name            TEXT,
        expertise_level         TEXT NOT NULL DEFAULT 'intermediate'
                                    CHECK (expertise_level IN ('beginner', 'intermediate', 'expert')),
        risk_tolerance          INTEGER NOT NULL DEFAULT 5
                                    CHECK (risk_tolerance BETWEEN 1 AND 10),
        values_json             TEXT NOT NULL DEFAULT '[]',
        life_situation          TEXT NOT NULL DEFAULT '',
        decision_patterns_json  TEXT NOT NULL DEFAULT '{}',
        onboarding_complete     INTEGER NOT NULL DEFAULT 0
                                    CHECK (onboarding_complete IN (0, 1)),
        subscription_tier       TEXT NOT NULL DEFAULT 'free'
                                    CHECK (subscription_tier IN ('free', 'pro')),
        stripe_customer_id      TEXT,
        graphs_this_month       INTEGER NOT NULL DEFAULT 0,
        month_reset_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        created_at              TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at              TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_user_profiles_email ON user_profiles(email)",
    """
    CREATE TABLE IF NOT EXISTS export_cache (
        id          TEXT PRIMARY KEY,
        session_id  TEXT NOT NULL,
        format      TEXT NOT NULL CHECK (format IN ('pdf', 'png')),
        file_path   TEXT NOT NULL,
        created_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        expires_at  TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_export_cache_session_id ON export_cache(session_id)",
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
        actionability_score REAL DEFAULT 0.0 CHECK (actionability_score BETWEEN 0.0 AND 1.0),
        FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_chunks_session_id ON chunks(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_source_url ON chunks(source_url)",
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        id UNINDEXED,
        content,
        source_url UNINDEXED,
        source_title,
        tokenize='porter unicode61'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(id, content, source_url, source_title)
        VALUES (new.id, new.content, new.source_url, COALESCE(new.source_title, ''));
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        DELETE FROM chunks_fts WHERE id = old.id;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        DELETE FROM chunks_fts WHERE id = old.id;
        INSERT INTO chunks_fts(id, content, source_url, source_title)
        VALUES (new.id, new.content, new.source_url, COALESCE(new.source_title, ''));
    END
    """,
    """
    INSERT INTO chunks_fts(id, content, source_url, source_title)
    SELECT c.id, c.content, c.source_url, COALESCE(c.source_title, '')
    FROM chunks c
    WHERE NOT EXISTS (
        SELECT 1
        FROM chunks_fts f
        WHERE f.id = c.id
    )
    """,
)


def apply_schema(connection: sqlite3.Connection) -> None:
    for statement in SCHEMA_STATEMENTS:
        connection.execute(statement)


MIGRATION_STATEMENTS = (
    # Existing v0.1 migration
    "ALTER TABLE nodes ADD COLUMN watchpoints_json TEXT NOT NULL DEFAULT '[]'",
    # ---------------------------------------------------------------------------
    # v0.2 migrations — safe to run multiple times (errors silently ignored)
    # ---------------------------------------------------------------------------
    "ALTER TABLE sessions ADD COLUMN user_id TEXT",
    "ALTER TABLE sessions ADD COLUMN domain TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE sessions ADD COLUMN horizon_months INTEGER NOT NULL DEFAULT 3",
    "ALTER TABLE sessions ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
    "ALTER TABLE sessions ADD COLUMN node_count INTEGER NOT NULL DEFAULT 0",
)


def apply_migrations(connection: sqlite3.Connection) -> None:
    """Run additive migrations that are safe to skip if already applied."""
    for statement in MIGRATION_STATEMENTS:
        try:
            connection.execute(statement)
            connection.commit()
        except Exception:
            # Column already exists – ignore
            pass
