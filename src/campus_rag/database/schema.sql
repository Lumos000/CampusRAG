PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    department TEXT NOT NULL DEFAULT '',
    published_at TEXT,
    crawled_at TEXT,
    local_path TEXT NOT NULL DEFAULT '',
    content_type TEXT NOT NULL DEFAULT '',
    sha256 TEXT NOT NULL DEFAULT '',
    version TEXT NOT NULL DEFAULT '',
    UNIQUE(source_url, sha256)
);

CREATE TABLE IF NOT EXISTS tables (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    locator TEXT NOT NULL DEFAULT '',
    header_levels INTEGER NOT NULL DEFAULT 1,
    unit TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    search_text TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS table_columns (
    id INTEGER PRIMARY KEY,
    table_id INTEGER NOT NULL REFERENCES tables(id) ON DELETE CASCADE,
    column_index INTEGER NOT NULL,
    original_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    data_type TEXT NOT NULL DEFAULT 'text',
    unit TEXT NOT NULL DEFAULT '',
    UNIQUE(table_id, column_index)
);

CREATE TABLE IF NOT EXISTS table_rows (
    id INTEGER PRIMARY KEY,
    table_id INTEGER NOT NULL REFERENCES tables(id) ON DELETE CASCADE,
    row_index INTEGER NOT NULL,
    row_label TEXT NOT NULL DEFAULT '',
    UNIQUE(table_id, row_index)
);

CREATE TABLE IF NOT EXISTS table_cells (
    id INTEGER PRIMARY KEY,
    row_id INTEGER NOT NULL REFERENCES table_rows(id) ON DELETE CASCADE,
    column_id INTEGER NOT NULL REFERENCES table_columns(id) ON DELETE CASCADE,
    raw_value TEXT NOT NULL DEFAULT '',
    normalized_value TEXT NOT NULL DEFAULT '',
    numeric_value TEXT,
    unit TEXT NOT NULL DEFAULT '',
    locator TEXT NOT NULL DEFAULT '',
    UNIQUE(row_id, column_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    table_id INTEGER REFERENCES tables(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK(kind IN ('text', 'table')),
    heading_path TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    content_hash TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_cells_row ON table_cells(row_id);
CREATE INDEX IF NOT EXISTS idx_columns_table ON table_columns(table_id);

CREATE TABLE IF NOT EXISTS audit_issues (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    severity TEXT NOT NULL,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_sources (
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_url TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY(doc_id, source_url)
);

CREATE TABLE IF NOT EXISTS ingestion_state (
    doc_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    content_key TEXT NOT NULL UNIQUE,
    completed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

