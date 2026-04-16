-- dosha_schema.sql
-- Dosha Assessment Database — LLM Constitutional Study
-- CLI: sqlite3 dosha_assessment.db < dosha_schema.sql

PRAGMA foreign_keys = ON;

-- ── Models ────────────────────────────────────────────────────────────────────
-- Insert model rows manually before running assessments
CREATE TABLE Models (
    model_id      TEXT PRIMARY KEY,   -- e.g. "claude-sonnet-4-5"
    display_name  TEXT NOT NULL,       -- e.g. "Claude Sonnet 4.5"
    provider      TEXT NOT NULL,       -- "anthropic" | "openai" | "google" | "ollama" | "mistral"
    api_type      TEXT NOT NULL,       -- "anthropic" | "openai_compat" | "ollama" | "google"
    default_temp  REAL,
    notes         TEXT
);

-- ── Probes ────────────────────────────────────────────────────────────────────
-- Populated automatically from probes.json by assess_dosha.py on first run
CREATE TABLE Probes (
    probe_id      TEXT PRIMARY KEY,    -- e.g. "V1", "P3a", "K10"
    category      TEXT NOT NULL CHECK(category IN ('vata','pitta','kapha')),
    name          TEXT NOT NULL,
    probe_type    TEXT NOT NULL        -- "single"|"multi"|"repeat"|"battery"|"sustained"
);

-- ── Responses ─────────────────────────────────────────────────────────────────
CREATE TABLE Responses (
    response_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id      TEXT NOT NULL REFERENCES Models(model_id),
    probe_id      TEXT NOT NULL REFERENCES Probes(probe_id),
    run_index     INTEGER NOT NULL CHECK(run_index BETWEEN 1 AND 3),
    response_text TEXT NOT NULL,
    full_context  TEXT,                -- JSON: full conversation history for multi-turn
    temperature   REAL,               -- actual temperature used
    created_at    TEXT DEFAULT (datetime('now')),
    UNIQUE(model_id, probe_id, run_index)
);

CREATE INDEX idx_responses_model  ON Responses(model_id);
CREATE INDEX idx_responses_probe  ON Responses(probe_id);

-- ── Scores ────────────────────────────────────────────────────────────────────
CREATE TABLE Scores (
    score_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    response_id   INTEGER NOT NULL UNIQUE REFERENCES Responses(response_id),
    g_T           REAL NOT NULL CHECK(g_T BETWEEN 0 AND 4),
    g_R           REAL NOT NULL CHECK(g_R BETWEEN 0 AND 4),
    g_S           REAL NOT NULL CHECK(g_S BETWEEN 0 AND 4),
    reasoning     TEXT,
    judge_model   TEXT NOT NULL,
    scored_at     TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_scores_response ON Scores(response_id);

-- ── GVectors ──────────────────────────────────────────────────────────────────
-- Computed aggregate G vectors per model. Updated after each full assessment run.
CREATE TABLE GVectors (
    model_id      TEXT PRIMARY KEY REFERENCES Models(model_id),
    g_T_mean      REAL,              -- mean raw tamas score across all probes × runs
    g_R_mean      REAL,
    g_S_mean      REAL,
    g_T_norm      REAL,              -- normalized to unit sphere
    g_R_norm      REAL,
    g_S_norm      REAL,
    g_T_var       REAL,              -- variance across probes (spread measure)
    g_R_var       REAL,
    g_S_var       REAL,
    n_probes      INTEGER,           -- number of probes included
    n_runs        INTEGER,           -- runs per probe averaged
    computed_at   TEXT DEFAULT (datetime('now'))
);

-- ── V_AssessmentDetail ────────────────────────────────────────────────────────
-- Convenience view for analysis queries
CREATE VIEW V_AssessmentDetail AS
SELECT
    r.response_id,
    r.model_id,
    m.display_name,
    m.provider,
    m.default_temp,
    r.probe_id,
    p.category,
    p.name       AS probe_name,
    p.probe_type,
    r.run_index,
    r.response_text,
    r.temperature,
    s.g_T,
    s.g_R,
    s.g_S,
    s.reasoning,
    s.judge_model,
    r.created_at
FROM Responses r
JOIN Models  m ON m.model_id = r.model_id
JOIN Probes  p ON p.probe_id = r.probe_id
LEFT JOIN Scores s ON s.response_id = r.response_id;
