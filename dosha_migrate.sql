-- dosha_migrate.sql
-- Run this on an existing dosha_assessment.db to add dosha scoring columns
-- CLI: sqlite3 dosha_assessment.db < dosha_migrate.sql

-- Add dosha columns to Scores (safe to run — ALTER TABLE ADD COLUMN is idempotent if column exists in SQLite 3.37+)
ALTER TABLE Scores ADD COLUMN d_V REAL;
ALTER TABLE Scores ADD COLUMN d_P REAL;
ALTER TABLE Scores ADD COLUMN d_K REAL;
ALTER TABLE Scores ADD COLUMN dosha_reasoning TEXT;

-- Create DoshaVectors table
CREATE TABLE IF NOT EXISTS DoshaVectors (
    model_id        TEXT PRIMARY KEY REFERENCES Models(model_id),
    vata_mean       REAL,
    pitta_mean      REAL,
    kapha_mean      REAL,
    vata_norm       REAL,
    pitta_norm      REAL,
    kapha_norm      REAL,
    vata_var        REAL,
    pitta_var       REAL,
    kapha_var       REAL,
    n_v_probes      INTEGER,
    n_p_probes      INTEGER,
    n_k_probes      INTEGER,
    computed_at     TEXT DEFAULT (datetime('now'))
);

SELECT 'Migration complete. Columns added: d_V, d_P, d_K, dosha_reasoning. Table created: DoshaVectors.';

-- ── Composite score columns (added April 2026) ─────────────────────────────
-- composite = 0.5 * mean + 0.5 * SD (within probe category)
ALTER TABLE DoshaVectors ADD COLUMN vata_sd         REAL;
ALTER TABLE DoshaVectors ADD COLUMN pitta_sd        REAL;
ALTER TABLE DoshaVectors ADD COLUMN kapha_sd        REAL;
ALTER TABLE DoshaVectors ADD COLUMN vata_composite  REAL;
ALTER TABLE DoshaVectors ADD COLUMN pitta_composite REAL;
ALTER TABLE DoshaVectors ADD COLUMN kapha_composite REAL;
ALTER TABLE DoshaVectors ADD COLUMN vata_comp_norm  REAL;
ALTER TABLE DoshaVectors ADD COLUMN pitta_comp_norm REAL;
ALTER TABLE DoshaVectors ADD COLUMN kapha_comp_norm REAL;

SELECT 'Migration 2 complete. Composite columns added to DoshaVectors.';
