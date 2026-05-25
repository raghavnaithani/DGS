# Decision Graph Simulator

Monorepo for the Decision Graph Simulator backend, frontend, and future desktop app.

## Phase 3 (Stateless Oracle) - Status

- Implemented ingestion pipeline (search → filter → crawl → chunk → embed → LanceDB).
- SQLite job tracking and ingestion endpoints added under `/v1/jobs` and `/v1/knowledge/ingest`.
- Default embedder switched to `ibm-granite/granite-embedding-30m-english` to reduce CPU memory usage.
- Concurrency, batch size, and search fanout controls exposed via `backend/app/config.py` as `INGESTION_MAX_WORKERS`, `SCRAPE_MAX_WORKERS`, `EMBEDDING_MAX_WORKERS`, `EMBEDDING_BATCH_SIZE`, and `SEARCH_MAX_RESULTS`.
- Current shipped defaults are tuned for stronger-machine throughput: `INGESTION_MAX_WORKERS=6`, `SCRAPE_MAX_WORKERS=6`, `EMBEDDING_MAX_WORKERS=2`, `EMBEDDING_BATCH_SIZE=16`, `SEARCH_MAX_RESULTS=30`.
- Tests updated and run locally; see `backend/tests` for ingestion-focused tests.

For details and validation artifacts see `completion status/phase3_v0.1_completion.md`.
