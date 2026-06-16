# Phase 7 — Tests, Bench, Matrix

**Branch**: `feat/aisaq`
**Status**: `- [ ]` not started
**Effort**: 1-2 weeks
**Depends on**: Phase 6 (AiSAQ label integration)
**Dependents**: Phase 8 (Vulkan)

> **Maintenance note**: Update task markers (`- [ ]` -> `- [~]` -> `- [x]`) as you progress. At phase completion, update the phase status in [AISAQ.md](AISAQ.md) and update `AGENTS.md` per the [AGENTS.md Update Protocol](AISAQ.md#agentsmd-update-protocol).

---

## Goal

Complete the AiSAQ test surface: SQL logic tests, unit tests, recall regression thresholds, and combo_matrix acceptance cells. This phase establishes the quality gate for the AiSAQ algorithm — it must not be marked "complete" until all tests pass and recall thresholds are met.

## Background

vindex has three test suites (per `AGENTS.md`):
- **SQL logic tests** (`make test`): `test/sql/<algo>/*.test` files.
- **Catch2 unit tests** (`make unit`): `test/unit/test_*.cpp` files.
- **Recall regression** (`make bench`): `test/bench/run_recall.py`, fails on Recall@10 regression.

Plus the **acceptance matrix** (`make matrix`): `test/sql/common/combo_matrix.test`, gated behind `VINDEX_RUN_MATRIX=1`.

This phase fills in the AiSAQ entries across all four.

## Scope

### In scope

- All SQL logic tests under `test/sql/aisaq/`.
- All AiSAQ-related additions to `test/unit/`.
- Recall bench entries in `test/bench/run_recall.py` (`CONFIGS` + `THRESHOLDS`).
- `combo_matrix.test` cells for `AISAQ x {pq, scann} x {l2sq, cosine, ip}`.
- Recall validation on siftsmall and sift1m datasets.

### Out of scope

- New datasets (only siftsmall and sift1m, matching existing algos).
- Performance benchmarks (latency/QPS) — recall only in this phase.
- Cross-algorithm comparison reports.

## Files

### New files

| Path | Purpose |
|---|---|
| `test/sql/aisaq/aisaq_basic.test` | CREATE INDEX + ORDER BY + LIMIT, plan shape, rerank |
| `test/sql/aisaq/aisaq_pq.test` | PQ quantizer specific paths |
| `test/sql/aisaq/aisaq_scann.test` | ScaNN quantizer specific paths |
| `test/sql/aisaq/aisaq_pragma.test` | `pragma_vindex_aisaq_index_info`; session overrides |
| `test/sql/aisaq/aisaq_persist.test` | Checkpoint + WAL round-trip (gated on persistence pragma) |
| `test/sql/aisaq/aisaq_filter.test` | Label-column filtering (from Phase 6) |
| `test/sql/aisaq/aisaq_inline.test` | `aisaq_inline_pq` knob |
| `test/sql/aisaq/aisaq_ram_budget.test` | `memory_limit` enforcement (BufferPool eviction) |
| `test/sql/aisaq/aisaq_errors.test` | Rejected quantizers, unknown options, out-of-range params |

### Changed files

| Path | Change |
|---|---|
| `test/bench/run_recall.py` | Add AiSAQ entries to `CONFIGS` and `THRESHOLDS` |
| `test/sql/common/combo_matrix.test` | Add 6 AiSAQ cells |
| `test/unit/test_aisaq_core.cpp` | Complete the test suite (started in Phase 5/6) |

## Implementation Steps

### Step 1: `aisaq_basic.test`

- [ ] `require vindex` header.
- [ ] `SET vindex_enable_experimental_persistence = true`.
- [ ] Create a table with `FLOAT[3]` vectors.
- [ ] `CREATE INDEX my_idx ON t1 USING AISAQ (vec) WITH (quantizer='pq', m=3, bits=8, rerank=10)`.
- [ ] `EXPLAIN SELECT ... ORDER BY array_distance(vec, [...]::FLOAT[3]) LIMIT 3` — verify `VINDEX_INDEX_SCAN` in plan.
- [ ] Query — verify correct results.
- [ ] `statement error` for `quantizer='flat'`.
- [ ] `statement error` for `quantizer='rabitq'`.
- [ ] `statement error` for unknown options.
- [ ] `statement error` for out-of-range `aisaq_r`, `aisaq_alpha`, etc.

### Step 2: `aisaq_pq.test`

- [ ] Build with `quantizer='pq'` at various `m` values (4, 8, 16, 32).
- [ ] Verify search works at each.
- [ ] Verify `bits=4` and `bits=8` both work.
- [ ] Verify the `pragma_vindex_aisaq_index_info` reports the correct quantizer and parameters.

### Step 3: `aisaq_scann.test`

- [ ] Build with `quantizer='scann'`, various `m` and `eta` values.
- [ ] Verify search works.
- [ ] Verify pragma reports correct parameters.

### Step 4: `aisaq_pragma.test`

- [ ] Build an index, call `pragma_vindex_aisaq_index_info`.
- [ ] Verify all metadata columns are populated correctly.
- [ ] `SET vindex_aisaq_beam_width = 16` — verify it affects search.
- [ ] `SET vindex_aisaq_l_search = 200` — verify it affects search.
- [ ] `SET vindex_aisaq_io_limit = 10` — verify it caps I/O.

### Step 5: `aisaq_persist.test`

- [ ] `SET vindex_enable_experimental_persistence = true`.
- [ ] Create a persistent database (`test.duckdb`).
- [ ] Build an AiSAQ index, insert data.
- [ ] Checkpoint, close, reopen.
- [ ] Verify the index still works (search returns correct results).
- [ ] WAL test: insert rows, close without checkpoint, reopen, verify WAL replay restores the index.

### Step 6: `aisaq_filter.test`

- [ ] (Already specified in Phase 6, Step 8 — finalize here if not already done.)
- [ ] Label-column designation, WHERE filter routing, correctness.

### Step 7: `aisaq_inline.test`

- [ ] Build with `aisaq_inline_pq=0` (default) — verify works.
- [ ] Build with `aisaq_inline_pq=8` — verify works.
- [ ] Build with `aisaq_inline_pq=64` (= aisaq_r) — verify works.
- [ ] Build with `aisaq_inline_pq=128` (> aisaq_r) — verify error.
- [ ] Compare recall across inline values — verify inline >= paged recall.

### Step 8: `aisaq_ram_budget.test`

- [ ] Build a 100K-vector index and verify it works under a tight `memory_limit`.
- [ ] `SET memory_limit='10MB'` — verify the index still builds (BufferPool evicts cold blocks to DB file).
- [ ] Verify search still returns correct results (BufferPool pages in blocks on demand).
- [ ] `SET vindex_aisaq_beam_width = 2` — verify reduced working set under memory pressure.
- [ ] Build the same index with default `memory_limit` — verify results are consistent.

### Step 9: `aisaq_errors.test`

- [ ] `quantizer='flat'` -> error.
- [ ] `quantizer='rabitq'` -> error.
- [ ] `aisaq_r=3` (below 4) -> error.
- [ ] `aisaq_r=300` (above 256) -> error.
- [ ] `aisaq_alpha=0.5` (below 1.0) -> error.
- [ ] `aisaq_alpha=3.0` (above 2.0) -> error.
- [ ] `aisaq_inline_pq=-1` -> error.
- [ ] `aisaq_inline_pq > aisaq_r` -> error.
- [ ] `aisaq_entry_points=0` -> error.
- [ ] `aisaq_entry_points=100` (above 64) -> error.
- [ ] `notreal=1` -> "Unknown option for AiSAQ index".

### Step 10: Complete `test/unit/test_aisaq_core.cpp`

- [ ] (Started in Phase 5/6 — ensure all test cases are implemented.)
- [ ] `BuildAndSearch` — basic correctness.
- [ ] `IoLimitCap` — I/O budget enforcement.
- [ ] `InlineVsPaged` — recall comparison.
- [ ] `RamBudgetEnforcement` — search under tight `memory_limit` (BufferPool evicts cold blocks).
- [ ] `PersistenceRoundTrip` — serialize/deserialize.
- [ ] `LabelMedoidSelection` — per-label medoids.
- [ ] `LabelFilterRecall` — recall under heavy filter.
- [ ] `EntryPointEffectiveness` — search with 16 entry points vs 1 entry point, verify recall improvement on clustered data.

### Step 11: Add recall bench entries

- [ ] In `test/bench/run_recall.py::CONFIGS`, add:
  ```python
  "aisaq-pq":         ("AISAQ", "WITH (quantizer='pq', m=64, bits=8, rerank=10, aisaq_r=64, aisaq_l=100)"),
  "aisaq-scann":      ("AISAQ", "WITH (quantizer='scann', m=64, bits=8, eta=4.0, rerank=10, aisaq_r=64, aisaq_l=100)"),
  "aisaq-pq-inline64": ("AISAQ", "WITH (quantizer='pq', m=64, bits=8, rerank=10, aisaq_r=64, aisaq_l=100, aisaq_inline_pq=64)"),
  ```
- [ ] In `THRESHOLDS`, add:
  ```python
  ("aisaq-pq",          "siftsmall"): 0.90,
  ("aisaq-scann",       "siftsmall"): 0.90,
  ("aisaq-pq-inline64", "siftsmall"): 0.90,
  ("aisaq-pq",          "sift1m"):    0.95,
  ("aisaq-scann",       "sift1m"):    0.95,
  ("aisaq-pq-inline64", "sift1m"):    0.95,
  ```
- [ ] Run `make bench` and verify all thresholds are met.
- [ ] If any threshold is not met: investigate, tune parameters, or adjust the threshold (with justification).

### Step 12: Add combo_matrix cells

- [ ] In `test/sql/common/combo_matrix.test`, add 6 cells:
  - `AISAQ x PQ x L2SQ`
  - `AISAQ x PQ x COSINE`
  - `AISAQ x PQ x IP`
  - `AISAQ x ScaNN x L2SQ`
  - `AISAQ x ScaNN x COSINE`
  - `AISAQ x ScaNN x IP`
- [ ] Each cell: build a small index, run a query, verify result.
- [ ] Run `make matrix` to verify all 6 cells pass.

### Step 13: Full regression run

- [ ] `make test` — all SQL tests pass.
- [ ] `make unit` — all unit tests pass.
- [ ] `make bench` — all recall thresholds met.
- [ ] `make matrix` — all combo_matrix cells pass.
- [ ] `scripts/format.sh --check` — formatting is clean.

### Step 14: Update `AGENTS.md`

- [ ] Add AiSAQ test commands to the "Commands" table.
- [ ] Note the recall thresholds for AiSAQ in the bench section.
- [ ] Add the combo_matrix cell count update.

## Acceptance Criteria

- [ ] All 9 SQL test files exist and pass (`make test`).
- [ ] All unit tests pass (`make unit`).
- [ ] Recall thresholds met on siftsmall (0.90) and sift1m (0.95) for all 3 AiSAQ configs (`make bench`).
- [ ] All 6 combo_matrix cells pass (`make matrix`).
- [ ] `scripts/format.sh --check` passes.
- [ ] No existing test regresses.
- [ ] `AGENTS.md` is updated.
- [ ] Phase status updated to `- [x]` in [AISAQ.md](AISAQ.md).

## Notes

- If recall on sift1m does not meet 0.95 for `aisaq-pq`, investigate before adjusting the threshold. Common causes: `aisaq_l` too low, `aisaq_r` too low, entry-point count too low, or a bug in the paged-PQ distance path.
- `aisaq-pq-inline64` should match or exceed `aisaq-pq` recall (inline codes are a superset of paged codes). If it doesn't, there's a bug in `FinalizeInlineCodes`.
- The combo_matrix cells are the backend-drift gate. If any cell fails, it indicates a quantizer/metric interaction bug.
