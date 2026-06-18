# Changelog

All notable changes to `vindex` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-18

Second tagged release. Built against DuckDB v1.5.3. Adds the AiSAQ
index (All-in-Storage ANNS with Product Quantization) for datasets
larger than RAM, the ScaNN quantizer, and the WHERE-label predicate
pushdown framework. Fixes a UBSan abort under HNSW + RaBitQ at common
configurations.

### Added

- **AiSAQ index** (`USING AISAQ`) — Vamana graph with PQ codes paged
  from storage on demand via DuckDB's `BlockManager` (AiSaqBlockStore).
  Keeps near-zero DRAM footprint for the PQ code structure; cold blocks
  evict via the BufferPool under `memory_limit` pressure. WITH options:
  `metric`, `quantizer`, `bits`, `rerank`, `aisaq_r`, `aisaq_l`,
  `aisaq_alpha`, `aisaq_inline_pq`, `aisaq_entry_points`, plus the
  `label_column` predicate-pushdown option. Accepts `quantizer='pq'`
  or `quantizer='scann'`; rejects `flat` and `rabitq`.
- **ScaNN quantizer** (`quantizer='scann'`) — anisotropic product
  quantization with `eta` parameter; same LUT-based distance path as
  PQ. `bits ∈ {4, 8}`, `m` defaults to `dim/4`.
- **Label-column filtering framework** — `WITH (label_column='col')`
  designates a BIGINT column for predicate pushdown. The optimizer in
  `optimize_scan.cpp` extracts `=`, `>`, `>=`, `<`, `<=`, `BETWEEN`,
  and folded conjuncts into a `LabelFilter` passed to
  `InitializeScan`. AiSAQ is the first consumer; IVF/SPANN/HNSW/DiskANN
  track as future milestones (M9-M12).
- **LUT-based distance** on `Quantizer` — `PopulateDistanceLUT` +
  `LUTDistance` virtuals enable batched distance evaluation for any
  LUT-capable quantizer (PQ, ScaNN). Default implementations are no-ops
  so flat/RaBitQ still work transparently.
- **AiSAQ session pragmas** — `vindex_aisaq_build_strategy`
  (`pq_buffer`/`exact_prune`/`paged`), `vindex_aisaq_beam_width`,
  `vindex_aisaq_io_limit`, `vindex_aisaq_l_build`.
- **Info pragma** — `pragma_vindex_aisaq_index_info()`.
- **AiSAQ recall / bench entries** — `run_recall.py --algo aisaq-{pq,scann}`;
  the recall harness covers all 60 (algo × quantizer × metric) cells
  via `make matrix`.

### Fixed

- **HNSW + DiskANN node alignment** — replaced every
  `reinterpret_cast<T*>(node+offset)` in the HNSW and DiskANN node
  accessors with `duckdb::Load<T>` / `duckdb::Store<T>`. Fixes UBSan
  `misaligned reference` aborts (and crashes on strict-alignment
  AArch64 cores) under HNSW + RaBitQ at any `dim × bits ∈ {3,4,8}`
  where `dim·bits ≡ 0 mod 8`. The previous fix padded node sizes to
  the next 8-byte boundary (4 wasted bytes per RaBitQ node, ~6.7% of
  the code section at dim=128/bits=3); the Load/Store approach has
  zero memory overhead. Disassembled `HnswCore::{SearchLayer, Insert,
  ConnectAndPrune}` before/after: `SearchLayer` shrinks by 12 B,
  `Insert` is byte-identical, `ConnectAndPrune` grows by 152 B (~38
  instructions, ≤0.5% of build time). The search hot path is unchanged
  to marginally faster.

### Changed

- HNSW/DiskANN internal accessor API: reference-returning accessors
  (`NodeRowId(n) = x`, `arr[i] = y`) replaced with value-returning
  Get/Set pairs (`SetNodeRowId(n, x)`, `SetNeighborAt(n, layer, i, y)`),
  matching the pattern AiSAQ already uses. `NodeCode` unchanged
  (already returned `data_ptr_t`).

### Removed

- `src/backend/` abstraction layer (Backend base class, CpuBackend,
  `GetActiveBackend` selector) and the `vindex_gpu_backend` session
  option. Added as Phase 4 scaffolding for a future Vulkan backend,
  removed before v0.2.0 release as dead code — no algorithm routed
  through it. Phase 8 (Vulkan) can resurrect the layer from git
  history when it actually wires to a consumer.

## [0.1.0] - 2026-04-28

First tagged release. Ships as an unsigned DuckDB extension built against
DuckDB v1.5.2 (`LOAD 'path/to/vindex.duckdb_extension'` with
`SET allow_unsigned_extensions = true;`).

### Added

- **HNSW index** (`USING HNSW`) — in-house graph implementation (`HnswCore`)
  over `IndexBlockStore`; replaces the `usearch` wrapper carried by
  upstream `duckdb-vss`. WITH options: `metric`, `quantizer`, `bits`,
  `rerank`, `M`, `M0`, `ef_construction`, `ef_search`.
- **IVF index** (`USING IVF`) — k-means++ centroids plus per-list posting
  buffers; supports IVF-Flat, IVF-RaBitQ, IVF-PQ. WITH options: `metric`,
  `quantizer`, `bits`, `rerank`, `nlist`, `nprobe`.
- **DiskANN index** (`USING DISKANN`) — Vamana graph with codes held
  out-of-band so graph blocks can evict past RAM via the DuckDB buffer
  pool. WITH options: `metric`, `quantizer`, `bits`, `rerank`,
  `diskann_r`, `diskann_l`, `diskann_alpha`. Accepts `quantizer='pq'` or
  `quantizer='rabitq'`; rejects `flat`.
- **RaBitQ quantizer** (`quantizer='rabitq'`) — rotated + bit-packed
  codes at `bits ∈ {1,2,3,4,5,7,8}`; 3-bit is the default and hits >99%
  Recall@10 on SIFT1M with `rerank=10`.
- **PQ quantizer** (`quantizer='pq'`) — classical product quantization
  with k-means++ per-segment codebooks; `bits ∈ {4, 8}`, `m` defaults to
  `dim/4`.
- **Flat quantizer** (`quantizer='flat'`) — float32 passthrough,
  bit-for-bit exact distances.
- **Rerank pass** — `WITH (rerank = N)` (or session pragma
  `SET vindex_rerank_multiple = N`) has the planner pull `k × N`
  candidates from the index and re-score against the authoritative
  `FLOAT[d]` column. Uniform `TOP_N ← PROJECTION ← VINDEX_INDEX_SCAN`
  plan shape across algorithms.
- **Session pragmas** — `vindex_ef_search`, `vindex_nprobe`,
  `vindex_diskann_l_search`, `vindex_rerank_multiple`,
  `vindex_enable_experimental_persistence`.
- **Info pragmas** — `pragma_vindex_hnsw_index_info()`,
  `pragma_vindex_ivf_index_info()`,
  `pragma_vindex_diskann_index_info()`.
- **Compact pragma** — `CALL vindex_compact_index('<idx>')` reclaims
  tombstoned entries (IVF rebuilds posting lists in place; HNSW/DiskANN
  currently clear the tombstone set and mark the index dirty).
- **Persistence** — indexes round-trip through checkpoint and WAL; state
  stream carries quantizer blob + core state + row mapping + tombstones.
- **Recall harness** — `make bench` downloads siftsmall on first run and
  fails non-zero on Recall@10 regressions. `run_recall.py --dataset
  sift1m` is wired but gated.
- **GitHub release pipeline** —
  `.github/workflows/MainDistributionPipeline.yml` builds the multi-arch
  matrix on tag push (`v*`) and publishes a GitHub Release with
  per-arch `vindex.<arch>.duckdb_extension` assets.

### Deprecated

The following names are retained as aliases of their `vindex_*`
replacements for at least one release; they will be removed in a
future version:

- `hnsw_enable_experimental_persistence` → `vindex_enable_experimental_persistence`
- `hnsw_ef_search` → `vindex_ef_search`
- `hnsw_compact_index(...)` → `vindex_compact_index(...)`
- `pragma_hnsw_index_info()` → `pragma_vindex_hnsw_index_info()`

[Unreleased]: https://github.com/Icemap/duckdb-vector-index/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Icemap/duckdb-vector-index/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Icemap/duckdb-vector-index/releases/tag/v0.1.0
