# Changelog

All notable changes to `vindex` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Phase 9: parallel AiSAQ CREATE INDEX

- **Parallel PQ encoding** (Pass 1) — `EncodePqCodes` now runs across
  `TaskScheduler::NumberOfThreads()` tasks via DuckDB's `TaskExecutor` push
  API. Page-aligned row-range partitioning keeps `WritePqPage` and flat-
  buffer writes disjoint across tasks. ~4.6× speedup on sift1m.
- **Parallel graph construction** (Pass 2) — Vamana `Construct` runs across
  N tasks with partitioned row ranges. Per-node spinlocks
  (`NodeSpinlock`) on `AiSaqCore` handle `ConnectAndPrune`'s reciprocal
  edges without a global rwlock, mirroring the hnswlib/DiskANN production
  pattern. Peak CPU 957% on a 10-core host (~10/10 cores engaged). ~7×
  speedup on sift1m Pass 2 vs serial.
- **Parallel post-build phases** — `FinalizeInlineCodes`,
  `ComputeLabelMedoids`, and `WriteAllGraphNodes` now run across N tasks
  with disjoint partitions. `ComputeEntryPoints` stays serial (bounded by
  `n_entry_points` ≤ 64). Also fixes a latent inefficiency: `ReadPqCode`
  now has the flat `build_codes_` fast path (mirrors `DistanceToCode` /
  `CopyBuildCode`), skipping `BufferManager::Pin` when the flat buffer is
  active.
- **Parallel quantizer training** — `PqQuantizer::Train` runs m=16
  independent k-means++ passes in parallel via `std::thread` work-queue
  with `hardware_concurrency()` cap. New `SetTrainThreads(n)` hint on the
  `Quantizer` base class; `AiSaqIndex::TrainQuantizer` sets it from
  `TaskScheduler::NumberOfThreads()`. ~4.9× speedup on sift1m (5.9s → 1.2s).
- **Precomputed cross-distance table** on `PqQuantizer` — the symmetric
  K×K centroid-pair distance table is built once at end of `Train()` /
  `Deserialize()`. `CodeDistance` becomes O(m) array lookups instead of
  O(m·sub_dim) simsimd FLOPs. For SIFT (m=16, K=256): 4MB table, ~5-10×
  faster per call. 22% Pass 2 wall-clock improvement on sift1m.
- **`VamanaTLS` per-thread scratch** — `visit_marks`, `visit_counter`,
  `prune_scratch`, and `rng` extracted from `AiSaqCore` and `DiskAnnCore`
  into a shared `src/include/vindex/vamana.hpp` struct. Pays down the
  `TODO(M6)` at `aisaq_core.hpp:29-31`. Threaded as a parameter through
  `BeamSearch` / `RobustPrune` / `ConnectAndPrune` build-path overloads.
- **Thread-safe `AiSaqBlockStore`** — `graph_node_count_` and
  `flat_build_mode_` are now `std::atomic`. `EnsureGraphCapacity(N)` is
  pre-called once in `Finalize` before parallel tasks spawn.
- **Phase timing instrumentation** — `train_quantizer`, `PQ encoding`,
  and `pass2_construct` wall-clock times logged at `vindex_log_level=info`.
- **New test** — `test/unit/test_vamana_concurrency.cpp` (3 cases: recall
  match vs serial, graph invariants, multi-thread stress). Existing tests
  verify byte-identical PQ codes, post-build state, and recall vs serial
  baselines.

### Changed

- `AiSaqCore::Insert` path split: serial `Insert` (live inserts) delegates
  to `InsertBuild` with the core's own `tls_`; parallel `InsertBuildRange`
  passes per-task `VamanaTLS` + per-task label maps that merge serially in
  `FinalizeParallelConstruct`.
- `AiSaqCore::ConnectAndPrune` writes forward edges immediately and acquires
  per-target `NodeSpinlock` for reciprocal edges. No global rwlock during
  parallel construction; rwlock only in the serial `FinalizeParallelConstruct`
  merge step.
- `AiSaqCore::BeamSearch` / `RobustPrune` / `ConnectAndPrune` build-path
  overloads take `VamanaTLS&` parameter; search-path stays single-threaded
  via the core's `tls_` member.

### Performance

End-to-end `CREATE INDEX ... USING AISAQ` on sift1m (1M × 128-d,
`pq_buffer` strategy, 10-core host):

| Phase | Before | After | Speedup |
|---|---|---|---|
| Quantizer training | 5.9s | 1.2s | 4.9× |
| PQ encode | 3.3s | 0.7s | 4.6× |
| Graph construct | ~140s | ~20s | 7× |
| Post-build | 2.8s | <0.1s | 28×+ |
| **Total** | **~152s** | **~22s** | **6.9×** |

Recall unchanged across all configurations: `aiasaq-pq` 0.9990,
`aiasaq-scann` 0.9960, `aisaq-pq-inline64` 0.9990 (siftsmall thresholds).

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
