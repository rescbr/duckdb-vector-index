# AISAQ Implementation Plan — Main Document

**Status**: PLANNING COMPLETE — ready for execution (revised)
**Last updated**: 2026-06-16
**Target workload**: 100M vectors x 768 dimensions on 32 GB RAM machines
**Estimated total effort**: 18-28 weeks (single engineer)

---

## Maintenance Protocol

> **READ THIS BEFORE TOUCHING ANY FILE IN `docs/plans/`.**

### Task Markers

Every phase document uses checkbox task markers:

- `- [ ]` — not started
- `- [~]` — in progress
- `- [x]` — completed
- `- [!]` — blocked (add a note explaining the blocker)

**When you start a task**: change `- [ ]` to `- [~]`.
**When you finish a task**: change `- [~]` to `- [x]`.
**When a task is blocked**: change to `- [!]` and add a note below the task.

### Update Cadence

1. **After every task completion**: update the task marker in the phase document.
2. **After every phase completion**: update the phase status in the [Phase Overview](#phase-overview) table below AND update `AGENTS.md` (see [AGENTS.md Update Protocol](#agentsmd-update-protocol)).
3. **When scope changes**: update this main document's [Locked Decisions](#locked-decisions) section and all affected phase documents. Do not silently change scope in a phase document without updating the main plan.

### AGENTS.md Update Protocol

`AGENTS.md` is the canonical project-understanding document. It must be updated at the **end of every phase** to keep the project understanding whole. Specifically:

| Phase | AGENTS.md sections to update |
|---|---|
| ~~1~~ | ~~ELIMINATED — no update needed~~ |
| 2 (feat/label-filter) | "Source wiring" diagram (new `SupportsLabelFilter` hook on `VectorIndex`); new "Label-column filtering" subsection |
| 3 (LUT virtuals) | "Source wiring" diagram (LUT virtuals on `Quantizer`); no new subsection needed |
| 4 (Backend interface) | "Source wiring" diagram (new `src/backend/` directory); new "Backend abstraction" subsection |
| 5 (AiSAQ core) | "Source wiring" diagram (new `src/algo/aisaq/` + `AiSaqBlockStore`); "Commands" table (new SQL/unit/bench entries); new "AiSAQ algorithm" subsection; "Planner gotcha" section (literal query vectors still applies) |
| 6 (AiSAQ labels) | Update "AiSAQ algorithm" subsection with label-column behavior; update "Planner gotcha" section |
| 7 (Tests + bench) | "Commands" table (new test commands); "Style" section if new conventions emerge |
| 8 (AiSAQ Vulkan) | "Source wiring" diagram (`src/backend/vulkan_backend.cpp`); new "Vulkan backend" subsection; "CI" section (new build flag `VINDEX_BACKEND_VULKAN`) |

The update at phase end is **mandatory and non-negotiable**. A phase is not "done" until `AGENTS.md` reflects the new state of the project.

---

## Executive Summary

Add AiSAQ (All-in-Storage ANNS with Product Quantization) support to the `vindex` DuckDB extension. AiSAQ is a Vamana-graph-based ANN algorithm that pages PQ codes from storage on demand, keeping only the graph topology and a small skeleton in DRAM. This enables indexing datasets larger than RAM.

The reference implementation is `ref/aisaq-diskann/` (KioxiaAmerica fork of DiskANN). The vindex port replaces the libaio/liburing SSD I/O layer with DuckDB's `BlockManager` (via `AiSaqBlockStore`), eliminating platform-specific aligned-IO code and working cross-platform from day one.

### Why AiSAQ over the existing DISKANN

The existing `DISKANN` algorithm in vindex keeps PQ codes in a contiguous in-RAM array (`DiskAnnCore::codes_`). Despite the "DiskANN" name, it is a DRAM-resident index. AiSAQ moves the PQ codes into `AiSaqBlockStore` blocks backed by DuckDB's `BlockManager`, so cold codes evict cleanly under memory pressure via the BufferPool. This is what enables the 100M x 768-dim workload on 32 GB RAM.

### Storage architecture

AiSAQ uses `AiSaqBlockStore` (wrapping DuckDB's `BlockManager` directly) for both graph nodes and PQ code pages. This bypasses `FixedSizeAllocator`/`IndexBlockStore` entirely — `FixedSizeAllocator` pins buffers permanently (the FIXME at `fixed_size_buffer.hpp:142` was never resolved in DuckDB). With `BlockManager`, each block is managed by the BufferPool: `BufferManager::Pin()` returns a `BufferHandle` (RAII), and when the handle goes out of scope the block is immediately eligible for eviction. This provides native eviction for both clean and dirty blocks during build AND search, without patching DuckDB.

The build has two passes:
1. **PQ encoding** (write-only): stream vectors from the table, encode each to a PQ code, write directly to persistent page blocks via `BlockManager::Write()`. No BufferPool overhead.
2. **Graph construction** (read-heavy): for each vector, BeamSearch reads existing PQ codes and graph nodes via `BufferManager::Pin()`. The BufferPool caches hot blocks and evicts cold ones to the DB file under `memory_limit` pressure.

---

## Locked Decisions

These decisions were finalized during planning and must not be changed without updating this document.

### Architecture

1. **New `AISAQ` `IndexType`** (TYPE_NAME `"AISAQ"`), parallel to `DISKANN`/`HNSW`/`IVF`/`SPANN`. Not an option on `DISKANN`.
2. **Vamana math is duplicated** in `AiSaqCore`, not shared with `DiskAnnCore`. The data planes diverge (paged codes vs in-RAM codes), making a shared core awkward. Extraction to `src/common/vamana.hpp` is tracked as M6.
3. **`src/backend/`** directory for the backend abstraction. Abstract `Backend` base class; concrete `CpuBackend` always compiled; concrete `VulkanBackend` gated by `-DVINDEX_BACKEND_VULKAN=ON` (Phase 8 only). No `IBackend`/`CpuBackendImpl` naming — follows project convention (`Quantizer`/`PqQuantizer`).
4. **No factory, no registry, no plugin system**. Backend selection is a hardcoded selector function `GetActiveBackend(ctx)` reading the `vindex_gpu_backend` session option. Runtime selection, compile-time inclusion.
5. **AiSAQ uses `AiSaqBlockStore`** (wrapping DuckDB's `BlockManager` directly), NOT `IndexBlockStore`/`FixedSizeAllocator`. `FixedSizeAllocator` pins buffers permanently (the FIXME at `fixed_size_buffer.hpp:142` was never resolved in DuckDB). `AiSaqBlockStore` uses `BufferManager::Pin()` / `BufferHandle` (RAII) which provides native eviction via the BufferPool. This enables both graph nodes and PQ code pages to be evicted under memory pressure during build AND search.
6. **Neighbor references are `uint32_t` internal_ids** (4 bytes), not `BlockId`/`IndexPointer` (8 bytes). Dense arithmetic indexing: `block_idx = internal_id / nodes_per_block`, `offset = (internal_id % nodes_per_block) * node_size`. This halves per-node neighbor storage (R=64: 272 B/node vs 528 B/node).

### Quantizers

7. **Accepted**: `pq`, `scann`. Both use chunked-LUT distance math.
8. **Rejected**: `flat` (defeats compression), `rabitq` (no LUT path; codes too small to benefit from paging).
9. **LUT virtuals on `Quantizer`**: `PopulateDistanceLUT` + `LUTDistance` added as improvement (additive, default fallback, no semantic change). Not AiSAQ-only. Upstreamable.

### AiSAQ parameters

10. **`aisaq_inline_pq` default = 0** (range [0, `aisaq_r`]). Non-zero values duplicate PQ codes into graph nodes AND keep them in pages — that is a full copy in both disk and DRAM cache, defeating AiSAQ's purpose over DiskANN. Users who want the latency/DRAM trade-off may set it explicitly.
11. **PQ page size = `Storage::BLOCK_SIZE`** (one DuckDB block, 256 KB default). Allocated via `BlockManager::AllocateBlock()` (persistent from creation). PQ encoding pass writes directly to DB file via `BlockManager::Write()`; graph build and search read via `BufferManager::Pin()`. Internal constant, no user knob.
12. **Entry-point count default = 16** (range [1, 64]). Computed via k-means at build time. The architectural difference from cuVS (no IVF pre-cluster layer) justifies a higher default than cuVS's per-shard "1-3".
13. **`aisaq_r` default = 64**, `aisaq_l` default = 100, `aisaq_alpha` default = 1.2 (mirrors DiskANN).

### Memory

14. **No per-index `ram_budget_mb` eviction queue.** Memory management relies on DuckDB's global `memory_limit` setting. The BufferPool automatically evicts cold blocks (graph node blocks + PQ page blocks) under memory pressure, writing them to the DB file. This enables indexing datasets larger than RAM without any custom eviction logic. The session options `vindex_aisaq_beam_width` and `vindex_aisaq_io_limit` control the per-query working set size (fewer simultaneous pins = less RAM per query).
15. **Two-pass build**: PQ encoding (write-only, bypasses BufferPool, writes directly to persistent blocks) followed by graph construction (read-heavy, BufferPool caches hot blocks). The encoding pass streams vectors from the DuckDB table, encodes each to a PQ code, and writes to page blocks via `BlockManager::Write()`. No BufferPool overhead during encoding.

### Filtering

16. **Label-column filtering is included in v0**. `WITH (label_column='cat_id')` designates a BIGINT column; the optimizer detects `WHERE` predicates on that column and routes them to the index scan.
17. **Label type: BIGINT only in v0**. VARCHAR tracked as M7.
18. **WHERE-clause operators supported in v0**: `=`, `>`, `>=`, `<`, `<=`, `BETWEEN`, and conjunctive folding (e.g., `> 10 AND < 100` folds into `RANGE[11, 99]`). All operators map to two `LabelFilter` kinds: `EQUALS` (single-value fast-path) or `RANGE` (inclusive `[lo, hi]`). Deferred to M7.5: `!=` (not-equal) and `IN (...)`. Subquery-sourced values always fall through to post-hoc filtering.
19. **Adaptive range medoid strategy**: `EQUALS` filters use the single label's medoid. `RANGE` filters use all matching labels' medoids as entry points when the matching count is <= `n_entry_points` (default 16); otherwise fall back to global k-means entry points with range-based candidate filtering.
20. **General WHERE-label optimizer is Phase 2** (branch `feat/label-filter`), upstreamable independently of AiSAQ. AiSAQ label integration is Phase 6.

### GPU

21. **GPU scaffolding = interface + CpuBackend only** in the initial AiSAQ plan. `VulkanBackend` is Phase 8 (`feat/aisaq-vulkan`). No CUDA/ROCm/Metal — only `"cpu"` and `"vulkan"` are referenced anywhere.
22. **Backend selection is runtime** (`vindex_gpu_backend` session option, default `"cpu"`); **backend inclusion is compile-time** (`#ifdef VINDEX_BACKEND_VULKAN`).

### Persistence

23. **Persistence via `BlockManager` block IDs.** Graph node blocks and PQ page blocks are allocated as persistent DuckDB blocks from creation. At checkpoint, `BlockManager` writes dirty blocks to the DB file natively. The block IDs are stored in the index state stream. On recovery, blocks are registered with `BlockManager::RegisterBlock()` and loaded lazily on first `Pin()`. Gated behind `vindex_enable_experimental_persistence` (default off), same as other index types.

### Out of scope (declined, not deferred)

24. **BFS node cache** (`_nhood_cache`/`_coord_cache`) — replaced by BufferPool.
25. **Global PQ-vector cache** (`_aisaq_pq_vectors_cache_buf`) — replaced by BufferPool.
26. **Rearrange permutation** (`aisaq_generate_vectors_rearrange_map`) — replaced by BlockManager's block packing.
27. **Patching DuckDB's `FixedSizeBuffer`** — investigated and declined. The FIXME at `fixed_size_buffer.hpp:142` could be resolved with ~8 lines, but it would only benefit AiSAQ (not other algorithms). Using `BlockManager` directly via `AiSaqBlockStore` achieves the same result without touching DuckDB.

### Future milestones (not in this plan)

- **M6**: Extract shared Vamana primitives to `src/common/vamana.hpp`.
- **M7**: VARCHAR label support.
- **M7.5**: `!=` (not-equal) and `IN (...)` label filter operators.
- **M8**: = Phase 8 (`VulkanBackend`).
- **M9**: IVF label filtering (filter during posting-list scan; ~2 days).
- **M10**: SPANN label filtering (filter during posting-list scan; ~2 days).
- **M11**: HNSW label filtering (filter during `SearchLayer` traversal; ~3 days).
- **M12**: DiskANN label filtering (filter during `BeamSearch` + optional per-label medoid; ~1 week).

---

## Branch Topology

```
main
  |
  |- feat/label-filter     Phase 2 (WHERE-label opt)     [upstreamable]
  |
  [merge to main]
  |
  '- feat/aisaq            Phases 3-7
      '- feat/aisaq-vulkan Phase 8 (VulkanBackend)
```

- `feat/label-filter` branches from `main` independently and can be PR'd upstream separately.
- `feat/aisaq` branches from `main` AFTER `feat/label-filter` is merged.
- `feat/aisaq-vulkan` branches from `feat/aisaq` after Phase 7 is complete.

> **Phase 1 (feat/unpin) was eliminated.** Investigation revealed DuckDB's
> `FixedSizeBuffer` has the same no-eviction FIXME that Phase 1 intended to
> fix, and patching it would only benefit AiSAQ (not HNSW/DiskANN/IVF/SPANN,
> which either don't use IndexBlockStore for large data or would thrash if
> their graph nodes were evicted). AiSAQ instead uses `AiSaqBlockStore`
> (wrapping `BlockManager` directly) which has native eviction via the
> BufferPool. See revised Phase 5 for details.

---

## Phase Overview

| Phase | Document | Branch | Effort | Status | Depends on |
|---|---|---|---|---|---|
| ~~1~~ | ~~[01-feat-unpin.md](01-feat-unpin.md)~~ | ~~`feat/unpin`~~ | ~~4-6 wks~~ | `- [x]` ELIMINATED | — |
| 2 | [02-feat-label-filter.md](02-feat-label-filter.md) | `feat/label-filter` | 3 wks | `- [x]` | — |
| 3 | [03-lut-virtuals.md](03-lut-virtuals.md) | `feat/aisaq` | 1 wk | `- [x]` | 2 |
| 4 | [04-backend-interface.md](04-backend-interface.md) | `feat/aisaq` | 1 wk | `- [x]` | 3 |
| 5 | [05-aisaq-core.md](05-aisaq-core.md) | `feat/aisaq` | 5-7 wks | `- [x]` | 4 |
| 6 | [06-aisaq-label-integration.md](06-aisaq-label-integration.md) | `feat/aisaq` | 2-3 wks | `- [ ]` | 5 |
| 7 | [07-tests-bench.md](07-tests-bench.md) | `feat/aisaq` | 1-2 wks | `- [ ]` | 6 |
| 8 | [08-aisaq-vulkan.md](08-aisaq-vulkan.md) | `feat/aisaq-vulkan` | 8-12 wks | `- [ ]` | 7 |

**Status legend**: `- [ ]` not started / `- [~]` in progress / `- [x]` complete / `- [!]` blocked

**Critical path**: Phase 2 -> merge -> Phase 3 -> 4 -> 5 -> 6 -> 7 -> 8.

---

## Cross-Phase Invariants

These must hold throughout all phases. Violating them requires updating this document.

1. **No mmap of external files.** All storage goes through DuckDB's `BlockManager` (per `AGENTS.md` and `index_block_store.hpp:32`). This satisfies community-extensions requirements. AiSAQ uses `BlockManager` directly via `AiSaqBlockStore`, not `FixedSizeAllocator`.
2. **Literal query vectors only** for index-scan rewrite. The `VINDEX_INDEX_SCAN` optimizer fires only when the query vector is a literal (e.g. `[...]::FLOAT[128]`). AiSAQ inherits this from the shared planner.
3. **Rerank invariant.** `WITH (rerank = N)` produces a uniform plan shape across all algorithms: `TOP_N (k) <- PROJECTION <- VINDEX_INDEX_SCAN (emits k x N row_ids)`. No "skip rerank" path for AiSAQ.
4. **Persistence is opt-in.** Checkpoint/WAL round-trip is gated behind `vindex_enable_experimental_persistence = true` (default off). AiSAQ inherits this.
5. **Style.** `.clang-format` mirrors DuckDB upstream: column 120, tabs for indentation, attached braces, right-aligned pointers. Format script touches `src/` and `test/unit/`; skips `src/include/third_party/`.
6. **Per-algorithm registration convention.** Each algorithm self-registers an `IndexType` + `TYPE_NAME` via `Register(ExtensionLoader&)` and `VectorIndexRegistry::Instance().RegisterTypeName(...)`. Optimizers in `src/common/optimize_*.cpp` pick it up automatically — do not edit them per-algorithm.
7. **Backend inclusion is compile-time.** `VulkanBackend` source is only compiled when `-DVINDEX_BACKEND_VULKAN=ON`. The `#ifdef VINDEX_BACKEND_VULKAN` guard appears in exactly one place: `src/backend/active_backend.hpp`. No other file may reference Vulkan symbols ungated.
8. **Node-size alignment.** Graph node sizes in `AiSaqBlockStore` must be multiples of 8 so that dense arithmetic indexing (`internal_id * node_size`) yields 8-byte-aligned offsets within blocks. See AGENTS.md "Node-size alignment invariant".

---

## Glossary

| Term | Definition |
|---|---|
| AiSAQ | All-in-Storage ANNS with Product Quantization. Kioxia's DiskANN fork that pages PQ codes from SSD on demand. |
| Vamana | The graph algorithm used by DiskANN/AiSAQ. Single-layer, RobustPrune with alpha relaxation, beam search. |
| ADC | Asymmetric Distance Computation. Precompute a LUT from the query, then gather per-code distances from the LUT. |
| LUT | Look-Up Table. The precomputed table of `256 x m_chunks` distances from a query to all PQ centroids. |
| BufferManager | DuckDB's cross-platform block cache. The substrate that replaces AiSAQ's libaio/liburing reader. |
| IndexBlockStore | vindex's addressable block storage layer on top of DuckDB's `FixedSizeAllocator`/`BlockManager`. |
| M3-complete | The (previously planned) work to make `IndexBlockStore::Unpin` actually evict buffers. Was Phase 1 of this plan; **eliminated** — AiSAQ uses `AiSaqBlockStore` instead. |
| FixedSizeAllocator | DuckDB internal allocator that packs same-size allocations into shared `FixedSizeBuffer`s. |
| `BlockId` | vindex alias for `IndexPointer` — 8-byte opaque handle to a node allocation in `IndexBlockStore`. AiSAQ does not use `BlockId` for neighbor references; it uses `uint32_t` internal_ids (4 bytes). |
| `NodeSizeId` | Tag identifying which registered node-size allocator a `BlockId` belongs to. |

---

## Reference

- **Reference code**: `ref/aisaq-diskann/` (gitignored, read-only)
- **Kioxia GPU blog**: `ref/kioxia-gpu-blog.txt`
- **Existing DiskANN**: `src/algo/diskann/` (the closest existing template for AiSAQ)
- **IndexBlockStore**: `src/include/vindex/index_block_store.hpp` + `src/common/index_block_store.cpp`
- **Quantizer base**: `src/include/vindex/quantizer.hpp`
- **Registry**: `src/common/vector_index_registry.cpp`
