# AGENTS.md — notes for OpenCode sessions working in `duckdb-vector-index`

Local-only. This file is gitignored alongside `CLAUDE.md` — do not commit it.
When README / Makefile / CMakeLists conflict with anything below, trust them.

## What this is

DuckDB extension `vindex` (vector similarity search). Fork of upstream
`duckdb-vss` with the `usearch` wrapper replaced by in-house kernels.
Published to `duckdb/community-extensions`. MIT.

Pinned in lockstep at **DuckDB v1.5.3 + extension-ci-tools v1.5.3**
(`scripts/bootstrap.sh` constants). Bump both together or the build breaks.

## First-time setup (do not skip)

`duckdb/` and `extension-ci-tools/` are git submodules and ship **empty** in
a fresh clone. Nothing will build until you run:

```sh
./scripts/bootstrap.sh   # adds + pins both submodules; safe to re-run
make                     # release build → build/release/extension/vindex/
```

The empty `duckdb/` and `extension-ci-tools/` dirs are not a failed clone.

The extension compiles with **C++17** (set via `CMAKE_CXX_STANDARD` in the
extension's `CMakeLists.txt`). DuckDB core keeps its own default — do not
remove the directory-level `set(CMAKE_CXX_STANDARD 17)`. For parallel
builds, pass `CMAKE_BUILD_PARALLEL_LEVEL=$(sysctl -n hw.ncpu)`.

## Commands

| task | command |
| --- | --- |
| release build | `make` |
| debug build | `make debug` |
| SQL logic tests (test/sql/) | `make test` |
| Catch2 unit tests (test/unit/) | `make unit` |
| recall regression (siftsmall) | `make bench` |
| single AiSAQ bench algo | `python3 test/bench/run_recall.py --algo aisaq-pq` |
| HnswCore vs usearch microbench | `make bench_hnsw_core` |
| acceptance matrix (all 60 algo×quant×metric cells) | `make matrix` |
| rewrite formatting | `make format` or `scripts/format.sh` |
| check formatting without modifying | `scripts/format.sh --check` |
| profile AiSAQ build (samply) | see "Profiling" below |

Notes an agent will get wrong without help:

- `make` already forwards `-DVINDEX_BUILD_UNIT=ON -DVINDEX_BUILD_BENCH=ON`
  via `EXT_FLAGS` in the Makefile. Do **not** pass these to cmake by hand.
- `make test`, `make unit`, and `make bench` are **three separate suites**.
  Running `make test` does not run unit or recall tests.
- SQL logic tests (`make test`) require a **release** build. DuckDB v1.5.3
  has debug-mode assertions in `ColumnDataCollection::Scan` that abort
  during index construction; use `make unit` (Catch2) for debug feedback.
- Unit tests and bench binaries are **silently skipped on Windows** (DuckDB's
  lib doesn't export the internal C++ symbols they link against). SQL logic
  tests still run there. See `CMakeLists.txt:59,67`.
- `make bench` downloads siftsmall over **FTP from INRIA** on first run into
  `test/bench/datasets/` (gitignored). Networks that block FTP will fail;
  the dataset must be obtained out-of-band.
- `make bench` exits non-zero on any Recall@10 regression. Thresholds live
  in the `THRESHOLDS` dict at the top of `test/bench/run_recall.py`. Add a
  new (algo, dataset) key before loosening.
- `combo_matrix.test` is **gated behind `VINDEX_RUN_MATRIX=1`** and skipped
  by `make test` (each cell pays the full build pipeline). Use `make matrix`.

### Running a single test

- One SQL test: `./build/release/test/unittest "test/sql/hnsw/hnsw_basic.test"`
- One unit test:  `ctest --test-dir build/release/extension/vindex -L unit -R <name> --output-on-failure`
- One bench algo: `python3 test/bench/run_recall.py --algo hnsw-rabitq3`

The `require vindex` header in each `.test` file is what gates execution on
extension load — leave it in new files.

## Profiling

### samply (macOS)

`samply` is a sampling profiler that uses the mach timer API. It produces
Firefox Profiler-format JSON that can be analyzed from the command line.

**Recording a profile** — always use `--unstable-presymbolicate` (embeds
symbol info in a `.syms.json` sidecar) and `-s` (save-only, no local server).
Use the DuckDB CLI binary (not the Python module) to avoid capturing Python
traces. Capture all threads (omit `--main-thread-only`) since DuckDB's
construction work runs on a worker thread, not the main thread.

```sh
# 1. Prep a DuckDB file with test data (one-time, via Python):
python3 -c "
import duckdb, sys; sys.path.insert(0,'test/bench'); import datasets
import pyarrow as pa
ds = datasets.load('siftsmall')
con = duckdb.connect('test/bench/datasets/siftsmall.duckdb')
con.execute('CREATE TABLE base (id INTEGER, vec FLOAT[128])')
t = pa.Table.from_arrays([pa.array(range(len(ds.base)), type=pa.int32()),
    pa.FixedSizeListArray.from_arrays(pa.array(ds.base.flatten(), type=pa.float32()), 128)],
    names=['id','vec'])
con.register('a', t); con.execute('INSERT INTO base SELECT * FROM a'); con.close()
"

# 2. Create a SQL script (profile_target.sql):
#    LOAD 'build/release/extension/vindex/vindex.duckdb_extension';
#    SET vindex_log_level = 'off';
#    ATTACH 'test/bench/datasets/siftsmall.duckdb' AS sift (READ_ONLY);
#    CREATE TABLE base AS SELECT * FROM sift.base;
#    CREATE INDEX idx ON base USING AISAQ(vec)
#      WITH (quantizer='pq', m=16, bits=8, aisaq_r=64, aisaq_l=100);

# 3. Record at high sample rate:
samply record --unstable-presymbolicate -r 10000 -s -o profile.json.gz \
  -- build/release/duckdb -f profile_target.sql

# 4. Analyze:
python3 scripts/analyze_profile.py profile.json.gz
```

**Reading the output** — the script reports self-time (exclusive CPU in
the function itself) and inclusive-time (function + all callees). Focus on:

- **Self-time** shows where CPU cycles are actually spent (hot leaf functions).
- **Inclusive-time** shows the call tree cost (which high-level functions
  dominate).
- Filter by `[duckdb]` vs `[vindex.duckdb_extension]` to distinguish DuckDB
  core from extension code.
- Inlined functions may attribute time to unexpected symbols — if a
  function's self-time seems too high or too low, check if it's a thin
  wrapper that the compiler inlined.

**Gotchas:**

- Symbol resolution uses the `.syms.json` sidecar (from
  `--unstable-presymbolicate`). Without it, addresses appear as `0x...`.
- The profile captures ALL threads. The construction work runs on a
  single worker thread — look for the thread with high `vindex_self%`.
- `--main-thread-only` will miss the construction thread entirely.
- Frame addresses in the profile are per-library RVAs; the analysis script
  resolves them via `known_addresses` from the syms sidecar.
- For long builds (sift1m: ~10 min), use `--duration 120` to capture a
  representative window instead of the full run.

### Logging during builds

```sql
-- Session option (overridden by VINDEX_LOG_LEVEL env var):
SET vindex_log_level = 'info';       -- phase milestones, throttled 2s
SET vindex_log_level = 'debug';      -- per-page granularity
SET vindex_log_level = 'profile';    -- per-insert timing breakdowns
```

Or via environment variable (overrides session option):
```sh
VINDEX_LOG_LEVEL=info python3 test/bench/run_recall.py --algo aisaq-pq
```

Output goes to stderr (sqllogictest safe). Default is `off`.

## Source wiring (not obvious from filenames)

```
src/vindex_extension.cpp        DUCKDB_CPP_EXTENSION_ENTRY; calls RegisterBuiltInAlgorithms
src/common/vector_index_registry.{cpp,hpp}  singleton of registered algo type names
src/algo/<name>/                each algo self-registers an IndexType + TYPE_NAME
src/quant/<name>/               pluggable quantizers (flat, rabitq, pq, scann)
src/common/optimize_*.cpp       shared planner rules; enumerate the registry, never hard-code algos
src/common/vector_index_scan.cpp  the only index-scan operator; all algos go through it
src/common/index_block_store.cpp  shared block/pager substrate (DiskANN, SPANN, HNSW)
src/algo/aisaq/aisaq_block_store.{cpp,hpp}  AiSAQ-specific storage via BlockManager (not IndexBlockStore)
src/include/vindex/logging.hpp  LogLevel enum + GetLogLevel() (env var + session option)
src/include/vindex/label_filter.hpp  LabelFilter struct (Phase 2/6 WHERE-label infra)
```

**To add a new algorithm**: create `src/algo/<name>/`, register an IndexType
in a `Register()` function, and append its TYPE_NAME to the registry from
`RegisterBuiltInAlgorithms`. The optimizers in `src/common/optimize_*.cpp`
pick it up automatically — do **not** edit them per-algorithm.

**To add a quantizer**: create `src/quant/<name>/`, register against each
algorithm that should accept it, and add cells to
`test/sql/common/combo_matrix.test` (this file is the acceptance matrix;
backend drift fails CI).

### Node-size alignment invariant

`IndexBlockStore` uses DuckDB's `FixedSizeAllocator`, which packs
same-size allocations at `segment_size` strides inside 8-byte-aligned
buffers. When `segment_size` (the node size returned by `NodeSize()`)
is a multiple of 8, every allocation is 8-byte aligned and all typed
fields at their natural offsets within the node are safe under UBSan.

**Every algorithm's `NodeSize()` must return a multiple of 8.** If a node
layout has a variable-length code section (like HNSW), pad it to the
next 8-byte boundary before the following typed array. The HNSW fix
(`hnsw_core.cpp:NodeSize`) is the reference; DiskANN's fixed
`16 + R*8` layout is already aligned.

### AiSAQ storage (AiSaqBlockStore, not IndexBlockStore)

AiSAQ uses `AiSaqBlockStore` (wrapping `BlockManager` directly) for both
graph nodes and PQ code pages — **not** `IndexBlockStore`/`FixedSizeAllocator`.
`FixedSizeAllocator` pins buffers permanently (the FIXME at
`fixed_size_buffer.hpp:142` was never resolved in DuckDB). `AiSaqBlockStore`
uses `BufferManager::Pin()` / `BufferHandle` (RAII) which provides native
eviction via the BufferPool — blocks become evictable when the `BufferHandle`
goes out of scope.

Neighbor references are `uint32_t` internal_ids (4 bytes, dense indexing),
not `BlockId`/`IndexPointer` (8 bytes). Arithmetic resolution:
`block_idx = internal_id / nodes_per_block`, `offset = internal_id % nodes_per_block * node_size`.

The build has two passes:
1. **PQ encoding** (write-only): writes directly to persistent blocks via
   `BlockManager::Write()` — no BufferPool overhead.
2. **Graph construction** (read-heavy): uses `BufferManager::Pin()` — the
   BufferPool caches hot blocks and evicts cold ones under `memory_limit`.

Memory management relies on DuckDB's global `memory_limit`. There is no
per-index `ram_budget_mb` eviction queue.

### AiSAQ build acceleration (three tiers)

The build uses a three-tier strategy to eliminate I/O overhead during Vamana
graph construction. The strategy is resolved in `ResolveBuildStrategy()` from
`WITH (build_strategy=...)` or `SET vindex_aisaq_build_strategy=...`:

- **`pq_buffer`** *(auto default when it fits)*: flat PQ code + flat graph
  node buffers in RAM. `DistanceToCode`, `CodeDistance`, and `PinNode` all
  use pointer arithmetic — zero BufferManager calls during build. Allocated
  in `EncodePqCodes`, activated in `ActivateBuildBuffers`, flushed in
  `FlushBuildNodes` (writes flat node buffer to blocks via
  `WriteAllGraphNodes`), freed in `ClearBuildBuffers`.
- **`exact_prune`** *(opt-in)*: same as `pq_buffer` plus a flat
  full-precision vector buffer. `RobustPrune` uses `RawDistance()` (exact
  O(dim) float loop) instead of PQ `CodeDistance` (O(code_size)). Mirrors
  the reference `aisaq-diskann` build path. 8× slower than `pq_buffer`;
  best graph topology quality.
- **`paged`** *(fallback)*: per-prune gather in `RobustPrune` — collects
  candidate PQ codes into a local buffer (O(L) Pins) before the O(L²)
  pairwise loop, instead of O(L²) per-pair Pins. Works at any scale.

When flat node buffer is active, `AiSaqBlockStore::SetFlatBuildMode(true)`
skips `EnsureGraphCapacity` during `AllocGraphNode` (just bumps the counter).
Blocks are lazily allocated in `WriteAllGraphNodes`.

`aisaq_l_build` defaults to `aisaq_r` (the Vamana paper recommendation).
This decouples build beam width from search beam width.

### AiSAQ label filtering

`WITH (label_column='cat')` designates a BIGINT column for predicate
pushdown. The optimizer in `optimize_scan.cpp` extracts `=`, `>`, `>=`, `<`,
`<=`, `BETWEEN`, and folded conjuncts into a `LabelFilter` passed to
`InitializeScan` → `AiSaqCore::Search`.

At build time, `ComputeLabelMedoids()` finds the member closest to each
label's centroid (PQ-code L2). At search time:
- **EQUALS**: single medoid entry point; neighbors with wrong label are
  skipped during graph expansion.
- **RANGE** with ≤ `n_entry_points` matching labels: all matching medoids
  as entry points (multi-start).
- **RANGE** with more matching labels: global entry points + range
  candidate filtering during expansion.

Label maps are serialized in the state stream (V2 format: `label`, `medoid`,
`members[]` triples). `WITH (label_column=...)` survives checkpoint/restart
via `stored_options_` + `IndexStorageInfo.options` fallback (the DuckDB
v1.5.x `IndexCatalogEntry::GetInfo()` bug workaround in `create_instance`).

## Planner gotcha: literal query vectors only

The `VINDEX_INDEX_SCAN` rewrite fires only when the query vector is a
**literal** (e.g. `[...]::FLOAT[128]`). Subquery-sourced vectors do not
trigger it. `test/bench/run_recall.py` builds literal casts deliberately for
this reason. Any new SQL test that expects the index path must do the same.
Reference: `test/sql/rabitq/rabitq_basic.test`.

## Persistence is opt-in

Checkpoint / WAL round-trip of indexes is gated behind the session pragma
`vindex_enable_experimental_persistence = true` (default off). SQL tests for
persistence flows must set it; see `hnsw_experimental_persistence.test` and
`hnsw_insert_wal.test`.

## Rerank invariant

`WITH (rerank = N)` (or `SET vindex_rerank_multiple = N`) produces a uniform
plan shape across all algorithms:

```
TOP_N (k) ← PROJECTION ← VINDEX_INDEX_SCAN (emits k × N row_ids)
```

There is no "skip rerank" path; the upstream operator is always exact
distance against the authoritative `FLOAT[d]` column. Enforced by
`test/sql/hnsw/hnsw_rerank.test` — keep new algos consistent.

## Style

`.clang-format` mirrors DuckDB upstream (commented at top): **column 120,
tabs for indentation, attached braces, right-aligned pointers**. The format
script only touches `src/` and `test/unit/`; it skips
`src/include/third_party/`. Do not format vendored headers.

## CI

- `.github/workflows/MainDistributionPipeline.yml` triggers **only on `v*`
  tag push** (or manual dispatch). There is no per-push CI; run `make test`
  locally before opening a PR.
- WASM archs are excluded (`exclude_archs: wasm_mvp;wasm_eh;wasm_threads`).
- Docs site (`docs-site/`, Astro) deploys via `docs.yml` **manual dispatch
  only** — never on push.

## Release checklist (per CHANGELOG)

1. Bump `version:` in `description.yml` and `CHANGELOG.md`.
2. Tag `vX.Y.Z`. The pipeline builds the multi-arch matrix and publishes a
   GitHub Release with `vindex.<arch>.duckdb_extension` assets.
3. Unsigned per-arch binaries are attached to the release; the signed
   community-extensions build is produced by the separate
   `duckdb/community-extensions` pipeline.

## Local reference checkouts (`ref/`, gitignored)

`ref/` holds read-only checkouts of external code used as design references
when working on the in-house kernels. Both are gitignored — don't edit them,
don't rely on either being present in a fresh clone.

- `ref/duckdb-vss/` — upstream `duckdb-vss` (the fork this repo replaced).
  Diff against it when you need to see what the `usearch` wrapper used to do.
- `ref/aisaq-diskann/` — [KioxiaAmerica/aisaq-diskann](https://github.com/KioxiaAmerica/aisaq-diskann),
  a DiskANN fork (AiSAQ: SSD-resident, DRAM-free, PQ codes paged on demand).
  Reference for the in-house `src/algo/diskann/`; `pq_flash_index.cpp` /
  `aisaq_pq_reader.cpp` / `linux_aligned_file_reader.cpp` map roughly to what
  `IndexBlockStore` + the vindex DiskANN core do.
  It is a **standalone CMake project**, not a DuckDB extension — its own
  toolchain (Intel MKL, OpenMP, Boost, liburing/libaio, tcmalloc) is
  unrelated to this repo's build. Consult it for algorithmic reference only;
  do not import its headers or link its objects.
