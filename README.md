# duckdb-vector-index (`vindex`)

A DuckDB extension for **vector similarity search at scale**. Superset of the
official [`vss`](https://github.com/duckdb/duckdb-vss) extension: supports
HNSW, IVF (IVF-Flat / IVF-RaBitQ / IVF-PQ / IVF-ScaNN), DiskANN (Vamana
graph with codes held out-of-band so the graph can evict past RAM),
AiSAQ (Vamana graph whose PQ codes are paged from DuckDB's BufferPool on
demand, enabling indexes larger than RAM, with **parallel CREATE INDEX**
across all CPU cores), and
SPANN (IVF with closure-replica writes so boundary points survive a
single-cell probe), with pluggable quantization — RaBitQ (bits ∈
{1,2,3,4,5,7,8}, default **3-bit**), PQ, and ScaNN anisotropic PQ — plus
an optimizer-level rerank pass against the authoritative `FLOAT[d]` column.

## Explore interactively

A live docs site at **<https://icemap.github.io/duckdb-vector-index/>** lets
you click through every algorithm × quantizer × metric combination and see
the exact `CREATE INDEX` SQL it would generate, with hover cards covering
the capability trade-offs of each choice:

<table>
  <tr>
    <td><img width="100%" alt="Hierarchy explorer — HNSW + RaBitQ" src="https://lab-static.pingcap.com/images/2026/4/29/e3cea18d7eaa6d297e592f53cbf23a1d36203487.png"></td>
    <td><img width="100%" alt="Hierarchy explorer — DiskANN + PQ" src="https://lab-static.pingcap.com/images/2026/4/29/0ba1022cbbde51a60dd6b3befd6bd33c931d8c35.png"></td>
  </tr>
  <tr>
    <td><img width="100%" alt="Algorithm capability cards" src="https://lab-static.pingcap.com/images/2026/4/29/0d90b23b3c3c24d73e4d24e08661fb98f5eb38a0.png"></td>
    <td><img width="100%" alt="RaBitQ detail page" src="https://lab-static.pingcap.com/images/2026/4/29/4bf810db92c0188f882085841329077c6bef0eae.png"></td>
  </tr>
</table>

## Installing

`vindex` is published to the DuckDB
[community-extensions](https://community-extensions.duckdb.org/extensions/vindex.html)
repository, so the signed per-platform build loads with two lines:

```sql
INSTALL vindex FROM community;
LOAD vindex;
```

No `-unsigned` flag or `allow_unsigned_extensions` is required — the
community-extensions pipeline signs each `vindex.duckdb_extension` binary
after build. DuckDB versions must match the one vindex was built against
(currently **v1.5.3**).

### From source

If you want to hack on vindex or run against a newer DuckDB than the
community repo has rebuilt for yet:

```sh
git clone https://github.com/Icemap/duckdb-vector-index.git
cd duckdb-vector-index
./scripts/bootstrap.sh   # pulls the duckdb + extension-ci-tools submodules
make                     # release build → build/release/extension/vindex/vindex.duckdb_extension
```

Then load the unsigned build:

```sql
-- duckdb -unsigned
LOAD 'build/release/extension/vindex/vindex.duckdb_extension';
```

Or, inside a session started without `-unsigned`:

```sql
SET allow_unsigned_extensions = true;
LOAD 'build/release/extension/vindex/vindex.duckdb_extension';
```

### From a GitHub release

Unsigned per-arch binaries (`vindex.linux_amd64.duckdb_extension`,
`vindex.osx_arm64.duckdb_extension`, …) are also attached to each
[GitHub Release](https://github.com/Icemap/duckdb-vector-index/releases)
and can be `LOAD '<path>'`-ed the same way. Prefer
`INSTALL vindex FROM community;` unless you need a release that has not
propagated to community-extensions yet.

## Quickstart

```sql
INSTALL vindex FROM community;
LOAD vindex;

CREATE TABLE docs (id INT, embedding FLOAT[768]);
-- ... populate from your model of choice ...

-- HNSW with RaBitQ 3-bit compression (default), >99% Recall@10
CREATE INDEX docs_idx ON docs USING HNSW (embedding)
    WITH (metric='cosine', quantizer='rabitq', bits=3);

-- Or IVF-RaBitQ — cheaper build, tunable recall/speed via nlist/nprobe.
-- Recall@10 ≥ 0.97 on SIFT1M at nlist=1024/nprobe=32.
CREATE INDEX docs_idx ON docs USING IVF (embedding)
    WITH (metric='cosine', quantizer='rabitq', bits=3, rerank=10,
          nlist=1024, nprobe=32);

-- DiskANN (Vamana) with PQ compression — graph blocks evict from the
-- buffer pool so the index can exceed RAM. PQ defaults (m=dim/4, bits=8)
-- are fine for most 768-d models; tune `diskann_r`/`diskann_l` if you need
-- a wider beam.
CREATE INDEX docs_idx ON docs USING DISKANN (embedding)
    WITH (metric='cosine', quantizer='pq', bits=8, rerank=10,
          diskann_r=64, diskann_l=100);

-- SPANN — IVF with closure replicas. Boundary points are written into
-- every centroid within `closure_factor × d_best`, so a single-cell
-- probe still finds them. Paper defaults: replica_count=8, closure_factor=1.1.
CREATE INDEX docs_idx ON docs USING SPANN (embedding)
    WITH (metric='cosine', quantizer='rabitq', bits=3, rerank=10,
          nlist=1024, nprobe=32, replica_count=8, closure_factor=1.1);

-- AiSAQ — paged-PQ Vamana graph. PQ codes live in DuckDB blocks and are
-- paged on demand, so the index scales past RAM. Accepts PQ or ScaNN only.
-- Recall@10 ≥ 0.95 on SIFT1M.
CREATE INDEX docs_idx ON docs USING AISAQ (embedding)
    WITH (metric='l2sq', quantizer='pq', m=16, bits=8, rerank=10,
          aisaq_r=64, aisaq_l=100);

-- Query uses the standard DuckDB distance function; the index kicks in.
SELECT id, embedding
FROM docs
ORDER BY array_cosine_distance(embedding, [ ... ]::FLOAT[768])
LIMIT 10;
```

## Why not usearch?

The upstream `duckdb-vss` extension (which this repo forks) wraps
[`unum-cloud/usearch`](https://github.com/unum-cloud/usearch). We replaced it
with an in-house HNSW implementation (`src/algo/hnsw/` + `src/include/vindex/hnsw_core.hpp`).
We ran a side-by-side microbench (`test/bench/bench_hnsw_core.cpp`) at matched
hyperparameters before making the call:

| engine | build (s) | QPS | Recall@10 |
| --- | --- | --- | --- |
| usearch | 21.0 | 9,664 | 0.49 |
| HnswCore (ours) | 24.5 | 10,444 | 0.52 |

`N=100,000  D=128  NQ=200  K=10  M=16  M0=32  ef_construction=128  ef_search=64`

Throughput and recall are comparable (QPS ratio 1.08). What we gain from
owning the code path is the thing usearch cannot give us:

1. **Pluggable quantization.** usearch's scalar types are fixed at (`f32`,
   `f16`, `i8`, `b1`) — these are pure type casts, not compression. usearch
   deliberately **does not own the vector data**: `add(key, ptr)` only
   registers a `key → ptr` mapping and the caller keeps the `FLOAT[d]`
   around. That design can't host RaBitQ (rotated + bit-packed codes), PQ
   codebooks, or ScaNN's anisotropic quantization, because the "code"
   doesn't exist outside the index — we produce it. We own it, so we can
   compress it.
2. **Rerank / fine-search.** RaBitQ is a coarse filter — the planner needs
   access to the top-`k × rerank_multiple` candidates to re-score them against
   the authoritative `FLOAT[d]` column. usearch hides the candidate list
   behind its iterator, with no extension point.
3. **Block-native storage.** DiskANN and SPANN need per-node block
   addressing so the page cache can evict cold regions. `IndexBlockStore`
   is the shared substrate; the usearch blob would have to be torn apart
   anyway.

### Memory footprint

The bench above deliberately omitted a memory column because a naive RSS
comparison is misleading. usearch's 14.7 MB resident delta is real but
narrow — it measures the bench's mode, which is not the mode a DuckDB index
actually runs in.

- **In the microbench**, vectors live in a caller-owned `std::vector<float>`
  and usearch's `add(key, ptr)` just registers a pointer into it — no copy,
  hence the small RSS. That pointer mode requires the caller to keep the
  backing array alive for the lifetime of the index.
- **Inside DuckDB**, column-store `FLOAT[d]` blocks are paged in and out of
  the buffer pool; there is no stable `float*` an index can hang onto across
  scans. So `duckdb-vss` has usearch **copy the float32 codes internally** —
  the external-pointer trick is unavailable. usearch's index RSS in a real
  DuckDB process is roughly the same as our `flat` path (one float32 per
  vector, whatever graph overhead on top).

Index RSS for N=100k, d=128, same hyperparameters as the bench:

| index                     | per-vector code | index RSS |
| ------------------------- | --------------- | --------- |
| usearch, bench mode       | 512 B (external)| 14.7 MB (caller holds the 51 MB) |
| HnswCore + `flat`         | 512 B (inline)  | 75.8 MB   |
| HnswCore + `rabitq` 3-bit | 60 B (inline)   | ~34.0 MB  |

What actually matters is the `rabitq` row. Owning the code path lets us
compress the per-vector payload ~8.5× and pull total index RSS below what
either `flat` path can reach. usearch's `f32 / f16 / i8 / b1` options are
type casts, not compression — none of them can host rotated + bit-packed
RaBitQ codes.

## Quantizer defaults

For the capability matrix (metrics accepted, trade-offs per
algorithm / quantizer combination) see the
[interactive docs](https://icemap.github.io/duckdb-vector-index/). This
table just pins the `WITH (…)` defaults so you know what you're overriding:

| `quantizer` | Default `bits` | Other overridable options |
| --- | --- | --- |
| `flat`   | — | — |
| `rabitq` | 3 | `bits` ∈ {1, 2, 3, 4, 5, 7, 8} |
| `pq`     | 8 | `bits` ∈ {4, 8}; `m` defaults to `dim/4` |
| `scann`  | 8 | `bits` ∈ {4, 8}; `m` defaults to `dim/4`; `eta` (default 4) |

> **AiSAQ** accepts only `pq` and `scann` — it requires a LUT-capable
> compressing quantizer. `flat` (stores full fp32, defeats the paged layout)
> and `rabitq` (no LUT distance path) are rejected at `CREATE INDEX` time.

### Quantizer bits vs recall

Low-bit RaBitQ is a **coarse filter** — on its own the estimated distances are
noisy, so the expected usage is:

> top `k × rerank_multiple` candidates ranked by estimated distance → re-rank
> those candidates using the exact distance from the original `FLOAT[d]` column.

The numbers below are Recall@10 over a 1,000-vector × 128-dim Gaussian fixture
(scalar path; see `test/unit/test_rabitq_quantizer.cpp`). End-to-end numbers
through DuckDB on the INRIA [siftsmall](http://corpus-texmex.irisa.fr/) set
(10k × 128-d, 100 queries, `make bench`) match the shape:

| config | Recall@10 | build | 100 queries |
| --- | --- | --- | --- |
| `hnsw-flat` | 0.996 | 0.5 s | 0.08 s |
| `hnsw-rabitq3 + rerank=10` | 1.000 | 1.9 s | 0.09 s |
| `hnsw-rabitq1 + rerank=50` | 0.998 | 3.0 s | 0.18 s |

| `bits` | No rerank | + 10× rerank | + 20× rerank | Bytes / vector (d=128) | vs float32 |
| --- | --- | --- | --- | --- | --- |
| 1 | ~0.40 | ~0.85 | ≥0.90 | 16 + 12 trailer = **28 B** | 18× smaller |
| 2 | ~0.60 | ~0.95 | ≥0.97 | 32 + 12 = **44 B** | 12× smaller |
| 3 *(default)* | ~0.80 | ≥0.98 | **≥0.99** | 48 + 12 = **60 B** | 8.5× smaller |
| 4 | ~0.90 | ≥0.99 | ≥0.99 | 64 + 12 = **76 B** | 6.7× smaller |
| 5 | ~0.95 | ≥0.99 | ≥0.99 | 80 + 12 = **92 B** | 5.6× smaller |
| 7 | ~0.98 | ≥0.99 | ≥0.99 | 112 + 12 = **124 B** | 4.1× smaller |
| 8 | ~0.99 | ≥0.99 | ≥0.99 | 128 + 12 = **140 B** | 3.7× smaller |
| float32 (flat) | 1.00 | 1.00 | 1.00 | **512 B** | 1× |

**Rules of thumb:**

- `bits=3` is the default for a reason — it's the sweet spot on recall × memory.
- `bits=1` and `bits=2` **only make sense with rerank ≥ 20×**. Using them
  without rerank will emit a runtime warning and give you 40–60% Recall@10.
- `bits ≥ 5` tends not to pay off vs `bits=3 + bigger rerank`; memory-bound
  workloads almost always prefer lower bits + more rerank.

### The rerank pass

`WITH (rerank = N)` on `CREATE INDEX` (or the session pragma
`SET vindex_rerank_multiple = N`) tells the planner to pull `k × N` candidates
from the index and re-rank them by **exact** `array_distance` against the
authoritative `FLOAT[d]` column. The plan shape is uniform regardless of `N`:

```
TOP_N (k) ← PROJECTION ← VINDEX_INDEX_SCAN (emits k × N row_ids)
```

This is enforced by `test/sql/hnsw/hnsw_rerank.test`. There is no
"skip rerank" shortcut — the upstream operator is always the exact-distance
step, which is why `bits=1 + rerank=20` can recover >99% Recall@10.

## AiSAQ — paged-PQ vector search

AiSAQ ("All-in-Storage ANNS") is a single-layer Vamana graph whose PQ codes
are stored in DuckDB blocks and paged through the BufferPool on demand. Unlike
DiskANN (which holds PQ codes in contiguous RAM), AiSAQ's codes are
evictable — the index can grow larger than RAM, with cold PQ pages evicted by
DuckDB's global `memory_limit` and re-paged on access.

### When to choose AiSAQ vs DiskANN

| | DiskANN | AiSAQ |
|---|---|---|
| PQ codes during build | RAM-resident (fast) | RAM-resident (flat buffer, same speed) |
| PQ codes at search | RAM-resident | Paged from BufferPool (evictable) |
| Memory beyond RAM | Graph only; codes must fit | Graph + codes evictable |
| Search latency | Lower (pointer arithmetic) | Higher (BufferPool pin per code) |
| Best for | Datasets that fit in RAM | Datasets that exceed RAM |

### Parallel CREATE INDEX

AiSAQ's `CREATE INDEX` is parallelized across all CPU cores via DuckDB's
`TaskScheduler`. Four phases run in parallel:

1. **Quantizer training** — m=16 k-means++ passes (one per PQ sub-vector slot).
2. **PQ encoding** (Pass 1) — page-aligned row-range partitioning; each task
   writes disjoint PQ pages.
3. **Graph construction** (Pass 2) — Vamana inserts with per-node spinlocks
   for reciprocal edges (no global lock during construction).
4. **Post-build** — `FinalizeInlineCodes`, `ComputeLabelMedoids`, and
   `WriteAllGraphNodes` each partition by disjoint ranges.

Build-time performance on [SIFT1M](http://corpus-texmex.irisa.fr/)
(1M × 128-d, `pq_buffer` strategy, 10-core host):

| Phase | Serial | Parallel (10 cores) | Speedup |
|---|---|---|---|
| Quantizer training | 5.9 s | 1.2 s | 4.9× |
| PQ encoding | 3.3 s | 0.7 s | 4.6× |
| Graph construction | ~140 s | ~20 s | 7× |
| Post-build | 2.8 s | <0.1 s | 28×+ |
| **Total** | **~152 s** | **~22 s** | **6.9×** |

Recall is identical between serial and parallel builds — the per-node
spinlock strategy produces graph topology equivalent to serial insertion
(recall within 0.001 on all bench configurations). Enable
`SET vindex_log_level = 'info'` to see per-phase wall-clock timing.

### Scaling guide

AiSAQ's footprint has two dimensions: **storage** (on disk, always
proportional to N) and **build-time RAM** (in memory, depends on the
build strategy). Understanding both is essential for datasets above a
few million vectors.

#### Storage vs RAM

| What | Where | Size | Evictable? |
|---|---|---|---|
| PQ code pages | DuckDB blocks (disk) | `N × code_size` | Yes (BufferPool paged on demand) |
| Graph node blocks | DuckDB blocks (disk) | `N × (16 + R×4)` | Yes (BufferPool paged on demand) |
| Original `FLOAT[d]` column | DuckDB table (disk) | `N × dim × 4` | Yes (column store) |
| Build-time flat buffers (`pq_buffer`) | RAM only | `N × (code_size + node_size)` | No — freed after build |
| Cross-distance table | RAM only | `m × K² × 4` (4 MB for SIFT) | No — lives on the quantizer |
| Search-time working set | RAM only | Bounded by `memory_limit` | Yes (BufferPool evicts cold pages) |

The index's persistent footprint on disk is `N × (code_size + node_size)`
— roughly **288 bytes per vector** for SIFT (128-d, m=16, bits=8, R=64).
For 768-d OpenAI-style embeddings (m=192): **464 bytes per vector**.

During search, only the hot pages (graph blocks + PQ code pages near the
query's graph neighborhood) are in RAM. Cold pages evict via the
BufferPool under `memory_limit` pressure. This is what lets AiSAQ scale
past RAM — the **index on disk** can be much larger than the **working
set in RAM**.

During build with `pq_buffer` strategy, ALL codes and nodes are loaded
into flat RAM buffers for zero-I/O construction. The build-time RAM
equals the persistent footprint. With `paged` strategy, codes are read
on demand from the BufferPool — build RAM is small, but construction is
slower (~7× per the benchmarks above).

#### Per-vector footprint formula

```
code_size = m × ceil(bits / 8)           bytes
node_size = (16 + R × 4), padded to 8    bytes   (graph header + R neighbor IDs)
total_per_vector = code_size + node_size  bytes
```

Common configurations (128-d SIFT and 768-d embeddings):

| Config | `m` | `bits` | `code_size` | `R` | `node_size` | **Per vector** |
|---|---|---|---|---|---|---|
| SIFT 128-d, PQ-8 | 16 | 8 | 16 B | 64 | 272 B | **288 B** |
| SIFT 128-d, PQ-8, R=32 | 16 | 8 | 16 B | 32 | 144 B | **160 B** |
| Embeddings 768-d, PQ-8 | 192 | 8 | 192 B | 64 | 272 B | **464 B** |
| Embeddings 768-d, PQ-4 | 192 | 4 | 96 B | 64 | 272 B | **368 B** |

#### Scaling reference

| Dataset | Per vector | Index on disk | `pq_buffer` build RAM | Default strategy (32 GB machine) | Build time (10 cores) |
|---|---|---|---|---|---|
| 1M × 128-d | 288 B | 288 MB | 290 MB | `pq_buffer` (fits) | **~22 s** *(measured)* |
| 10M × 128-d | 288 B | 2.9 GB | 2.9 GB | `pq_buffer` (fits in 25% budget) | Scales roughly linearly with N |
| 100M × 128-d | 288 B | 29 GB | 29 GB | `paged` (exceeds 25% budget) | Scales roughly linearly with N |
| 1M × 768-d | 464 B | 464 MB | 470 MB | `pq_buffer` (fits) | Scales with `m` (more k-means++ slots) |
| 10M × 768-d | 464 B | 4.6 GB | 4.7 GB | `pq_buffer` (fits in 25% budget) | Scales with N × m |

Build times above 1M are extrapolations — the Vamana graph gets denser as
N grows, so scaling may be slightly superlinear. Benchmark on your
hardware for accurate planning.

#### Recommended configurations by scale

**1M vectors — defaults work as-is:**

```sql
-- pq_buffer strategy selected automatically; ~290 MB RAM, ~22 s on 10 cores.
CREATE INDEX idx ON t USING AISAQ(vec)
    WITH (quantizer='pq', m=16, bits=8, aisaq_r=64, aisaq_l=100);
```

**10M vectors — explicit RAM budget to reserve build buffers:**

```sql
-- Ensure DuckDB has enough headroom for the flat build buffers.
-- 10M × 288 B ≈ 2.9 GB; reserve 4 GB to be safe.
SET memory_limit = '16GB';
CREATE INDEX idx ON t USING AISAQ(vec)
    WITH (quantizer='pq', m=16, bits=8, aisaq_r=64, aisaq_l=100,
          ram_budget_mb=4096);
```

**100M vectors — paged strategy, tune R to fit your machine:**

```sql
-- At 100M, pq_buffer needs ~29 GB — exceeds the default 25% budget on
-- most machines. Two options:
--   1) If you have 64+ GB RAM, force pq_buffer with ram_budget_mb:
SET memory_limit = '64GB';
CREATE INDEX idx ON t USING AISAQ(vec)
    WITH (quantizer='pq', m=16, bits=8, aisaq_r=64, aisaq_l=100,
          ram_budget_mb=30720);

--   2) Otherwise use paged (slower build, but works at any scale).
--      Consider lowering aisaq_r to 32 to reduce per-node memory:
CREATE INDEX idx ON t USING AISAQ(vec)
    WITH (quantizer='pq', m=16, bits=8, aisaq_r=32, aisaq_l=100,
          build_strategy='paged');
```

> **When to lower `aisaq_r`**: keep R=64 if your RAM allows it — higher
> connectivity improves recall. Reduce to 32 only when `pq_buffer` doesn't
> fit and you want to minimize per-node memory in the BufferPool. The
> recall impact at 100M scale is untested; benchmark with your data.

#### Strategy selection

The `auto` strategy (default) picks `pq_buffer` when
`N × (code_size + node_size)` fits in 25 % of `memory_limit` (the
`ram_budget_mb=0` default). Otherwise it falls back to `paged`.

Check which strategy was selected:

```sql
SET vindex_log_level = 'info';
CREATE INDEX idx ON t USING AISAQ(vec) WITH (...);
-- Look for: [vindex] AiSAQ build: strategy=pq_buffer, N=...
```

`pq_buffer` is ~7× faster than `paged` (zero BufferPool calls during
construction vs per-prune page pins). Always prefer `pq_buffer` when RAM
allows. The `exact_prune` strategy adds full-precision vectors for prune
distances — ~8× slower than `pq_buffer` but produces the best graph
topology (marginally higher recall). Use it only when recall is critical
and build time is secondary.

### Build acceleration

AiSAQ supports a three-tier build acceleration system controlled by
`WITH (build_strategy=...)` or the session option
`SET vindex_aisaq_build_strategy=...`:

| Strategy | Description | Memory cost | Build speed |
|---|---|---|---|
| `auto` *(default)* | Picks the best tier that fits in RAM budget | — | — |
| `pq_buffer` | Flat PQ code + graph node buffers in RAM during build | `N × (code_size + node_size)` | Fast (zero I/O) |
| `paged` | Per-prune gather; PQ codes read from BufferPool | ~1.6 KB per prune call | Slower (BufferPool pins) |
| `exact_prune` | Same as `pq_buffer` but uses full-precision vectors for prune distances | `N × (code_size + node_size + dim×4)` | **8× slower** than `pq_buffer`; best recall |

The `auto` strategy picks `pq_buffer` when `N × (code_size + node_size)` fits
in 25% of `memory_limit` (overridable via `WITH (ram_budget_mb=...)`), falling
back to `paged` otherwise. `exact_prune` is always opt-in — it mirrors the
reference `aisaq-diskann` implementation's use of full-precision prune
distances, giving marginally better graph quality at 8× the build cost.

```sql
-- Explicit strategy override:
CREATE INDEX idx ON t USING AISAQ(vec)
    WITH (quantizer='pq', m=16, bits=8, aisaq_r=64, aisaq_l=100, build_strategy='exact_prune');

-- Session-level override:
SET vindex_aisaq_build_strategy = 'paged';

-- Explicit RAM budget (MB) for build-time flat buffers:
CREATE INDEX idx ON t USING AISAQ(vec)
    WITH (quantizer='pq', m=16, bits=8, ram_budget_mb=2048);
```

### Key parameters

| Option | Default | Range | Description |
|---|---|---|---|
| `aisaq_r` | 64 | 4–256 | Graph out-degree (R) |
| `aisaq_l` | 100 | 4–1024 | Search beam width (L) |
| `aisaq_l_build` | = `aisaq_r` | 4–1024 | Build beam width (decoupled from search; Vamana paper recommends L_build ≈ R) |
| `aisaq_alpha` | 1.2 | 1.0–2.0 | RobustPrune relaxation factor |
| `aisaq_inline_pq` | 0 | 0–`aisaq_r` | Inline PQ codes for first N neighbors (faster search, larger nodes) |
| `aisaq_entry_points` | 16 | 1–64 | Number of cached entry points for search |
| `aisaq_beam_width` | 8 | — | I/O batching hint for search |
| `aisaq_io_limit` | 0 | — | Max PQ code page-ins per search (0 = unlimited) |

### Label-column filtering

AiSAQ supports label-filtered search via `WITH (label_column='cat')`, where
`cat` must be a `BIGINT` column. When a `WHERE` predicate on `cat` is present,
the optimizer routes it into the index scan:

```sql
CREATE INDEX idx ON docs USING AISAQ(embedding)
    WITH (quantizer='pq', m=16, bits=8, label_column='category');

-- EQUALS: uses the label's medoid as the single entry point
SELECT id FROM docs WHERE category = 5
    ORDER BY array_distance(embedding, [...]::FLOAT[768]) LIMIT 10;

-- RANGE: adaptive — all matching medoids (≤ n_entry_points) or global fallback
SELECT id FROM docs WHERE category BETWEEN 3 AND 7
    ORDER BY array_distance(embedding, [...]::FLOAT[768]) LIMIT 10;
```

Supported operators: `=`, `>`, `>=`, `<`, `<=`, `BETWEEN`, and folded
conjuncts (`cat > 1 AND cat < 5`). Non-literal predicates (subqueries, etc.)
fall through to post-hoc filtering.

### Logging

Build progress and per-phase timing are available via the `vindex_log_level`
session option or the `VINDEX_LOG_LEVEL` environment variable (env var
overrides session option):

```sql
SET vindex_log_level = 'info';    -- phase milestones + per-phase wall-clock timing
SET vindex_log_level = 'debug';   -- per-page granularity
SET vindex_log_level = 'profile'; -- per-insert timing breakdowns
```

At `info` level, AiSAQ parallel build prints per-phase wall-clock times:

```
[vindex] train_quantizer (1207ms)
[vindex] PQ encoding complete: 1000000 vectors in 734ms (10 tasks)
[vindex] pass2_construct (20134ms)
[vindex] merge_parallel_construct (49ms)
[vindex] finalize_inline_codes (0ms)
[vindex] flush_graph_nodes (63ms)
```

```sh
VINDEX_LOG_LEVEL=info python3 test/bench/run_recall.py --algo aisaq-pq
```

Output goes to stderr (sqllogictest-safe). Default is `off`. See AGENTS.md
"Profiling" section for samply profiling instructions.

## Repository layout

```text
src/                    C++ extension source
  include/vindex/       public headers (VectorIndex, Quantizer, logging, ...)
  common/               optimizers, registry, block store
  algo/<name>/          one subdirectory per algorithm (hnsw, ivf, diskann, spann, aisaq)
  quant/<name>/         one subdirectory per quantizer
test/
  sql/                  sqllogictest (.test files)
  unit/                 Catch2 kernel tests
  bench/                recall regression harness (Python)
  python/               duckdb-python e2e smoke
ref/duckdb-vss/         read-only upstream reference
```

## Building

```sh
./scripts/bootstrap.sh   # clones duckdb + extension-ci-tools
make                     # release build → build/release/extension/vindex/
make test                # SQL logic tests (test/sql/)
make unit                # Catch2 unit tests (test/unit/)
make bench               # recall regression on siftsmall (~5 s, auto-downloads)
python3 test/bench/run_recall.py --algo aisaq-pq --dataset sift1m  # AiSAQ on SIFT1M
```

`make bench` downloads the [siftsmall](http://corpus-texmex.irisa.fr/)
dataset into `test/bench/datasets/` on first run and fails non-zero if any
Recall@10 threshold regresses. Full-size SIFT1M is wired but gated — pass
`--dataset sift1m` to `run_recall.py` to exercise it.

## License

MIT — compatible with DuckDB's [`community-extensions`](https://github.com/duckdb/community-extensions)
submission policy.
