# Phase 5 — AiSAQ Core (No Labels)

**Branch**: `feat/aisaq`
**Status**: `- [x]` complete
**Effort**: 5-7 weeks
**Depends on**: Phase 3 (LUT virtuals), Phase 4 (Backend interface)
**Dependents**: Phase 6 (label integration), Phase 7 (tests + bench)

> **Maintenance note**: Update task markers (`- [ ]` -> `- [~]` -> `- [x]`) as you progress. At phase completion, update the phase status in [AISAQ.md](AISAQ.md) and update `AGENTS.md` per the [AGENTS.md Update Protocol](AISAQ.md#agentsmd-update-protocol).

---

## Goal

Implement the AiSAQ algorithm: Vamana graph + paged-PQ search, with PQ codes stored in DuckDB blocks managed by `BlockManager` (via `AiSaqBlockStore`), rather than in a contiguous in-RAM array. This is the defining trait that distinguishes AiSAQ from the existing `DISKANN` and enables indexing datasets larger than RAM.

This phase delivers a functional AiSAQ algorithm that supports `pq` and `scann` quantizers, multi-entry-point search, and the two-pass build (PQ encoding + graph construction). Label-column filtering is Phase 6.

## Background

### Why AiSAQ uses AiSaqBlockStore (not IndexBlockStore/FixedSizeAllocator)

The existing `DiskAnnCore` stores PQ codes in a contiguous in-RAM `vector<data_t> codes_` and graph nodes in `IndexBlockStore` (wrapping `FixedSizeAllocator`). `FixedSizeAllocator` pins every buffer permanently — the FIXME at `fixed_size_buffer.hpp:142` confirms DuckDB intended to fix this but never did.

AiSAQ cannot use `FixedSizeAllocator` because:
1. During build, both graph nodes and PQ codes must be evictable under memory pressure (100M vectors × 768 dim doesn't fit in 32 GB RAM).
2. `FixedSizeAllocator` can't evict dirty buffers (no public `Serialize()` or `Unpin()` API).
3. Patching DuckDB to add eviction was investigated and declined (see AISAQ.md decision #17).

`AiSaqBlockStore` wraps `BlockManager` directly. Each block is a persistent DuckDB block managed by the BufferPool. `BufferManager::Pin()` returns a `BufferHandle` (RAII); when the handle goes out of scope, the block is immediately eligible for eviction. The BufferPool handles writing evicted blocks to the DB file and reading them back on demand.

### Two-pass build

1. **PQ encoding pass** (write-only): stream vectors from the DuckDB table, encode each to a PQ code (m bytes), write directly to persistent page blocks via `BlockManager::Write()`. No BufferPool involvement — pure sequential writes to the DB file.

2. **Graph construction pass** (read-heavy): for each vector, online Vamana insert via BeamSearch. Reads existing PQ codes and graph nodes via `BufferManager::Pin()`. The BufferPool caches hot blocks (search frontier) and evicts cold ones under `memory_limit` pressure.

### Node layout (uint32_t neighbors)

Neighbors are stored as `uint32_t internal_ids` (4 bytes), not `BlockId`/`IndexPointer` (8 bytes). Dense arithmetic indexing resolves internal_id to a block + offset:

```
block_idx = internal_id / nodes_per_block
offset    = (internal_id % nodes_per_block) * node_size
block_id  = graph_block_handles_[block_idx]
```

Node layout:
```
[0]      int64_t  row_id              8
[8]      uint32_t internal_id         4  (redundant with array position but useful for debugging)
[12]     uint16_t neighbor_count      2
[14]     uint16_t inline_pq_count     2
[16]     uint32_t neighbors[R]        R × 4
[16+R*4] uint8_t  inline_pq[...]      (optional, default empty)
```

Node size for R=64, inline_pq=0: **272 bytes** (was 528 with 8-byte BlockId neighbors).
nodes_per_block: floor(Storage::BLOCK_SIZE / node_size) = floor(262144 / 272) = **963 nodes/block**.

Node size must be a multiple of 8 (see Cross-Phase Invariant #8). For R=64: 16 + 64×4 = 272 (already 8-aligned). For odd R, pad to next 8-byte boundary.

### Distance computation in search

For each neighbor encountered during beam search:

1. **Inline code** (first `inline_pq_count` neighbors): code bytes are inside the graph node block we already pinned. Zero additional I/O.
2. **Paged code** (all other neighbors): compute `page_idx = internal_id / codes_per_page`, `offset = (internal_id % codes_per_page) * code_size`. `BufferHandle handle = block_store_.PinPqPage(page_idx)`. Read code at offset. Compute distance. Handle goes out of scope → evictable.

With `inline_pq_count = 0` (the default), all neighbors use path 2.

## Scope

### In scope

- `AiSaqBlockStore`: block-granularity storage via `BlockManager` with native pin/unpin/eviction.
- `AiSaqCore`: Vamana graph algorithm + paged-PQ search loop.
- `AiSaqIndex`: `VectorIndex` subclass, TYPE_NAME = `"AISAQ"`.
- `PhysicalCreateAiSaqIndex`: parallel sink operator for `CREATE INDEX ... USING AISAQ`.
- Two-pass build: PQ encoding (direct write) + graph construction (BufferPool-cached).
- `AiSaqIndex::CreatePlan`: WITH-clause validation.
- `pragma_vindex_aisaq_index_info`: diagnostic table function.
- Registration in `vector_index_registry.cpp`.
- Persistence via block IDs in the state stream.
- Session options: `vindex_aisaq_beam_width`, `vindex_aisaq_io_limit`, `vindex_aisaq_l_search`.
- Routing batched math through `CpuBackend` (from Phase 4).

### Out of scope

- Label-column filtering (Phase 6).
- Recall bench thresholds (Phase 7).
- `combo_matrix.test` cells (Phase 7).

## Files

### New files

| Path | Purpose |
|---|---|
| `src/algo/aisaq/CMakeLists.txt` | Wires source files |
| `src/algo/aisaq/aisaq_module.hpp` | Declares `Register(ExtensionLoader&)` |
| `src/algo/aisaq/aisaq_register.cpp` | Trampoline: `RegisterIndex(db)` + `RegisterPragmas(loader)` |
| `src/algo/aisaq/aisaq_block_store.hpp` | `AiSaqBlockStore` declaration |
| `src/algo/aisaq/aisaq_block_store.cpp` | `AiSaqBlockStore` implementation |
| `src/algo/aisaq/aisaq_index.hpp` | `AiSaqIndex : VectorIndex` declaration |
| `src/algo/aisaq/aisaq_index.cpp` | `AiSaqIndex` implementation + `RegisterIndex` |
| `src/algo/aisaq/aisaq_core.hpp` | Internal: `AiSaqCore` declaration |
| `src/algo/aisaq/aisaq_core.cpp` | Vamana graph + paged-PQ search |
| `src/algo/aisaq/aisaq_build.hpp` | `PhysicalCreateAiSaqIndex` declaration |
| `src/algo/aisaq/aisaq_build.cpp` | Parallel sink operator (two-pass build) |
| `src/algo/aisaq/aisaq_plan.cpp` | `AiSaqIndex::CreatePlan` (WITH validation) |
| `src/algo/aisaq/aisaq_pragmas.cpp` | `pragma_vindex_aisaq_index_info` |
| `test/unit/test_aisaq_core.cpp` | Unit tests for AiSaqCore + AiSaqBlockStore |

### Changed files

| Path | Change |
|---|---|
| `src/common/vector_index_registry.cpp` | Add `#include "algo/aisaq/aisaq_module.hpp"` + `aisaq::Register(loader);` |
| `src/CMakeLists.txt` | Add `add_subdirectory(algo/aisaq)` (via `src/algo/CMakeLists.txt`) |

## API Surfaces

### `src/algo/aisaq/aisaq_block_store.hpp` (new)

```cpp
namespace duckdb {
namespace vindex {
namespace aisaq {

// AiSaqBlockStore — block-granularity storage backed by DuckDB's BlockManager.
// Each allocation is exactly one Storage::BLOCK_SIZE block. Pin returns a
// BufferHandle (RAII); unpin on destruction → block eligible for eviction.
// Blocks are persistent from creation (allocated via BlockManager::AllocateBlock).
class AiSaqBlockStore {
public:
	AiSaqBlockStore(BlockManager &block_manager, BufferManager &buffer_manager);

	// --- Graph node blocks ---
	// Dense packing: nodes_per_block consecutive nodes per block.
	// internal_id → (block_idx, offset) via arithmetic.

	idx_t RegisterGraphLayout(idx_t node_size);
	// Returns nodes_per_block. Must be called before AllocGraphNode.

	uint32_t AllocGraphNode();
	// Allocates a new internal_id. Grows graph_block_handles_ as needed.
	// Returns the new internal_id.

	BufferHandle PinGraphNode(uint32_t internal_id) const;
	// Returns a BufferHandle for the block containing this node.
	// Caller computes: ptr = handle.Ptr() + (internal_id % nodes_per_block) * node_size_

	void EnsureGraphCapacity(uint32_t up_to_internal_id);
	// Pre-allocates blocks to hold up_to_internal_id + 1 nodes.

	// --- PQ code page blocks ---
	// One page per block. codes_per_page = BLOCK_SIZE / code_size.

	void RegisterPqLayout(idx_t code_size);
	// Sets code_size_ and codes_per_page_.

	void WritePqPage(uint32_t page_idx, const_data_ptr_t data);
	// Direct write to a persistent block via BlockManager::Write().
	// Used during the PQ encoding pass (no BufferPool involvement).

	BufferHandle PinPqPage(uint32_t page_idx) const;
	// Returns a BufferHandle for the PQ page block.
	// Used during graph build and search (BufferPool-cached).

	void EnsurePqCapacity(uint32_t up_to_page_idx);
	// Pre-allocates blocks to hold up_to_page_idx + 1 pages.

	// --- Introspection ---

	idx_t GraphNodeCount() const { return graph_node_count_; }
	idx_t PqPageCount() const { return pq_page_handles_.size(); }
	idx_t NodesPerBlock() const { return nodes_per_block_; }
	idx_t CodesPerPage() const { return codes_per_page_; }
	idx_t NodeSize() const { return node_size_; }
	idx_t CodeSize() const { return code_size_; }

	// --- Persistence ---

	void SerializeState(vector<data_t> &out) const;
	// Serializes: node_size, code_size, graph_node_count, graph_block_ids[],
	//             pq_page_count, pq_block_ids[].

	void DeserializeState(const_data_ptr_t in, idx_t size);
	// Deserializes and registers all blocks with BlockManager.

private:
	BlockManager &block_manager_;
	BufferManager &buffer_manager_;

	// Graph nodes
	idx_t node_size_ = 0;
	idx_t nodes_per_block_ = 0;
	uint32_t graph_node_count_ = 0;
	vector<shared_ptr<BlockHandle>> graph_block_handles_;
	// Write buffer for sequential node writes during build.
	AllocatedData graph_write_buffer_;
	uint32_t graph_write_buffer_used_ = 0;

	// PQ code pages
	idx_t code_size_ = 0;
	idx_t codes_per_page_ = 0;
	vector<shared_ptr<BlockHandle>> pq_page_handles_;

	block_id_t AllocatePersistentBlock();
};

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
```

### `src/algo/aisaq/aisaq_core.hpp` (internal)

```cpp
struct AiSaqCoreParams {
	idx_t dim = 0;
	uint16_t R = 64;          // graph out-degree
	uint16_t L = 100;         // beam width (build + default search)
	float alpha = 1.2f;       // RobustPrune relaxation
	uint16_t inline_pq_count = 0;
	uint16_t beam_width = 8;  // search beam width
	uint16_t n_entry_points = 16;
	uint64_t seed = 0xA15A6ULL;
};

class AiSaqCore {
public:
	AiSaqCore(AiSaqCoreParams params, Quantizer &quantizer, AiSaqBlockStore &store);

	struct Candidate {
		int64_t row_id;
		float distance;
	};

	// Vamana online insert. Returns the new internal_id.
	uint32_t Insert(int64_t row_id, const float *vec);

	// Paged-PQ beam search. `query_lut` is the pre-populated LUT.
	vector<Candidate> Search(const float *query_lut, idx_t k, idx_t L_search,
	                         idx_t beam_width, idx_t io_limit) const;

	// Finalize inline PQ codes after all inserts.
	void FinalizeInlineCodes();

	// Compute entry points via k-means (delegates to Backend).
	void ComputeEntryPoints();

	// Serialize/deserialize core state (graph topology metadata, entry points).
	void SerializeState(vector<data_t> &out) const;
	void DeserializeState(const_data_ptr_t in, idx_t size);

	idx_t Size() const { return size_; }
	const AiSaqCoreParams &Params() const { return params_; }

	// Node layout helpers (offsets within a pinned graph node block).
	static constexpr idx_t kRowIdOffset = 0;
	static constexpr idx_t kInternalIdOffset = 8;
	static constexpr idx_t kNeighborCountOffset = 12;
	static constexpr idx_t kInlinePqCountOffset = 14;
	static constexpr idx_t kNeighborArrayOffset = 16;

	idx_t StaticNodeSize() const;

private:
	// Node field accessors (operate on data_ptr_t from a pinned block).
	static int64_t GetRowId(data_ptr_t node) {
		return LoadUnaligned<int64_t>(node + kRowIdOffset);
	}
	static void SetRowId(data_ptr_t node, int64_t val) {
		StoreUnaligned<int64_t>(node + kRowIdOffset, val);
	}
	static uint32_t GetNeighbor(data_ptr_t node, idx_t i) {
		return LoadUnaligned<uint32_t>(node + kNeighborArrayOffset + i * sizeof(uint32_t));
	}
	static void SetNeighbor(data_ptr_t node, idx_t i, uint32_t val) {
		StoreUnaligned<uint32_t>(node + kNeighborArrayOffset + i * sizeof(uint32_t), val);
	}
	// ... similar for neighbor_count, inline_pq_count, internal_id ...

	// PQ code access: returns a PqCodeRef holding the BufferHandle.
	struct PqCodeRef {
		BufferHandle handle;
		const_data_ptr_t ptr;
	};
	PqCodeRef ReadPqCode(uint32_t internal_id) const;

	// Beam search (Vamana greedy with paged-PQ distances).
	vector<Candidate> BeamSearch(const float *query_lut, idx_t L,
	                             idx_t beam_width, idx_t io_limit) const;

	// RobustPrune + ConnectAndPrune (duplicated from DiskAnnCore with
	// TODO(M6): extract to src/common/vamana.hpp).
	vector<Candidate> RobustPrune(const float *query_lut, vector<Candidate> candidates,
	                              idx_t R, float alpha) const;
	void ConnectAndPrune(uint32_t new_internal_id, const float *new_lut,
	                     const vector<Candidate> &selected);

	AiSaqCoreParams params_;
	Quantizer &quantizer_;
	AiSaqBlockStore &store_;
	idx_t code_size_ = 0;

	// Entry points (k-means centroids from build).
	struct EntryPoint {
		uint32_t internal_id;
		vector<uint8_t> code;  // inline copy of the PQ code
	};
	vector<EntryPoint> entry_points_;

	uint32_t size_ = 0;

	// Visit tracking (epoch-based, same as DiskAnnCore/HnswCore).
	mutable vector<uint32_t> visit_marks_;
	mutable uint32_t visit_counter_ = 0;
	mutable std::mt19937_64 rng_;
};
```

### `src/algo/aisaq/aisaq_index.hpp`

```cpp
class AiSaqIndex : public VectorIndex {
public:
	static constexpr const char *TYPE_NAME = "AISAQ";

	// ... constructor (mirrors DiskAnnIndex signature) ...

	static PhysicalOperator &CreatePlan(PlanIndexInput &input);

	// VectorIndex contract
	MetricKind GetMetricKind() const override;
	idx_t GetVectorSize() const override;
	idx_t GetRerankMultiple(ClientContext &context) const override;
	unique_ptr<IndexScanState> InitializeScan(float *query_vector, idx_t limit,
	                                          ClientContext &context) override;
	idx_t Scan(IndexScanState &state, Vector &result, idx_t result_offset = 0) override;
	// ... InitializeMultiScan, ExecuteMultiScan, etc. ...

	// Mutation
	void Construct(DataChunk &input, Vector &row_ids, idx_t thread_idx);
	void Compact() override;

	// Quantizer training
	void TrainQuantizer(ColumnDataCollection &collection, idx_t sample_cap = 65536);

private:
	unique_ptr<AiSaqBlockStore> block_store_;
	unique_ptr<Quantizer> quantizer_;
	unique_ptr<AiSaqCore> core_;
	MetricKind metric_ {MetricKind::L2SQ};
	idx_t dim_ = 0;
	AiSaqCoreParams params_ {};
	idx_t rerank_multiple_ = 1;
	// ... rwlock, tombstones_, etc. (mirrors DiskAnnIndex) ...
};
```

## Implementation Steps

### Step 1: Implement `AiSaqBlockStore`

- [ ] Create `aisaq_block_store.hpp` and `aisaq_block_store.cpp`.
- [ ] `AllocatePersistentBlock()`: call `block_manager_.AllocateBlock()`, register via `block_manager_.RegisterBlock(block_id)`, return block_id.
- [ ] `RegisterGraphLayout(node_size)`: compute `nodes_per_block_ = Storage::BLOCK_SIZE / node_size`. Assert `node_size % 8 == 0`.
- [ ] `EnsureGraphCapacity(internal_id)`: while `graph_block_handles_.size() * nodes_per_block_ <= internal_id`, allocate a new persistent block and append its handle.
- [ ] `AllocGraphNode()`: `EnsureGraphCapacity(graph_node_count_)`, return `graph_node_count_++`.
- [ ] `PinGraphNode(internal_id)`: compute `block_idx = internal_id / nodes_per_block_`, return `buffer_manager_.Pin(graph_block_handles_[block_idx])`.
- [ ] `RegisterPqLayout(code_size)`: compute `codes_per_page_ = Storage::BLOCK_SIZE / code_size`.
- [ ] `WritePqPage(page_idx, data)`: allocate block if needed, `block_manager_.Write(buffer, block_id)` directly (bypasses BufferPool).
- [ ] `PinPqPage(page_idx)`: return `buffer_manager_.Pin(pq_page_handles_[page_idx])`.
- [ ] `SerializeState` / `DeserializeState`: serialize block IDs + counts.

### Step 2: Implement `AiSaqCore` — node layout and PQ page access

- [ ] `StaticNodeSize()`: return `kNeighborArrayOffset + params_.R * sizeof(uint32_t)` padded to 8.
- [ ] `ReadPqCode(internal_id)`: `page_idx = internal_id / codes_per_page`, `offset = (internal_id % codes_per_page) * code_size_`. Return `{store_.PinPqPage(page_idx), handle.Ptr() + offset}`.
- [ ] Implement node field accessors using `LoadUnaligned` / `StoreUnaligned`.

### Step 3: Implement `AiSaqCore::Insert` — Vamana online insert

- [ ] Allocate a new graph node via `store_.AllocGraphNode()`.
- [ ] Pin the graph node block, write row_id, internal_id, neighbor_count=0, inline_pq_count.
- [ ] If not the first insert: run `BeamSearch` to find candidates, `RobustPrune`, `ConnectAndPrune`.
- [ ] Use `BufferHandle` for all block access — RAII ensures blocks are unpinned after each insert.

> **Note**: This is adapted from `DiskAnnCore::Insert` with the data plane changed from `codes_` array access to `ReadPqCode`. Add `// TODO(M6): extract Vamana primitives to src/common/vamana.hpp` at the top.

### Step 4: Implement `AiSaqCore::BeamSearch` — paged-PQ beam search

- [ ] Port the beam search loop from `DiskAnnCore::BeamSearch`.
- [ ] Replace `DistanceToCode(internal_id)` with `ReadPqCode(internal_id)` + `quantizer_.LUTDistance(code, query_lut)`.
- [ ] Each PQ code access: create a `PqCodeRef` (holds BufferHandle), compute distance, release. No stale pointer risk.
- [ ] Graph node access: pin block, read neighbor list, expand frontier, release.
- [ ] Track `io_count` and abort if `io_limit > 0 && io_count >= io_limit`.

### Step 5: Implement `AiSaqCore::RobustPrune` and `ConnectAndPrune`

- [ ] Port from `DiskAnnCore`. Replace `CodeDistance(a, b)` (which reads `codes_`) with `ReadPqCode` + `quantizer_.CodeDistance`.
- [ ] Add the `TODO(M6)` extraction comment.

### Step 6: Implement `AiSaqCore::FinalizeInlineCodes`

- [ ] Walk every node. For the first `inline_pq_count` neighbors: read their code from the PQ page, write it into the node's inline region.
- [ ] Called once after all inserts.

### Step 7: Implement `AiSaqCore::Search`

- [ ] Pick the best entry point: compute `LUTDistance(entry_point.code, query_lut)` for all entry points, pick the minimum.
- [ ] Call `BeamSearch(query_lut, L_search, beam_width, io_limit)`.
- [ ] Truncate to k candidates.
- [ ] Resolve `internal_id -> row_id` by pinning the graph node and reading `GetRowId`.

### Step 8: Implement entry-point computation

- [ ] At build finalize: run k-means (via `CpuBackend::ComputeEntryPoints`) with `k = n_entry_points` on a sample of the vectors.
- [ ] For each centroid: find the nearest `internal_id` in the index, store it as an `EntryPoint` with its PQ code copied inline.

### Step 9: Implement two-pass build in `PhysicalCreateAiSaqIndex`

- [ ] Mirror `PhysicalCreateDiskAnnIndex` from `src/algo/diskann/diskann_build.{cpp,hpp}`.
- [ ] **Pass 1 (PQ encoding)**: sink rows into `ColumnDataCollection`, train quantizer, then stream vectors: encode each to PQ code, write to page blocks via `block_store_.WritePqPage()`.
- [ ] **Pass 2 (graph construction)**: for each vector, preprocess query, `core_->Insert(row_id, vec)`. The BufferPool caches hot blocks.
- [ ] Construction is single-threaded through the index rwlock (same as DiskANN).

### Step 10: Implement `AiSaqIndex` — lifecycle

- [ ] Constructor: parse options, create quantizer (reject flat/rabitq), create `AiSaqBlockStore`, create `AiSaqCore`.
- [ ] `Construct`: per-row `core_->Insert(row_id, vec)`, take `rwlock.GetExclusiveLock()`.
- [ ] `TrainQuantizer`: sample up to 65536 vectors, call `quantizer_->Train`.
- [ ] `InitializeScan` / `Scan`: resolve L_search from `vindex_aisaq_l_search`, preprocess query, populate LUT, call `core_->Search`, filter tombstones.

### Step 11: Implement `AiSaqIndex::CreatePlan` — WITH validation

- [ ] Accept and validate all options listed in [Locked Decisions](AISAQ.md#locked-decisions):
  - `metric`, `quantizer`, `m`, `bits`, `eta`, `rerank`
  - `aisaq_r` (default 64, range [4, 256])
  - `aisaq_l` (default 100, range [4, 1024])
  - `aisaq_alpha` (default 1.2, range [1.0, 2.0])
  - `aisaq_inline_pq` (default 0, range [0, aisaq_r])
  - `aisaq_beam_width` (default 8)
  - `aisaq_io_limit` (default 0)
  - `aisaq_entry_points` (default 16, range [1, 64])
- [ ] Validate exactly one expression of type `FLOAT[N]`.
- [ ] Gate persistent DB creation on `vindex_enable_experimental_persistence`.
- [ ] Assemble `Projection -> Filter(IS NOT NULL) -> PhysicalCreateAiSaqIndex` pipeline.

### Step 12: Implement persistence (block ID stream)

- [ ] `AiSaqBlockStore::SerializeState`: write node_size, code_size, counts, and all block IDs.
- [ ] `AiSaqBlockStore::DeserializeState`: read counts and block IDs, register each with `block_manager_.RegisterBlock(block_id)`.
- [ ] `AiSaqIndex::SerializeToDisk` / `SerializeToWAL`: mirror DiskANN's pattern; store the block store state + core state + quantizer blob.

### Step 13: Implement registration

- [ ] In `aisaq_index.cpp::RegisterIndex`:
  - Build `IndexType` with `name = "AISAQ"`, `create_instance` lambda, `create_plan = AiSaqIndex::CreatePlan`.
  - Register session options: `vindex_aisaq_beam_width`, `vindex_aisaq_io_limit`, `vindex_aisaq_l_search`.
  - `VectorIndexRegistry::Instance().RegisterTypeName("AISAQ")`.
- [ ] In `src/common/vector_index_registry.cpp`:
  - Add `#include "algo/aisaq/aisaq_module.hpp"`.
  - Add `aisaq::Register(loader);` after `diskann::Register(loader);`.

### Step 14: Implement `pragma_vindex_aisaq_index_info`

- [ ] Mirror `pragma_vindex_diskann_index_info`.
- [ ] Columns: index_name, table_name, quantizer, metric, dim, count, R, L, alpha, inline_pq, beam_width, n_entry_points, n_graph_blocks, n_pq_pages, nodes_per_block, codes_per_page.

### Step 15: Write basic unit tests

- [ ] Create `test/unit/test_aisaq_core.cpp`.
- [ ] Test cases:
  - [ ] `BuildAndSearch`: build a small index (100 vectors, dim=128, PQ m=16), search k=10, verify results are reasonable.
  - [ ] `IoLimitCap`: set `io_limit=5`, verify the search pins at most 5 PQ page blocks.
  - [ ] `InlineVsPaged`: build with `inline_pq_count=0` and `inline_pq_count=R`, verify recall is at least as good with inline.
  - [ ] `BlockStoreRoundTrip`: write PQ pages, read them back via Pin, verify data integrity.
  - [ ] `PersistenceRoundTrip`: build, serialize block IDs, deserialize, search — verify identical results.

### Step 16: Update `AGENTS.md`

- [ ] Add `src/algo/aisaq/` + `AiSaqBlockStore` to the source wiring diagram.
- [ ] Add a new "AiSAQ algorithm" subsection.
- [ ] Document the `AISAQ` index type, its options, and the `aisaq_inline_pq` default of 0.
- [ ] Note that AiSAQ uses `BlockManager` directly, not `IndexBlockStore`.
- [ ] Add AiSAQ-specific commands to the "Commands" table.

## Acceptance Criteria

- [ ] `CREATE INDEX ... USING AISAQ` works with `quantizer='pq'` and `quantizer='scann'`.
- [ ] `quantizer='flat'` and `quantizer='rabitq'` are rejected with clear errors.
- [ ] `ORDER BY array_distance(vec, [...]::FLOAT[N]) LIMIT k` produces correct results.
- [ ] `EXPLAIN` shows `VINDEX_INDEX_SCAN` in the plan.
- [ ] Two-pass build works: PQ encoding writes directly to blocks, graph construction reads via BufferPool.
- [ ] `aisaq_inline_pq = 0` (default) works; `aisaq_inline_pq = N` works.
- [ ] Entry points are computed via k-means and used at search time.
- [ ] Persistence round-trips through checkpoint and WAL (block IDs in state stream).
- [ ] `pragma_vindex_aisaq_index_info` returns correct metadata.
- [ ] Unit tests pass.
- [ ] `AGENTS.md` is updated.
- [ ] Phase status updated to `- [x]` in [AISAQ.md](AISAQ.md).

## Open Questions

- **Q1**: Should `WritePqPage` use `BlockManager::Write` (direct, bypasses BufferPool) or `BufferManager::Pin` + write + `Unpin` (BufferPool-managed)? **Proposed**: `BlockManager::Write` for the encoding pass (write-only, no caching needed), `BufferManager::Pin` for graph build and search (read caching). Verify `BlockManager::Write` is accessible from extension code.
- **Q2**: Should the graph construction pass pre-load PQ codes into the BufferPool before starting (to warm the cache), or let the BufferPool load pages on demand? **Proposed**: on-demand — the BufferPool's LRU handles the working set naturally.
- **Q3**: Should the `vindex_gpu_backend` session option be read at query time (per-query backend) or at index-construction time (per-index backend)? **Proposed**: per-query, so the same index can be searched with different backends.
