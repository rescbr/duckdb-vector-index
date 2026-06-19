#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"

#include "algo/aisaq/aisaq_block_store.hpp"
#include "vindex/label_filter.hpp"
#include "vindex/metric.hpp"
#include "vindex/quantizer.hpp"
#include "vindex/unaligned.hpp"
#include "vindex/vamana.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>

namespace duckdb {
class DatabaseInstance;
namespace vindex {
namespace aisaq {

// ---------------------------------------------------------------------------
// AiSaqCore — single-layer Vamana graph whose PQ codes are paged from DuckDB
// blocks on demand (the defining AiSAQ trait). Mirrors DiskAnnCore's algorithm
// but replaces the contiguous in-RAM `codes_` array with on-demand page reads
// through AiSaqBlockStore, and stores neighbor references as dense uint32_t
// internal_ids (4 bytes) rather than 8-byte BlockId/IndexPointers.
//
// TODO(M6): extract the shared Vamana primitives (BeamSearch / RobustPrune /
// ConnectAndPrune) into src/common/vamana.hpp — DiskAnnCore and AiSaqCore
// duplicate the algorithm with only the data-plane access swapped.
//
// Algorithmic reference: Suhas Jayaram Subramanya et al., "DiskANN: Fast
// Accurate Billion-point Nearest Neighbor Search on a Single Node", NeurIPS
// 2019 — the AiSAQ variant pages PQ codes from BlockManager-backed blocks
// (SSD-resident) instead of pinning them in DRAM.
// ---------------------------------------------------------------------------

struct AiSaqCoreParams {
	idx_t dim = 0;
	uint16_t R = 64;              // graph out-degree
	uint16_t L = 100;             // beam width (build + default search)
	float alpha = 1.2f;           // RobustPrune relaxation
	uint16_t inline_pq_count = 0; // codes inlined into the node for the first N neighbors
	uint16_t beam_width = 8;      // search beam width (I/O batching hint)
	uint16_t n_entry_points = 16;
	uint16_t L_build = 0; // 0 = use R (Vamana paper recommendation)
	uint64_t seed = 0xA15A6ULL;
};

// ---------------------------------------------------------------------------
// NodeSpinlock — per-node spinlock for parallel ConnectAndPrune (Phase 9
// Task 5.5).
//
// Phase 9 Task 5's deferred-reciprocity + global rwlock strategy capped CPU
// at ~250% on a 10-core host because every periodic apply serialized through
// one global lock. Task 5.5 replaces it with per-node spinlocks: each task
// acquires the target node's lock individually when writing the back-edge to
// that node's neighbor list, while the forward-edge write to the new node is
// lock-free (each new_internal_id is exclusive to one task). No global lock
// is held on the parallel insert path; the only serialization left is the
// per-node critical section, which is sub-microsecond.
//
// Spinlock (not std::mutex): the critical section is append-or-re-prune of a
// ≤R-neighbor list. std::mutex would syscall on every acquisition; PAUSE
// (x86) / yield (arm64) keeps the contended thread on-core. Same pattern as
// hnswlib (`link_list_locks_`, hnswalg.h:3142) and DiskANN (`inter_insert`,
// index_build.cpp) — both production Vamana/HNSW implementations.
//
// 4 bytes per node → 4 MB for sift1m. Stored in AiSaqCore::node_locks_,
// separate from build_nodes_ (which evicts via BufferManager in paged mode)
// so lock storage has a stable address across the build.
//
// TTAS (test-and-test-and-set) pattern: the spinning thread reloads the flag
// with relaxed ordering before each CAS retry, so it doesn't re-dirty the
// cache line on every iteration. The CAS itself uses acquire-on-success /
// relaxed-on-failure; unlock uses release. This guarantees that everything
// done inside the critical section is visible to the next acquirer.
//
// Deadlock safety: at most ONE lock is held at a time per task (the loop in
// ConnectAndPrune acquires, writes, releases per neighbor). With no nested
// lock acquisition, deadlock is impossible regardless of acquisition order.
// ---------------------------------------------------------------------------
struct NodeSpinlock {
	std::atomic<uint32_t> flag{0};

	void lock() {
		for (;;) {
			uint32_t expected = 0;
			if (flag.compare_exchange_weak(expected, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
				return;
			}
			// Spin-wait for the holder to release before retrying, to reduce
			// cache-line bouncing under contention.
			while (flag.load(std::memory_order_relaxed) != 0) {
#if defined(__x86_64__) || defined(__i386__)
				__builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
				__builtin_arm_yield();
#else
				std::this_thread::yield();
#endif
			}
		}
	}

	void unlock() {
		flag.store(0, std::memory_order_release);
	}
};

class AiSaqCore {
  public:
	AiSaqCore(AiSaqCoreParams params, Quantizer &quantizer, AiSaqBlockStore &store);

	struct Candidate {
		int64_t row_id;
		float distance;
	};

	// Vamana online insert (serial / test path). The PQ code for this
	// internal_id must already be present in the block store (written by the
	// PQ encoding pass). Returns the allocated internal_id.
	// label = INT64_MIN means "no label". Uses the core's own `tls_` member
	// and the global label maps — single-threaded only. For the parallel
	// build path, use InsertBuild.
	uint32_t Insert(int64_t row_id, const float *vec, int64_t label = INT64_MIN);

	// Vamana online insert with explicit per-thread state (Phase 9 Task 5
	// parallel build path). The caller supplies:
	//   - internal_id: must equal the row's PQ slot index (Pass 1 wrote the
	//     code there). Allocation is the caller's responsibility — typically
	//     the build event pre-reserves [0, N) via AllocGraphNodeRange so
	//     internal_id = row_ordinal deterministically.
	//   - tls: per-thread VamanaTLS scratch. Must outlive the call.
	//   - local_id2label / local_label2ids: per-task label maps. Merged into
	//     the core's global maps serially by the caller; BeamSearch during
	//     build never reads label maps (label_filter is null), so per-task
	//     isolation is sound.
	//
	// The FIRST call (size_ == 0) claims the entry point. The caller must
	// serialize the first InsertBuild across all tasks (typically the build
	// event runs it in Schedule() before spawning tasks) — concurrent first
	// calls would race on entry_internal_ / size_. Subsequent calls are
	// safe to run concurrently across tasks on disjoint internal_id ranges.
	// Does NOT bump size_ — the caller sets size_ = N at the end.
	//
	// Reciprocal edges are applied inline via per-node spinlocks in
	// ConnectAndPrune (Phase 9 Task 5.5). The caller MUST have pre-sized
	// node_locks_ to N via ResizeNodeLocks before spawning tasks.
	uint32_t InsertBuild(uint32_t internal_id, int64_t row_id, const float *vec, int64_t label, vamana::VamanaTLS &tls,
	                     unordered_map<uint32_t, int64_t> &local_id2label,
	                     unordered_map<int64_t, vector<uint32_t>> &local_label2ids);

	// Merge per-task label maps into the core's global maps. Called serially
	// from the build event's FinishEvent after all construct tasks finish.
	// Safe to call multiple times (e.g., once per task's local maps).
	void MergeLabelMaps(const unordered_map<uint32_t, int64_t> &id2label,
	                    const unordered_map<int64_t, vector<uint32_t>> &label2ids);

	// Set the final node count after parallel construction. Tasks don't bump
	// size_ during InsertBuild (that would race); the build event sets it
	// once after all tasks finish + deferred reciprocity is applied.
	void SetSize(uint32_t size) {
		size_ = size;
	}

	// Paged-PQ approximate kNN. `query_lut` is a pre-populated LUT
	// (Quantizer::PopulateDistanceLUT). L_search overrides params_.L when > 0.
	// beam_width / io_limit bound the I/O fan-out; io_limit == 0 means unbounded.
	//
	// When label_filter is active:
	//   EQUALS(value):
	//     Entry point = label_medoids_[value] (single targeted start).
	//     Candidates with label != value are skipped during graph expansion.
	//
	//   RANGE(lo, hi):
	//     Count matching labels N = |{l : lo <= l <= hi}|.
	//     If N <= params_.n_entry_points:
	//       Entry points = {label_medoids_[l] for all matching l}.
	//     Else:
	//       Entry points = global k-means entry points (ComputeEntryPoints).
	//     Candidates with label outside [lo, hi] are skipped during expansion.
	//
	//   NONE:
	//     Global entry points; no candidate filtering.
	vector<Candidate> Search(const float *query_lut, idx_t k, idx_t L_search, idx_t beam_width, idx_t io_limit,
	                         const LabelFilter &label_filter = LabelFilter::None()) const;

	// Copy the PQ codes of the first `inline_pq_count` neighbors into each
	// node's inline region. No-op when inline_pq_count == 0. Parallelised
	// across TaskScheduler::NumberOfThreads() tasks with disjoint id ranges
	// when SetDatabase has been called; otherwise runs serially.
	void FinalizeInlineCodes();

	// Populate entry_points_ with up to n_entry_points spread-out nodes whose
	// PQ codes are cached inline for the entry-point distance probe at search.
	// Output is bit-identical to the serial path; the loop itself stays serial
	// (bounded by n_entry_points ≤ 64 — TaskExecutor overhead exceeds the
	// work) but entry_points_ is pre-sized so the per-iteration write is
	// concurrency-safe regardless of caller threading.
	void ComputeEntryPoints();

	// Compute per-label medoids after all inserts. For each label, finds the
	// member whose PQ code is closest to the label's centroid (approximated
	// by averaging PQ codes). Also builds sorted_labels_ for range queries.
	// Parallelised across disjoint label partitions with per-task local maps
	// merged serially at the end (unordered_map concurrent writes are unsafe).
	void ComputeLabelMedoids();

	// Inject the DatabaseInstance used to spawn parallel post-build tasks.
	// When unset (nullptr), FinalizeInlineCodes / ComputeLabelMedoids run
	// serially — used by unit tests that build AiSaqCore without a full
	// DatabaseInstance.
	void SetDatabase(DatabaseInstance *db) {
		db_ = db;
	}

	// Count labels in [lo, hi] via binary search on sorted_labels_.
	idx_t CountLabelsInRange(int64_t lo, int64_t hi) const;

	// Collect medoids for labels in [lo, hi], up to max_count.
	// Returns empty if the count exceeds max_count (caller should fall back
	// to global entry points).
	vector<uint32_t> GetMedoidsInRange(int64_t lo, int64_t hi, idx_t max_count) const;

	// Whether any labels have been registered.
	bool HasLabels() const {
		return !label_to_internal_ids_.empty();
	}

	// --- build-time acceleration buffers -------------------------------
	// Tier 2: flat PQ code buffer. When active, CodeDistance and
	// DistanceToCode index directly into this buffer instead of paging
	// through BufferManager.
	void SetBuildCodes(const uint8_t *codes, idx_t count) {
		build_codes_ = codes;
		build_codes_count_ = count;
	}
	// Tier 3: flat full-precision vectors. When active, RobustPrune uses
	// exact distances for topology decisions (mirrors ref/aisaq-diskann).
	void SetBuildVectors(const float *vecs, idx_t dim, MetricKind metric) {
		build_vectors_ = vecs;
		build_metric_ = metric;
	}
	// Tier 2: flat graph node buffer. When active, PinNode returns a pointer
	// into this buffer instead of going through BufferManager. The buffer is
	// flushed to AiSaqBlockStore after construction completes.
	void SetBuildNodes(data_ptr_t nodes) {
		build_nodes_ = nodes;
	}
	void ClearBuildBuffers() {
		build_codes_ = nullptr;
		build_codes_count_ = 0;
		build_vectors_ = nullptr;
		build_nodes_ = nullptr;
	}
	bool HasBuildCodes() const {
		return build_codes_ != nullptr;
	}
	// Pre-size internal buffers for construction (avoids repeated realloc).
	void PrepareForBuild(idx_t estimated_count) {
		tls_.PrepareForBuild(estimated_count);
	}

	// Pre-size the per-node spinlock array. MUST be called once,
	// single-threaded, before any parallel InsertBuild task spawns (the
	// parallel build path in aisaq_build.cpp calls this from Schedule()). The
	// array is indexed by internal_id and must not be reallocated while tasks
	// hold references to its elements.
	//
	// Also used by the serial Insert() path to lazily grow the array past the
	// build-time size for live inserts (no concurrent threads there, so
	// reallocating is safe).
	void ResizeNodeLocks(idx_t count);

	// Per-task workers for parallel post-build phases (Phase 9 Task 3).
	// Public so the AiSaqFinalizeInlineTask / AiSaqComputeMedoidsTask classes
	// defined in the .cpp can invoke them without friend declarations —
	// mirrors the AiSaqIndex::EncodePqRange pattern from Task 2.
	// FinalizeInlineCodesRange: processes nodes [id_start, id_end) — writes
	// are disjoint across tasks because each task touches a distinct id range
	// and only writes to its own nodes' inline regions.
	void FinalizeInlineCodesRange(idx_t id_start, idx_t id_end) const;
	// ComputeLabelMedoidsRange: processes labels [start, end) of `labels`,
	// writing results into the caller-supplied local map (per-task). Reads
	// from label_to_internal_ids_ are const; writes go to the disjoint local
	// map. Merged into label_medoids_ serially after all tasks complete.
	void ComputeLabelMedoidsRange(const vector<int64_t> &labels, idx_t start, idx_t end,
	                              unordered_map<int64_t, uint32_t> &local_out) const;

	idx_t Size() const {
		return size_;
	}
	const AiSaqCoreParams &Params() const {
		return params_;
	}

	// --- serialization -----------------------------------------------------
	void SerializeState(vector<data_t> &out) const;
	void DeserializeState(const_data_ptr_t in, idx_t size);

	// Node layout offsets within a pinned graph-node block.
	static constexpr idx_t kRowIdOffset = 0;
	static constexpr idx_t kInternalIdOffset = 8;
	static constexpr idx_t kNeighborCountOffset = 12;
	static constexpr idx_t kInlinePqCountOffset = 14;
	static constexpr idx_t kNeighborArrayOffset = 16;

	idx_t StaticNodeSize() const;

  private:
	// Node field accessors — operate on a pointer into a *pinned* graph block.
	static int64_t GetRowId(data_ptr_t node) {
		return LoadUnaligned<int64_t>(node + kRowIdOffset);
	}
	static void SetRowId(data_ptr_t node, int64_t val) {
		StoreUnaligned<int64_t>(node + kRowIdOffset, val);
	}
	static uint32_t GetInternalId(data_ptr_t node) {
		return LoadUnaligned<uint32_t>(node + kInternalIdOffset);
	}
	static void SetInternalId(data_ptr_t node, uint32_t val) {
		StoreUnaligned<uint32_t>(node + kInternalIdOffset, val);
	}
	static uint16_t GetNeighborCount(data_ptr_t node) {
		return LoadUnaligned<uint16_t>(node + kNeighborCountOffset);
	}
	static void SetNeighborCount(data_ptr_t node, uint16_t val) {
		StoreUnaligned<uint16_t>(node + kNeighborCountOffset, val);
	}
	static uint16_t GetInlinePqCount(data_ptr_t node) {
		return LoadUnaligned<uint16_t>(node + kInlinePqCountOffset);
	}
	static void SetInlinePqCount(data_ptr_t node, uint16_t val) {
		StoreUnaligned<uint16_t>(node + kInlinePqCountOffset, val);
	}
	static uint32_t GetNeighbor(data_ptr_t node, idx_t i) {
		return LoadUnaligned<uint32_t>(node + kNeighborArrayOffset + i * sizeof(uint32_t));
	}
	static void SetNeighbor(data_ptr_t node, idx_t i, uint32_t val) {
		StoreUnaligned<uint32_t>(node + kNeighborArrayOffset + i * sizeof(uint32_t), val);
	}

	// RAII PQ code reference — keeps the page pinned while the code is in use.
	struct PqCodeRef {
		BufferHandle handle;
		const_data_ptr_t ptr;
	};
	PqCodeRef ReadPqCode(uint32_t internal_id) const;

	// Resolve an internal_id to a pointer inside a freshly pinned graph block.
	// Caller must keep the returned BufferHandle alive while using the pointer.
	struct NodeRef {
		BufferHandle handle;
		data_ptr_t ptr;
	};
	NodeRef PinNode(uint32_t internal_id) const {
		if (build_nodes_) {
			return {BufferHandle(), build_nodes_ + idx_t(internal_id) * store_.NodeSize()};
		}
		auto handle = store_.PinGraphNode(internal_id);
		const idx_t off = (idx_t(internal_id) % store_.NodesPerBlock()) * store_.NodeSize();
		// Capture the pointer BEFORE moving the handle out — Ptr() on a
		// moved-from BufferHandle trips the IsValid() assertion.
		data_ptr_t ptr = handle.Ptr() + off;
		return {std::move(handle), ptr};
	}

	// Pick the best entry point for a query LUT.
	uint32_t PickEntryPoint(const float *query_lut) const;

	// Estimate distance from the query LUT to a node's code. Uses flat
	// build buffer (Tier 2) when available, else pages via BufferManager.
	float DistanceToCode(const float *query_lut, uint32_t internal_id) const {
		if (build_codes_) {
			return quantizer_.LUTDistance(build_codes_ + idx_t(internal_id) * code_size_, query_lut);
		}
		auto ref = ReadPqCode(internal_id);
		return quantizer_.LUTDistance(ref.ptr, query_lut);
	}

	// Code-to-code distance. Tier 3 uses full-precision vectors; Tier 2
	// uses flat PQ buffer; Tier 1 pages via BufferManager (callers should
	// use per-prune gather pattern via CopyBuildCode).
	float CodeDistance(uint32_t a, uint32_t b) const {
		if (build_vectors_) {
			return RawDistance(build_vectors_ + idx_t(a) * params_.dim, build_vectors_ + idx_t(b) * params_.dim);
		}
		if (build_codes_) {
			return quantizer_.CodeDistance(build_codes_ + idx_t(a) * code_size_, build_codes_ + idx_t(b) * code_size_);
		}
		auto ra = ReadPqCode(a);
		auto rb = ReadPqCode(b);
		return quantizer_.CodeDistance(ra.ptr, rb.ptr);
	}

	// Copy the PQ code for internal_id into dst. Uses flat buffer when
	// available (Tier 2), else reads from block store (1 BufferManager Pin).
	// Used by the per-prune gather pattern (Tier 1).
	void CopyBuildCode(uint32_t internal_id, uint8_t *dst) const {
		if (build_codes_) {
			std::memcpy(dst, build_codes_ + idx_t(internal_id) * code_size_, code_size_);
		} else {
			auto ref = ReadPqCode(internal_id);
			std::memcpy(dst, ref.ptr, code_size_);
		}
	}

	// Full-precision distance for Tier 3 prune.
	float RawDistance(const float *a, const float *b) const;

	// Vamana greedy beam search. Returns up to `L` nearest candidates
	// (ascending distance). Candidates carry internal_id in the row_id field;
	// the caller resolves to table row_id via PinNode + GetRowId.
	//
	// forced_entry_points: when non-null and non-empty, use these as start
	// points instead of PickEntryPoint (for label-mediated multi-start).
	// label_filter: when non-null, skip neighbors whose label doesn't match.
	//
	// `tls` is per-thread scratch (visit marks + epoch). Single-threaded
	// callers pass `tls_` (the core member); parallel build-path callers
	// pass their own per-task VamanaTLS&.
	vector<Candidate> BeamSearch(const float *query_lut, idx_t L, idx_t io_limit, vamana::VamanaTLS &tls,
	                             const vector<uint32_t> *forced_entry_points = nullptr,
	                             const LabelFilter *label_filter = nullptr) const;

	vector<Candidate> RobustPrune(const float *query_lut, vector<Candidate> candidates, idx_t R, float alpha,
	                              vamana::VamanaTLS &tls) const;

	// Write forward edges (new_internal_id -> each selected neighbor) and
	// apply reciprocal edges inline. Each target neighbor's per-node spinlock
	// is acquired individually while its back-edge is appended (or its
	// neighbor list is re-pruned on overflow). At most one lock is held at a
	// time per task — deadlock-free.
	void ConnectAndPrune(uint32_t new_internal_id, const float *new_lut, const vector<Candidate> &selected,
	                     vamana::VamanaTLS &tls);

	AiSaqCoreParams params_;
	Quantizer &quantizer_;
	AiSaqBlockStore &store_;
	idx_t code_size_ = 0;

	// Entry points populated by ComputeEntryPoints (used at search time).
	struct EntryPoint {
		uint32_t internal_id = 0;
		vector<uint8_t> code; // inline copy of the PQ code
	};
	vector<EntryPoint> entry_points_;
	uint32_t entry_internal_ = 0; // build-time entry (= first inserted node)

	uint32_t size_ = 0;

	// Build-time acceleration buffers (non-owning; owned by AiSaqIndex).
	// When non-null, code access skips BufferManager entirely.
	const uint8_t *build_codes_ = nullptr;
	idx_t build_codes_count_ = 0;
	const float *build_vectors_ = nullptr;
	MetricKind build_metric_ = MetricKind::L2SQ;
	data_ptr_t build_nodes_ = nullptr;

	// Label bookkeeping (populated when labels are provided at insert time).
	// internal_id → label (for candidate filtering during search).
	unordered_map<uint32_t, int64_t> internal_id_to_label_;
	// label → list of internal_ids belonging to that label.
	unordered_map<int64_t, vector<uint32_t>> label_to_internal_ids_;
	// label → medoid internal_id (node closest to the label's centroid).
	unordered_map<int64_t, uint32_t> label_medoids_;
	// Sorted label keys for efficient range queries (lower_bound/upper_bound).
	vector<int64_t> sorted_labels_;

	// Per-call Vamana scratch (visit-mark table, prune gather buffer, RNG).
	// Marked mutable because BeamSearch / RobustPrune are const but bump the
	// epoch and reuse the scratch. Used by:
	//   - Search() — public query path, single-threaded per call.
	//   - Insert() — public serial test path; passes tls_ into InsertBuild.
	// The parallel build path (InsertBuild) does NOT touch this member —
	// each task owns its own VamanaTLS and passes it in explicitly.
	mutable vamana::VamanaTLS tls_;

	// Per-node spinlocks for parallel ConnectAndPrune (Phase 9 Task 5.5).
	// Sized to N before parallel tasks spawn via ResizeNodeLocks(N); grown
	// lazily by Insert() for the serial live path. Indexed by internal_id.
	// Stored as unique_ptr<T[]> (not vector<T>) because std::atomic is
	// non-copyable AND non-movable — vector resize/insert would fail to
	// compile, while unique_ptr<T[]>.reset(new T[n]) default-constructs in
	// place via `new T[n]`.
	unique_ptr<NodeSpinlock[]> node_locks_;
	idx_t node_locks_count_ = 0;

	// Database instance for TaskScheduler access. When null, the post-build
	// phases fall back to the serial path (used by unit tests that don't
	// stand up a full DuckDB instance).
	DatabaseInstance *db_ = nullptr;
};

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
