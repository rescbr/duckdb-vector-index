#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"

#include "algo/aisaq/aisaq_block_store.hpp"
#include "vindex/label_filter.hpp"
#include "vindex/quantizer.hpp"
#include "vindex/unaligned.hpp"

#include <cstdint>
#include <random>

namespace duckdb {
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
	uint64_t seed = 0xA15A6ULL;
};

class AiSaqCore {
  public:
	AiSaqCore(AiSaqCoreParams params, Quantizer &quantizer, AiSaqBlockStore &store);

	struct Candidate {
		int64_t row_id;
		float distance;
	};

	// Vamana online insert. The PQ code for this internal_id must already be
	// present in the block store (written by the PQ encoding pass). Returns
	// the allocated internal_id. label = INT64_MIN means "no label".
	uint32_t Insert(int64_t row_id, const float *vec, int64_t label = INT64_MIN);

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
	// node's inline region. No-op when inline_pq_count == 0.
	void FinalizeInlineCodes();

	// Populate entry_points_ with up to n_entry_points spread-out nodes whose
	// PQ codes are cached inline for the entry-point distance probe at search.
	void ComputeEntryPoints();

	// Compute per-label medoids after all inserts. For each label, finds the
	// member whose PQ code is closest to the label's centroid (approximated
	// by averaging PQ codes). Also builds sorted_labels_ for range queries.
	void ComputeLabelMedoids();

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
		auto handle = store_.PinGraphNode(internal_id);
		const idx_t off = (idx_t(internal_id) % store_.NodesPerBlock()) * store_.NodeSize();
		// Capture the pointer BEFORE moving the handle out — Ptr() on a
		// moved-from BufferHandle trips the IsValid() assertion.
		data_ptr_t ptr = handle.Ptr() + off;
		return {std::move(handle), ptr};
	}

	// Pick the best entry point for a query LUT.
	uint32_t PickEntryPoint(const float *query_lut) const;

	// Estimate distance from the query LUT to a node's code (pages in the code).
	float DistanceToCode(const float *query_lut, uint32_t internal_id) const {
		auto ref = ReadPqCode(internal_id);
		return quantizer_.LUTDistance(ref.ptr, query_lut);
	}

	// Code-to-code distance (pags both codes in).
	float CodeDistance(uint32_t a, uint32_t b) const {
		auto ra = ReadPqCode(a);
		auto rb = ReadPqCode(b);
		return quantizer_.CodeDistance(ra.ptr, rb.ptr);
	}

	// Vamana greedy beam search. Returns up to `L` nearest candidates
	// (ascending distance). Candidates carry internal_id in the row_id field;
	// the caller resolves to table row_id via PinNode + GetRowId.
	//
	// forced_entry_points: when non-null and non-empty, use these as start
	// points instead of PickEntryPoint (for label-mediated multi-start).
	// label_filter: when non-null, skip neighbors whose label doesn't match.
	vector<Candidate> BeamSearch(const float *query_lut, idx_t L, idx_t io_limit,
	                             const vector<uint32_t> *forced_entry_points = nullptr,
	                             const LabelFilter *label_filter = nullptr) const;

	vector<Candidate> RobustPrune(const float *query_lut, vector<Candidate> candidates, idx_t R, float alpha) const;

	void ConnectAndPrune(uint32_t new_internal_id, const float *new_lut, const vector<Candidate> &selected);

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

	// Label bookkeeping (populated when labels are provided at insert time).
	// internal_id → label (for candidate filtering during search).
	unordered_map<uint32_t, int64_t> internal_id_to_label_;
	// label → list of internal_ids belonging to that label.
	unordered_map<int64_t, vector<uint32_t>> label_to_internal_ids_;
	// label → medoid internal_id (node closest to the label's centroid).
	unordered_map<int64_t, uint32_t> label_medoids_;
	// Sorted label keys for efficient range queries (lower_bound/upper_bound).
	vector<int64_t> sorted_labels_;

	mutable vector<uint32_t> visit_marks_;
	mutable uint32_t visit_counter_ = 0;
	mutable std::mt19937_64 rng_;
};

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
