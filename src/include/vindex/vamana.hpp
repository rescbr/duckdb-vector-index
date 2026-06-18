#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/vector.hpp"

#include <cstdint>
#include <random>

namespace duckdb {
namespace vindex {
namespace vamana {

// ---------------------------------------------------------------------------
// VamanaTLS — per-thread scratch for the Vamana build/search primitives.
//
// Each thread doing BeamSearch / RobustPrune / ConnectAndPrune MUST have its
// own instance. Today (Task 1) the AiSAQ and DiskANN core classes each hold
// one mutable `tls_` member and route their existing per-call scratch
// through it. This preserves bit-identical behavior while paying down the
// field-level duplication called out by the TODO(M6) at aisaq_core.hpp:29.
//
// In Task 5 (parallel Construct), `tls_` is lifted out of the core classes
// into a per-thread parameter — which is what makes concurrent BeamSearch /
// RobustPrune safe by construction (no shared mutable scratch).
//
// Fields:
//   visit_marks   — epoch-tagged "have we visited this node id?" table.
//                   Each entry holds the `visit_counter` value of the most
//                   recent search that touched it, so we don't need to clear
//                   the table on every call.
//   visit_counter — monotonic epoch tag. Wraps every 2^32 searches and re-
//                   zeroes the table via NextVisitEpoch().
//   prune_scratch — reusable PQ code buffer for RobustPrune's per-prune
//                   gather pattern (AiSAQ Tier 1 path). DiskANN's RobustPrune
//                   does not currently use it; Task 5 will when it
//                   parallelizes DiskANN's prune hot loop.
//   rng           — per-thread RNG. Both cores currently seed it from
//                   params_.seed but do not yet read it; kept here so the
//                   seed plumbing doesn't have to change again in Task 5.
// ---------------------------------------------------------------------------

struct VamanaTLS {
	vector<uint32_t> visit_marks;
	uint32_t visit_counter = 0;
	vector<uint8_t> prune_scratch; // reused across RobustPrune calls
	std::mt19937_64 rng;

	explicit VamanaTLS(uint64_t seed = 0xA15A6ULL) : rng(seed) {
	}

	// Bump visit epoch, handling 2^32 overflow by re-zeroing the marks table.
	// Returns the new epoch tag to compare against visit_marks[id].
	uint32_t NextVisitEpoch();

	// Pre-size visit_marks for an estimated build size and reset the epoch.
	void PrepareForBuild(idx_t estimated_count);
};

// ---------------------------------------------------------------------------
// Heap item types — today both cores define identical anonymous-namespace
// copies of these. Centralizing them keeps the min-vs-max convention in one
// place; they are NOT yet consumed by the cores (each BeamSearch still
// instantiates its priority_queue with its local types). Task 2+ collapses
// those once the primitives themselves are shared.
// ---------------------------------------------------------------------------

// Min-heap frontier (pop closest first).
struct FrontierItem {
	float dist;
	uint32_t internal_id;
};
struct FrontierCmp {
	bool operator()(const FrontierItem &a, const FrontierItem &b) const {
		return a.dist > b.dist;
	}
};

// Max-heap working set (pop farthest first).
struct WorkingItem {
	float dist;
	uint32_t internal_id;
};
struct WorkingCmp {
	bool operator()(const WorkingItem &a, const WorkingItem &b) const {
		return a.dist < b.dist;
	}
};

// The candidate type both cores use. Defined here for reference; each core
// keeps its own nested `Candidate` so the existing public API (which refers
// to AiSaqCore::Candidate / DiskAnnCore::Candidate) stays intact.
struct Candidate {
	int64_t row_id;
	float distance;
};

} // namespace vamana
} // namespace vindex
} // namespace duckdb
