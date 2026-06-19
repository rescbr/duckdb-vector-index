#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/vector.hpp"

#include "vindex/metric.hpp"
#include "vindex/quantizer.hpp"

namespace duckdb {
namespace vindex {
namespace pq {

// ---------------------------------------------------------------------------
// PqQuantizer — classical product quantization.
//
// Splits each d-dim vector into `m` sub-vectors (each of dim d/m) and learns
// `2^bits` centroids per sub-vector via k-means++. A `code` is the tuple of
// nearest-centroid ids, packed into ceil(m * bits / 8) bytes.
//
// At query time PreprocessQuery precomputes the asymmetric distance table
// (ADC): for each sub-vector slot s and each centroid id c in 0..(2^bits-1),
// store the sub-distance between the query's sub-vector and centroid c.
// EstimateDistance then just sums `m` table lookups — a single gather pass,
// extremely fast.
//
// bits ∈ {4, 8}. 8 is the default (256 centroids / slot, <1% distance error
// on typical SIFT/GIST features); 4 halves RAM at the cost of lower recall
// and is intended for DiskANN-sized workloads where code RAM is the binding
// constraint.
// ---------------------------------------------------------------------------

class PqQuantizer : public Quantizer {
public:
	// `m` must divide `dim` exactly. Caller passes these through from
	// CREATE INDEX WITH-options (`m`, `bits`, `metric`). Training is deferred
	// to Train().
	PqQuantizer(MetricKind metric, idx_t dim, uint8_t m, uint8_t bits, uint64_t seed = 0xC0DE1234ULL);

	void Train(const float *samples, idx_t n, idx_t dim) override;

	void Encode(const float *vec, data_ptr_t code_out) const override;
	float EstimateDistance(const_data_ptr_t code, const float *query_preproc) const override;
	float CodeDistance(const_data_ptr_t code_a, const_data_ptr_t code_b) const override;
	void PreprocessQuery(const float *query, float *out) const override;

	idx_t CodeSize() const override;
	idx_t QueryWorkspaceSize() const override;
	MetricKind Metric() const override {
		return metric_;
	}
	QuantizerKind Kind() const override {
		return QuantizerKind::PQ;
	}

	void Serialize(vector<data_t> &out) const override;
	void Deserialize(const_data_ptr_t in, idx_t size) override;

	// Phase 3: LUT virtuals.
	void PopulateDistanceLUT(const float *query_preproc, float *lut_out) const override;
	float LUTDistance(const_data_ptr_t code, const float *lut) const override;
	idx_t LUTSize() const override;

	// Introspection for tests / pragmas.
	uint8_t M() const {
		return m_;
	}
	uint8_t Bits() const {
		return bits_;
	}

private:
	// Extract centroid id for slot `s` from a packed code.
	uint32_t ReadCode(const_data_ptr_t code, idx_t s) const;
	// Write centroid id for slot `s` into a packed code.
	void WriteCode(data_ptr_t code, idx_t s, uint32_t cid) const;

	// Assign a float sub-vector to its nearest centroid (in codebook_[s]).
	uint32_t NearestCentroid(idx_t s, const float *sub) const;

	idx_t SubDim() const {
		return dim_ / idx_t(m_);
	}
	idx_t CentroidsPerSlot() const {
		return idx_t(1) << bits_;
	}

	MetricKind metric_;
	idx_t dim_ = 0;
	uint8_t m_ = 0;     // number of sub-vectors
	uint8_t bits_ = 0;  // bits per sub-vector
	uint64_t seed_;
	bool trained_ = false;

	// Flat codebook: for slot s, codebook_[s * (2^bits) * sub_dim .. ] holds
	// `2^bits` centroids of `sub_dim` floats each.
	vector<float> codebook_;

	// Phase 9.5 iter 1: precomputed centroid-pair distances.
	// cross_distance_table_[s * K * K + ca * K + cb] = sub-distance between
	// centroid ca and cb in slot s. Symmetric (ca,cb) == (cb,ca) for all
	// supported metrics (L2SQ and IP via -Dot).
	//
	// Built once at end of Train() / Deserialize(). Turns CodeDistance from
	// O(m * sub_dim) simsimd work into O(m) array lookups — ~5-10x faster
	// for typical SIFT/GIST configs (m=16, K=256, sub_dim=8).
	//
	// Memory: m * K * K * sizeof(float). For SIFT: 16 * 256 * 256 * 4 = 4 MB.
	vector<float> cross_distance_table_;

	void BuildCrossDistanceTable();
};

} // namespace pq
} // namespace vindex
} // namespace duckdb
