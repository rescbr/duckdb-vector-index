#pragma once

#include "duckdb/common/array.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/typedefs.hpp"

namespace duckdb {
class ClientContext;
} // namespace duckdb

namespace duckdb {
namespace vindex {

constexpr std::array<const char *, 2> kKnownBackends = {"cpu", "vulkan"};

class Backend {
  public:
	virtual ~Backend() = default;
	virtual const string &Name() const = 0;

	// Train a quantizer on a sample of vectors.
	virtual void TrainQuantizer(class Quantizer &quantizer, const float *samples, idx_t n, idx_t dim) = 0;

	// Compute k entry-point centroids via k-means clustering.
	virtual void ComputeEntryPoints(const float *vectors, idx_t n, idx_t dim, idx_t k, float *centroids_out) = 0;

	// Populate the PQ/ScaNN distance LUT from a preprocessed query.
	virtual void PopulateDistanceLUT(const class Quantizer &quantizer, const float *query_preproc, float *lut_out,
	                                 idx_t lut_size) = 0;

	// Batched LUT distance: compute distances for n_codes codes against
	// a pre-populated LUT. Results in distances_out[0..n_codes-1].
	virtual void BatchLUTDistance(const class Quantizer &quantizer, const_data_ptr_t codes, idx_t n_codes,
	                              const float *lut, float *distances_out) = 0;
};

} // namespace vindex
} // namespace duckdb
