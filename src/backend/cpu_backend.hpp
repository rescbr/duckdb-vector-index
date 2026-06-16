#pragma once

#include "backend.hpp"

namespace duckdb {
namespace vindex {

class CpuBackend : public Backend {
  public:
	const string &Name() const override;

	void TrainQuantizer(class Quantizer &quantizer, const float *samples, idx_t n, idx_t dim) override;
	void ComputeEntryPoints(const float *vectors, idx_t n, idx_t dim, idx_t k, float *centroids_out) override;
	void PopulateDistanceLUT(const class Quantizer &quantizer, const float *query_preproc, float *lut_out,
	                         idx_t lut_size) override;
	void BatchLUTDistance(const class Quantizer &quantizer, const_data_ptr_t codes, idx_t n_codes, const float *lut,
	                      float *distances_out) override;
};

} // namespace vindex
} // namespace duckdb
