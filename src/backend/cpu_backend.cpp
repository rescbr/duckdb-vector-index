#include "backend/cpu_backend.hpp"

#include "duckdb/common/exception.hpp"

#include "algo/ivf/kmeans.hpp"
#include "vindex/quantizer.hpp"

namespace duckdb {
namespace vindex {

const string &CpuBackend::Name() const {
	static const string name = "cpu";
	return name;
}

void CpuBackend::TrainQuantizer(Quantizer &quantizer, const float *samples, idx_t n, idx_t dim) {
	quantizer.Train(samples, n, dim);
}

void CpuBackend::ComputeEntryPoints(const float *vectors, idx_t n, idx_t dim, idx_t k, float *centroids_out) {
	ivf::KMeansPlusPlus(vectors, n, dim, k, /*seed=*/42, /*max_iters=*/25, centroids_out);
}

void CpuBackend::PopulateDistanceLUT(const Quantizer &quantizer, const float *query_preproc, float *lut_out,
                                     idx_t lut_size) {
	(void)lut_size;
	quantizer.PopulateDistanceLUT(query_preproc, lut_out);
}

void CpuBackend::BatchLUTDistance(const Quantizer &quantizer, const_data_ptr_t codes, idx_t n_codes, const float *lut,
                                  float *distances_out) {
	const idx_t code_size = quantizer.CodeSize();
	for (idx_t i = 0; i < n_codes; i++) {
		distances_out[i] = quantizer.LUTDistance(codes, lut);
		codes += code_size;
	}
}

} // namespace vindex
} // namespace duckdb
