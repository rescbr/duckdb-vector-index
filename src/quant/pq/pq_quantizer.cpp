#include "vindex/pq_quantizer.hpp"

#include "algo/ivf/kmeans.hpp"

#include "duckdb/common/exception.hpp"

#define SIMSIMD_NATIVE_F16 0
#define SIMSIMD_NATIVE_BF16 0
#include "simsimd/simsimd.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace duckdb {
namespace vindex {
namespace pq {

namespace {

inline float L2sq(const float *a, const float *b, idx_t dim) {
	simsimd_distance_t d = 0;
	simsimd_l2sq_f32(reinterpret_cast<const simsimd_f32_t *>(a), reinterpret_cast<const simsimd_f32_t *>(b), dim, &d);
	return float(d);
}

inline float Dot(const float *a, const float *b, idx_t dim) {
	simsimd_distance_t d = 0;
	simsimd_dot_f32(reinterpret_cast<const simsimd_f32_t *>(a), reinterpret_cast<const simsimd_f32_t *>(b), dim, &d);
	return float(d);
}

} // namespace

PqQuantizer::PqQuantizer(MetricKind metric, idx_t dim, uint8_t m, uint8_t bits, uint64_t seed)
    : metric_(metric), dim_(dim), m_(m), bits_(bits), seed_(seed) {
	if (metric_ == MetricKind::COSINE) {
		// Classical PQ supports L2SQ and IP cleanly. COSINE would require
		// per-vector norm storage; left out for now to keep the code small.
		throw InvalidInputException("vindex: PQ quantizer does not support cosine metric (use L2SQ or IP)");
	}
	if (m_ == 0) {
		throw InvalidInputException("vindex: PQ 'm' must be > 0");
	}
	if (dim_ % idx_t(m_) != 0) {
		throw InvalidInputException("vindex: PQ requires dim (%llu) to be divisible by m (%d)",
		                            (unsigned long long)dim_, int(m_));
	}
	if (bits_ != 4 && bits_ != 8) {
		throw InvalidInputException("vindex: PQ 'bits' must be 4 or 8 (got %d)", int(bits_));
	}
}

idx_t PqQuantizer::CodeSize() const {
	return (idx_t(m_) * bits_ + 7) / 8;
}

idx_t PqQuantizer::QueryWorkspaceSize() const {
	return idx_t(m_) * CentroidsPerSlot();
}

uint32_t PqQuantizer::ReadCode(const_data_ptr_t code, idx_t s) const {
	if (bits_ == 8) {
		return static_cast<uint32_t>(code[s]);
	}
	// bits_ == 4 — two codes per byte, slot 0 = low nibble, slot 1 = high.
	const idx_t byte_off = s / 2;
	const uint8_t shift = (s % 2) * 4;
	return (static_cast<uint32_t>(code[byte_off]) >> shift) & 0x0Fu;
}

void PqQuantizer::WriteCode(data_ptr_t code, idx_t s, uint32_t cid) const {
	if (bits_ == 8) {
		code[s] = static_cast<uint8_t>(cid & 0xFFu);
		return;
	}
	const idx_t byte_off = s / 2;
	const uint8_t shift = (s % 2) * 4;
	const uint8_t nibble = static_cast<uint8_t>(cid & 0x0Fu);
	code[byte_off] = static_cast<uint8_t>((code[byte_off] & ~(0x0Fu << shift)) | (nibble << shift));
}

uint32_t PqQuantizer::NearestCentroid(idx_t s, const float *sub) const {
	const idx_t sub_dim = SubDim();
	const idx_t k = CentroidsPerSlot();
	const float *slot_book = codebook_.data() + s * k * sub_dim;
	uint32_t best = 0;
	float best_d = std::numeric_limits<float>::infinity();
	for (idx_t c = 0; c < k; c++) {
		const float d = L2sq(sub, slot_book + c * sub_dim, sub_dim);
		if (d < best_d) {
			best_d = d;
			best = uint32_t(c);
		}
	}
	return best;
}

void PqQuantizer::Train(const float *samples, idx_t n, idx_t dim) {
	if (dim != dim_) {
		throw InternalException("PqQuantizer::Train dim mismatch (%llu vs %llu)", (unsigned long long)dim,
		                        (unsigned long long)dim_);
	}
	const idx_t sub_dim = SubDim();
	const idx_t k = CentroidsPerSlot();
	codebook_.assign(idx_t(m_) * k * sub_dim, 0.0f);

	// Per-slot training: extract column-block of size (n × sub_dim) and run
	// k-means++. For n < k (rare — tests only), KMeansPlusPlus pads with
	// sample repeats; we still get a usable codebook.
	vector<float> sub_buffer(n * sub_dim);
	for (idx_t s = 0; s < idx_t(m_); s++) {
		for (idx_t i = 0; i < n; i++) {
			std::memcpy(sub_buffer.data() + i * sub_dim, samples + i * dim + s * sub_dim,
			            sub_dim * sizeof(float));
		}
		// Distinct seed per slot — otherwise every slot's k-means picks the
		// same initial seeds (identical RNG stream).
		const uint64_t slot_seed = seed_ ^ (0x9E3779B97F4A7C15ULL * (s + 1));
		ivf::KMeansPlusPlus(sub_buffer.data(), n, sub_dim, k, slot_seed, /*max_iters=*/25,
		                    codebook_.data() + s * k * sub_dim);
	}
	trained_ = true;

	// Phase 9.5 iter 1: precompute centroid-pair distances so CodeDistance
	// becomes O(m) lookups instead of O(m * sub_dim) simsimd work. This is
	// the dominant cost during Vamana construction (RobustPrune's O(L²)
	// pairwise loop calls CodeDistance ~4B times on sift1m).
	BuildCrossDistanceTable();
}

void PqQuantizer::BuildCrossDistanceTable() {
	const idx_t sub_dim = SubDim();
	const idx_t k = CentroidsPerSlot();
	cross_distance_table_.assign(idx_t(m_) * k * k, 0.0f);
	for (idx_t s = 0; s < idx_t(m_); s++) {
		const float *book = codebook_.data() + s * k * sub_dim;
		float *table = cross_distance_table_.data() + s * k * k;
		for (idx_t a = 0; a < k; a++) {
			const float *ca = book + a * sub_dim;
			// Diagonal is 0 for L2SQ and -|ca|² for IP; just store 0 for both
			// since CodeDistance callers only care about relative ordering.
			// (IP stores -Dot, so diagonal would be -|ca|²; the absolute value
			// doesn't affect RobustPrune's ranking. Skip for cache density.)
			table[a * k + a] = 0.0f;
			for (idx_t b = a + 1; b < k; b++) {
				const float *cb = book + b * sub_dim;
				float d;
				switch (metric_) {
				case MetricKind::L2SQ:
					d = L2sq(ca, cb, sub_dim);
					break;
				case MetricKind::IP:
					d = -Dot(ca, cb, sub_dim);
					break;
				case MetricKind::COSINE:
					throw InternalException("PqQuantizer: cosine unreachable");
				default:
					d = 0.0f;
				}
				table[a * k + b] = d;
				table[b * k + a] = d; // symmetric
			}
		}
	}
}

void PqQuantizer::Encode(const float *vec, data_ptr_t code_out) const {
	if (!trained_) {
		throw InternalException("PqQuantizer::Encode called before Train");
	}
	const idx_t sub_dim = SubDim();
	std::memset(code_out, 0, CodeSize());
	for (idx_t s = 0; s < idx_t(m_); s++) {
		const uint32_t cid = NearestCentroid(s, vec + s * sub_dim);
		WriteCode(code_out, s, cid);
	}
}

void PqQuantizer::PreprocessQuery(const float *query, float *out) const {
	if (!trained_) {
		throw InternalException("PqQuantizer::PreprocessQuery called before Train");
	}
	// ADC table: out[s * K + c] = d(query_sub_s, centroid[s][c]).
	//   L2SQ: L2 squared distance.
	//   IP:   -dot(query_sub, centroid)  (DuckDB stores *negative* IP).
	const idx_t sub_dim = SubDim();
	const idx_t k = CentroidsPerSlot();
	for (idx_t s = 0; s < idx_t(m_); s++) {
		const float *q_sub = query + s * sub_dim;
		const float *slot_book = codebook_.data() + s * k * sub_dim;
		float *row = out + s * k;
		for (idx_t c = 0; c < k; c++) {
			const float *cen = slot_book + c * sub_dim;
			switch (metric_) {
			case MetricKind::L2SQ:
				row[c] = L2sq(q_sub, cen, sub_dim);
				break;
			case MetricKind::IP:
				row[c] = -Dot(q_sub, cen, sub_dim);
				break;
			case MetricKind::COSINE:
				throw InternalException("PqQuantizer: cosine unreachable");
			}
		}
	}
}

float PqQuantizer::EstimateDistance(const_data_ptr_t code, const float *query_preproc) const {
	const idx_t k = CentroidsPerSlot();
	float acc = 0.0f;
	for (idx_t s = 0; s < idx_t(m_); s++) {
		const uint32_t cid = ReadCode(code, s);
		acc += query_preproc[s * k + cid];
	}
	return acc;
}

float PqQuantizer::CodeDistance(const_data_ptr_t code_a, const_data_ptr_t code_b) const {
	// Phase 9.5 iter 1: O(m) lookup path via cross_distance_table_. Falls back
	// to the O(m * sub_dim) codebook path if the table wasn't built (e.g.,
	// old serialized state loaded without migration — Train() always builds it
	// for new indexes).
	if (!cross_distance_table_.empty()) {
		const idx_t k = CentroidsPerSlot();
		const float *table_base = cross_distance_table_.data();
		float acc = 0.0f;
		for (idx_t s = 0; s < idx_t(m_); s++) {
			const uint32_t ca = ReadCode(code_a, s);
			const uint32_t cb = ReadCode(code_b, s);
			acc += table_base[s * k * k + ca * k + cb];
		}
		return acc;
	}
	// Legacy fallback — same as pre-Phase-9.5.
	const idx_t sub_dim = SubDim();
	const idx_t k = CentroidsPerSlot();
	float acc = 0.0f;
	for (idx_t s = 0; s < idx_t(m_); s++) {
		const uint32_t ca = ReadCode(code_a, s);
		const uint32_t cb = ReadCode(code_b, s);
		const float *book = codebook_.data() + s * k * sub_dim;
		switch (metric_) {
		case MetricKind::L2SQ:
			acc += L2sq(book + ca * sub_dim, book + cb * sub_dim, sub_dim);
			break;
		case MetricKind::IP:
			acc += -Dot(book + ca * sub_dim, book + cb * sub_dim, sub_dim);
			break;
		case MetricKind::COSINE:
			throw InternalException("PqQuantizer: cosine unreachable");
		}
	}
	return acc;
}

void PqQuantizer::Serialize(vector<data_t> &out) const {
	// {kind:u8, metric:u8, m:u8, bits:u8, dim:u64, codebook:float32[m*K*sub_dim]}
	const idx_t header = sizeof(uint8_t) * 4 + sizeof(uint64_t);
	const idx_t book_bytes = codebook_.size() * sizeof(float);
	out.resize(header + book_bytes);
	auto ptr = out.data();
	ptr[0] = static_cast<uint8_t>(QuantizerKind::PQ);
	ptr[1] = static_cast<uint8_t>(metric_);
	ptr[2] = m_;
	ptr[3] = bits_;
	const uint64_t d = dim_;
	std::memcpy(ptr + 4, &d, sizeof(d));
	if (book_bytes > 0) {
		std::memcpy(ptr + header, codebook_.data(), book_bytes);
	}
}

void PqQuantizer::Deserialize(const_data_ptr_t in, idx_t size) {
	const idx_t header = sizeof(uint8_t) * 4 + sizeof(uint64_t);
	if (size < header) {
		throw InternalException("PqQuantizer::Deserialize: blob too small (%llu)", (unsigned long long)size);
	}
	const auto kind = static_cast<QuantizerKind>(in[0]);
	if (kind != QuantizerKind::PQ) {
		throw InternalException("PqQuantizer::Deserialize: wrong kind tag %d", int(in[0]));
	}
	metric_ = static_cast<MetricKind>(in[1]);
	m_ = in[2];
	bits_ = in[3];
	uint64_t d;
	std::memcpy(&d, in + 4, sizeof(d));
	dim_ = d;
	if (m_ == 0 || dim_ % idx_t(m_) != 0 || (bits_ != 4 && bits_ != 8)) {
		throw InternalException("PqQuantizer::Deserialize: invalid header (m=%d bits=%d dim=%llu)", int(m_),
		                        int(bits_), (unsigned long long)dim_);
	}
	const idx_t sub_dim = SubDim();
	const idx_t k = CentroidsPerSlot();
	const idx_t book_floats = idx_t(m_) * k * sub_dim;
	const idx_t book_bytes = book_floats * sizeof(float);
	if (size != header + book_bytes) {
		throw InternalException("PqQuantizer::Deserialize: blob size mismatch (got %llu, expected %llu)",
		                        (unsigned long long)size, (unsigned long long)(header + book_bytes));
	}
	codebook_.assign(book_floats, 0.0f);
	if (book_bytes > 0) {
		std::memcpy(codebook_.data(), in + header, book_bytes);
	}
	trained_ = true;
	// Regenerate the cross-distance table from the restored codebook.
	BuildCrossDistanceTable();
}

// ---------------------------------------------------------------------------
// Phase 3: LUT virtuals
// ---------------------------------------------------------------------------

void PqQuantizer::PopulateDistanceLUT(const float *query_preproc, float *lut_out) const {
	const idx_t size = LUTSize();
	std::memcpy(lut_out, query_preproc, size * sizeof(float));
}

float PqQuantizer::LUTDistance(const_data_ptr_t code, const float *lut) const {
	const idx_t k = CentroidsPerSlot();
	float acc = 0.0f;
	for (idx_t s = 0; s < idx_t(m_); s++) {
		const uint32_t cid = ReadCode(code, s);
		acc += lut[s * k + cid];
	}
	return acc;
}

idx_t PqQuantizer::LUTSize() const {
	return idx_t(m_) * CentroidsPerSlot();
}

} // namespace pq
} // namespace vindex
} // namespace duckdb
