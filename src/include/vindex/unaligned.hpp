#pragma once

#include "duckdb/common/typedefs.hpp"

#include <cstring>

namespace duckdb {
namespace vindex {

// Safe unaligned loads/stores for typed node fields.
//
// Why this exists: AiSaqBlockStore packs N consecutive nodes into a single
// DuckDB block via arithmetic indexing. Even though node_size is required to
// be a multiple of 8 (the alignment invariant), typed sub-fields (e.g. the
// uint16_t neighbor_count at offset 12) sit at their natural offsets and a
// bare `*reinterpret_cast<T*>(ptr)` would be UB / sanitizer noise on any
// field that isn't on its natural boundary. The packed-struct wrapper lowers
// the member alignment to 1, so the compiler emits a byte-wise (or its
// preferred unaligned) access — UBSan-clean on GCC/Clang.
template <typename T> T LoadUnaligned(const_data_ptr_t ptr) {
	struct __attribute__((packed)) Packed {
		T v;
	};
	Packed p;
	std::memcpy(&p, ptr, sizeof(T));
	return p.v;
}

template <typename T> void StoreUnaligned(data_ptr_t ptr, T value) {
	struct __attribute__((packed)) Packed {
		T v;
	};
	Packed p;
	p.v = value;
	std::memcpy(ptr, &p, sizeof(T));
}

} // namespace vindex
} // namespace duckdb
