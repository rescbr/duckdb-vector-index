#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/vector.hpp"

#include <climits>
#include <algorithm>

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// LabelFilter — a unified representation of label-column predicates routed
// to the index scan by the optimizer.
//
// Supported in v0:
//   EQUALS  — label_col = value
//   RANGE   — label_col >= lo AND label_col <= hi
//             (covers >, >=, <, <=, BETWEEN, and folded conjuncts)
//
// Future (M7.5):
//   NOT_EQUALS — label_col != value
//   IN_LIST    — label_col IN (values...)
//
// The optimizer folds all supported comparison operators into EQUALS or
// RANGE before calling SetLabelFilter. The index never sees raw comparison
// types.
// ---------------------------------------------------------------------------

struct LabelFilter {
	enum class Kind : uint8_t {
		NONE,
		EQUALS,
		RANGE,
		NOT_EQUALS, // M7.5
		IN_LIST     // M7.5
	};

	Kind kind = Kind::NONE;
	int64_t value = 0;       // EQUALS, NOT_EQUALS
	int64_t lo = INT64_MIN;  // RANGE (inclusive)
	int64_t hi = INT64_MAX;  // RANGE (inclusive)
	vector<int64_t> values;  // IN_LIST (M7.5; sorted for binary search)

	bool IsActive() const {
		return kind != Kind::NONE;
	}

	// Test whether a given label value matches this filter.
	bool Matches(int64_t label) const {
		switch (kind) {
			case Kind::NONE:
				return true;
			case Kind::EQUALS:
				return label == value;
			case Kind::RANGE:
				return label >= lo && label <= hi;
			case Kind::NOT_EQUALS:
				return label != value;
			case Kind::IN_LIST:
				return std::binary_search(values.begin(), values.end(), label);
		}
		return false;
	}

	// Factory helpers.
	static LabelFilter Equals(int64_t v) {
		LabelFilter f;
		f.kind = Kind::EQUALS;
		f.value = v;
		return f;
	}

	static LabelFilter Range(int64_t lo_, int64_t hi_) {
		LabelFilter f;
		f.kind = Kind::RANGE;
		f.lo = lo_;
		f.hi = hi_;
		return f;
	}

	static LabelFilter None() {
		return LabelFilter {};
	}
};

} // namespace vindex
} // namespace duckdb
