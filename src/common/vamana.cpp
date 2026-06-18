#include "vindex/vamana.hpp"

#include <algorithm>

namespace duckdb {
namespace vindex {
namespace vamana {

uint32_t VamanaTLS::NextVisitEpoch() {
	visit_counter++;
	if (visit_counter == 0) {
		// Wrapped around 2^32 — every existing mark is suspect; reset and
		// start the next epoch at 1 so the "uninitialized == 0" sentinel
		// stays valid.
		std::fill(visit_marks.begin(), visit_marks.end(), 0);
		visit_counter = 1;
	}
	return visit_counter;
}

void VamanaTLS::PrepareForBuild(idx_t estimated_count) {
	visit_marks.assign(estimated_count, 0);
	visit_counter = 0;
}

} // namespace vamana
} // namespace vindex
} // namespace duckdb
