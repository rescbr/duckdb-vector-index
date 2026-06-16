#pragma once

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/main/client_context.hpp"

#include "backend/backend.hpp"
#include "backend/cpu_backend.hpp"

namespace duckdb {
namespace vindex {

// GetActiveBackend - returns the backend selected by the vindex_gpu_backend
// session option. Hardcoded switch; no factory, no registry.
inline unique_ptr<Backend> GetActiveBackend(ClientContext &ctx) {
	Value v;
	ctx.TryGetCurrentSetting("vindex_gpu_backend", v);
	const auto name = v.IsNull() ? "cpu" : StringUtil::Lower(v.GetValue<string>());

	if (name == "cpu") {
		return make_uniq<CpuBackend>();
	}
#ifdef VINDEX_BACKEND_VULKAN
	if (name == "vulkan") {
		return make_uniq<VulkanBackend>();
	}
#endif
	throw BinderException("Unknown vindex_gpu_backend '%s'. Known backends: cpu%s", name,
#ifdef VINDEX_BACKEND_VULKAN
	                      ", vulkan"
#else
	                      ""
#endif
	);
}

} // namespace vindex
} // namespace duckdb
