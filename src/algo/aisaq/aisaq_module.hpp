#pragma once

namespace duckdb {
class ExtensionLoader;
} // namespace duckdb

namespace duckdb {
namespace vindex {
namespace aisaq {

void Register(ExtensionLoader &loader);

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
