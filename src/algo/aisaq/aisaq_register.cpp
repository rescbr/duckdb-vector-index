#include "algo/aisaq/aisaq_module.hpp"

#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace vindex {
namespace aisaq {

// Defined in aisaq_index.cpp / aisaq_pragmas.cpp.
void RegisterIndex(DatabaseInstance &db);
void RegisterPragmas(ExtensionLoader &loader);

void Register(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();

	RegisterIndex(db);
	RegisterPragmas(loader);
}

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
