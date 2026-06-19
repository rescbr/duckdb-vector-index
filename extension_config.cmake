# This file is included by DuckDB's build system.
# It specifies which extension(s) to load into the duckdb build.

# GCC 14+ stricter -fno-common enforcement causes "multiple definition"
# linker errors when both libvindex_extension.a and libduckdb_static.a emit
# the same constexpr class members (LogicalType::BIGINT,
# TableCatalogEntry::Name, etc.). --allow-multiple-definition is safe
# because both definitions are identical constexpr values.
# See: docs/gcc-14-vindex.md
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "14")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-multiple-definition")
endif()

# The vindex extension lives in this repository.
duckdb_extension_load(vindex
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
        LOAD_TESTS
        )

# Any extra extensions that should be built alongside for testing purposes can
# be loaded here, e.g.:
# duckdb_extension_load(json)
