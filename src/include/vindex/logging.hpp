#pragma once

#include "duckdb/common/enum_util.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"

#include <cstdio>
#include <cstdlib>

namespace duckdb {
namespace vindex {

// ---------------------------------------------------------------------------
// LogLevel — controls build-time (and eventually search-time) stderr output.
//
// Resolution order:
//   1. VINDEX_LOG_LEVEL environment variable (overrides everything)
//   2. vindex_log_level session option (SET vindex_log_level = 'info')
//   3. Default: OFF (no output — sqllogictest safe)
// ---------------------------------------------------------------------------

enum class LogLevel : uint8_t {
	OFF = 0,
	INFO = 1,
	DBG = 2,
	PROFILE = 3
};

inline LogLevel ParseLogLevel(const string &s) {
	auto lower = StringUtil::Lower(s);
	if (lower == "off") {
		return LogLevel::OFF;
	}
	if (lower == "info") {
		return LogLevel::INFO;
	}
	if (lower == "debug") {
		return LogLevel::DBG;
	}
	if (lower == "profile") {
		return LogLevel::PROFILE;
	}
	return LogLevel::OFF;
}

inline LogLevel GetLogLevel(ClientContext &context) {
	// Environment variable takes precedence.
	if (const char *env = std::getenv("VINDEX_LOG_LEVEL")) {
		return ParseLogLevel(env);
	}
	// Fall back to session option.
	Value v;
	if (context.TryGetCurrentSetting("vindex_log_level", v)) {
		if (!v.IsNull() && v.type() == LogicalType::VARCHAR) {
			return ParseLogLevel(v.GetValue<string>());
		}
	}
	return LogLevel::OFF;
}

// Convenience predicates.
inline bool LogInfo(LogLevel l) {
	return static_cast<uint8_t>(l) >= static_cast<uint8_t>(LogLevel::INFO);
}
inline bool LogDebug(LogLevel l) {
	return static_cast<uint8_t>(l) >= static_cast<uint8_t>(LogLevel::DBG);
}
inline bool LogProfile(LogLevel l) {
	return l == LogLevel::PROFILE;
}

} // namespace vindex
} // namespace duckdb
