# Phase 2 — General WHERE-Label Optimizer

**Branch**: `feat/label-filter`
**Status**: `- [x]` complete
**Effort**: 3 weeks
**Depends on**: nothing
**Dependents**: Phase 6 (AiSAQ label integration builds on this)
**Upstreamable**: yes — general infrastructure for all `VectorIndex` subclasses

> **Maintenance note**: Update task markers (`- [ ]` -> `- [~]` -> `- [x]`) as you progress. At phase completion, update the phase status in [AISAQ.md](AISAQ.md) and update `AGENTS.md` per the [AGENTS.md Update Protocol](AISAQ.md#agentsmd-update-protocol).

---

## Goal

Add a general mechanism for vector indexes to support label-column filtering at the index-scan level, rather than as a post-hoc filter. This benefits any `VectorIndex` subclass that opts in via a virtual hook, and is upstreamable to upstream vindex independently of AiSAQ.

Supported operators in v0:

| SQL operator | Filter kind | Example |
|---|---|---|
| `=` | `EQUALS` | `WHERE cat = 42` |
| `>` | `RANGE` | `WHERE cat > 10` |
| `>=` | `RANGE` | `WHERE cat >= 10` |
| `<` | `RANGE` | `WHERE cat < 100` |
| `<=` | `RANGE` | `WHERE cat <= 100` |
| `BETWEEN` | `RANGE` | `WHERE cat BETWEEN 10 AND 100` |
| `> AND <` (combined) | `RANGE` (folded) | `WHERE cat > 10 AND cat < 100` -> `RANGE[11, 99]` |

Deferred to M7.5: `!=` (not-equal) and `IN (...)`. The `LabelFilter` struct is forward-compatible with these.

Today, the optimizer in `src/common/optimize_scan.cpp` only handles `ORDER BY array_distance(...) LIMIT k`. This phase extends it to also detect label-column predicates and route them into the index scan.

## Background

The current optimizer rewrite path (from `AGENTS.md`):

> The `VINDEX_INDEX_SCAN` rewrite fires only when the query vector is a **literal** (e.g. `[...]::FLOAT[128]`).

This phase adds a second dimension to the rewrite: label-column filter predicates. The optimizer must:

1. Detect comparison predicates (`=`, `>`, `<`, `>=`, `<=`, `BETWEEN`) on the designated label column.
2. Extract the literal BIGINT bounds.
3. Fold multiple conjunctive predicates on the same column into a single `RANGE` filter where applicable (e.g., `> 10 AND < 100` -> `RANGE[11, 99]`).
4. Route the resulting `LabelFilter` to the scan state via a new virtual hook on `VectorIndex`.

Non-literal filters (subqueries, expressions) and unsupported operators (`!=`, `IN`) fall through to DuckDB's normal post-filter execution. This preserves correctness at the cost of recall for heavy filters.

## Scope

### In scope

- New `LabelFilter` struct in `src/include/vindex/label_filter.hpp` (forward-compatible with M7.5).
- New virtual hooks on `VectorIndex`: `SupportsLabelFilter()`, `SetLabelFilter()`, `GetLabelColumn()`.
- New `WITH (label_column = 'col_name')` option parsing for all vector index types (generic, not AiSAQ-specific).
- Planner rewrite in `src/common/optimize_scan.cpp` to detect:
  - `WHERE label_col = literal_BIGINT`
  - `WHERE label_col > literal_BIGINT` / `>=` / `<` / `<=`
  - `WHERE label_col BETWEEN literal AND literal`
  - Combined conjuncts: `WHERE label_col > A AND label_col < B` -> folded `RANGE`
- Predicate folding: two conjunctive range predicates on the same label column are merged into one `RANGE` filter.
- BIGINT literals only.
- SQL logic tests for each operator.
- Unit tests for the planner detection + folding logic.

### Out of scope

- VARCHAR labels (tracked as M7).
- `!=` (not-equal) and `IN (...)` (tracked as M7.5; `LabelFilter` struct is forward-compatible).
- Per-label medoid selection (that's Phase 6, AiSAQ-specific).
- Multi-column filters (`WHERE col1 = X AND col2 = Y`).
- The actual AiSAQ integration (Phase 6).

### Forward compatibility (M7.5)

The `LabelFilter` struct is designed to accommodate `!=` and `IN` without API changes:

```cpp
struct LabelFilter {
    enum class Kind : uint8_t {
        NONE,
        EQUALS,
        RANGE,
        // --- M7.5 (not implemented in v0) ---
        NOT_EQUALS,
        IN_LIST
    };
    Kind kind = Kind::NONE;
    int64_t value = 0;            // EQUALS, NOT_EQUALS (M7.5)
    int64_t lo = INT64_MIN;       // RANGE (inclusive)
    int64_t hi = INT64_MAX;       // RANGE (inclusive)
    vector<int64_t> values;       // IN_LIST (M7.5; unused in v0)
};
```

The optimizer and index implementations only handle `NONE`, `EQUALS`, `RANGE` in v0. The unused enum values and the `values` vector are carried so the serialization format and API don't change when M7.5 lands.

## Files

### New files

| Path | Purpose |
|---|---|
| `src/include/vindex/label_filter.hpp` | `LabelFilter` struct definition |
| `test/sql/common/label_filter_optimizer.test` | Tests the optimizer rewrite for all operators |
| `test/unit/test_label_filter_optimizer.cpp` | Planner-level unit tests for literal extraction + folding |

### Changed files

| Path | Change |
|---|---|
| `src/include/vindex/vector_index.hpp` | Include `label_filter.hpp`; add `SupportsLabelFilter()`, `SetLabelFilter(LabelFilter)`, `GetLabelColumn()` virtuals |
| `src/common/vector_index.cpp` | Parse `label_column` option from `WITH (...)` map |
| `src/common/optimize_scan.cpp` | Detect all operators + predicate folding; route to `SetLabelFilter` |
| `src/algo/hnsw/hnsw_plan.cpp` | Accept `label_column` option (validate column exists, BIGINT type) |
| `src/algo/ivf/ivf_plan.cpp` | Same |
| `src/algo/diskann/diskann_plan.cpp` | Same |
| `src/algo/spann/spann_plan.cpp` | Same |

## API Surfaces

### `src/include/vindex/label_filter.hpp` (new)

```cpp
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
        NOT_EQUALS,  // M7.5
        IN_LIST      // M7.5
    };

    Kind kind = Kind::NONE;
    int64_t value = 0;              // EQUALS, NOT_EQUALS
    int64_t lo = INT64_MIN;         // RANGE (inclusive)
    int64_t hi = INT64_MAX;         // RANGE (inclusive)
    vector<int64_t> values;         // IN_LIST (M7.5; sorted for binary search)

    bool IsActive() const { return kind != Kind::NONE; }

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
```

### `src/include/vindex/vector_index.hpp` (additions)

```cpp
#include "vindex/label_filter.hpp"

class VectorIndex : public BoundIndex {
public:
    // ... existing API ...

    // --- Label filtering (Phase 2: general infrastructure) ---

    // Override to return true if this index type supports label-filtered
    // search. Default: false (no algorithm supports it until it opts in).
    virtual bool SupportsLabelFilter() const {
        return false;
    }

    // Called by the optimizer when a label-column predicate is detected.
    // The index should record the filter and use it to scope subsequent
    // InitializeScan / ExecuteMultiScan calls. Default: no-op.
    //
    // label_column is the column name designated at CREATE INDEX time via
    // WITH (label_column = '...'). It's passed here so the index can verify
    // the optimizer's detection matches its own designation.
    virtual void SetLabelFilter(IndexScanState &state, const LabelFilter &filter,
                                const string &label_column) const {}

    // Returns the column name designated as the label column at CREATE INDEX
    // time, or empty string if none. Used by the optimizer to match WHERE
    // predicates.
    virtual const string &GetLabelColumn() const {
        static const string empty;
        return empty;
    }
};
```

### `src/common/optimize_scan.cpp` (detection + folding logic)

The optimizer scans the filter expression children of the `PhysicalFilter` above the index scan. For each conjunctive filter predicate on the designated label column:

**Operator detection** — each comparison maps to a partial range:

| DuckDB ExpressionType | Mapping |
|---|---|
| `COMPARE_EQUAL` | `EQUALS(value)` |
| `COMPARE_GREATERTHAN` | `RANGE(value + 1, INT64_MAX)` |
| `COMPARE_GREATERTHANOREQUALTO` | `RANGE(value, INT64_MAX)` |
| `COMPARE_LESSTHAN` | `RANGE(INT64_MIN, value - 1)` |
| `COMPARE_LESSTHANOREQUALTO` | `RANGE(INT64_MIN, value)` |
| `COMPARE_BETWEEN` (if DuckDB preserves it) | `RANGE(lo, hi)` |

**Predicate folding** — two conjunctive predicates on the same label column are merged by intersecting their ranges:
- `> 10 AND < 100` -> `RANGE(max(11, ...), min(99, ...))` = `RANGE[11, 99]`
- `>= 5 AND <= 50` -> `RANGE[5, 50]`
- `> 10 AND > 20` -> `RANGE[21, INT64_MAX]` (take the tighter lower bound)
- `= 42 AND > 10` -> `EQUALS(42)` (equality is tighter than any range; verify 42 satisfies the range)

If folding produces an empty range (e.g., `> 100 AND < 50`), the result is unsatisfiable — the optimizer can short-circuit to an empty result set.

```cpp
// Pseudocode for the detection + folding in optimize_scan.cpp:
static LabelFilter TryExtractLabelFilter(PhysicalFilter &filter,
                                         const VectorIndex &index) {
    if (!index.SupportsLabelFilter()) return LabelFilter::None();
    const auto &label_col = index.GetLabelColumn();
    if (label_col.empty()) return LabelFilter::None();

    // Collect all predicates on the label column.
    // Each becomes a (possibly partial) LabelFilter.
    vector<LabelFilter> partials;
    vector<idx_t> consumed_indices;  // indices into filter.expressions

    for (idx_t i = 0; i < filter.expressions.size(); i++) {
        auto &expr = filter.expressions[i];
        LabelFilter partial;

        if (TryMatchComparison(expr, label_col, partial)) {
            partials.push_back(partial);
            consumed_indices.push_back(i);
        }
        // Also try BETWEEN if DuckDB preserves it as a single expression:
        else if (TryMatchBetween(expr, label_col, partial)) {
            partials.push_back(partial);
            consumed_indices.push_back(i);
        }
    }

    if (partials.empty()) return LabelFilter::None();

    // Fold all partials into one LabelFilter.
    LabelFilter result = FoldLabelFilters(partials);
    if (!result.IsActive()) return LabelFilter::None();  // unsatisfiable

    // Remove consumed predicates from the PhysicalFilter.
    RemoveConsumedPredicates(filter, consumed_indices);

    return result;
}

// Fold multiple partial filters into one by intersection.
// Equality dominates: if any partial is EQUALS(v), and v satisfies all
// range partials, the result is EQUALS(v). Otherwise unsatisfiable.
static LabelFilter FoldLabelFilters(const vector<LabelFilter> &partials) {
    if (partials.size() == 1) return partials[0];

    // Check for equality dominance.
    for (const auto &p : partials) {
        if (p.kind == LabelFilter::Kind::EQUALS) {
            // Verify the value satisfies all other partials.
            for (const auto &q : partials) {
                if (&p == &q) continue;
                if (!q.Matches(p.value)) return LabelFilter::None();  // unsatisfiable
            }
            return p;  // EQUALS is the tightest.
        }
    }

    // All partials are RANGE; intersect them.
    int64_t lo = INT64_MIN, hi = INT64_MAX;
    for (const auto &p : partials) {
        if (p.kind != LabelFilter::Kind::RANGE) continue;
        lo = std::max(lo, p.lo);
        hi = std::min(hi, p.hi);
    }
    if (lo > hi) return LabelFilter::None();  // empty intersection
    return LabelFilter::Range(lo, hi);
}
```

## Implementation Steps

### Step 1: Create `LabelFilter` struct

- [ ] Create `src/include/vindex/label_filter.hpp` with the struct shown in [API Surfaces](#api-surfaces).
- [ ] Include the `Matches()`, `IsActive()`, and factory helpers (`Equals`, `Range`, `None`).
- [ ] Include M7.5 forward-compatibility fields (`NOT_EQUALS`, `IN_LIST`, `values` vector) with a comment marking them as unused in v0.

### Step 2: Add virtual hooks to `VectorIndex`

- [ ] Add `#include "vindex/label_filter.hpp"` to `vector_index.hpp`.
- [ ] Add `SupportsLabelFilter()`, `SetLabelFilter(const LabelFilter &, const string &)`, `GetLabelColumn()` to `VectorIndex` with the default implementations shown above.
- [ ] Add a `label_column_` string member to `VectorIndex` base class (protected).
- [ ] Add a protected setter `SetLabelColumn(const string &col)` for subclasses to call during construction.

### Step 3: Parse `label_column` option

- [ ] In `src/common/vector_index.cpp`, during index construction, check for `label_column` in the `WITH (...)` options map.
- [ ] If present: validate it's a VARCHAR, store it via `SetLabelColumn(value)`.
- [ ] Validate the column exists in the table and is of type BIGINT. If not, throw `BinderException("label_column '%s' must be a BIGINT column")`.
- [ ] If absent: `label_column_` remains empty (no label filtering).

### Step 4: Add `label_column` to all algorithm plan validators

For each algorithm's `CreatePlan` function, add `label_column` to the accepted options list:

- [ ] `src/algo/hnsw/hnsw_plan.cpp`: accept `label_column` in the options loop.
- [ ] `src/algo/ivf/ivf_plan.cpp`: same.
- [ ] `src/algo/diskann/diskann_plan.cpp`: same.
- [ ] `src/algo/spann/spann_plan.cpp`: same.

Each should accept `label_column` as a VARCHAR and let the generic `vector_index.cpp` constructor do the validation.

### Step 5: Implement comparison operator matching

- [ ] Study the existing optimizer in `src/common/optimize_scan.cpp` to understand where the `PhysicalFilter` sits relative to the index scan.
- [ ] Implement `TryMatchComparison(expr, label_col, partial)`:
  - Detect `COMPARE_EQUAL`, `COMPARE_GREATERTHAN`, `COMPARE_GREATERTHANOREQUALTO`, `COMPARE_LESSTHAN`, `COMPARE_LESSTHANOREQUALTO`.
  - Verify one side is a column ref matching `label_col` and the other is a BIGINT literal.
  - Handle reversed operands (`42 > cat` is the same as `cat < 42`).
  - Map to the appropriate `LabelFilter` per the table in [API Surfaces](#api-surfaces).
- [ ] Implement `TryMatchBetween(expr, label_col, partial)`:
  - Check if DuckDB preserves `BETWEEN` as a single expression type or desugars it to `>= AND <=`.
  - If preserved: extract both bounds, create `RANGE(lo, hi)`.
  - If desugared: the two conjuncts will be caught by `TryMatchComparison` and folded in Step 6.

> **Note**: Verify DuckDB's expression representation for `BETWEEN` during implementation. DuckDB may desugar `x BETWEEN a AND b` into `x >= a AND x <= b` during binding. In that case, `TryMatchBetween` is unnecessary and the two conjuncts are handled by `TryMatchComparison` + folding.

### Step 6: Implement predicate folding

- [ ] Implement `FoldLabelFilters(partials)` as shown in [API Surfaces](#api-surfaces).
- [ ] Equality dominance: if any partial is `EQUALS(v)`, verify `v` satisfies all other partials; if yes, result is `EQUALS(v)`.
- [ ] Range intersection: if all partials are `RANGE`, intersect to `RANGE(max_lo, min_hi)`.
- [ ] Empty intersection: return `LabelFilter::None()` (unsatisfiable; optimizer can short-circuit).

### Step 7: Wire into the main optimizer rewrite

- [ ] In the main optimizer rewrite: after identifying a `VectorIndex` for the `ORDER BY array_distance(...) LIMIT k` rewrite, call `TryExtractLabelFilter(filter, index)`.
- [ ] If the result is active: call `index.SetLabelFilter(scan_state, filter, label_col)`.
- [ ] Remove consumed predicates from the `PhysicalFilter` (or remove the entire `PhysicalFilter` if all predicates were consumed).
- [ ] If the result is `NONE` or no predicates matched: leave the `PhysicalFilter` unchanged.

### Step 8: Handle the scan-state plumbing

- [ ] The `IndexScanState` base class carries a `LabelFilter label_filter` member (default: `LabelFilter::None()`).
- [ ] `InitializeScan` and `ExecuteMultiScan` implementations read this filter to scope their search.
- [ ] For algorithms that don't override `SupportsLabelFilter()` (i.e., return false), the label filter is ignored and predicates stay in the `PhysicalFilter` for post-hoc execution.

### Step 9: Write optimizer unit tests

- [ ] Create `test/unit/test_label_filter_optimizer.cpp`.
- [ ] Test cases:
  - [ ] `DetectEquals`: `WHERE cat = 42` -> `EQUALS(42)`.
  - [ ] `DetectGreaterThan`: `WHERE cat > 10` -> `RANGE(11, INT64_MAX)`.
  - [ ] `DetectGreaterOrEqual`: `WHERE cat >= 10` -> `RANGE(10, INT64_MAX)`.
  - [ ] `DetectLessThan`: `WHERE cat < 100` -> `RANGE(INT64_MIN, 99)`.
  - [ ] `DetectLessOrEqual`: `WHERE cat <= 100` -> `RANGE(INT64_MIN, 100)`.
  - [ ] `DetectBetween`: `WHERE cat BETWEEN 10 AND 100` -> `RANGE(10, 100)`.
  - [ ] `FoldConjuncts`: `WHERE cat > 10 AND cat < 100` -> `RANGE(11, 99)`.
  - [ ] `FoldOverlappingRanges`: `WHERE cat >= 5 AND cat <= 50` -> `RANGE(5, 50)`.
  - [ ] `FoldTighterLowerBound`: `WHERE cat > 10 AND cat > 20` -> `RANGE(21, INT64_MAX)`.
  - [ ] `FoldEqualityWithRange`: `WHERE cat = 42 AND cat > 10` -> `EQUALS(42)`.
  - [ ] `FoldUnsatisfiable`: `WHERE cat > 100 AND cat < 50` -> `NONE` (empty result).
  - [ ] `RejectNonLabelColumn`: `WHERE other_col = 42` -> not consumed (stays in PhysicalFilter).
  - [ ] `RejectVarcharLiteral`: `WHERE cat = 'foo'` -> not consumed (label must be BIGINT).
  - [ ] `RejectSubquery`: `WHERE cat = (SELECT ...)` -> not consumed (non-literal).
  - [ ] `ReversedOperands`: `WHERE 42 = cat` and `WHERE 100 > cat` -> detected correctly.
  - [ ] `MixedPredicates`: `WHERE cat > 10 AND other_col = 5` -> only `cat > 10` consumed, `other_col = 5` stays.
  - [ ] `NoLabelColumn`: index created without `label_column` -> no filter consumed.

### Step 10: Write SQL logic tests

- [ ] Create `test/sql/common/label_filter_optimizer.test`.
- [ ] Test that `EXPLAIN` shows the filter is consumed by the index scan when matched.
- [ ] Test each operator: `=`, `>`, `>=`, `<`, `<=`, `BETWEEN`.
- [ ] Test folded conjuncts: `> 10 AND < 100`.
- [ ] Test that `EXPLAIN` shows the filter remains as a `PhysicalFilter` when not matched.
- [ ] Test correctness: results are the same whether or not the filter is index-routed.

> **Note**: Since no real algorithm overrides `SupportsLabelFilter()` in this phase (that's Phase 6 for AiSAQ), the SQL tests verify the *absence* of index-routed filtering (the filter stays as `PhysicalFilter` until an algorithm opts in). Full integration tests come in Phase 6.

### Step 11: Update `AGENTS.md`

- [ ] Add a new "Label-column filtering" subsection under "Source wiring".
- [ ] Document the `WITH (label_column = 'col')` option as shared across all vector index types.
- [ ] Document the supported operators: `=`, `>`, `>=`, `<`, `<=`, `BETWEEN`, and conjunctive folding.
- [ ] Document the unsupported operators (deferred): `!=` (M7.5), `IN` (M7.5), VARCHAR labels (M7).
- [ ] Note that no algorithm uses label filtering yet (AiSAQ will be the first in Phase 6).

## Acceptance Criteria

- [ ] `LabelFilter` struct exists in `src/include/vindex/label_filter.hpp` with `EQUALS`, `RANGE`, and forward-compatible `NOT_EQUALS` / `IN_LIST` kinds.
- [ ] `VectorIndex` has `SupportsLabelFilter()`, `SetLabelFilter(const LabelFilter &, ...)`, `GetLabelColumn()` virtuals with safe defaults.
- [ ] `WITH (label_column = 'col')` option is parsed for all algorithm types.
- [ ] `optimize_scan.cpp` detects `=`, `>`, `>=`, `<`, `<=`, `BETWEEN` on the designated label column.
- [ ] Two conjunctive range predicates on the same label column are folded into one `RANGE`.
- [ ] Non-matching filters fall through to `PhysicalFilter` unchanged.
- [ ] Unit tests pass (`make unit`).
- [ ] SQL logic tests pass (`make test`).
- [ ] No existing test regresses.
- [ ] `AGENTS.md` is updated per the protocol.
- [ ] Phase status updated to `- [x]` in [AISAQ.md](AISAQ.md).

## Open Questions

- **Q1**: Should the `label_column` designation be per-index (stored in `IndexCatalogEntry`) or per-session? Per-index is correct — the column is bound at `CREATE INDEX` time.
- **Q2**: What happens if the user drops the label column after creating the index? DuckDB should prevent this via column dependency tracking. Verify during implementation.
- **Q3**: Does DuckDB preserve `BETWEEN` as a single expression type, or desugar it to `>= AND <=` during binding? If desugared, `TryMatchBetween` is unnecessary and the folding logic handles it. Verify during implementation.
- **Q4**: Should we support `WHERE label_col IS NULL` as a filter? Not in v0 — NULL handling adds complexity. Defer.
