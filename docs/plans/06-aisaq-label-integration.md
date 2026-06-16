# Phase 6 — AiSAQ Label Integration

**Branch**: `feat/aisaq`
**Status**: `- [ ]` not started
**Effort**: 2-3 weeks
**Depends on**: Phase 5 (AiSAQ core), Phase 2 (general WHERE-label optimizer)
**Dependents**: Phase 7 (tests + bench)

---

## Goal

Make AiSAQ the first algorithm to opt into the general WHERE-label filter infrastructure from Phase 2. At build time, AiSAQ reads the designated label column, maintains a per-label medoid list, and persists the label map. At search time, when the optimizer routes a `LabelFilter` to the scan state, AiSAQ narrows its entry-point selection and candidate filtering based on the filter kind (`EQUALS` or `RANGE`).

This preserves recall under heavy filtering — the primary use case being "find similar items within a specific category" or "find similar items in a range of categories".

## Background

Phase 2 added the general infrastructure:
- `LabelFilter` struct (`src/include/vindex/label_filter.hpp`) supporting `EQUALS` and `RANGE` kinds.
- `VectorIndex::SupportsLabelFilter()` virtual (default false).
- `VectorIndex::SetLabelFilter(state, const LabelFilter &, label_column)` virtual (default no-op).
- `VectorIndex::GetLabelColumn()` virtual (default empty string).
- `optimize_scan.cpp` detects `=`, `>`, `>=`, `<`, `<=`, `BETWEEN`, and folds conjunctive range predicates. Routes the resulting `LabelFilter` to the index.

Phase 5 built AiSAQ without labels (`n_labels = 0` placeholder in the state stream). This phase fills in the label-aware behavior for both `EQUALS` and `RANGE` filter kinds.

## Adaptive Range Medoid Strategy

The key design question is how `RANGE` filters affect entry-point selection at search time. An `EQUALS` filter maps to exactly one label, so one medoid — a clean fast-path. A `RANGE` filter can match many labels, requiring a different approach.

### Strategy

```
At search time, given a LabelFilter:

  EQUALS(value):
    → Use the single label's medoid as the entry point.
    → Filter candidates: label == value.
    → Best recall (single targeted starting point).

  RANGE(lo, hi):
    → Count matching labels: N = |{label : lo <= label <= hi}|.
    →
    → If N <= n_entry_points (default 16):
        → Use ALL matching labels' medoids as entry points.
        → High recall (multi-start from every relevant region of the graph).
    →
    → If N > n_entry_points:
        → Fall back to the global k-means entry points (from Phase 5).
        → Filter candidates: lo <= label <= hi during graph expansion.
        → Recall is lower than multi-medoid but bounded I/O at search start.
```

**Rationale for the threshold**: using all matching medoids means N starting points, each requiring one PQ-code pin for the initial distance computation. With N = 16 (the default entry-point count), this is the same cost as the unfiltered search's entry-point evaluation. Beyond that, the marginal recall gain diminishes while I/O cost grows linearly.

**Documentation requirement**: the adaptive strategy must be documented in:
1. `aisaq_core.hpp` — as a comment block above `Search`.
2. `aisaq_plan.cpp` — in the `aisaq_entry_points` option documentation (note that this parameter also controls the range-medoid threshold).
3. `AGENTS.md` — in the "AiSAQ algorithm" subsection.

## Scope

### In scope

- AiSAQ overrides `SupportsLabelFilter() -> true`.
- `WITH (label_column = 'cat_id')` validation in `aisaq_plan.cpp` (column must exist, must be BIGINT).
- Build-time: read the label column alongside the vector; maintain `(internal_id -> label)` and `(label -> medoid)` maps; compute per-label medoid lists.
- Search-time:
  - `EQUALS` filter: medoid fast-path (single label's medoid, exact candidate filtering).
  - `RANGE` filter: adaptive multi-medoid strategy (all matching medoids up to `n_entry_points`, then global entry points + range candidate filtering).
- Persistence: the `n_labels` field in the state stream is populated; per-label medoid lists are serialized.
- SQL logic tests for all filter kinds (`=`, `>`, `>=`, `<`, `<=`, `BETWEEN`, folded conjuncts).
- Unit tests for medoid selection correctness and recall under filtering.

### Out of scope

- VARCHAR labels (M7).
- `!=` (not-equal) and `IN (...)` (M7.5). The `LabelFilter` struct from Phase 2 is forward-compatible.
- Multi-column labels.
- Label-based partitioning of the graph (the graph remains global; only entry-point selection and candidate filtering are label-aware).

## Files

### Changed files

| Path | Change |
|---|---|
| `src/algo/aisaq/aisaq_index.hpp` | Add label-related members: `label_column_`, `label_to_internal_ids_`, per-label medoid map |
| `src/algo/aisaq/aisaq_index.cpp` | Override `SupportsLabelFilter`, `SetLabelFilter(LabelFilter)`, `GetLabelColumn`; implement label-aware scan |
| `src/algo/aisaq/aisaq_plan.cpp` | Accept `label_column` option; validate column exists and is BIGINT |
| `src/algo/aisaq/aisaq_build.hpp` | Extend `PhysicalCreateAiSaqIndex` to read the label column |
| `src/algo/aisaq/aisaq_build.cpp` | Read label column in parallel sink; pass to `AiSaqCore::Insert` |
| `src/algo/aisaq/aisaq_core.hpp` | Add label parameter to `Insert`; add per-label medoid computation; add adaptive range entry-point selection |
| `src/algo/aisaq/aisaq_core.cpp` | Maintain `(row_id, label)` map; medoid computation; adaptive search entry-point selection; candidate filtering |
| `test/unit/test_aisaq_core.cpp` | Add label-filter correctness tests (EQUALS + RANGE) |
| `test/sql/aisaq/aisaq_filter.test` | New: SQL logic tests for all filter operators |

## API Surfaces

### `AiSaqIndex` additions

```cpp
#include "vindex/label_filter.hpp"

class AiSaqIndex : public VectorIndex {
public:
    // Phase 6: label filter hooks
    bool SupportsLabelFilter() const override { return !label_column_.empty(); }
    const string &GetLabelColumn() const override { return label_column_; }
    void SetLabelFilter(IndexScanState &state, const LabelFilter &filter,
                        const string &label_column) const override;

private:
    string label_column_;  // empty = no label filtering

    // label -> set of internal_ids belonging to that label.
    // Built at construction time; used for medoid selection and candidate filtering.
    unordered_map<int64_t, vector<uint32_t>> label_to_internal_ids_;

    // label -> medoid internal_id (the node closest to the label's centroid).
    unordered_map<int64_t, uint32_t> label_to_medoid_;
};
```

### `AiSaqCore` additions

```cpp
#include "vindex/label_filter.hpp"

class AiSaqCore {
public:
    // Insert now takes an optional label. label = INT64_MIN means "no label".
    uint32_t Insert(int64_t row_id, const float *vec, int64_t label);

    // Search now takes a LabelFilter. When the filter is active:
    //
    //   EQUALS(value):
    //     Entry point = label_to_medoid_[value].
    //     Candidate filter: label == value.
    //
    //   RANGE(lo, hi):
    //     Count matching labels N = |{l : lo <= l <= hi}|.
    //     If N <= params_.n_entry_points:
    //       Entry points = {label_to_medoid_[l] for all matching l}.
    //     Else:
    //       Entry points = global k-means entry points (from Phase 5).
    //     Candidate filter: lo <= label <= hi.
    //
    //   NONE:
    //     Entry points = global k-means entry points.
    //     No candidate filtering.
    vector<Candidate> Search(const float *query_lut, idx_t k, idx_t L_search,
                             idx_t beam_width, idx_t io_limit,
                             const LabelFilter &label_filter = LabelFilter::None()) const;

    // Compute per-label medoids after all inserts.
    void ComputeLabelMedoids();

    // Count labels matching a RANGE filter.
    idx_t CountLabelsInRange(int64_t lo, int64_t hi) const;

    // Get the medoids for all labels matching a RANGE filter.
    vector<uint32_t> GetMedoidsInRange(int64_t lo, int64_t hi, idx_t max_count) const;

private:
    // internal_id -> label (for candidate filtering during search).
    unordered_map<uint32_t, int64_t> internal_id_to_label_;

    // label -> medoid internal_id.
    unordered_map<int64_t, uint32_t> label_medoids_;

    // Sorted label keys for efficient range queries.
    vector<int64_t> sorted_labels_;
};
```

## Implementation Steps

### Step 1: Add label support to `aisaq_plan.cpp`

- [ ] Accept `label_column` as a VARCHAR option.
- [ ] If present: look up the column in the table; verify it exists and is BIGINT.
- [ ] If the column doesn't exist: `BinderException("AiSAQ label_column '%s' not found in table")`.
- [ ] If the column is not BIGINT: `BinderException("AiSAQ label_column '%s' must be BIGINT")`.
- [ ] Store the column name for the build pipeline.

### Step 2: Extend `PhysicalCreateAiSaqIndex` to read the label column

- [ ] The parallel sink currently reads `(vec, row_id)` pairs. Extend to read `(vec, row_id, label)`.
- [ ] The label column id is resolved from the `label_column` option during planning.
- [ ] Pass the label through to `AiSaqIndex::Construct` and then to `AiSaqCore::Insert`.

### Step 3: Implement `AiSaqCore::Insert` with labels

- [ ] Extend the `Insert` signature to take `int64_t label`.
- [ ] Store `internal_id_to_label_[new_internal_id] = label`.
- [ ] If `label != INT64_MIN`: add `new_internal_id` to `label_to_internal_ids_[label]`.
- [ ] Otherwise: no label bookkeeping (the index is label-free).

### Step 4: Build `sorted_labels_` for range queries

- [ ] After all inserts are complete (before `ComputeLabelMedoids`), build `sorted_labels_` from the keys of `label_to_internal_ids_`.
- [ ] Sort ascending.
- [ ] This enables `std::lower_bound` / `std::upper_bound` for efficient `CountLabelsInRange` and `GetMedoidsInRange`.

### Step 5: Implement `ComputeLabelMedoids`

- [ ] Called once after all inserts and after `FinalizeInlineCodes`.
- [ ] For each label in `label_to_internal_ids_`:
  - Compute the centroid of all member vectors (or approximate via their PQ codes).
  - Find the member whose PQ code is closest to the centroid (by `LUTDistance` or `CodeDistance`).
  - Store as `label_medoids_[label] = best_internal_id`.

### Step 6: Implement range query helpers

- [ ] `CountLabelsInRange(lo, hi)`: use `std::lower_bound` / `std::upper_bound` on `sorted_labels_` to count labels in `[lo, hi]`. O(log N).
- [ ] `GetMedoidsInRange(lo, hi, max_count)`: collect up to `max_count` medoids for labels in `[lo, hi]`. Returns empty if count exceeds `max_count`.

### Step 7: Implement label-aware `Search`

- [ ] If `label_filter.kind == NONE`: unchanged behavior from Phase 5 (global entry points, no candidate filtering).
- [ ] If `label_filter.kind == EQUALS`:
  - Look up `label_medoids_[label_filter.value]`. If not found, return empty results (no vectors with that label).
  - Start `BeamSearch` from that single medoid.
  - During neighbor expansion: skip any neighbor whose `internal_id_to_label_[nid] != label_filter.value`.
- [ ] If `label_filter.kind == RANGE`:
  - Count matching labels via `CountLabelsInRange(lo, hi)`.
  - **If count <= `params_.n_entry_points`**: collect matching medoids via `GetMedoidsInRange(lo, hi, params_.n_entry_points)`. Start `BeamSearch` from ALL of them (multi-start). During expansion: skip any neighbor whose label is outside `[lo, hi]`.
  - **If count > `params_.n_entry_points`**: fall back to global k-means entry points (from Phase 5). During expansion: skip any neighbor whose label is outside `[lo, hi]`.
- [ ] Document the adaptive strategy in a comment block above `Search`.

### Step 8: Implement `AiSaqIndex::SetLabelFilter`

- [ ] Store the `LabelFilter` in the scan state (Phase 2 added the `label_filter` member to `IndexScanState`).
- [ ] In `InitializeScan` / `ExecuteMultiScan`: read the filter from the scan state and pass it to `core_->Search`.

### Step 9: Update persistence format

- [ ] The state stream already has a `n_labels` field (set to 0 in Phase 5).
- [ ] Populate it: serialize `label_medoids_` as `(label: int64, medoid: uint32)` pairs.
- [ ] Serialize `internal_id_to_label_` as part of the `(row_id, internal_id, label)` tuples.
- [ ] `ReadStateStream`: deserialize the label map into `AiSaqCore`, rebuild `sorted_labels_`.

### Step 10: Write SQL logic tests

- [ ] Create `test/sql/aisaq/aisaq_filter.test`.
- [ ] Test cases for `EQUALS`:
  - [ ] Create an index with `WITH (label_column='category')`.
  - [ ] Query with `WHERE category = 1`: verify results are all from category 1.
  - [ ] Query with `WHERE category = 2`: verify different results from category 1.
  - [ ] Query with `WHERE category = 999` (nonexistent): verify empty results.
  - [ ] `EXPLAIN` the filtered query: verify the filter is consumed by the index scan.
- [ ] Test cases for `RANGE` (each operator):
  - [ ] `WHERE category > 5`: verify results are all from categories 6+.
  - [ ] `WHERE category >= 5`: verify results include category 5.
  - [ ] `WHERE category < 5`: verify results are all from categories 0-4.
  - [ ] `WHERE category <= 5`: verify results include category 5.
  - [ ] `WHERE category BETWEEN 3 AND 7`: verify results are from categories 3-7.
  - [ ] `WHERE category > 3 AND category < 7` (folded): verify same as BETWEEN 4 AND 6.
  - [ ] `EXPLAIN` each: verify the filter is consumed by the index scan.
- [ ] Test fallback cases:
  - [ ] Query without WHERE: verify all categories represented.
  - [ ] Query with `WHERE non_label_col = 5`: verify NOT consumed (stays as PhysicalFilter).
  - [ ] Query with `WHERE category = (subquery)`: verify NOT consumed (non-literal).

### Step 11: Write unit tests

- [ ] Add to `test/unit/test_aisaq_core.cpp`:
  - [ ] `LabelMedoidSelection`: build with 3 labels, verify each label gets a distinct medoid.
  - [ ] `EqualsFilterRecall`: build with skewed labels (90% label 1, 10% label 2), query label 2 with `EQUALS` — verify recall is high.
  - [ ] `RangeFilterMultiMedoid`: build with 10 labels, query `RANGE(1, 5)` — verify all matching labels' medoids are used as entry points (count <= n_entry_points).
  - [ ] `RangeFilterFallback`: build with 100 labels, query `RANGE(1, 100)` — verify fallback to global entry points (count > n_entry_points).
  - [ ] `RangeFilterCorrectness`: verify `WHERE cat > 5` returns only labels > 5; verify `BETWEEN` returns only labels in range.
  - [ ] `FoldedRangeCorrectness`: verify `cat > 3 AND cat < 7` returns same results as `cat BETWEEN 4 AND 6`.
  - [ ] `NoLabelFallback`: build without labels, query — verify it works as in Phase 5.
  - [ ] `PersistenceWithLabels`: build with labels, serialize, deserialize, query with `EQUALS` and `RANGE` filters — verify identical results.

### Step 12: Update `AGENTS.md`

- [ ] Update the "AiSAQ algorithm" subsection with label-column behavior.
- [ ] Document the `WITH (label_column = 'col')` option.
- [ ] Document the supported operators: `=`, `>`, `>=`, `<`, `<=`, `BETWEEN`, folded conjuncts.
- [ ] Document the adaptive range medoid strategy:
  - `EQUALS`: single label's medoid.
  - `RANGE` with matching labels <= `n_entry_points`: all matching medoids as entry points.
  - `RANGE` with matching labels > `n_entry_points`: global entry points + range candidate filtering.
- [ ] Note: `!=` and `IN` are M7.5; VARCHAR labels are M7.

## Acceptance Criteria

- [ ] `WITH (label_column = 'cat')` works when `cat` is a BIGINT column.
- [ ] `WHERE cat = X` routes through the index scan (visible in `EXPLAIN`).
- [ ] `WHERE cat > X`, `>=`, `<`, `<=`, `BETWEEN` all route through the index scan.
- [ ] Folded conjuncts (`> 3 AND < 7`) route as a single `RANGE` filter.
- [ ] `EQUALS` queries use the label's medoid as the single entry point.
- [ ] `RANGE` queries with few matching labels use multi-medoid entry points.
- [ ] `RANGE` queries with many matching labels fall back to global entry points.
- [ ] Filtered queries return only matching-label results.
- [ ] Recall under heavy filtering is significantly better than post-hoc filtering.
- [ ] Non-literal filters fall through to post-hoc correctly.
- [ ] Persistence round-trips the label map.
- [ ] `make test` and `make unit` pass.
- [ ] `AGENTS.md` is updated.
- [ ] Phase status updated to `- [x]` in [AISAQ.md](AISAQ.md).
