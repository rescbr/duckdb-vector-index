# Phase 3 — Quantizer LUT Virtuals

**Branch**: `feat/aisaq`
**Status**: `- [x]` complete
**Effort**: 1 week
**Depends on**: Phase 2 merged to main
**Dependents**: Phase 4 (Backend interface uses these virtuals), Phase 5 (AiSAQ uses these for batched distance)

> **Maintenance note**: Update task markers (`- [ ]` -> `- [~]` -> `- [x]`) as you progress. At phase completion, update the phase status in [AISAQ.md](AISAQ.md) and update `AGENTS.md` per the [AGENTS.md Update Protocol](AISAQ.md#agentsmd-update-protocol).

---

## Goal

Expose the existing LUT-based PQ distance computation as explicitly-named virtual methods on `Quantizer`. This is an **improvement, not a semantic change** — `PqQuantizer::EstimateDistance` already uses the LUT internally (`pq_quantizer.cpp:167-175`). The new virtuals just make the LUT semantics explicit and accessible to callers (like AiSAQ) who want to batch distances over many codes with one precomputed LUT.

## Why this is an improvement, not a new implementation

Looking at the current code:

```cpp
// pq_quantizer.cpp:167-175 — EXISTING code
float PqQuantizer::EstimateDistance(const_data_ptr_t code, const float *query_preproc) const {
    const idx_t k = CentroidsPerSlot();
    float acc = 0.0f;
    for (idx_t s = 0; s < idx_t(m_); s++) {
        const uint32_t cid = ReadCode(code, s);
        acc += query_preproc[s * k + cid];  // <-- this IS a LUT gather
    }
    return acc;
}
```

The `query_preproc` parameter **is** the LUT. `EstimateDistance` **is** a LUT gather. The new virtuals just:
1. Name the LUT computation explicitly (`PopulateDistanceLUT` = what `PreprocessQuery` already does).
2. Name the gather explicitly (`LUTDistance` = what `EstimateDistance` already does).
3. Allow the gather to be called in a tight loop over many codes without per-code function call overhead.

No new code paths. No precision changes. No side-effects. Existing algorithms are unaffected because the default implementations fall back to the existing `PreprocessQuery` + `EstimateDistance` path.

## Scope

### In scope

- Two new virtual methods on `Quantizer` with default fallback implementations.
- `PqQuantizer` override: expose the existing LUT path explicitly.
- `ScannQuantizer` override: anisotropic variant (same LUT math, different LUT population due to rotation + anisotropic weighting).
- `FlatQuantizer` and `RabitqQuantizer`: inherit the default fallback (no override needed).
- Unit test extensions verifying `LUTDistance` matches `EstimateDistance` to float tolerance.

### Out of scope

- Batched multi-code distance method (`BatchLUTDistance`) — that's on the `Backend` interface (Phase 4), not on `Quantizer`.
- Changes to how `PreprocessQuery` works internally.
- Any change to existing algorithm code paths (HNSW, IVF, DiskANN, SPANN continue using `EstimateDistance` unchanged).

## Files

### Changed files

| Path | Change |
|---|---|
| `src/include/vindex/quantizer.hpp` | Add `PopulateDistanceLUT` + `LUTDistance` virtuals with defaults |
| `src/include/vindex/pq_quantizer.hpp` | Declare overrides |
| `src/quant/pq/pq_quantizer.cpp` | Implement overrides (delegates to existing internals) |
| `src/include/vindex/scann_quantizer.hpp` | Declare overrides |
| `src/quant/scann/scann_quantizer.cpp` | Implement overrides |
| `test/unit/test_pq_quantizer.cpp` | Add LUT equivalence tests |
| `test/unit/test_scann_quantizer.cpp` | Add LUT equivalence tests |

## API Surfaces

### `src/include/vindex/quantizer.hpp` (additions)

```cpp
class Quantizer {
public:
    // ... existing API ...

    // --- LUT-based distance (Phase 3) ---

    // Populate a look-up table of per-chunk distances from the (already
    // preprocessed) query to all PQ centroids. The LUT has layout:
    //   lut[s * CentroidsPerSlot() + c] = distance(query_chunk_s, centroid_c)
    // for s in [0, m) and c in [0, CentroidsPerSlot()).
    //
    // Default implementation: no-op (signals LUT path unavailable). PqQuantizer
    // and ScannQuantizer override with the actual computation.
    //
    // `lut_out` must have room for LUTSize() floats.
    virtual void PopulateDistanceLUT(const float *query_preproc, float *lut_out) const;

    // Estimated distance from a code to a query, using a pre-populated LUT.
    // This is a gather-and-sum over the LUT entries indexed by the code's
    // centroid ids. Must produce the same value as EstimateDistance(code,
    // query_preproc) for the same query.
    //
    // Default implementation: falls back to EstimateDistance (ignores the LUT).
    virtual float LUTDistance(const_data_ptr_t code, const float *lut) const;

    // Size of the LUT in floats. Equal to m * CentroidsPerSlot() for PQ/ScaNN.
    // Default: 0 (signals LUT path unavailable).
    virtual idx_t LUTSize() const;
};
```

### `src/include/vindex/pq_quantizer.hpp` (additions)

```cpp
class PqQuantizer : public Quantizer {
public:
    // ... existing API ...

    // Phase 3: LUT virtuals.
    void PopulateDistanceLUT(const float *query_preproc, float *lut_out) const override;
    float LUTDistance(const_data_ptr_t code, const float *lut) const override;
    idx_t LUTSize() const override;

private:
    // ... existing members ...
};
```

## Implementation Steps

### Step 1: Add virtuals to `Quantizer` base

- [ ] Add `PopulateDistanceLUT`, `LUTDistance`, `LUTSize` to `src/include/vindex/quantizer.hpp`.
- [ ] Default implementations in `src/quant/quantizer.cpp` (or inline in header):
  - `PopulateDistanceLUT`: no-op.
  - `LUTDistance`: `return EstimateDistance(code, /* query_preproc = */ lut);` — wait, this is wrong because `lut` is not `query_preproc`. The fallback should be: `return EstimateDistance(code, query_preproc_cached_)` — but we don't have a cached query. **Resolution**: the default `LUTDistance` cannot fall back correctly without a query. Instead, the default should throw or return NaN, and callers must check `LUTSize() > 0` before using the LUT path. Document this.
  - `LUTSize`: return 0.

> **Important design note**: The default `LUTDistance` cannot meaningfully fall back because it doesn't have the original query — only the LUT. The correct pattern is:
> - Callers check `quantizer.LUTSize() > 0` before using LUT path.
> - If `LUTSize() == 0`: use `PreprocessQuery` + `EstimateDistance` (the existing path).
> - If `LUTSize() > 0`: use `PopulateDistanceLUT` + `LUTDistance` (the new path).
> - Both paths produce identical distances for PQ/ScaNN quantizers.

### Step 2: Implement `PqQuantizer` overrides

- [ ] `PopulateDistanceLUT`: copy `query_preproc` into `lut_out` (they're the same thing for PQ — `PreprocessQuery` already writes the LUT). Size = `LUTSize()`.
- [ ] `LUTDistance`: identical to the existing `EstimateDistance` gather loop.
- [ ] `LUTSize`: return `m_ * CentroidsPerSlot()`.

```cpp
void PqQuantizer::PopulateDistanceLUT(const float *query_preproc, float *lut_out) const {
    const idx_t size = LUTSize();
    memcpy(lut_out, query_preproc, size * sizeof(float));
}

float PqQuantizer::LUTDistance(const_data_ptr_t code, const float *lut) const {
    // Identical to EstimateDistance — the LUT IS query_preproc for PQ.
    const idx_t k = CentroidsPerSlot();
    float acc = 0.0f;
    for (idx_t s = 0; s < idx_t(m_); s++) {
        const uint32_t cid = ReadCode(code, s);
        acc += lut[s * k + cid];
    }
    return acc;
}

idx_t PqQuantizer::LUTSize() const {
    return idx_t(m_) * CentroidsPerSlot();
}
```

### Step 3: Implement `ScannQuantizer` overrides

- [ ] Study `src/quant/scann/scann_quantizer.cpp` to understand how ScaNN's anisotropic LUT differs from PQ's.
- [ ] `PopulateDistanceLUT`: same as PQ but with anisotropic weighting applied during population (the rotation is already in `PreprocessQuery`).
- [ ] `LUTDistance`: same gather as PQ.
- [ ] `LUTSize`: same as PQ (`m * CentroidsPerSlot()`).

### Step 4: Write equivalence tests

- [ ] In `test/unit/test_pq_quantizer.cpp`, add:
  - `LUTEquivalence`: for random queries and codes, verify `LUTDistance(code, lut) == EstimateDistance(code, query_preproc)` within float epsilon.
  - `LUTSizeCorrect`: verify `LUTSize() == m * 2^bits`.
  - `LUTPopulatedCorrectly`: verify `PopulateDistanceLUT` produces the same bytes as `PreprocessQuery`.
- [ ] In `test/unit/test_scann_quantizer.cpp`, add the same three tests for ScaNN.

### Step 5: Verify no regression

- [ ] `make test` — all SQL tests pass.
- [ ] `make unit` — all unit tests pass.
- [ ] `make bench` — recall thresholds unchanged.

### Step 6: Update `AGENTS.md`

- [ ] Add LUT virtuals to the quantizer description in "Source wiring".
- [ ] Note that the LUT path is opt-in (`LUTSize() > 0` check).

## Acceptance Criteria

- [ ] `Quantizer` has `PopulateDistanceLUT`, `LUTDistance`, `LUTSize` virtuals.
- [ ] `PqQuantizer` and `ScannQuantizer` override all three.
- [ ] `FlatQuantizer` and `RabitqQuantizer` inherit defaults (`LUTSize() = 0`).
- [ ] LUT distance matches `EstimateDistance` within float tolerance for PQ and ScaNN.
- [ ] No existing test regresses.
- [ ] `AGENTS.md` is updated.
- [ ] Phase status updated to `- [x]` in [AISAQ.md](AISAQ.md).
