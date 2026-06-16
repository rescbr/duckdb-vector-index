# Phase 4 — Backend Interface

**Branch**: `feat/aisaq`
**Status**: `- [x]` complete
**Effort**: 1 week
**Depends on**: Phase 3 (LUT virtuals)
**Dependents**: Phase 5 (AiSAQ core routes all batched math through this)

> **Maintenance note**: Update task markers (`- [ ]` -> `- [~]` -> `- [x]`) as you progress. At phase completion, update the phase status in [AISAQ.md](AISAQ.md) and update `AGENTS.md` per the [AGENTS.md Update Protocol](AISAQ.md#agentsmd-update-protocol).

---

## Goal

Create the `src/backend/` abstraction layer with an abstract `Backend` base class and a concrete `CpuBackend` implementation. This establishes the interface that AiSAQ will route its batched math through, and that a future `VulkanBackend` (Phase 8) can plug into.

**Design constraints** (locked decisions from planning):
- No factory, no registry, no plugin system.
- Backend selection is runtime (`vindex_gpu_backend` session option).
- Backend inclusion is compile-time (`#ifdef VINDEX_BACKEND_VULKAN`).
- Naming follows project convention: `Backend` (abstract, no `I` prefix), `CpuBackend` / `VulkanBackend` (concrete, no `Impl` suffix).
- Directory: `src/backend/` (singular, matching `src/algo/`, `src/quant/`).

## Scope

### In scope

- `src/backend/` directory with CMakeLists.
- Abstract `Backend` base class declaring the operations that benefit from GPU acceleration.
- Concrete `CpuBackend` that delegates to existing CPU code paths.
- `active_backend.hpp` with the `GetActiveBackend(ctx)` selector function.
- Session option `vindex_gpu_backend` (VARCHAR, default `"cpu"`).
- `kKnownBackends = {"cpu", "vulkan"}` constant (vulkan not compiled in this phase).
- Unit tests verifying `CpuBackend` delegates correctly.

### Out of scope

- `VulkanBackend` implementation (Phase 8).
- Wiring AiSAQ to use the backend (Phase 5).
- Actual GPU acceleration.

## Files

### New files

| Path | Purpose |
|---|---|
| `src/backend/CMakeLists.txt` | Wires `cpu_backend.cpp` into `EXTENSION_SOURCES` |
| `src/backend/backend.hpp` | Abstract `Backend` base class + `kKnownBackends` |
| `src/backend/cpu_backend.hpp` | Concrete `CpuBackend` declaration |
| `src/backend/cpu_backend.cpp` | `CpuBackend` implementation (delegates to existing code) |
| `src/backend/active_backend.hpp` | `GetActiveBackend(ctx)` selector function |
| `test/unit/test_backend.cpp` | Unit tests for `CpuBackend` delegation |

### Changed files

| Path | Change |
|---|---|
| `src/CMakeLists.txt` | Add `add_subdirectory(backend)` |

## API Surfaces

### `src/backend/backend.hpp` (new)

```cpp
#pragma once

#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/array.hpp"
#include "duckdb/common/unique_ptr.hpp"

namespace duckdb {
class ClientContext;
} // namespace duckdb

namespace duckdb {
namespace vindex {

// Known backend identifiers. v0 ships CpuBackend only; VulkanBackend is
// tracked as Phase 8. Additional backends would be appended here if needed.
constexpr std::array<const char *, 2> kKnownBackends = {"cpu", "vulkan"};

// ---------------------------------------------------------------------------
// Backend — abstract interface for batched math operations that may benefit
// from GPU acceleration.
//
// Design:
//   - Direct inheritance. No factory, no registry, no plugin system.
//   - Runtime selection via vindex_gpu_backend session option.
//   - Compile-time inclusion of VulkanBackend via #ifdef VINDEX_BACKEND_VULKAN.
//   - CpuBackend is always compiled in and is the default.
//
// Performance:
//   - Dispatch happens once per query or once per build, not per code.
//   - Inner loops (BatchLUTDistance, etc.) process many codes per call,
//     amortizing virtual dispatch to zero.
// ---------------------------------------------------------------------------
class Backend {
public:
    virtual ~Backend() = default;

    virtual const string &Name() const = 0;

    // --- Build-time operations (where GPU sees the biggest win) ---

    // Train a quantizer on a sample of vectors. Delegates to
    // Quantizer::Train on CPU; GPU implementation would use batched k-means.
    virtual void TrainQuantizer(class Quantizer &quantizer, const float *samples,
                                idx_t n, idx_t dim) = 0;

    // Compute k entry-point centroids via k-means clustering. Returns
    // k * dim floats in centroids_out.
    virtual void ComputeEntryPoints(const float *vectors, idx_t n, idx_t dim,
                                    idx_t k, float *centroids_out) = 0;

    // --- Search-time operations (batched) ---

    // Populate the PQ/ScaNN distance LUT from a preprocessed query.
    // Delegates to Quantizer::PopulateDistanceLUT on CPU.
    virtual void PopulateDistanceLUT(const class Quantizer &quantizer,
                                     const float *query_preproc,
                                     float *lut_out, idx_t lut_size) = 0;

    // Batched LUT distance: compute distances for n_codes codes against
    // a pre-populated LUT. Results in distances_out[0..n_codes-1].
    //
    // This is the hot inner loop of AiSAQ search. CPU implementation is a
    // tight gather-and-sum; GPU implementation would be a compute shader.
    virtual void BatchLUTDistance(const_data_ptr_t codes, idx_t n_codes,
                                  const float *lut, float *distances_out) = 0;

    // --- Vamana graph build (placeholder; filled in Phase 5) ---
    // The full signature is defined when AiSAQ core is wired in Phase 5.
    // For now this is a pure virtual that CpuBackend stubs out.
};

} // namespace vindex
} // namespace duckdb
```

### `src/backend/cpu_backend.hpp` (new)

```cpp
#pragma once

#include "backend.hpp"

namespace duckdb {
namespace vindex {

// CpuBackend — concrete Backend that delegates all operations to existing
// CPU code paths. This is the default and always-compiled backend.
class CpuBackend : public Backend {
public:
    const string &Name() const override;

    void TrainQuantizer(class Quantizer &quantizer, const float *samples,
                        idx_t n, idx_t dim) override;
    void ComputeEntryPoints(const float *vectors, idx_t n, idx_t dim,
                            idx_t k, float *centroids_out) override;
    void PopulateDistanceLUT(const class Quantizer &quantizer,
                             const float *query_preproc,
                             float *lut_out, idx_t lut_size) override;
    void BatchLUTDistance(const_data_ptr_t codes, idx_t n_codes,
                          const float *lut, float *distances_out) override;
};

} // namespace vindex
} // namespace duckdb
```

### `src/backend/active_backend.hpp` (new)

```cpp
#pragma once

#include "duckdb/main/client_context.hpp"
#include "duckdb/common/unique_ptr.hpp"

#include "backend.hpp"
#include "cpu_backend.hpp"

namespace duckdb {
namespace vindex {

// GetActiveBackend — returns the backend selected by the vindex_gpu_backend
// session option. Hardcoded switch; no factory, no registry.
//
// Compile-time: if VINDEX_BACKEND_VULKAN is defined, VulkanBackend is
// available for selection. Otherwise only CpuBackend is returned.
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
    throw BinderException("Unknown vindex_gpu_backend '%s'. Known backends: cpu%s",
                          name,
#ifdef VINDEX_BACKEND_VULKAN
                          ", vulkan"
#else
                          ""
#endif
                          );
}

} // namespace vindex
} // namespace duckdb
```

## Implementation Steps

### Step 1: Create `src/backend/` directory and CMake

- [ ] Create `src/backend/CMakeLists.txt`:
  ```cmake
  set(EXTENSION_SOURCES
      ${EXTENSION_SOURCES}
      ${CMAKE_CURRENT_SOURCE_DIR}/cpu_backend.cpp
      PARENT_SCOPE)
  ```
- [ ] Add `add_subdirectory(backend)` to `src/CMakeLists.txt`.

### Step 2: Implement `Backend` abstract class

- [ ] Create `src/backend/backend.hpp` with the interface shown above.
- [ ] Include `kKnownBackends` constant.

### Step 3: Implement `CpuBackend`

- [ ] Create `src/backend/cpu_backend.hpp` and `cpu_backend.cpp`.
- [ ] `TrainQuantizer`: call `quantizer.Train(samples, n, dim)`.
- [ ] `ComputeEntryPoints`: call into the existing k-means implementation (from `src/quant/` — find the shared k-means used by PQ training). Run with `k` clusters, return centroids.
- [ ] `PopulateDistanceLUT`: call `quantizer.PopulateDistanceLUT(query_preproc, lut_out)`.
- [ ] `BatchLUTDistance`: tight loop calling `quantizer.LUTDistance(code_i, lut)` for each code.
  ```cpp
  void CpuBackend::BatchLUTDistance(const_data_ptr_t codes, idx_t n_codes,
                                    const float *lut, float *distances_out) {
      // The Quantizer instance is needed for LUTDistance. It should be
      // captured alongside the LUT. Resolution: pass a Quantizer reference
      // into BatchLUTDistance, or bind it at backend construction time.
      // For Phase 4, use the Quantizer-from-LUT approach: the caller
      // passes both quantizer and lut.
      // See Open Questions Q1.
  }
  ```
- [ ] `Name()`: return `"cpu"`.

### Step 4: Implement `GetActiveBackend`

- [ ] Create `src/backend/active_backend.hpp` with the inline selector shown above.
- [ ] Register the `vindex_gpu_backend` session option (VARCHAR, default `"cpu"`) in `src/common/vector_index_registry.cpp` during `RegisterBuiltInAlgorithms`.
- [ ] Verify: requesting `"cpu"` returns `CpuBackend`.
- [ ] Verify: requesting `"vulkan"` (when not compiled) throws with a clear message.
- [ ] Verify: requesting an unknown name throws with the list of known backends.

### Step 5: Write unit tests

- [ ] Create `test/unit/test_backend.cpp`.
- [ ] Test cases:
  - [ ] `CpuBackendName`: `backend.Name() == "cpu"`.
  - [ ] `CpuBackendTrainQuantizer`: train a PqQuantizer via CpuBackend, verify it matches direct training.
  - [ ] `CpuBackendPopulateLUT`: populate LUT via CpuBackend, verify it matches `quantizer.PopulateDistanceLUT`.
  - [ ] `CpuBackendBatchLUTDistance`: batch distances via CpuBackend, verify each matches `quantizer.LUTDistance`.
  - [ ] `CpuBackendComputeEntryPoints`: k-means via CpuBackend on known data, verify centroids.
  - [ ] `GetActiveBackendDefault`: no session option set → returns CpuBackend.
  - [ ] `GetActiveBackendCpu`: `SET vindex_gpu_backend='cpu'` → returns CpuBackend.
  - [ ] `GetActiveBackendUnknown`: `SET vindex_gpu_backend='bogus'` → throws BinderException.

### Step 6: Update `AGENTS.md`

- [ ] Add `src/backend/` to the source wiring diagram.
- [ ] Add a "Backend abstraction" subsection.
- [ | Document the `vindex_gpu_backend` session option.
- [ ] Note that `VulkanBackend` is gated behind `-DVINDEX_BACKEND_VULKAN` (Phase 8).

## Acceptance Criteria

- [ ] `src/backend/` directory exists with `backend.hpp`, `cpu_backend.{hpp,cpp}`, `active_backend.hpp`.
- [ ] `CpuBackend` delegates all operations to existing CPU code paths.
- [ ] `GetActiveBackend(ctx)` returns `CpuBackend` by default.
- [ ] `vindex_gpu_backend` session option is registered.
- [ ] Unknown backend names throw `BinderException` with the list of known backends.
- [ ] Unit tests pass.
- [ ] `make test` and `make unit` — no regressions.
- [ ] `AGENTS.md` is updated.
- [ ] Phase status updated to `- [x]` in [AISAQ.md](AISAQ.md).

## Open Questions

- **Q1**: Should `BatchLUTDistance` take a `Quantizer&` parameter (needed for the gather to know code layout), or should the caller pre-bind the quantizer to the backend? **Proposed**: add `Quantizer&` as a parameter. The backend is stateless; the quantizer carries the code layout. Update the signature:
  ```cpp
  virtual void BatchLUTDistance(const class Quantizer &quantizer,
                                const_data_ptr_t codes, idx_t n_codes,
                                const float *lut, float *distances_out) = 0;
  ```
- **Q2**: Should `ComputeEntryPoints` use the same k-means as PQ training (from `src/quant/`) or a separate implementation? **Proposed**: reuse the existing k-means to avoid duplication. Find the shared k-means entry point during implementation.
