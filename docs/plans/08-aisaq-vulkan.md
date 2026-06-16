# Phase 8 — AiSAQ Vulkan Backend

**Branch**: `feat/aisaq-vulkan` (branches from `feat/aisaq` after Phase 7)
**Status**: `- [ ]` not started
**Effort**: 8-12 weeks
**Depends on**: Phase 7 (all AiSAQ tests pass)
**Dependents**: nothing (terminal phase for this plan)

> **Maintenance note**: Update task markers (`- [ ]` -> `- [~]` -> `- [x]`) as you progress. At phase completion, update the phase status in [AISAQ.md](AISAQ.md) and update `AGENTS.md` per the [AGENTS.md Update Protocol](AISAQ.md#agentsmd-update-protocol).

---

## Goal

Implement a concrete `VulkanBackend` that accelerates AiSAQ's batched math operations via Vulkan compute shaders. This is the GPU acceleration phase — the equivalent of what cuVS provides to Kioxia's AiSAQ, but implemented from scratch in Vulkan for portability across GPU vendors (NVIDIA, AMD, Intel, ARM Mali, Apple via MoltenVK).

Per the [Kioxia blog](../../ref/kioxia-gpu-blog.txt), the primary acceleration target is **index build** (Vamana graph construction, PQ training, entry-point k-means), where cuVS achieves 7.8x speedup. Search-time acceleration is secondary (Kioxia doesn't GPU-accelerate search because their bottleneck is SSD I/O, not compute — but on vindex's BufferManager substrate, compute may be a more significant fraction).

## Background

### Why Vulkan (not CUDA)

From the locked decisions in [AISAQ.md](AISAQ.md):
- cuVS is CUDA-only and would break community-extensions portability.
- The WASM arch exclusion in AGENTS.md already constrains the build.
- Vulkan compute shaders are cross-vendor (NVIDIA, AMD, Intel, ARM, Apple via MoltenVK).
- The reference (`ref/aisaq-diskann/`) uses libaio/liburing which is Linux-only; vindex already replaced that with BufferManager. Vulkan is the same principle applied to the compute layer.

### What cuVS accelerates (from the Kioxia blog)

> "Index build was accelerated using NVIDIA cuVS, leveraging cuVS support for Vamana index build. cuVS was additionally used to accelerate the generation of other index elements employing k-means clustering, such as PQ vector generation, and clustering for entry point generation."

Three accelerated build phases:
1. **Vamana graph construction** (CPU 31d -> 4xH100 4d). Internally: batched PQ-LUT distance across the beam-search frontier.
2. **PQ codebook training** (k-means with 256 centroids per chunk).
3. **Entry-point generation** (k-means on the dataset).

Search is not GPU-accelerated in Kioxia's stack. We may optionally accelerate the `BatchLUTDistance` search kernel if profiling shows it's a bottleneck on the BufferManager substrate.

## Scope

### In scope

- `src/backend/vulkan_backend.{hpp,cpp}`: concrete `VulkanBackend` class.
- Vulkan compute shaders for the accelerated kernels.
- CMake integration with `find_package(Vulkan)` gated behind `-DVINDEX_BACKEND_VULKAN=ON`.
- Runtime fallback: if `vindex_gpu_backend='vulkan'` is requested but no Vulkan-capable device is found, throw a clear error (or optionally fall back to CPU — TBD).
- Unit tests verifying `VulkanBackend` produces identical results to `CpuBackend`.

### Out of scope

- CUDA, ROCm, Metal backends (only `"cpu"` and `"vulkan"` are in `kKnownBackends`).
- GPU-resident index storage (all data stays in DuckDB's BufferManager; GPU is used as a compute accelerator only, with explicit host-to-device transfers).
- Multi-GPU support (single Vulkan device).
- Real-time ray tracing or graphics pipelines (compute only).

## Files

### New files

| Path | Purpose |
|---|---|
| `src/backend/vulkan_backend.hpp` | `VulkanBackend` class declaration |
| `src/backend/vulkan_backend.cpp` | `VulkanBackend` implementation |
| `src/backend/vulkan_context.hpp` | Vulkan device/queue/pipeline management |
| `src/backend/vulkan_context.cpp` | Implementation |
| `src/backend/shaders/kmeans.comp` | Compute shader: k-means cluster assignment + centroid update |
| `src/backend/shaders/pq_lut_populate.comp` | Compute shader: populate PQ distance LUT from query |
| `src/backend/shaders/pq_lut_distance.comp` | Compute shader: batched LUT gather + sum |
| `src/backend/shaders/vamana_beam_search.comp` | Compute shader: batched frontier expansion (the hard one) |
| `src/backend/CMakeLists.txt` (update) | Conditional Vulkan sources |
| `test/unit/test_vulkan_backend.cpp` | Equivalence tests vs CpuBackend |

### Changed files

| Path | Change |
|---|---|
| `src/backend/active_backend.hpp` | Add `#ifdef VINDEX_BACKEND_VULKAN` block with `VulkanBackend` include |
| `src/backend/CMakeLists.txt` | Add conditional Vulkan sources + `find_package(Vulkan)` |
| `CMakeLists.txt` | Add `option(VINDEX_BACKEND_VULKAN "Enable Vulkan GPU backend" OFF)` |

## Architecture

### Vulkan context management

```cpp
class VulkanContext {
public:
    static VulkanContext &Instance();  // singleton; lazy-init on first use

    VkDevice Device() const;
    VkQueue ComputeQueue() const;
    VkCommandPool CommandPool() const;

    // Pipeline management: compile-on-first-use, cached.
    VkPipeline GetPipeline(const string &shader_path);
    VkShaderModule CompileShader(const string &spv_path);

    // Buffer management.
    VkBuffer CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage);
    void *MapBuffer(VkBuffer buf);
    void UnmapBuffer(VkBuffer buf);
    void DestroyBuffer(VkBuffer buf);

    // Submit + wait (synchronous compute for v0).
    void SubmitAndWait(VkCommandBuffer cmd);

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue compute_queue_ = VK_NULL_HANDLE;
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    unordered_map<string, VkPipeline> pipeline_cache_;
};
```

### Data flow

All accelerated operations follow the same pattern:

1. Host prepares input data (vectors, codes, LUT) in host-visible buffers.
2. `vkCmdCopyBuffer` to device-local memory (or use host-visible for small data).
3. `vkCmdBindPipeline` + `vkCmdBindDescriptorSets` + `vkCmdDispatch`.
4. `vkCmdCopyBuffer` back to host-visible.
5. `vkQueueSubmit` + `vkQueueWaitIdle`.
6. Host reads results.

For v0, all operations are **synchronous** (submit + wait). Async/pipelined compute is a future optimization.

### Shader specifications

#### `kmeans.comp` — k-means iteration

```
inputs:
  - points: float[N * dim] (host -> device)
  - centroids: float[K * dim] (device, updated in-place)
  - N, K, dim: push constants

workgroup: (256, 1, 1)
global_size: (ceil(N / 256) * 256, 1, 1)

per-invocation:
  - Assign one point to nearest centroid (reduce over K).
  - Atomically accumulate sum + count into centroid accumulator buffers.

post-dispatch (host):
  - Second dispatch: divide centroid sums by counts.
  - Read back updated centroids.
```

#### `pq_lut_populate.comp` — PQ LUT population

```
inputs:
  - query_preprocessed: float[m * centroids_per_slot] (host -> device)
  - codebook: float[m * centroids_per_slot * sub_dim] (device, persistent)
  - m, centroids_per_slot, sub_dim: push constants

output:
  - lut: float[m * centroids_per_slot] (device -> host)

workgroup: (256, 1, 1)
global_size: (ceil(m * centroids_per_slot / 256) * 256, 1, 1)

per-invocation:
  - Compute distance between one query chunk and one centroid.
  - Write to lut[slot * centroids_per_slot + centroid_id].
```

#### `pq_lut_distance.comp` — Batched LUT gather

```
inputs:
  - codes: uint8[N * code_size] (host -> device)
  - lut: float[m * centroids_per_slot] (device, from previous shader)
  - N, code_size, m, centroids_per_slot: push constants

output:
  - distances: float[N] (device -> host)

workgroup: (64, 1, 1)
global_size: (ceil(N / 64) * 64, 1, 1)

per-invocation:
  - Gather m LUT entries for one code.
  - Sum into distances[thread_id].
```

#### `vamana_beam_search.comp` — Batched frontier expansion

This is the most complex shader. It parallelizes the inner loop of Vamana beam search: given a frontier of F nodes, each with up to R neighbors, compute PQ distances for all F*R neighbor codes in parallel.

```
inputs:
  - frontier_codes: uint8[F * R * code_size] (the PQ codes of all frontier neighbors)
  - lut: float[m * centroids_per_slot] (device)
  - F, R, code_size, m, centroids_per_slot: push constants

output:
  - neighbor_distances: float[F * R] (device -> host)

workgroup: (256, 1, 1)
global_size: (ceil(F * R / 256) * 256, 1, 1)

per-invocation:
  - Compute LUT gather for one neighbor code.
  - Write to neighbor_distances[frontier_idx * R + neighbor_idx].

post-dispatch (host):
  - Host reads back distances.
  - Host updates the priority queue (sequential, not GPU-accelerated — the retset is small).
  - Host determines next frontier.
  - Repeat.
```

> **Note**: The retset/priority-queue management stays on the CPU. Only the batched distance computation is GPU-accelerated. This matches cuVS's architecture.

## Implementation Steps

### Step 1: CMake integration

- [ ] Add `option(VINDEX_BACKEND_VULKAN "Enable Vulkan GPU backend" OFF)` to root `CMakeLists.txt`.
- [ ] In `src/backend/CMakeLists.txt`:
  ```cmake
  if(VINDEX_BACKEND_VULKAN)
      find_package(Vulkan REQUIRED)
      target_sources(vindex_backend PRIVATE
          vulkan_backend.cpp
          vulkan_context.cpp)
      target_compile_definitions(vindex_backend PRIVATE VINDEX_BACKEND_VULKAN=1)
      target_link_libraries(vindex_backend PRIVATE Vulkan::Vulkan)
      # Compile shaders to SPIR-V at build time
      file(GLOB GLSL_SHADERS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp")
      foreach(SHADER ${GLSL_SHADERS})
          get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
          add_custom_command(
              OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv"
              COMMAND glslangValidator -V ${SHADER} -o "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv"
              DEPENDS ${SHADER})
          list(APPEND SPIRV_FILES "${CMAKE_CURRENT_BINARY_DIR}/${SHADER_NAME}.spv")
      endforeach()
      add_custom_target(vindex_shaders DEPENDS ${SPIRV_FILES})
      add_dependencies(vindex_backend vindex_shaders)
  endif()
  ```

### Step 2: Implement `VulkanContext`

- [ ] Create `vulkan_context.{hpp,cpp}`.
- [ ] Instance creation with validation layers (debug only).
- [ ] Physical device selection (pick first Vulkan-capable GPU).
- [ ] Logical device + compute queue.
- [ ] Command pool + buffer allocation helpers.
- [ ] Pipeline cache (compile-on-first-use).
- [ ] Singleton access (`VulkanContext::Instance()`).
- [ ] Graceful failure: if no Vulkan device, throw on first use with a clear message.

### Step 3: Implement `kmeans.comp` shader

- [ ] Write GLSL compute shader.
- [ ] Compile to SPIR-V (via `glslangValidator`).
- [ ] Test: run k-means on CPU and GPU, verify identical centroids.

### Step 4: Implement `VulkanBackend::ComputeEntryPoints`

- [ ] Upload vectors to device.
- [ ] Initialize random centroids.
- [ ] Iterate: dispatch `kmeans.comp` for assignment, dispatch centroid-update, read back.
- [ ] Return final centroids.

### Step 5: Implement `VulkanBackend::TrainQuantizer`

- [ ] PQ training is k-means per chunk. Reuse `kmeans.comp` with per-chunk dispatches.
- [ ] Or: write a multi-chunk variant that trains all `m` chunks in one dispatch.
- [ ] Write back codebook to the `Quantizer` object.

### Step 6: Implement `pq_lut_populate.comp` shader

- [ ] Write GLSL compute shader.
- [ ] Test: populate LUT on CPU and GPU, verify identical values.

### Step 7: Implement `pq_lut_distance.comp` shader

- [ ] Write GLSL compute shader.
- [ ] Test: batch distances on CPU and GPU, verify identical values.

### Step 8: Implement `VulkanBackend::PopulateDistanceLUT` + `BatchLUTDistance`

- [ ] Wire the shaders from Steps 6-7.
- [ ] Benchmark vs `CpuBackend::BatchLUTDistance` — identify the breakeven batch size where GPU transfer overhead is amortized.

### Step 9: Implement `vamana_beam_search.comp` shader

- [ ] Write GLSL compute shader for batched frontier distance computation.
- [ ] This is the most complex kernel — iterate on correctness.
- [ ] Test: given a known frontier, verify GPU distances match CPU distances.

### Step 10: Implement `VulkanBackend::BuildVamanaGraph`

- [ ] The outer Vamana insert loop stays on CPU (sequential by nature).
- [ ] Per-insert: batch the beam-search frontier distances on GPU.
- [ ] Profile: measure speedup vs CPU-only build.
- [ ] Expected: significant speedup on large datasets (the 7.8x from Kioxia's blog, though likely less on smaller datasets due to transfer overhead).

### Step 11: Write equivalence tests

- [ ] Create `test/unit/test_vulkan_backend.cpp`.
- [ ] For each backend operation:
  - Run on `CpuBackend`, run on `VulkanBackend`.
  - Verify results match within float tolerance.
- [ ] Test cases:
  - [ ] `KMeansEquivalence`: same data, same K -> same centroids.
  - [ ] `LUTPopulateEquivalence`: same query -> same LUT.
  - [ ] `BatchLUTDistanceEquivalence`: same codes + LUT -> same distances.
  - [ ] `VamanaBuildEquivalence`: same vectors + params -> equivalent graph (same recall).
- [ ] Tests are gated behind `#ifdef VINDEX_BACKEND_VULKAN` (only compiled when Vulkan is enabled).

### Step 12: Integration test

- [ ] Build with `-DVINDEX_BACKEND_VULKAN=ON`.
- [ ] `SET vindex_gpu_backend='vulkan'`.
- [ ] Build an AiSAQ index on sift1m.
- [ ] Verify recall matches the CPU build (within tolerance).
- [ ] Measure build time speedup vs CPU.

### Step 13: Update `AGENTS.md`

- [ ] Add `src/backend/vulkan_backend.cpp` to source wiring.
- [ ] Add a "Vulkan backend" subsection.
- [ | Document `-DVINDEX_BACKEND_VULKAN=ON` build flag.
- [ ] Document `vindex_gpu_backend='vulkan'` session option.
- [ ] Note: community-extensions builds are Vulkan-free (the flag is off by default).

### Step 14: Update CI

- [ ] The `MainDistributionPipeline.yml` does NOT enable Vulkan (community-extensions constraint).
- [ ] Add an optional Vulkan CI job (manual dispatch) that builds with `-DVINDEX_BACKEND_VULKAN=ON` and runs equivalence tests.
- [ ] WASM archs remain excluded.

## Acceptance Criteria

- [ ] `VulkanBackend` compiles when `-DVINDEX_BACKEND_VULKAN=ON`.
- [ ] `VulkanBackend` does not compile (and adds no Vulkan dependency) when the flag is off.
- [ ] `SET vindex_gpu_backend='vulkan'` selects `VulkanBackend` at runtime.
- [ ] All backend operations produce identical results to `CpuBackend` (within float tolerance).
- [ ] AiSAQ index built with Vulkan backend achieves the same recall as CPU-built (sift1m threshold 0.95).
- [ ] Build-time speedup is measurable on sift1m (target: at least 2x, aspirational: 5x+).
- [ ] No regressions in the default (CPU) build.
- [ ] `AGENTS.md` is updated.
- [ ] Phase status updated to `- [x]` in [AISAQ.md](AISAQ.md).

## Open Questions

- **Q1**: Should we use `VulkanMemoryAllocator` (VMA) for buffer management, or roll our own? VMA is the standard but adds a dependency. **Proposed**: use VMA — it handles memory types and suballocation correctly, which is hard to get right manually.
- **Q2**: Should the shaders be compiled at build time (via `glslangValidator`) or shipped as pre-compiled SPIR-V? **Proposed**: build-time compilation, with pre-compiled SPIR-V as a fallback if `glslangValidator` is not found.
- **Q3**: Should there be a "Vulkan available but no suitable GPU" fallback to CPU? **Proposed**: no — throw a clear error. Users who requested Vulkan explicitly should know it didn't work. Automatic fallback hides configuration mistakes.
- **Q4**: For the Vamana build, should the entire graph be GPU-resident during construction, or should we keep it on host and only upload frontier batches? **Proposed**: host-resident graph (via BufferManager), upload only frontier PQ codes per iteration. This keeps memory bounded and works with the existing IndexBlockStore substrate.
- **Q5**: Should we support Vulkan descriptor indexing (`VK_EXT_descriptor_indexing`) for large codebooks? **Proposed**: not in v0 — the codebook fits in a single SSBO. Revisit if dim > 4096 or m > 256.
