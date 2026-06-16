# Phase 1 — M3-complete: IndexBlockStore Eviction — ELIMINATED

**Status**: `- [x]` ELIMINATED — superseded by revised Phase 5
**Date eliminated**: 2026-06-16

> **This phase is no longer part of the plan.** Investigation revealed that
> DuckDB's `FixedSizeBuffer` has the same no-eviction FIXME (at
> `fixed_size_buffer.hpp:142`) that this phase intended to solve. Patching it
> would only benefit AiSAQ — the existing algorithms (HNSW, DiskANN, IVF,
> SPANN) either don't use `IndexBlockStore` for their large data or would
> thrash if their graph nodes were evicted (HNSW search requires rapid
> random access to graph nodes; evicting them defeats the algorithm's
> purpose).
>
> Instead, AiSAQ uses `AiSaqBlockStore` (wrapping `BlockManager` directly)
> which provides native eviction via the BufferPool without any DuckDB
> patch. See the revised Phase 5 for the full design.
>
> The investigation findings are preserved below for reference.

> **Maintenance note**: Update task markers (`- [ ]` -> `- [~]` -> `- [x]`) as you progress. At phase completion, update the phase status in [AISAQ.md](AISAQ.md) and update `AGENTS.md` per the [AGENTS.md Update Protocol](AISAQ.md#agentsmd-update-protocol).

---

## Goal

Make `IndexBlockStore::Unpin` actually do something. Today it is a literal no-op (`src/common/index_block_store.cpp:202-206`). After this phase, unpinned buffers become eligible for LRU eviction, and each `IndexBlockStore` instance can enforce a per-index RAM budget.

This is the prerequisite for indexing datasets larger than RAM. Without it, BufferManager thrashes against DuckDB's global `memory_limit` at the target workload (100M x 768-dim on 32 GB RAM).

## Background

The current state (from `index_block_store.hpp:101-106`):

```cpp
// Pin returns a pointer valid as long as the caller holds the store
// reference. In the M1 implementation the underlying FixedSizeAllocator
// keeps every buffer in memory, so Unpin is a no-op. When M3 enables disk
// eviction, these become proper pin/unpin reference counts.
data_ptr_t Pin(BlockId id);
void Unpin(BlockId id);
```

The `PinFast` cache (`index_block_store.hpp:107-127`) caches raw `data_ptr_t` base pointers keyed by `(tag, buffer_id)`, assuming buffers never move. This assumption breaks once eviction is possible.

The `FixedSizeAllocator::Get(raw, dirty=true)` call at `index_block_store.cpp:199` pins the DuckDB `FixedSizeBuffer` internally but we never call the corresponding unpin. DuckDB's `FixedSizeBuffer` has its own refcount — we need to decrement it when our usage refcount hits zero.

## Scope

### In scope

- Real reference counting on `IndexBlockStore::Pin` / `Unpin`.
- RAII `BufferPin` guard returned from pin operations (prevents leaks on exceptions).
- Per-`IndexBlockStore` LRU eviction queue.
- Dirty buffer flush-on-eviction (write back to `BlockManager` before evicting).
- `PinFast` cache invalidation when a buffer is evicted.
- Thread safety: refcount + cache + eviction queue under concurrent access.
- New APIs: `SetRamBudget(bytes)`, `ApproxResidentSize()`, `EvictCold()`.
- Migration of all existing algorithms (HNSW, IVF, DiskANN, SPANN) to use `BufferPin` RAII guards.
- New session option: `vindex_default_ram_budget_mb` (default 0 = unlimited).
- Comprehensive unit tests for eviction correctness.

### Out of scope

- Cross-index eviction coordination (each `IndexBlockStore` manages its own budget independently).
- Compression of evicted buffers (eviction = full unload, not compress-in-place).
- Async prefetch of likely-needed buffers before re-pin.

## Files

### New files

| Path | Purpose |
|---|---|
| `src/include/vindex/buffer_ref.hpp` | RAII `BufferPin` guard |
| `test/unit/test_index_block_store_eviction.cpp` | Eviction correctness, refcount, cache invalidation, dirty flush, concurrency, budget enforcement |

### Changed files

| Path | Change |
|---|---|
| `src/include/vindex/index_block_store.hpp` | Add `BufferPin`, `SetRamBudget`, `ApproxResidentSize`, `EvictCold`; update `Pin`/`PinFast` to return `BufferPin`; replace raw `pin_cache_` with eviction-aware cache; document `ram_budget_mb` |
| `src/common/index_block_store.cpp` | Implement refcount, LRU eviction queue, dirty flush, thread safety, cache invalidation |
| `src/include/vindex/hnsw_core.hpp` | Migrate `PinFast` call sites to `BufferPin` |
| `src/algo/hnsw/hnsw_index.cpp` | Migrate pin call sites |
| `src/include/vindex/diskann_core.hpp` | Migrate `PinFast` call sites to `BufferPin` |
| `src/algo/diskann/diskann_index.cpp` | Migrate pin call sites |
| `src/include/vindex/ivf_core.hpp` | Migrate pin call sites |
| `src/algo/ivf/ivf_index.cpp` | Migrate pin call sites |
| `src/include/vindex/spann_core.hpp` | Migrate pin call sites |
| `src/algo/spann/spann_index.cpp` | Migrate pin call sites |
| `src/common/vector_index.cpp` | Wire `ram_budget_mb` option parsing into `IndexBlockStore::SetRamBudget` |
| `test/unit/test_index_block_store.cpp` | Update existing tests to use `BufferPin` |
| `test/unit/test_hnsw_core.cpp` | Update if pin patterns changed |
| `test/unit/test_diskann_core.cpp` | Update if pin patterns changed |

## API Surfaces

### `src/include/vindex/buffer_ref.hpp` (new)

```cpp
namespace duckdb {
namespace vindex {

// RAII guard for a pinned IndexBlockStore buffer. Constructed by
// IndexBlockStore::Pin / PinFast. Destructor calls Unpin. Move-only;
// copying is deleted to prevent double-unpin.
class BufferPin {
public:
    BufferPin() = default;
    BufferPin(IndexBlockStore *store, BlockId id, data_ptr_t ptr)
        : store_(store), id_(id), ptr_(ptr), valid_(true) {}
    ~BufferPin();

    // Move-only.
    BufferPin(BufferPin &&other) noexcept;
    BufferPin &operator=(BufferPin &&other) noexcept;
    BufferPin(const BufferPin &) = delete;
    BufferPin &operator=(const BufferPin &) = delete;

    data_ptr_t Get() const {
        D_ASSERT(valid_);
        return ptr_;
    }
    data_ptr_t operator->() const { return Get(); }
    bool IsValid() const { return valid_; }
    BlockId Id() const { return id_; }

    // Explicit release before destructor.
    void Release();

private:
    IndexBlockStore *store_ = nullptr;
    BlockId id_ {};
    data_ptr_t ptr_ = nullptr;
    bool valid_ = false;
};

} // namespace vindex
} // namespace duckdb
```

### `src/include/vindex/index_block_store.hpp` (additions)

```cpp
class IndexBlockStore {
public:
    // ... existing API ...

    // --- M3-complete: eviction-aware pin/unpin ---

    // Pin a node, returning an RAII guard. The guard's destructor calls
    // Unpin. This is the preferred API for all callers after M3-complete.
    BufferPin PinGuard(BlockId id);

    // PinFast equivalent returning an RAII guard.
    BufferPin PinFastGuard(BlockId id);

    // Legacy raw-pointer pins — still available but require an explicit
    // Unpin call. Deprecated; new code should use PinGuard / PinFastGuard.
    data_ptr_t Pin(BlockId id);   // existing, unchanged signature
    void Unpin(BlockId id);       // existing, now actually does work

    // --- Per-index RAM budget ---

    // Set the soft per-index cap on BufferManager-resident bytes. When
    // the index's resident set exceeds budget, the LRU eviction queue
    // evicts cold (refcount == 0) buffers until back under budget.
    //
    // ram_budget = 0 disables the per-index cap and relies entirely on
    // DuckDB's global memory_limit setting. A positive value activates
    // the per-index LRU.
    void SetRamBudget(idx_t bytes);

    // Approximate bytes currently resident (sum of all non-evicted
    // buffers belonging to this store). For diagnostics and budget
    // enforcement.
    idx_t ApproxResidentSize() const;

    // Force-evict cold buffers until under budget. Called automatically
    // after every Unpin that brings refcount to 0. Exposed for tests
    // and for explicit compaction.
    idx_t EvictCold(idx_t target_bytes = 0);

private:
    // ... existing members ...

    // M3-complete additions:
    idx_t ram_budget_ = 0;  // 0 = unlimited
    mutable std::mutex evict_mutex_;

    // Per-buffer refcount, keyed by (tag, buffer_id).
    struct BufferState {
        std::atomic<uint32_t> refcount {0};
        bool dirty = false;
        // LRU linked-list iterators (valid when refcount == 0).
        uint64_t lru_seq = 0;  // monotonically increasing on last-unpin
    };
    // Keyed by packed (tag << 56 | buffer_id).
    mutable std::mutex state_mutex_;
    unordered_map<uint64_t, BufferState> buffer_states_;
    uint64_t lru_counter_ = 0;

    void DecrementRefcount(BlockId id);
    void FlushBuffer(BlockId id);  // write dirty buffer to BlockManager
    void InvalidatePinCacheForBuffer(NodeSizeId tag, idx_t buffer_id);
};
```

## Implementation Steps

### Step 1: Investigate DuckDB `FixedSizeBuffer` internals

- [ ] Read `duckdb/src/include/duckdb/storage/fixed_size_buffer.hpp` and understand the existing refcount / unload mechanism.
- [ ] Read `duckdb/src/storage/fixed_size_allocator.cpp` to find how `Get()` pins and what the corresponding unpin call is.
- [ ] Document findings in a comment at the top of `src/common/index_block_store.cpp`.
- [ ] Verify: does `FixedSizeAllocator` expose an `Unpin` or `UnloadBuffer` method? If not, identify the internal API we need to call.
- [ ] Verify: what happens when we unload a buffer that is still mapped by another `FixedSizeAllocator`? (Should not happen — each allocator owns its own buffers.)

> **Note**: This investigation step is critical. The entire eviction mechanism depends on knowing how DuckDB's `FixedSizeBuffer` manages its internal pin count. Do not skip.

### Step 2: Implement `BufferPin` RAII guard

- [ ] Create `src/include/vindex/buffer_ref.hpp` with the `BufferPin` class shown in [API Surfaces](#api-surfaces).
- [ ] Implement move constructor / move assignment (transfer ownership, invalidate source).
- [ ] Implement destructor calling `store_->Unpin(id_)` if `valid_`.
- [ ] Implement `Release()` for explicit early release.
- [ ] Unit-test the RAII semantics in isolation (mock store).

### Step 3: Implement refcount tracking in `IndexBlockStore`

- [ ] Add `buffer_states_` map and `state_mutex_` to `IndexBlockStore`.
- [ ] In `Pin(BlockId id)`: increment `buffer_states_[key].refcount` under `state_mutex_`; mark dirty if the existing `Pin` passes `dirty=true`.
- [ ] In `Unpin(BlockId id)`: decrement refcount under `state_mutex_`; if it hits 0, set `lru_seq = ++lru_counter_` and call `EvictCold()` if over budget.
- [ ] Add `DecrementRefcount(BlockId id)` private helper for the shared logic.
- [ ] Add `PinGuard(BlockId id)` and `PinFastGuard(BlockId id)` returning `BufferPin`.

### Step 4: Implement LRU eviction

- [ ] Add `EvictCold(idx_t target_bytes)` method.
- [ ] Walk `buffer_states_` for entries with `refcount == 0`, sorted by `lru_seq` ascending (oldest first).
- [ ] For each victim: call `FlushBuffer(id)` if dirty, then unload via DuckDB's `FixedSizeBuffer` API, then `InvalidatePinCacheForBuffer(tag, buffer_id)`.
- [ ] Continue until `ApproxResidentSize() <= target_bytes` (or `<= ram_budget_` if `target_bytes == 0`).
- [ ] Return the number of bytes evicted.

### Step 5: Implement dirty buffer flush

- [ ] Add `FlushBuffer(BlockId id)` private method.
- [ ] Write the buffer's contents back to `BlockManager` via the appropriate DuckDB storage API (likely `BlockManager::Write` or `FixedSizeAllocator::FlushBuffer` — confirm from Step 1).
- [ ] Mark `buffer_states_[key].dirty = false` after successful flush.

### Step 6: Update `PinFast` cache for eviction awareness

- [ ] `PinFast(id)` must still work as a fast path, but the cached pointer must be validated against eviction state.
- [ ] Add `InvalidatePinCacheForBuffer(NodeSizeId tag, idx_t buffer_id)`: clears `pin_cache_[tag][buffer_id]`.
- [ ] Call this from `EvictCold` before unloading the buffer.
- [ ] `PinFastSlow(id)` must re-resolve through the full path after eviction (the existing slow path already does this via `FixedSizeAllocator::Get`).
- [ ] **Critical invariant**: after this step, `PinFast` may return a stale pointer if a buffer was evicted between cache population and use. The cache invalidation in `EvictCold` prevents this, but verify with the concurrency tests.

### Step 7: Implement `SetRamBudget` / `ApproxResidentSize`

- [ ] `SetRamBudget(idx_t bytes)`: store `ram_budget_ = bytes`; immediately call `EvictCold()` if currently over budget.
- [ ] `ApproxResidentSize()`: count buffers in `buffer_states_` that have not been evicted. Track this as a running counter (`resident_bytes_`) updated on pin/evict for O(1) reporting.
- [ ] Wire the `WITH (ram_budget_mb = X)` option from `vector_index.cpp` into `SetRamBudget(X * 1024 * 1024)` at index construction time.

### Step 8: Thread safety review

- [ ] Audit all `buffer_states_` accesses to confirm they are under `state_mutex_`.
- [ ] Audit all `pin_cache_` writes to confirm proper invalidation.
- [ ] Confirm `EvictCold` does not deadlock with concurrent `Pin` (the eviction loop may need to release `state_mutex_` during `FlushBuffer` to avoid blocking readers).
- [ ] Consider: should `state_mutex_` be a spinlock for the hot pin/unpin path? Benchmark.

### Step 9: Migrate HNSW to `BufferPin`

- [ ] Find all `PinFast` call sites in `src/include/vindex/hnsw_core.hpp` and `src/algo/hnsw/hnsw_index.cpp`.
- [ ] Replace raw `data_ptr_t` usage with `BufferPin` RAII guards.
- [ ] Verify: HNSW search/build still produces identical results.
- [ ] Run `make test` and `make unit` — zero regressions.

### Step 10: Migrate IVF to `BufferPin`

- [ ] Same as Step 9 but for `src/include/vindex/ivf_core.hpp` and `src/algo/ivf/ivf_index.cpp`.
- [ ] Verify: `make test` and `make unit` — zero regressions.

### Step 11: Migrate DiskANN to `BufferPin`

- [ ] Same as Step 9 but for `src/include/vindex/diskann_core.hpp` and `src/algo/diskann/diskann_index.cpp`.
- [ ] Pay special attention to `DiskAnnCore::BeamSearch` at `diskann_core.cpp:60-141` — every `PinFast` call in the hot loop.
- [ ] Verify: `make test` and `make unit` — zero regressions.

### Step 12: Migrate SPANN to `BufferPin`

- [ ] Same as Step 9 but for `src/include/vindex/spann_core.hpp` and `src/algo/spann/spann_index.cpp`.
- [ ] Verify: `make test` and `make unit` — zero regressions.

### Step 13: Wire `ram_budget_mb` into `WITH (...)` options

- [ ] In `src/common/vector_index.cpp`, parse the `ram_budget_mb` option from the `WITH (...)` map.
- [ ] Call `store_->SetRamBudget(bytes)` during index construction (after `IndexBlockStore` is created but before any allocations).
- [ ] Validate: `ram_budget_mb` must be a non-negative integer.
- [ ] This option is shared across all algorithms (HNSW, IVF, DiskANN, SPANN, and future AiSAQ).

### Step 14: Write eviction unit tests

- [ ] Create `test/unit/test_index_block_store_eviction.cpp`.
- [ ] Test cases (each as a separate `TEST_CASE`):
  - [ ] `RefcountBasic`: Pin twice, Unpin once (still resident), Unpin again (eligible for eviction).
  - [ ] `EvictionUnderBudget`: Set budget smaller than total allocations, verify cold buffers are evicted.
  - [ ] `PinFastCacheInvalidation`: Pin, Unpin, Evict, PinFast again — must re-resolve through slow path.
  - [ ] `DirtyBufferFlush`: Pin dirty, modify, Unpin, Evict — verify contents are flushed and recoverable on re-pin.
  - [ ] `ConcurrentPinUnpin`: Multiple threads pinning/unpinning the same buffer — no race, no double-evict.
  - [ ] `BudgetEnforcement`: Allocate 100 buffers, set budget to hold 10, verify exactly 90 cold ones are evicted and 10 hot ones stay.
  - [ ] `RePinAfterEviction`: Evict a buffer, Pin it again — must get a valid pointer with correct contents.
  - [ ] `LRUOrder`: Unpin buffer A, then B, then C. Evict one. Must evict A (oldest LRU).

### Step 15: Regression test all algorithms under tight budget

- [ ] For each algorithm (HNSW, IVF, DiskANN, SPANN):
  - [ ] Build an index with `WITH (ram_budget_mb=1)` (very tight).
  - [ ] Run the standard recall bench.
  - [ ] Verify recall is within 1% of the unlimited-budget baseline.
- [ ] If any algorithm's recall degrades significantly, investigate (likely a missing `BufferPin` migration or a stale-pointer bug).

### Step 16: Update `AGENTS.md`

- [ ] Add `src/include/vindex/buffer_ref.hpp` to the source wiring diagram.
- [ ] Add a new "Buffer eviction and `ram_budget_mb`" subsection explaining the per-index RAM budget mechanism.
- [ ] Update the "Commands" table if new test commands are needed.
- [ ] Note: `ram_budget_mb` is now a valid `WITH (...)` option for all vector index types.

## Acceptance Criteria

This phase is complete when ALL of the following hold:

- [ ] `IndexBlockStore::Unpin` is no longer a no-op. Unpinned buffers are eligible for eviction.
- [ ] `BufferPin` RAII guard exists and is used by all four existing algorithms.
- [ ] `SetRamBudget` / `ApproxResidentSize` / `EvictCold` APIs are implemented and tested.
- [ ] `WITH (ram_budget_mb = X)` option is parsed and enforced for all algorithms.
- [ ] All existing SQL logic tests pass (`make test`).
- [ ] All existing unit tests pass (`make unit`).
- [ ] New eviction unit tests pass.
- [ ] Recall bench shows no regression under tight budgets (`WITH (ram_budget_mb=1)`) for any algorithm.
- [ ] `AGENTS.md` is updated per the protocol.
- [ ] Phase status updated to `- [x]` in [AISAQ.md](AISAQ.md).

## Open Questions

- **Q1**: Does DuckDB's `FixedSizeBuffer` expose a public unload/evict API, or do we need to use `BufferPool` / `BufferManager` directly? Resolve in Step 1.
- **Q2**: Should the LRU queue be per-`NodeSizeId` (tag) or global across all tags in one store? Per-tag gives better granularity for mixed-workload indexes (graph nodes vs PQ pages); global is simpler. Start with global, revisit if profiling shows problems.
- **Q3**: Should `BufferPin` support a "no-evict" pin mode for build-time hot loops? Probably not for v0 — the eviction check is cheap and build-time pins are short-lived.
