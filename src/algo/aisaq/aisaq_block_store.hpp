#pragma once

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/storage/block_manager.hpp"
#include "duckdb/storage/buffer/buffer_handle.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/storage_info.hpp"

namespace duckdb {
class DatabaseInstance;
namespace vindex {
namespace aisaq {

// AiSaqBlockStore — block-granularity storage backed by DuckDB's BlockManager.
//
// Two storage types:
//   1. Graph nodes: packed N consecutive nodes per block (dense arithmetic indexing)
//   2. PQ code pages: one block per page
//
// Pin returns BufferHandle (RAII — unpin on destruction → evictable).
// Blocks are persistent from creation (allocated via BlockManager::GetFreeBlockId).
class AiSaqBlockStore {
  public:
	AiSaqBlockStore(BlockManager &block_manager, BufferManager &buffer_manager);

	// --- Graph node blocks ---
	// Dense packing: nodes_per_block consecutive nodes per block.
	// internal_id → (block_idx, offset) via arithmetic.

	void RegisterGraphLayout(idx_t node_size);
	// Must be called before AllocGraphNode. Computes nodes_per_block_.

	uint32_t AllocGraphNode();
	// Allocates a new internal_id via an atomic fetch_add — safe to call
	// concurrently from multiple tasks. When flat_build_mode_ is set, skips
	// EnsureGraphCapacity (blocks allocated lazily in WriteAllGraphNodes).
	// In paged mode, EnsureGraphCapacity(id) is still invoked per-call, but a
	// prior PreAllocateGraphCapacity/EnsureGraphCapacity(N) call sizes the
	// block vectors once so the per-call path is a no-op (vector reads only).

	void SetFlatBuildMode(bool enabled) {
		flat_build_mode_.store(enabled, std::memory_order_relaxed);
	}

	void EnsureGraphCapacity(uint32_t up_to_internal_id);
	// Pre-allocates blocks to hold up_to_internal_id + 1 nodes.

	BufferHandle PinGraphNode(uint32_t internal_id) const;
	// Returns a BufferHandle for the block containing this node.
	// Caller computes: ptr = handle.Ptr() + (internal_id % nodes_per_block_) * node_size_

	idx_t NodeSize() const {
		return node_size_;
	}
	idx_t NodesPerBlock() const {
		return nodes_per_block_;
	}

	// --- PQ code page blocks ---
	// One page per block. codes_per_page = BLOCK_SIZE / code_size.

	void RegisterPqLayout(idx_t code_size);

	void WritePqPage(uint32_t page_idx, const_data_ptr_t data);
	// Direct write to a persistent block via BlockManager::Write().
	// Used during the PQ encoding pass (no BufferPool involvement).

	BufferHandle PinPqPage(uint32_t page_idx) const;
	// Returns a BufferHandle for the PQ page block.
	// Used during graph build and search (BufferPool-cached).

	void EnsurePqCapacity(uint32_t up_to_page_idx);

	idx_t CodeSize() const {
		return code_size_;
	}
	idx_t CodesPerPage() const {
		return codes_per_page_;
	}

	// --- Introspection ---

	idx_t GraphNodeCount() const {
		return graph_node_count_.load(std::memory_order_relaxed);
	}
	idx_t GraphBlockCount() const {
		return graph_block_handles_.size();
	}
	idx_t PqPageCount() const {
		return pq_page_block_ids_.size();
	}

	// --- Persistence ---

	// Bulk-write flat graph nodes from a build-time RAM buffer to block-store
	// blocks. Called after construction when a flat node buffer was used.
	// Creates the necessary blocks (lazily — skipped during construction
	// when flat_build_mode_ is set) and copies the data sequentially.
	// Parallelised across TaskScheduler::NumberOfThreads() tasks with
	// disjoint block_idx ranges when SetDatabase has been called.
	void WriteAllGraphNodes(const uint8_t *flat, idx_t count);

	// Per-task worker: writes blocks [block_start, block_end) from the flat
	// buffer. Public so the AiSaqWriteNodesTask defined in the .cpp can call
	// it without a friend declaration.
	void WriteAllGraphNodesRange(const uint8_t *flat, idx_t count, idx_t block_start, idx_t block_end);

	// Inject the DatabaseInstance for parallel WriteAllGraphNodes. When null
	// (default; used by unit tests without a full DuckDB instance), the write
	// stays serial.
	void SetDatabase(DatabaseInstance *db) {
		db_ = db;
	}

	// Convert all transient blocks to persistent DuckDB blocks.
	// Called at checkpoint (SerializeToDisk). After this call, all
	// block IDs in graph_block_ids_ and pq_page_block_ids_ are valid
	// persistent block IDs.
	void ConvertToPersistent(QueryContext context);

	void SerializeState(vector<data_t> &out) const;
	void DeserializeState(const_data_ptr_t in, idx_t size);

  private:
	block_id_t AllocatePersistentBlock();

	BlockManager &block_manager_;
	BufferManager &buffer_manager_;

	// Optional DatabaseInstance for parallel WriteAllGraphNodes. When null,
	// writes stay serial (unit-test path).
	DatabaseInstance *db_ = nullptr;

	// Graph nodes
	idx_t node_size_ = 0;
	idx_t nodes_per_block_ = 0;
	atomic<uint32_t> graph_node_count_{0};
	vector<shared_ptr<BlockHandle>> graph_block_handles_;
	vector<block_id_t> graph_block_ids_;
	atomic<bool> flat_build_mode_{false};

	// PQ code pages
	idx_t code_size_ = 0;
	idx_t codes_per_page_ = 0;
	vector<block_id_t> pq_page_block_ids_;
	vector<shared_ptr<BlockHandle>> pq_page_handles_;
};

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
