#include "algo/aisaq/aisaq_block_store.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "duckdb/parallel/task_scheduler.hpp"

#include <algorithm>
#include <cstring>

namespace duckdb {
namespace vindex {
namespace aisaq {

namespace {

// Per-task worker for parallel WriteAllGraphNodes (Phase 9 Task 3 bonus).
class AiSaqWriteNodesTask final : public BaseExecutorTask {
  public:
	AiSaqWriteNodesTask(TaskExecutor &executor, AiSaqBlockStore &store, const uint8_t *flat, idx_t count,
	                    idx_t block_start, idx_t block_end)
	    : BaseExecutorTask(executor), store_(store), flat_(flat), count_(count), block_start_(block_start),
	      block_end_(block_end) {
	}

	void ExecuteTask() override {
		store_.WriteAllGraphNodesRange(flat_, count_, block_start_, block_end_);
	}

	string TaskType() const override {
		return "AiSaqWriteNodesTask";
	}

  private:
	AiSaqBlockStore &store_;
	const uint8_t *flat_;
	idx_t count_;
	idx_t block_start_;
	idx_t block_end_;
};

} // namespace

AiSaqBlockStore::AiSaqBlockStore(BlockManager &block_manager, BufferManager &buffer_manager)
    : block_manager_(block_manager), buffer_manager_(buffer_manager) {
}

// ---------------------------------------------------------------------------
// Graph node blocks
// ---------------------------------------------------------------------------

void AiSaqBlockStore::RegisterGraphLayout(idx_t node_size) {
	D_ASSERT(node_size > 0);
	D_ASSERT(node_size % 8 == 0); // alignment invariant
	node_size_ = node_size;
	const idx_t block_size = block_manager_.GetBlockSize();
	nodes_per_block_ = block_size / node_size;
	D_ASSERT(nodes_per_block_ > 0);
}

void AiSaqBlockStore::EnsureGraphCapacity(uint32_t up_to_internal_id) {
	const uint32_t needed_blocks = (up_to_internal_id / static_cast<uint32_t>(nodes_per_block_)) + 1;
	while (graph_block_handles_.size() < needed_blocks) {
		auto handle = buffer_manager_.Allocate(MemoryTag::ART_INDEX, &block_manager_, false);
		auto block_handle = handle.GetBlockHandle();
		// Zero the block so unread nodes have deterministic content.
		std::memset(handle.Ptr(), 0, block_manager_.GetBlockSize());
		graph_block_handles_.push_back(block_handle);
		graph_block_ids_.push_back(INVALID_BLOCK); // transient; assigned at checkpoint
		                                           // Release the allocation pin — the BufferPool caches it.
		                                           // Subsequent Pin() calls re-pin on demand.
	}
}

uint32_t AiSaqBlockStore::AllocGraphNode() {
	// fetch_add(1) returns the prior value and bumps by 1 — identical to the
	// old `graph_node_count_++` return semantics, but safe under concurrent
	// callers (Phase 9 Task 4). relaxed ordering is sufficient: the caller
	// synchronises on the returned disjoint internal_id, not on the counter.
	if (flat_build_mode_.load(std::memory_order_relaxed)) {
		return graph_node_count_.fetch_add(1, std::memory_order_relaxed);
	}
	const uint32_t id = graph_node_count_.fetch_add(1, std::memory_order_relaxed);
	EnsureGraphCapacity(id);
	return id;
}

uint32_t AiSaqBlockStore::AllocGraphNodeRange(uint32_t count) {
	// Batched reservation. Same atomic semantics as AllocGraphNode, just
	// fetch_add(count) instead of fetch_add(1). Returned start = prior value,
	// so the caller owns [start, start + count). When count == 0, returns
	// the current counter without mutation.
	if (count == 0) {
		return graph_node_count_.load(std::memory_order_relaxed);
	}
	const uint32_t start = graph_node_count_.fetch_add(count, std::memory_order_relaxed);
	if (!flat_build_mode_.load(std::memory_order_relaxed)) {
		EnsureGraphCapacity(start + count - 1);
	}
	return start;
}

void AiSaqBlockStore::WriteAllGraphNodesRange(const uint8_t *flat, idx_t count, idx_t block_start, idx_t block_end) {
	for (idx_t block_idx = block_start; block_idx < block_end; block_idx++) {
		D_ASSERT(block_idx < graph_block_handles_.size());
		auto handle = buffer_manager_.Pin(graph_block_handles_[block_idx]);
		const idx_t start_node = block_idx * nodes_per_block_;
		const idx_t end_node = std::min(start_node + nodes_per_block_, count);
		const idx_t bytes = (end_node - start_node) * node_size_;
		std::memcpy(handle.Ptr(), flat + start_node * node_size_, bytes);
	}
}

void AiSaqBlockStore::WriteAllGraphNodes(const uint8_t *flat, idx_t count) {
	// Lazily allocate blocks skipped during flat-build construction.
	if (count > 0) {
		EnsureGraphCapacity(static_cast<uint32_t>(count - 1));
	}
	// Serial fallback for unit tests without a DatabaseInstance, or when the
	// block count is too small to amortise TaskExecutor scheduling overhead.
	const idx_t total_blocks = (count + nodes_per_block_ - 1) / nodes_per_block_;
	const bool serial = db_ == nullptr || total_blocks < 4;
	if (serial) {
		WriteAllGraphNodesRange(flat, count, 0, total_blocks);
		return;
	}
	auto &scheduler = TaskScheduler::GetScheduler(*db_);
	const size_t num_threads = std::max<size_t>(1, NumericCast<size_t>(scheduler.NumberOfThreads()));
	const size_t actual_threads = std::min(num_threads, NumericCast<size_t>(total_blocks));
	TaskExecutor executor(scheduler);
	for (size_t t = 0; t < actual_threads; t++) {
		const idx_t block_start = idx_t(t) * total_blocks / actual_threads;
		const idx_t block_end = idx_t(t + 1) * total_blocks / actual_threads;
		if (block_start >= block_end) {
			continue;
		}
		executor.ScheduleTask(make_uniq<AiSaqWriteNodesTask>(executor, *this, flat, count, block_start, block_end));
	}
	executor.WorkOnTasks();
}

BufferHandle AiSaqBlockStore::PinGraphNode(uint32_t internal_id) const {
	D_ASSERT(internal_id < graph_node_count_.load(std::memory_order_relaxed));
	const idx_t block_idx = internal_id / static_cast<uint32_t>(nodes_per_block_);
	D_ASSERT(block_idx < graph_block_handles_.size());
	auto handle = graph_block_handles_[block_idx];
	return buffer_manager_.Pin(handle);
}

// ---------------------------------------------------------------------------
// PQ code page blocks
// ---------------------------------------------------------------------------

void AiSaqBlockStore::RegisterPqLayout(idx_t code_size) {
	D_ASSERT(code_size > 0);
	code_size_ = code_size;
	const idx_t block_size = block_manager_.GetBlockSize();
	codes_per_page_ = block_size / code_size;
	D_ASSERT(codes_per_page_ > 0);
}

void AiSaqBlockStore::EnsurePqCapacity(uint32_t up_to_page_idx) {
	while (pq_page_handles_.size() <= up_to_page_idx) {
		auto handle = buffer_manager_.Allocate(MemoryTag::ART_INDEX, &block_manager_, false);
		auto block_handle = handle.GetBlockHandle();
		std::memset(handle.Ptr(), 0, block_manager_.GetBlockSize());
		pq_page_handles_.push_back(block_handle);
		pq_page_block_ids_.push_back(INVALID_BLOCK);
	}
}

void AiSaqBlockStore::WritePqPage(uint32_t page_idx, const_data_ptr_t data) {
	EnsurePqCapacity(page_idx);
	auto handle = buffer_manager_.Pin(pq_page_handles_[page_idx]);
	std::memcpy(handle.Ptr(), data, codes_per_page_ * code_size_);
	// handle released → block is dirty + evictable
}

BufferHandle AiSaqBlockStore::PinPqPage(uint32_t page_idx) const {
	D_ASSERT(page_idx < pq_page_handles_.size());
	// BufferManager::Pin takes shared_ptr<BlockHandle>& (non-const), so copy
	// the handle into a local — indexing a const member vector yields a const
	// reference that won't bind.
	auto handle = pq_page_handles_[page_idx];
	return buffer_manager_.Pin(handle);
}

// ---------------------------------------------------------------------------
// Persistence (block ID serialization — basic stub for Phase 5)
// ---------------------------------------------------------------------------

namespace {

template <typename T> void AppendPacked(vector<data_t> &out, T val) {
	const auto *p = reinterpret_cast<const data_t *>(&val);
	out.insert(out.end(), p, p + sizeof(T));
}

template <typename T> T ConsumePacked(const_data_ptr_t &cur, const_data_ptr_t end) {
	if (cur + sizeof(T) > end) {
		throw InternalException("AiSaqBlockStore: state stream truncated");
	}
	T val;
	std::memcpy(&val, cur, sizeof(T));
	cur += sizeof(T);
	return val;
}

} // namespace

// ---------------------------------------------------------------------------
// Persistence — convert transient blocks to persistent
// ---------------------------------------------------------------------------

void AiSaqBlockStore::ConvertToPersistent(QueryContext context) {
	for (idx_t i = 0; i < graph_block_handles_.size(); i++) {
		if (graph_block_ids_[i] != INVALID_BLOCK) {
			continue;
		}
		auto bid = block_manager_.GetFreeBlockId();
		auto new_handle = block_manager_.ConvertToPersistent(context, bid, graph_block_handles_[i]);
		graph_block_handles_[i] = new_handle;
		graph_block_ids_[i] = bid;
	}
	for (idx_t i = 0; i < pq_page_handles_.size(); i++) {
		if (pq_page_block_ids_[i] != INVALID_BLOCK) {
			continue;
		}
		auto bid = block_manager_.GetFreeBlockId();
		auto new_handle = block_manager_.ConvertToPersistent(context, bid, pq_page_handles_[i]);
		pq_page_handles_[i] = new_handle;
		pq_page_block_ids_[i] = bid;
	}
}

void AiSaqBlockStore::SerializeState(vector<data_t> &out) const {
	AppendPacked<uint64_t>(out, static_cast<uint64_t>(node_size_));
	AppendPacked<uint64_t>(out, static_cast<uint64_t>(graph_node_count_.load(std::memory_order_relaxed)));
	AppendPacked<uint64_t>(out, static_cast<uint64_t>(graph_block_ids_.size()));
	for (block_id_t bid : graph_block_ids_) {
		AppendPacked<int64_t>(out, static_cast<int64_t>(bid));
	}
	AppendPacked<uint64_t>(out, static_cast<uint64_t>(code_size_));
	AppendPacked<uint64_t>(out, static_cast<uint64_t>(pq_page_block_ids_.size()));
	for (block_id_t bid : pq_page_block_ids_) {
		AppendPacked<int64_t>(out, static_cast<int64_t>(bid));
	}
}

void AiSaqBlockStore::DeserializeState(const_data_ptr_t in, idx_t size) {
	const_data_ptr_t cur = in;
	const_data_ptr_t end = in + size;

	node_size_ = static_cast<idx_t>(ConsumePacked<uint64_t>(cur, end));
	graph_node_count_ = static_cast<uint32_t>(ConsumePacked<uint64_t>(cur, end));
	const auto n_graph_blocks = ConsumePacked<uint64_t>(cur, end);
	graph_block_ids_.clear();
	graph_block_handles_.clear();
	for (uint64_t i = 0; i < n_graph_blocks; i++) {
		auto bid = static_cast<block_id_t>(ConsumePacked<int64_t>(cur, end));
		graph_block_ids_.push_back(bid);
		if (bid != INVALID_BLOCK) {
			graph_block_handles_.push_back(block_manager_.RegisterBlock(bid));
		} else {
			graph_block_handles_.push_back(nullptr);
		}
	}
	const idx_t block_size = block_manager_.GetBlockSize();
	nodes_per_block_ = node_size_ > 0 ? block_size / node_size_ : 0;

	code_size_ = static_cast<idx_t>(ConsumePacked<uint64_t>(cur, end));
	codes_per_page_ = code_size_ > 0 ? block_size / code_size_ : 0;
	const auto n_pq_pages = ConsumePacked<uint64_t>(cur, end);
	pq_page_block_ids_.clear();
	pq_page_handles_.clear();
	for (uint64_t i = 0; i < n_pq_pages; i++) {
		auto bid = static_cast<block_id_t>(ConsumePacked<int64_t>(cur, end));
		pq_page_block_ids_.push_back(bid);
		if (bid != INVALID_BLOCK) {
			pq_page_handles_.push_back(block_manager_.RegisterBlock(bid));
		} else {
			pq_page_handles_.push_back(nullptr);
		}
	}
}

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
