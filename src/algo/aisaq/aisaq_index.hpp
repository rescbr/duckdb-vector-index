#pragma once

#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/helper.hpp"
#include "duckdb/common/string.hpp"
#include "duckdb/common/typedefs.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/storage_lock.hpp"
#include "duckdb/storage/table/scan_state.hpp"

#include "algo/aisaq/aisaq_block_store.hpp"
#include "algo/aisaq/aisaq_core.hpp"
#include "vindex/index_block_store.hpp"
#include "vindex/logging.hpp"
#include "vindex/quantizer.hpp"
#include "vindex/vector_index.hpp"

namespace duckdb {
class ColumnDataCollection;
namespace vindex {
namespace aisaq {

class AiSaqIndex : public VectorIndex {
  public:
	static constexpr const char *TYPE_NAME = "AISAQ";

	AiSaqIndex(const string &name, IndexConstraintType index_constraint_type, const vector<column_t> &column_ids,
	           TableIOManager &table_io_manager, const vector<unique_ptr<Expression>> &unbound_expressions,
	           AttachedDatabase &db, const case_insensitive_map_t<Value> &options,
	           const IndexStorageInfo &info = IndexStorageInfo(), idx_t estimated_cardinality = 0);

	static PhysicalOperator &CreatePlan(PlanIndexInput &input);

	// --- VectorIndex contract ------------------------------------------------
	MetricKind GetMetricKind() const override;
	idx_t GetVectorSize() const override;
	idx_t GetRerankMultiple(ClientContext &context) const override;

	bool SupportsLabelFilter() const override {
		return !label_column_.empty();
	}

	unique_ptr<IndexScanState> InitializeScan(float *query_vector, idx_t limit, ClientContext &context,
	                                          const LabelFilter &label_filter) override;
	idx_t Scan(IndexScanState &state, Vector &result, idx_t result_offset = 0) override;

	unique_ptr<IndexScanState> InitializeMultiScan(ClientContext &context) override;
	idx_t ExecuteMultiScan(IndexScanState &state, float *query_vector, idx_t limit) override;
	const Vector &GetMultiScanResult(IndexScanState &state) override;
	void ResetMultiScan(IndexScanState &state) override;

	// --- AiSAQ-specific ------------------------------------------------------
	void Construct(DataChunk &input, Vector &row_ids, idx_t thread_idx, Vector *labels = nullptr);
	void Compact() override;

	void TrainQuantizer(ColumnDataCollection &collection, idx_t sample_cap = 65536);
	// Pass 1 of the two-pass build: encode every vector to a PQ code and write
	// directly to the block store's PQ pages. Spawns NumberOfThreads() tasks,
	// each owning a disjoint page-aligned row range. Output is bit-identical to
	// the single-threaded version.
	void EncodePqCodes(ColumnDataCollection &collection);
	// Per-task PQ encode worker. Public so the AiSaqPqEncodeTask defined in the
	// .cpp can call it without friend declarations.
	void EncodePqRange(ColumnDataCollection &collection, idx_t row_start, idx_t row_end);
	// Build-finalize hooks forwarded to the core.
	void FinalizeInlineCodes() {
		if (core_) {
			core_->FinalizeInlineCodes();
		}
	}
	void ComputeEntryPoints() {
		if (core_) {
			core_->ComputeEntryPoints();
		}
	}
	void ComputeLabelMedoids() {
		if (core_) {
			core_->ComputeLabelMedoids();
		}
	}

	// --- build-time acceleration --------------------------------------------
	// Resolve build strategy (paged/pq_buffer/exact_prune) based on
	// dataset size, available memory, and user options. Called from Finalize.
	void ResolveBuildStrategy(ClientContext &context, idx_t N);
	// Pre-size the block store's graph block vectors to hold N nodes, so that
	// the per-node EnsureGraphCapacity call inside AllocGraphNode is a no-op
	// during the parallel construct pass (Phase 9 Task 4). Must be called once,
	// single-threaded, after Pass 1 knows N and before any construct task
	// spawns. Safe to call in flat-build mode too (it just pre-sizes vectors
	// that flat mode skips touching during AllocGraphNode).
	void PreAllocateGraphCapacity(idx_t N) {
		if (block_store_) {
			block_store_->EnsureGraphCapacity(static_cast<uint32_t>(N));
		}
	}
	// Populate flat PQ codes + full-precision vectors during EncodePqCodes.
	// Activate the buffers on the core.
	void ActivateBuildBuffers();
	// Free build-time buffers after construction completes.
	void ClearBuildBuffers();
	// Flush flat graph node buffer to block store.
	void FlushBuildNodes();
	void SetPqEncodeProgress(atomic<idx_t> *counter) {
		pq_encode_progress_ = counter;
	}
	void SetBuildLogLevel(LogLevel l) {
		build_log_level_ = l;
	}
	LogLevel GetBuildLogLevel() const {
		return build_log_level_;
	}
	idx_t QuantizerCodeSize() const {
		return quantizer_ ? quantizer_->CodeSize() : 0;
	}

	void VerifyBuffers(IndexLock &lock) override;

	// --- DuckDB BoundIndex hooks --------------------------------------------
	ErrorData Append(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) override;
	void CommitDrop(IndexLock &index_lock) override;
	void Delete(IndexLock &lock, DataChunk &entries, Vector &row_identifiers) override;
	ErrorData Insert(IndexLock &lock, DataChunk &data, Vector &row_ids) override;

	IndexStorageInfo SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) override;
	IndexStorageInfo SerializeToWAL(const case_insensitive_map_t<Value> &options) override;

	idx_t GetInMemorySize(IndexLock &state) override;
	bool MergeIndexes(IndexLock &state, BoundIndex &other_index) override;
	void Vacuum(IndexLock &state) override;
	void Verify(IndexLock &state) override;
	string ToString(IndexLock &state, bool display_ascii = false) override;
	void VerifyAllocations(IndexLock &state) override;
	string GetConstraintViolationMessage(VerifyExistenceType verify_type, idx_t failed_index,
	                                     DataChunk &input) override {
		return "Constraint violation in AISAQ index";
	}

	void SetDirty() {
		is_dirty = true;
	}
	void SyncSize() {
		index_size = core_ ? core_->Size() : 0;
	}

	idx_t Count() const {
		return core_ ? core_->Size() : 0;
	}
	idx_t GraphDegree() const {
		return params_.R;
	}
	idx_t BeamWidth() const {
		return params_.L;
	}
	float Alpha() const {
		return params_.alpha;
	}
	idx_t InlinePqCount() const {
		return params_.inline_pq_count;
	}
	idx_t EntryPoints() const {
		return params_.n_entry_points;
	}
	idx_t GraphBlockCount() const {
		return block_store_ ? block_store_->GraphBlockCount() : 0;
	}
	idx_t PqPageCount() const {
		return block_store_ ? block_store_->PqPageCount() : 0;
	}
	idx_t NodesPerBlock() const {
		return block_store_ ? block_store_->NodesPerBlock() : 0;
	}
	idx_t CodesPerPage() const {
		return block_store_ ? block_store_->CodesPerPage() : 0;
	}
	const string &QuantizerName() const;

  private:
	unique_ptr<AiSaqBlockStore> block_store_;
	unique_ptr<IndexBlockStore> state_store_; // for state stream (LinkedBlock chain)
	unique_ptr<Quantizer> quantizer_;
	unique_ptr<AiSaqCore> core_;

	BlockId state_root_;

	MetricKind metric_{MetricKind::L2SQ};
	idx_t dim_ = 0;
	AiSaqCoreParams params_{};
	idx_t rerank_multiple_ = 1;
	idx_t beam_width_ = 8;
	idx_t io_limit_ = 0;

	// Maps table row_id → internal_id (for delete/tombstone bookkeeping).
	unordered_map<row_t, uint32_t> row_to_internal_;
	unordered_set<row_t> tombstones_;

	bool is_dirty = false;
	case_insensitive_map_t<Value> stored_options_;

	// Build-time acceleration buffers (Tier 2/3). Allocated in EncodePqCodes,
	// activated on core_, freed after construction.
	enum class BuildStrategy { PAGED, PQ_BUFFER, EXACT_PRUNE };
	BuildStrategy build_strategy_ = BuildStrategy::PAGED;
	vector<uint8_t> build_codes_buffer_;
	vector<float> build_vectors_buffer_;
	vector<uint8_t> build_nodes_buffer_;

	// Build-time progress + logging.
	atomic<idx_t> *pq_encode_progress_ = nullptr;
	LogLevel build_log_level_ = LogLevel::OFF;

	StorageLock rwlock;
	atomic<idx_t> index_size = {0};

	void InitFromOptions(const case_insensitive_map_t<Value> &options, idx_t vector_size, MetricKind metric);

	void WriteStateStream();
	void ReadStateStream(BlockId root);
	void PersistToDisk();
};

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
