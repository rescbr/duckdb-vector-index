#include "algo/aisaq/aisaq_index.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/enums/index_constraint_type.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/validity_mask.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_size.hpp"
#include "duckdb/execution/index/index_type.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/partial_block_manager.hpp"
#include "duckdb/storage/storage_info.hpp"
#include "duckdb/storage/table_io_manager.hpp"

#include "algo/aisaq/aisaq_module.hpp"
#include "vindex/vector_index_registry.hpp"

#include <chrono>
#include <cstring>

namespace duckdb {
namespace vindex {
namespace aisaq {

namespace {

MetricKind ParseMetric(const case_insensitive_map_t<Value> &options) {
	auto it = options.find("metric");
	if (it == options.end()) {
		return MetricKind::L2SQ;
	}
	const auto raw = StringUtil::Lower(it->second.GetValue<string>());
	if (raw == "l2sq") {
		return MetricKind::L2SQ;
	}
	if (raw == "cosine") {
		return MetricKind::COSINE;
	}
	if (raw == "ip") {
		return MetricKind::IP;
	}
	throw BinderException("vindex: unknown metric '%s' (expected 'l2sq', 'cosine', or 'ip')", raw);
}

idx_t ParseRerankMultiple(const case_insensitive_map_t<Value> &options) {
	auto it = options.find("rerank");
	if (it == options.end()) {
		return 1;
	}
	const int64_t v = it->second.GetValue<int64_t>();
	if (v < 1) {
		throw BinderException("vindex: 'rerank' must be >= 1 (got %lld)", (long long)v);
	}
	return idx_t(v);
}

AiSaqCoreParams ParseCoreParams(const case_insensitive_map_t<Value> &options, idx_t dim) {
	AiSaqCoreParams p;
	p.dim = dim;
	auto r_it = options.find("aisaq_r");
	if (r_it != options.end()) {
		const int64_t v = r_it->second.GetValue<int64_t>();
		if (v < 4 || v > 256) {
			throw BinderException("AiSAQ 'aisaq_r' out of range (4..256)");
		}
		p.R = uint16_t(v);
	}
	auto l_it = options.find("aisaq_l");
	if (l_it != options.end()) {
		const int64_t v = l_it->second.GetValue<int64_t>();
		if (v < 4 || v > 1024) {
			throw BinderException("AiSAQ 'aisaq_l' out of range (4..1024)");
		}
		p.L = uint16_t(v);
	}
	if (p.L < p.R) {
		p.L = p.R;
	}
	auto lb_it = options.find("aisaq_l_build");
	if (lb_it != options.end()) {
		const int64_t v = lb_it->second.GetValue<int64_t>();
		if (v != 0 && (v < 4 || v > 1024)) {
			throw BinderException("AiSAQ 'aisaq_l_build' out of range (4..1024, or 0 for auto=R)");
		}
		p.L_build = uint16_t(v);
	}
	auto a_it = options.find("aisaq_alpha");
	if (a_it != options.end()) {
		const double v = a_it->second.GetValue<double>();
		if (v < 1.0 || v > 2.0) {
			throw BinderException("AiSAQ 'aisaq_alpha' out of range (1.0..2.0)");
		}
		p.alpha = float(v);
	}
	auto inl_it = options.find("aisaq_inline_pq");
	if (inl_it != options.end()) {
		const int64_t v = inl_it->second.GetValue<int64_t>();
		if (v < 0 || v > p.R) {
			throw BinderException("AiSAQ 'aisaq_inline_pq' out of range (0..aisaq_r)");
		}
		p.inline_pq_count = uint16_t(v);
	}
	auto bw_it = options.find("aisaq_beam_width");
	if (bw_it != options.end()) {
		p.beam_width = uint16_t(bw_it->second.GetValue<int64_t>());
	}
	auto io_it = options.find("aisaq_io_limit");
	if (io_it != options.end()) {
		// stored on the index; the core Search receives it per-query.
		(void)io_it;
	}
	auto ep_it = options.find("aisaq_entry_points");
	if (ep_it != options.end()) {
		const int64_t v = ep_it->second.GetValue<int64_t>();
		if (v < 1 || v > 64) {
			throw BinderException("AiSAQ 'aisaq_entry_points' out of range (1..64)");
		}
		p.n_entry_points = uint16_t(v);
	}
	return p;
}

idx_t ParseIoLimit(const case_insensitive_map_t<Value> &options) {
	auto it = options.find("aisaq_io_limit");
	if (it == options.end()) {
		return 0;
	}
	return idx_t(it->second.GetValue<int64_t>());
}

const char *QuantizerKindName(QuantizerKind k) {
	switch (k) {
	case QuantizerKind::FLAT:
		return "flat";
	case QuantizerKind::PQ:
		return "pq";
	case QuantizerKind::RABITQ:
		return "rabitq";
	case QuantizerKind::SCANN:
		return "scann";
	}
	return "unknown";
}

idx_t SampleFloats(ColumnDataCollection &collection, idx_t dim, idx_t sample_cap, vector<float> &buf) {
	const idx_t total = collection.Count();
	const idx_t n = std::min<idx_t>(total, sample_cap);
	buf.clear();
	if (n == 0) {
		return 0;
	}
	buf.resize(n * dim);

	DataChunk scan_chunk;
	collection.InitializeScanChunk(scan_chunk);
	ColumnDataScanState scan_state;
	collection.InitializeScan(scan_state, {0}, ColumnDataScanProperties::ALLOW_ZERO_COPY);

	idx_t copied = 0;
	while (copied < n && collection.Scan(scan_state, scan_chunk)) {
		const auto chunk_size = scan_chunk.size();
		if (chunk_size == 0) {
			continue;
		}
		scan_chunk.Flatten();
		auto &vec_vec = scan_chunk.data[0];
		auto &child_vec = ArrayVector::GetEntry(vec_vec);
		auto *vec_data = FlatVector::GetData<float>(child_vec);
		const idx_t take = std::min<idx_t>(chunk_size, n - copied);
		for (idx_t i = 0; i < take; i++) {
			if (FlatVector::IsNull(vec_vec, i)) {
				std::memset(buf.data() + (copied + i) * dim, 0, dim * sizeof(float));
			} else {
				std::memcpy(buf.data() + (copied + i) * dim, vec_data + i * dim, dim * sizeof(float));
			}
		}
		copied += take;
	}
	return copied;
}

} // namespace

//------------------------------------------------------------------------------
// Construction
//------------------------------------------------------------------------------

void AiSaqIndex::InitFromOptions(const case_insensitive_map_t<Value> &options, idx_t vector_size, MetricKind metric) {
	metric_ = metric;
	dim_ = vector_size;
	params_ = ParseCoreParams(options, vector_size);
	rerank_multiple_ = ParseRerankMultiple(options);
	io_limit_ = ParseIoLimit(options);
	quantizer_ = CreateQuantizer(options, metric_, dim_);
	// AiSAQ pags PQ codes from blocks on demand — its entire storage story
	// depends on a compressing, LUT-capable quantizer. FLAT stores full fp32
	// (defeating the >RAM layout) and has no LUT; RaBitQ has no LUT path
	// (LUTSize() == 0). Reject both up front so the failure is clear.
	if (quantizer_->Kind() == QuantizerKind::FLAT) {
		throw BinderException(
		    "AiSAQ requires a compressing LUT-capable quantizer: use WITH (quantizer='pq') or (quantizer='scann'). "
		    "FLAT would store full fp32 codes and defeat AiSAQ's paged-PQ layout.");
	}
	if (quantizer_->Kind() == QuantizerKind::RABITQ) {
		throw BinderException(
		    "AiSAQ does not support quantizer='rabitq': RaBitQ has no LUT distance path. Use (quantizer='pq') or "
		    "(quantizer='scann').");
	}
	ParseLabelColumn(options);
	beam_width_ = params_.beam_width;
	core_ = make_uniq<AiSaqCore>(params_, *quantizer_, *block_store_);
}

//------------------------------------------------------------------------------
// Build strategy resolution (Tier 1/2/3)
//------------------------------------------------------------------------------

void AiSaqIndex::ResolveBuildStrategy(ClientContext &context, idx_t N) {
	const idx_t code_size = quantizer_->CodeSize();
	const idx_t pq_bytes = N * code_size;
	const idx_t fp_bytes = N * dim_ * sizeof(float);
	const idx_t node_bytes = core_ ? N * core_->StaticNodeSize() : 0;

	// Read explicit strategy from WITH options or session.
	string strategy_str = "auto";
	auto strat_it = stored_options_.find("build_strategy");
	if (strat_it != stored_options_.end() && !strat_it->second.IsNull()) {
		strategy_str = StringUtil::Lower(strat_it->second.GetValue<string>());
	} else {
		Value strat_val;
		if (context.TryGetCurrentSetting("vindex_aisaq_build_strategy", strat_val)) {
			if (!strat_val.IsNull() && strat_val.type() == LogicalType::VARCHAR) {
				strategy_str = StringUtil::Lower(strat_val.GetValue<string>());
			}
		}
	}

	// Read RAM budget from WITH options or session.
	int64_t ram_budget_mb = 0;
	auto ram_it = stored_options_.find("ram_budget_mb");
	if (ram_it != stored_options_.end() && !ram_it->second.IsNull()) {
		ram_budget_mb = ram_it->second.GetValue<int64_t>();
	} else {
		Value ram_val;
		if (context.TryGetCurrentSetting("vindex_aisaq_ram_budget_mb", ram_val)) {
			if (!ram_val.IsNull() && ram_val.type() == LogicalType::BIGINT) {
				ram_budget_mb = ram_val.GetValue<int64_t>();
			}
		}
	}

	// Compute budget in bytes.
	idx_t budget;
	if (ram_budget_mb > 0) {
		budget = idx_t(ram_budget_mb) * 1024 * 1024;
	} else {
		auto &bm = BufferManager::GetBufferManager(context);
		idx_t max_mem = bm.GetMaxMemory();
		if (max_mem == idx_t(-1)) {
			budget = 1024 * 1024 * 1024; // 1 GB default for unlimited memory
		} else {
			budget = max_mem / 4; // 25% of memory_limit
		}
	}

	// Resolve strategy.
	if (strategy_str == "exact_prune") {
		build_strategy_ = BuildStrategy::EXACT_PRUNE;
	} else if (strategy_str == "pq_buffer") {
		build_strategy_ = BuildStrategy::PQ_BUFFER;
	} else if (strategy_str == "paged") {
		build_strategy_ = BuildStrategy::PAGED;
	} else {
		// auto: prefer pq_buffer (best speed/recall balance). exact_prune is opt-in only.
		if (pq_bytes + node_bytes <= budget) {
			build_strategy_ = BuildStrategy::PQ_BUFFER;
		} else {
			build_strategy_ = BuildStrategy::PAGED;
		}
	}
}

void AiSaqIndex::ActivateBuildBuffers() {
	if (!core_) {
		return;
	}
	if (!build_codes_buffer_.empty()) {
		core_->SetBuildCodes(build_codes_buffer_.data(), build_codes_buffer_.size() / quantizer_->CodeSize());
	}
	if (!build_vectors_buffer_.empty()) {
		core_->SetBuildVectors(build_vectors_buffer_.data(), dim_, metric_);
	}
	// Tier 2+: allocate flat graph node buffer so PinNode skips BufferManager.
	if (build_strategy_ != BuildStrategy::PAGED) {
		const idx_t node_size = core_->StaticNodeSize();
		const idx_t N = quantizer_->CodeSize() > 0 ? build_codes_buffer_.size() / quantizer_->CodeSize() : 0;
		if (N > 0) {
			build_nodes_buffer_.assign(N * node_size, 0);
			core_->SetBuildNodes(build_nodes_buffer_.data());
			block_store_->SetFlatBuildMode(true);
			core_->PrepareForBuild(N);
		}
	}
}

void AiSaqIndex::ClearBuildBuffers() {
	if (core_) {
		core_->ClearBuildBuffers();
	}
	build_codes_buffer_.clear();
	build_codes_buffer_.shrink_to_fit();
	build_vectors_buffer_.clear();
	build_vectors_buffer_.shrink_to_fit();
	build_nodes_buffer_.clear();
	build_nodes_buffer_.shrink_to_fit();
}

void AiSaqIndex::FlushBuildNodes() {
	if (!build_nodes_buffer_.empty() && block_store_) {
		block_store_->SetFlatBuildMode(false);
		block_store_->WriteAllGraphNodes(build_nodes_buffer_.data(), block_store_->GraphNodeCount());
	}
}

AiSaqIndex::AiSaqIndex(const string &name, IndexConstraintType index_constraint_type,
                       const vector<column_t> &column_ids, TableIOManager &table_io_manager,
                       const vector<unique_ptr<Expression>> &unbound_expressions, AttachedDatabase &db,
                       const case_insensitive_map_t<Value> &options, const IndexStorageInfo &info,
                       idx_t estimated_cardinality)
    : VectorIndex(name, TYPE_NAME, index_constraint_type, column_ids, table_io_manager, unbound_expressions, db) {

	if (index_constraint_type != IndexConstraintType::NONE) {
		throw NotImplementedException("AiSAQ indexes do not support unique or primary key constraints");
	}

	auto &block_manager = table_io_manager.GetIndexBlockManager();
	auto &buffer_manager = block_manager.GetBufferManager();
	block_store_ = make_uniq<AiSaqBlockStore>(block_manager, buffer_manager);
	state_store_ = make_uniq<IndexBlockStore>(block_manager);
	stored_options_ = options;

	D_ASSERT(logical_types.size() == 1);
	auto &vector_type = logical_types[0];
	D_ASSERT(vector_type.id() == LogicalTypeId::ARRAY);
	const auto vector_size = ArrayType::GetSize(vector_type);
	const auto vector_child_type = ArrayType::GetChildType(vector_type);
	if (vector_child_type.id() != LogicalTypeId::FLOAT) {
		throw NotImplementedException("AiSAQ indexes currently support only FLOAT[] vectors");
	}

	const auto metric = ParseMetric(options);
	InitFromOptions(options, vector_size, metric);

	// Inject the DatabaseInstance so the core + block store can spawn
	// parallel post-build tasks via TaskScheduler (Phase 9 Task 3). When null,
	// the post-build phases fall back to their serial paths.
	if (core_) {
		core_->SetDatabase(&db.GetDatabase());
	}
	if (block_store_) {
		block_store_->SetDatabase(&db.GetDatabase());
	}

	auto lock = rwlock.GetExclusiveLock();
	if (info.IsValid()) {
		state_root_.Set(info.root);
		state_store_->Init(info);
		if (state_root_.Get() != 0) {
			ReadStateStream(state_root_);
		}
	}
	index_size = core_->Size();
	(void)estimated_cardinality;
}

//------------------------------------------------------------------------------
// Metadata
//------------------------------------------------------------------------------

idx_t AiSaqIndex::GetVectorSize() const {
	return dim_;
}

MetricKind AiSaqIndex::GetMetricKind() const {
	return metric_;
}

idx_t AiSaqIndex::GetRerankMultiple(ClientContext &context) const {
	Value v;
	if (context.TryGetCurrentSetting("vindex_rerank_multiple", v)) {
		if (!v.IsNull() && v.type() == LogicalType::BIGINT) {
			const auto session_val = v.GetValue<int64_t>();
			if (session_val > 0) {
				return idx_t(session_val);
			}
		}
	}
	return rerank_multiple_ == 0 ? 1 : rerank_multiple_;
}

const string &AiSaqIndex::QuantizerName() const {
	static const string empty;
	if (!quantizer_) {
		return empty;
	}
	static string name;
	name = QuantizerKindName(quantizer_->Kind());
	return name;
}

//------------------------------------------------------------------------------
// Scans
//------------------------------------------------------------------------------

struct AiSaqIndexScanState : public IndexScanState {
	idx_t current_row = 0;
	idx_t total_rows = 0;
	unique_array<row_t> row_ids = nullptr;
};

static idx_t ResolveLSearch(ClientContext &context, idx_t fallback) {
	Value v;
	if (context.TryGetCurrentSetting("vindex_aisaq_l_search", v)) {
		if (!v.IsNull() && v.type() == LogicalType::BIGINT) {
			const auto val = v.GetValue<int64_t>();
			if (val > 0) {
				return idx_t(val);
			}
		}
	}
	return fallback;
}

unique_ptr<IndexScanState> AiSaqIndex::InitializeScan(float *query_vector, idx_t limit, ClientContext &context,
                                                      const LabelFilter &label_filter) {
	auto state = make_uniq<AiSaqIndexScanState>();
	const idx_t L_search = ResolveLSearch(context, params_.L);

	vector<float> qpre(quantizer_->QueryWorkspaceSize());
	quantizer_->PreprocessQuery(query_vector, qpre.data());
	vector<float> lut(quantizer_->LUTSize());
	quantizer_->PopulateDistanceLUT(qpre.data(), lut.data());

	auto lock = rwlock.GetSharedLock();
	const idx_t oversample = limit + tombstones_.size();
	auto cands = core_->Search(lut.data(), oversample, L_search, beam_width_, io_limit_, label_filter);

	vector<row_t> hits;
	hits.reserve(limit);
	for (auto &c : cands) {
		if (hits.size() >= limit) {
			break;
		}
		if (tombstones_.count(row_t(c.row_id))) {
			continue;
		}
		hits.push_back(row_t(c.row_id));
	}

	state->current_row = 0;
	state->total_rows = hits.size();
	state->row_ids = make_uniq_array<row_t>(hits.size());
	for (idx_t i = 0; i < hits.size(); i++) {
		state->row_ids[i] = hits[i];
	}
	return std::move(state);
}

idx_t AiSaqIndex::Scan(IndexScanState &state, Vector &result, idx_t result_offset) {
	auto &scan_state = state.Cast<AiSaqIndexScanState>();
	idx_t count = 0;
	auto row_ids = FlatVector::GetData<row_t>(result) + result_offset;
	while (count < STANDARD_VECTOR_SIZE && scan_state.current_row < scan_state.total_rows) {
		row_ids[count++] = scan_state.row_ids[scan_state.current_row++];
	}
	return count;
}

struct AiSaqMultiScanState final : IndexScanState {
	Vector vec;
	vector<row_t> row_ids;
	idx_t L_search;
	explicit AiSaqMultiScanState(idx_t L_p) : vec(LogicalType::ROW_TYPE, nullptr), L_search(L_p) {
	}
};

unique_ptr<IndexScanState> AiSaqIndex::InitializeMultiScan(ClientContext &context) {
	const idx_t L_search = ResolveLSearch(context, params_.L);
	return make_uniq<AiSaqMultiScanState>(L_search);
}

idx_t AiSaqIndex::ExecuteMultiScan(IndexScanState &state_p, float *query_vector, idx_t limit) {
	auto &state = state_p.Cast<AiSaqMultiScanState>();
	vector<float> qpre(quantizer_->QueryWorkspaceSize());
	quantizer_->PreprocessQuery(query_vector, qpre.data());
	vector<float> lut(quantizer_->LUTSize());
	quantizer_->PopulateDistanceLUT(qpre.data(), lut.data());

	vector<AiSaqCore::Candidate> cands;
	{
		auto lock = rwlock.GetSharedLock();
		const idx_t oversample = limit + tombstones_.size();
		cands = core_->Search(lut.data(), oversample, state.L_search, beam_width_, io_limit_);
	}
	const auto offset = state.row_ids.size();
	for (auto &c : cands) {
		if (state.row_ids.size() - offset >= limit) {
			break;
		}
		if (tombstones_.count(row_t(c.row_id))) {
			continue;
		}
		state.row_ids.push_back(row_t(c.row_id));
	}
	return state.row_ids.size() - offset;
}

const Vector &AiSaqIndex::GetMultiScanResult(IndexScanState &state) {
	auto &scan_state = state.Cast<AiSaqMultiScanState>();
	FlatVector::SetData(scan_state.vec, (data_ptr_t)scan_state.row_ids.data());
	return scan_state.vec;
}

void AiSaqIndex::ResetMultiScan(IndexScanState &state) {
	auto &scan_state = state.Cast<AiSaqMultiScanState>();
	scan_state.row_ids.clear();
}

//------------------------------------------------------------------------------
// Construction + mutation
//------------------------------------------------------------------------------

void AiSaqIndex::CommitDrop(IndexLock &index_lock) {
	auto lock = rwlock.GetExclusiveLock();
	core_.reset();
	block_store_.reset();
	row_to_internal_.clear();
	tombstones_.clear();
	// Re-create an empty block store + core so the index can be reused.
	auto &block_manager = table_io_manager.GetIndexBlockManager();
	auto &buffer_manager = block_manager.GetBufferManager();
	block_store_ = make_uniq<AiSaqBlockStore>(block_manager, buffer_manager);
	core_ = make_uniq<AiSaqCore>(params_, *quantizer_, *block_store_);
	index_size = 0;
}

void AiSaqIndex::TrainQuantizer(ColumnDataCollection &collection, idx_t sample_cap) {
	auto lock = rwlock.GetExclusiveLock();
	vector<float> buf;
	const idx_t n = SampleFloats(collection, dim_, sample_cap, buf);
	// Phase 9.5 iter 2a: parallelize per-slot k-means++ across cores.
	// PqQuantizer::Train uses the thread hint to spawn workers; each slot's
	// k-means is fully independent (disjoint codebook region, private
	// sub_buffer).
	auto &scheduler = TaskScheduler::GetScheduler(db.GetDatabase());
	quantizer_->SetTrainThreads(uint32_t(scheduler.NumberOfThreads()));
	quantizer_->Train(buf.data(), n, dim_);
}

// Per-task PQ encoder. Each task processes rows [row_start, row_end) of the
// collection — a page-aligned, disjoint range — and writes the corresponding
// PQ pages + flat-buffer slots. See AiSaqIndex::EncodePqCodes for the
// partitioning rationale.
class AiSaqPqEncodeTask final : public BaseExecutorTask {
  public:
	AiSaqPqEncodeTask(TaskExecutor &executor, AiSaqIndex &index, ColumnDataCollection &collection, idx_t row_start,
	                  idx_t row_end)
	    : BaseExecutorTask(executor), index_(index), collection_(collection), row_start_(row_start), row_end_(row_end) {
	}

	void ExecuteTask() override {
		index_.EncodePqRange(collection_, row_start_, row_end_);
	}

	string TaskType() const override {
		return "AiSaqPqEncodeTask";
	}

  private:
	AiSaqIndex &index_;
	ColumnDataCollection &collection_;
	idx_t row_start_;
	idx_t row_end_;
};

void AiSaqIndex::EncodePqCodes(ColumnDataCollection &collection) {
	// Pass 1 does not touch graph state — no rwlock needed. Each task owns a
	// disjoint page range; the only shared mutable state is the atomic
	// pq_encode_progress_ counter and the pre-allocated flat build buffers
	// (whose slot ranges are also disjoint across tasks).
	const idx_t code_size = quantizer_->CodeSize();
	const idx_t codes_per_page = block_store_->CodesPerPage();
	const idx_t N = collection.Count();

	// Tier 2/3: allocate flat PQ code buffer if strategy requires it.
	const bool use_pq_buffer = (build_strategy_ != BuildStrategy::PAGED);
	const bool use_fp_buffer = (build_strategy_ == BuildStrategy::EXACT_PRUNE);
	if (use_pq_buffer) {
		build_codes_buffer_.resize(N * code_size);
	}
	if (use_fp_buffer) {
		build_vectors_buffer_.resize(N * dim_);
	}

	if (LogInfo(build_log_level_)) {
		const char *strat_name = build_strategy_ == BuildStrategy::EXACT_PRUNE ? "exact_prune"
		                         : build_strategy_ == BuildStrategy::PQ_BUFFER ? "pq_buffer"
		                                                                       : "paged";
		fprintf(stderr, "[vindex] AiSAQ build: strategy=%s, N=%llu, pq_buffer=%lluMB", strat_name,
		        (unsigned long long)N, (unsigned long long)(N * code_size / (1024 * 1024)));
		if (use_pq_buffer) {
			fprintf(stderr, ", node_buffer=%lluMB", (unsigned long long)(N * core_->StaticNodeSize() / (1024 * 1024)));
		}
		if (use_fp_buffer) {
			fprintf(stderr, ", fp_buffer=%lluMB", (unsigned long long)(N * dim_ * 4 / (1024 * 1024)));
		}
		fprintf(stderr, "\n");
		fprintf(stderr, "[vindex] PQ encoding: 0/%llu\n", (unsigned long long)N);
	}

	// Pre-allocate every PQ page once, up front. Removes the lazy-block-
	// allocation race in WritePqPage (each call now just memcpys into an
	// already-owned block) and guarantees disjoint page ownership across tasks.
	const idx_t total_pages = (N + codes_per_page - 1) / codes_per_page;
	if (total_pages > 0) {
		block_store_->EnsurePqCapacity(static_cast<uint32_t>(total_pages - 1));
	}

	// Spawn N tasks partitioned by page-aligned row range. Each task owns
	// pages [p_start, p_end) which corresponds to rows
	// [p_start * codes_per_page, min(p_end * codes_per_page, N)). Page writes
	// are non-overlapping by construction; flat-buffer writes are
	// non-overlapping because slot ranges are disjoint.
	auto &scheduler = TaskScheduler::GetScheduler(db.GetDatabase());
	const size_t num_threads = std::max<size_t>(1, NumericCast<size_t>(scheduler.NumberOfThreads()));

	const auto encode_start = std::chrono::steady_clock::now();
	TaskExecutor executor(scheduler);
	if (N > 0) {
		for (size_t t = 0; t < num_threads; t++) {
			const uint32_t p_start = static_cast<uint32_t>(t * total_pages / num_threads);
			const uint32_t p_end = static_cast<uint32_t>((t + 1) * total_pages / num_threads);
			if (p_start >= p_end) {
				continue;
			}
			const idx_t row_start = idx_t(p_start) * codes_per_page;
			const idx_t row_end = std::min(idx_t(p_end) * codes_per_page, N);
			executor.ScheduleTask(make_uniq<AiSaqPqEncodeTask>(executor, *this, collection, row_start, row_end));
		}
		executor.WorkOnTasks();
	}
	if (LogInfo(build_log_level_)) {
		const double elapsed =
		    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - encode_start).count();
		fprintf(stderr, "[vindex] PQ encoding complete: %llu vectors in %.0fms (%llu tasks)\n", (unsigned long long)N,
		        elapsed, (unsigned long long)num_threads);
	}

	if (pq_encode_progress_) {
		pq_encode_progress_->store(N);
	}
}

void AiSaqIndex::EncodePqRange(ColumnDataCollection &collection, idx_t row_start, idx_t row_end) {
	if (row_start >= row_end) {
		return;
	}

	const idx_t code_size = quantizer_->CodeSize();
	const idx_t codes_per_page = block_store_->CodesPerPage();
	const idx_t dim = dim_;
	const bool use_pq_buffer = (build_strategy_ != BuildStrategy::PAGED);
	const bool use_fp_buffer = (build_strategy_ == BuildStrategy::EXACT_PRUNE);

	// Per-task single-page buffer. Zero-initialised so the trailing slots of the
	// final partial page (if this task owns it) match the serial path exactly.
	vector<data_t> page(codes_per_page * code_size, 0);
	uint32_t current_page_idx = static_cast<uint32_t>(row_start / codes_per_page);
	idx_t slot_in_page = 0; // row_start is page-aligned, so always starts at 0.
	idx_t processed = 0;

	ColumnDataScanState scan_state;
	collection.InitializeScan(scan_state, {0}, ColumnDataScanProperties::ALLOW_ZERO_COPY);
	DataChunk scan_chunk;
	collection.InitializeScanChunk(scan_chunk);

	auto process_chunk = [&](DataChunk &chunk, idx_t chunk_base) {
		const auto chunk_size = chunk.size();
		if (chunk_size == 0) {
			return;
		}
		chunk.Flatten();
		auto &vec_vec = chunk.data[0];
		auto &child_vec = ArrayVector::GetEntry(vec_vec);
		const auto array_size = ArrayType::GetSize(vec_vec.GetType());
		auto vec_child_data = FlatVector::GetData<float>(child_vec);
		for (idx_t i = 0; i < chunk_size; i++) {
			const idx_t local_slot = chunk_base + i;
			if (local_slot < row_start) {
				continue;
			}
			if (local_slot >= row_end) {
				return;
			}
			if (FlatVector::IsNull(vec_vec, i)) {
				std::memset(page.data() + slot_in_page * code_size, 0, code_size);
			} else {
				quantizer_->Encode(vec_child_data + i * array_size, page.data() + slot_in_page * code_size);
			}
			// Copy to flat build buffers (Tier 2/3). Slot ranges are disjoint
			// across tasks so no synchronisation is needed.
			if (use_pq_buffer) {
				std::memcpy(build_codes_buffer_.data() + local_slot * code_size, page.data() + slot_in_page * code_size,
				            code_size);
			}
			if (use_fp_buffer) {
				std::memcpy(build_vectors_buffer_.data() + local_slot * dim, vec_child_data + i * array_size,
				            dim * sizeof(float));
			}
			slot_in_page++;
			processed++;
			if (slot_in_page >= codes_per_page) {
				block_store_->WritePqPage(current_page_idx, page.data());
				current_page_idx++;
				slot_in_page = 0;
				std::memset(page.data(), 0, page.size());
			}
		}
	};

	// Position the scan at the chunk containing row_start. The chunk may begin
	// before row_start (page boundaries aren't necessarily chunk boundaries);
	// process_chunk discards leading rows outside [row_start, row_end).
	if (row_start > 0) {
		if (!collection.Seek(row_start, scan_state, scan_chunk)) {
			return;
		}
		process_chunk(scan_chunk, scan_state.current_row_index);
	}

	while (collection.Scan(scan_state, scan_chunk)) {
		if (scan_state.current_row_index >= row_end) {
			break;
		}
		process_chunk(scan_chunk, scan_state.current_row_index);
	}

	// Flush the final partial page (matches the serial tail-flush at line 646).
	if (slot_in_page > 0) {
		block_store_->WritePqPage(current_page_idx, page.data());
	}

	if (pq_encode_progress_) {
		pq_encode_progress_->fetch_add(processed);
	}
}

void AiSaqIndex::Construct(DataChunk &input, Vector &row_ids, idx_t /*thread_idx*/, Vector *labels) {
	D_ASSERT(row_ids.GetType().InternalType() == ROW_TYPE);
	D_ASSERT(logical_types[0] == input.data[0].GetType());
	is_dirty = true;

	const auto count = input.size();
	input.Flatten();
	auto &vec_vec = input.data[0];
	auto &vec_child_vec = ArrayVector::GetEntry(vec_vec);
	const auto array_size = ArrayType::GetSize(vec_vec.GetType());
	auto vec_child_data = FlatVector::GetData<float>(vec_child_vec);
	auto rowid_data = FlatVector::GetData<row_t>(row_ids);
	const int64_t *label_data = nullptr;
	if (labels) {
		labels->Flatten(count);
		label_data = FlatVector::GetData<int64_t>(*labels);
	}

	auto lock = rwlock.GetExclusiveLock();
	for (idx_t i = 0; i < count; i++) {
		if (FlatVector::IsNull(vec_vec, i)) {
			continue;
		}
		const auto rowid = rowid_data[i];
		auto tomb_it = tombstones_.find(rowid);
		if (tomb_it != tombstones_.end()) {
			tombstones_.erase(tomb_it);
		}
		const int64_t label = (label_data && !FlatVector::IsNull(*labels, i)) ? label_data[i] : INT64_MIN;
		const uint32_t internal_id = core_->Insert(int64_t(rowid), vec_child_data + (i * array_size), label);
		row_to_internal_[rowid] = internal_id;
	}
	index_size = core_->Size();
}

// ---------------------------------------------------------------------------
// Phase 9 Task 5.5: parallel Pass 2 (Construct) coordination.
//
// The single-task path (above) takes the rwlock per-chunk and serializes every
// insert. The parallel path replaces it with:
//
//   1. LeaderInsertEntry (serial, in Schedule): reserve [0, N), insert row 0
//      (or first non-NULL) as the entry point.
//   2. N parallel InsertBuildRange tasks (in ExecuteTask): each task scans a
//      disjoint row range, calls AiSaqCore::InsertBuild with its own VamanaTLS,
//      accumulates per-task local label maps.
//   3. PushParallelConstructResults: each task moves its local maps into its
//      slot in parallel_slots_.
//   4. FinalizeParallelConstruct (serial, in FinishEvent, under rwlock):
//      merge maps, set final size.
//
// No rwlock is held during the parallel insert phase. Graph-state safety is
// handled by per-node spinlocks inside AiSaqCore::ConnectAndPrune — each
// back-edge write takes only the target node's lock (sub-µs critical section).
// Forward edges are written to task-exclusive new internal_ids with no lock.
// ---------------------------------------------------------------------------

void AiSaqIndex::InitParallelConstructCollector(idx_t num_slots) {
	parallel_slots_.clear();
	parallel_slots_.resize(num_slots);
}

idx_t AiSaqIndex::LeaderInsertEntry(ClientContext & /*context*/, ColumnDataCollection &collection) {
	// Scan from row 0 forward until we find a non-NULL row. That row becomes
	// the entry point (entry_internal_ = its row_ordinal).
	ColumnDataScanState scan_state;
	collection.InitializeScan(scan_state, ColumnDataScanProperties::ALLOW_ZERO_COPY);
	DataChunk scan_chunk;
	collection.InitializeScanChunk(scan_chunk);

	const idx_t total_n = collection.Count();
	if (total_n == 0) {
		return DConstants::INVALID_INDEX;
	}

	const idx_t row_id_col = !label_column_.empty() ? 2 : 1;
	const bool has_labels = !label_column_.empty();

	vamana::VamanaTLS leader_tls(params_.seed);
	unordered_map<row_t, uint32_t> leader_row_to_internal;
	unordered_map<uint32_t, int64_t> leader_id2label;
	unordered_map<int64_t, vector<uint32_t>> leader_label2ids;

	idx_t entry_row = DConstants::INVALID_INDEX;
	while (collection.Scan(scan_state, scan_chunk)) {
		const auto chunk_size = scan_chunk.size();
		if (chunk_size == 0) {
			continue;
		}
		scan_chunk.Flatten();
		auto &vec_vec = scan_chunk.data[0];
		auto &child_vec = ArrayVector::GetEntry(vec_vec);
		const auto array_size = ArrayType::GetSize(vec_vec.GetType());
		auto vec_child_data = FlatVector::GetData<float>(child_vec);
		auto rowid_data = FlatVector::GetData<row_t>(scan_chunk.data[row_id_col]);
		const int64_t *label_data = has_labels ? FlatVector::GetData<int64_t>(scan_chunk.data[1]) : nullptr;

		for (idx_t i = 0; i < chunk_size; i++) {
			if (FlatVector::IsNull(vec_vec, i)) {
				continue;
			}
			const idx_t row_ordinal = scan_state.current_row_index + i;
			const auto rowid = rowid_data[i];
			const int64_t label =
			    (label_data && !FlatVector::IsNull(scan_chunk.data[1], i)) ? label_data[i] : INT64_MIN;

			// InsertBuild hits the "size_ == 0" branch and sets entry_internal_.
			// (ConnectAndPrune is never reached for the leader insert — the
			// first-call branch returns early — so no per-node locks are
			// taken here. node_locks_ is already sized by ResizeNodeLocks in
			// Schedule() before this call.)
			core_->InsertBuild(static_cast<uint32_t>(row_ordinal), int64_t(rowid), vec_child_data + i * array_size,
			                   label, leader_tls, leader_id2label, leader_label2ids);
			leader_row_to_internal[rowid] = static_cast<uint32_t>(row_ordinal);
			entry_row = row_ordinal;
			break;
		}
		if (entry_row != DConstants::INVALID_INDEX) {
			break;
		}
	}

	// Stash leader outputs into slot 0 (caller will merge them in Finalize).
	if (entry_row != DConstants::INVALID_INDEX && !parallel_slots_.empty()) {
		parallel_slots_[0].row_to_internal = std::move(leader_row_to_internal);
		parallel_slots_[0].id2label = std::move(leader_id2label);
		parallel_slots_[0].label2ids = std::move(leader_label2ids);
	}
	return entry_row;
}

void AiSaqIndex::InsertBuildRange(ClientContext & /*context*/, ColumnDataCollection &collection, idx_t row_start,
                                  idx_t row_end, idx_t skip_row, vamana::VamanaTLS &tls,
                                  unordered_map<row_t, uint32_t> &local_row_to_internal,
                                  unordered_map<uint32_t, int64_t> &local_id2label,
                                  unordered_map<int64_t, vector<uint32_t>> &local_label2ids, atomic<idx_t> &built_count,
                                  size_t /*thread_id*/) {
	if (row_start >= row_end) {
		return;
	}

	const idx_t row_id_col = !label_column_.empty() ? 2 : 1;
	const bool has_labels = !label_column_.empty();

	ColumnDataScanState scan_state;
	collection.InitializeScan(scan_state, ColumnDataScanProperties::ALLOW_ZERO_COPY);
	DataChunk scan_chunk;
	collection.InitializeScanChunk(scan_chunk);

	auto process_chunk = [&](DataChunk &chunk, idx_t chunk_base) {
		const auto chunk_size = chunk.size();
		if (chunk_size == 0) {
			return;
		}
		chunk.Flatten();
		auto &vec_vec = chunk.data[0];
		auto &child_vec = ArrayVector::GetEntry(vec_vec);
		const auto array_size = ArrayType::GetSize(vec_vec.GetType());
		auto vec_child_data = FlatVector::GetData<float>(child_vec);
		auto rowid_data = FlatVector::GetData<row_t>(chunk.data[row_id_col]);
		const int64_t *label_data = has_labels ? FlatVector::GetData<int64_t>(chunk.data[1]) : nullptr;

		for (idx_t i = 0; i < chunk_size; i++) {
			const idx_t row_ordinal = chunk_base + i;
			if (row_ordinal < row_start) {
				continue;
			}
			if (row_ordinal >= row_end) {
				return;
			}
			if (FlatVector::IsNull(vec_vec, i)) {
				continue;
			}
			if (row_ordinal == skip_row) {
				continue; // already inserted by the leader
			}

			const auto rowid = rowid_data[i];
			const int64_t label = (label_data && !FlatVector::IsNull(chunk.data[1], i)) ? label_data[i] : INT64_MIN;

			// InsertBuild's ConnectAndPrune acquires per-node spinlocks
			// inline for back-edge writes; no global lock, no per-task
			// deferred accumulator (Task 5.5).
			core_->InsertBuild(static_cast<uint32_t>(row_ordinal), int64_t(rowid), vec_child_data + i * array_size,
			                   label, tls, local_id2label, local_label2ids);
			local_row_to_internal[rowid] = static_cast<uint32_t>(row_ordinal);
			built_count.fetch_add(1, std::memory_order_relaxed);
		}
	};

	// Position the scan at the chunk containing row_start. The chunk may
	// begin before row_start (chunk boundaries aren't necessarily row-aligned);
	// process_chunk discards leading rows outside [row_start, row_end).
	if (row_start > 0) {
		if (!collection.Seek(row_start, scan_state, scan_chunk)) {
			return;
		}
		process_chunk(scan_chunk, scan_state.current_row_index);
	}

	while (collection.Scan(scan_state, scan_chunk)) {
		if (scan_state.current_row_index >= row_end) {
			break;
		}
		process_chunk(scan_chunk, scan_state.current_row_index);
	}
}

void AiSaqIndex::PushParallelConstructResults(idx_t thread_id, unordered_map<row_t, uint32_t> &&local_row_to_internal,
                                              unordered_map<uint32_t, int64_t> &&local_id2label,
                                              unordered_map<int64_t, vector<uint32_t>> &&local_label2ids) {
	D_ASSERT(idx_t(thread_id) < parallel_slots_.size());
	auto &slot = parallel_slots_[thread_id];
	slot.row_to_internal = std::move(local_row_to_internal);
	slot.id2label = std::move(local_id2label);
	slot.label2ids = std::move(local_label2ids);
}

void AiSaqIndex::FinalizeParallelConstruct(idx_t total_n) {
	auto lock = rwlock.GetExclusiveLock();

	// (1) Merge per-task label maps into core_'s global maps.
	for (const auto &slot : parallel_slots_) {
		core_->MergeLabelMaps(slot.id2label, slot.label2ids);
	}

	// (2) Merge per-task row_to_internal into index's row_to_internal_.
	for (const auto &slot : parallel_slots_) {
		for (const auto &kv : slot.row_to_internal) {
			row_to_internal_[kv.first] = kv.second;
		}
	}

	// (3) Set the final node count. Tasks didn't bump size_ during InsertBuild
	// (that would race); set it once here. Per-node spinlocks in
	// ConnectAndPrune applied every back-edge inline during the parallel
	// phase — no serial reciprocity apply is needed (Task 5.5).
	core_->SetSize(static_cast<uint32_t>(total_n));
	index_size = core_->Size();

	// Drop the per-task slots (frees memory before post-build phases run).
	parallel_slots_.clear();
	parallel_slots_.shrink_to_fit();
}

void AiSaqIndex::Compact() {
	is_dirty = true;
	auto lock = rwlock.GetExclusiveLock();
}

void AiSaqIndex::Delete(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	is_dirty = true;
	const auto count = input.size();
	rowid_vec.Flatten(count);
	auto row_id_data = FlatVector::GetData<row_t>(rowid_vec);

	auto _lock = rwlock.GetExclusiveLock();
	for (idx_t i = 0; i < count; i++) {
		const auto rid = row_id_data[i];
		if (row_to_internal_.count(rid)) {
			tombstones_.insert(rid);
		}
	}
}

ErrorData AiSaqIndex::Insert(IndexLock &lock, DataChunk &input, Vector &rowid_vec) {
	// Live single-row inserts: encode the PQ code then insert the graph node.
	Construct(input, rowid_vec, 0);
	return ErrorData{};
}

ErrorData AiSaqIndex::Append(IndexLock &lock, DataChunk &appended_data, Vector &row_identifiers) {
	DataChunk expression_result;
	expression_result.Initialize(Allocator::DefaultAllocator(), logical_types);
	ExecuteExpressions(appended_data, expression_result);
	Construct(expression_result, row_identifiers, 0);
	return ErrorData{};
}

//------------------------------------------------------------------------------
// Persistence (experimental)
//------------------------------------------------------------------------------

static constexpr uint64_t kAiSaqStateMagic = 0x3153414944415153ULL; // "SAQIDAIS1"

void AiSaqIndex::WriteStateStream() {
	if (state_root_.Get() == 0) {
		state_root_ = BlockId{};
	}
	auto writer = state_store_->BeginStream(state_root_);
	state_root_ = writer->Root();

	const uint64_t magic = kAiSaqStateMagic;
	writer->Write(reinterpret_cast<const_data_ptr_t>(&magic), sizeof(magic));

	const uint64_t rerank = uint64_t(rerank_multiple_);
	writer->Write(reinterpret_cast<const_data_ptr_t>(&rerank), sizeof(rerank));

	// Quantizer blob
	vector<data_t> qblob;
	quantizer_->Serialize(qblob);
	const uint32_t qsize = uint32_t(qblob.size());
	writer->Write(reinterpret_cast<const_data_ptr_t>(&qsize), sizeof(qsize));
	if (qsize > 0) {
		writer->Write(qblob.data(), qsize);
	}

	// Block store state (node_size, code_size, block IDs)
	vector<data_t> bs_blob;
	block_store_->SerializeState(bs_blob);
	const uint32_t bs_size = uint32_t(bs_blob.size());
	writer->Write(reinterpret_cast<const_data_ptr_t>(&bs_size), sizeof(bs_size));
	if (bs_size > 0) {
		writer->Write(bs_blob.data(), bs_size);
	}

	// Core state (entry points, params, size)
	vector<data_t> cblob;
	core_->SerializeState(cblob);
	const uint32_t csize = uint32_t(cblob.size());
	writer->Write(reinterpret_cast<const_data_ptr_t>(&csize), sizeof(csize));
	if (csize > 0) {
		writer->Write(cblob.data(), csize);
	}

	// row_to_internal map
	const uint64_t rsize = row_to_internal_.size();
	writer->Write(reinterpret_cast<const_data_ptr_t>(&rsize), sizeof(rsize));
	for (const auto &kv : row_to_internal_) {
		const int64_t rid = kv.first;
		const uint32_t iid = kv.second;
		writer->Write(reinterpret_cast<const_data_ptr_t>(&rid), sizeof(rid));
		writer->Write(reinterpret_cast<const_data_ptr_t>(&iid), sizeof(iid));
	}

	// tombstones
	const uint64_t tsize = tombstones_.size();
	writer->Write(reinterpret_cast<const_data_ptr_t>(&tsize), sizeof(tsize));
	for (const auto &t : tombstones_) {
		const int64_t rid = t;
		writer->Write(reinterpret_cast<const_data_ptr_t>(&rid), sizeof(rid));
	}
}

void AiSaqIndex::ReadStateStream(BlockId root) {
	auto reader = state_store_->OpenStream(root);

	uint64_t magic = 0;
	reader->Read(reinterpret_cast<data_ptr_t>(&magic), sizeof(magic));
	if (magic != kAiSaqStateMagic) {
		throw InternalException("AiSaqIndex: unrecognized state stream (magic mismatch)");
	}

	uint64_t rerank = 1;
	reader->Read(reinterpret_cast<data_ptr_t>(&rerank), sizeof(rerank));
	rerank_multiple_ = idx_t(rerank == 0 ? 1 : rerank);

	uint32_t qsize = 0;
	reader->Read(reinterpret_cast<data_ptr_t>(&qsize), sizeof(qsize));
	if (qsize > 0) {
		vector<data_t> qblob(qsize);
		reader->Read(qblob.data(), qsize);
		quantizer_->Deserialize(qblob.data(), qsize);
	}

	uint32_t bs_size = 0;
	reader->Read(reinterpret_cast<data_ptr_t>(&bs_size), sizeof(bs_size));
	if (bs_size > 0) {
		vector<data_t> bs_blob(bs_size);
		reader->Read(bs_blob.data(), bs_size);
		block_store_->DeserializeState(bs_blob.data(), bs_size);
	}

	uint32_t csize = 0;
	reader->Read(reinterpret_cast<data_ptr_t>(&csize), sizeof(csize));
	if (csize > 0) {
		vector<data_t> cblob(csize);
		reader->Read(cblob.data(), csize);
		core_->DeserializeState(cblob.data(), csize);
	}

	uint64_t rsize = 0;
	reader->Read(reinterpret_cast<data_ptr_t>(&rsize), sizeof(rsize));
	row_to_internal_.clear();
	row_to_internal_.reserve(rsize);
	for (uint64_t i = 0; i < rsize; i++) {
		int64_t rid = 0;
		uint32_t iid = 0;
		reader->Read(reinterpret_cast<data_ptr_t>(&rid), sizeof(rid));
		reader->Read(reinterpret_cast<data_ptr_t>(&iid), sizeof(iid));
		row_to_internal_.emplace(row_t(rid), iid);
	}

	uint64_t tsize = 0;
	reader->Read(reinterpret_cast<data_ptr_t>(&tsize), sizeof(tsize));
	tombstones_.clear();
	tombstones_.reserve(tsize);
	for (uint64_t i = 0; i < tsize; i++) {
		int64_t rid = 0;
		reader->Read(reinterpret_cast<data_ptr_t>(&rid), sizeof(rid));
		tombstones_.insert(row_t(rid));
	}
}

void AiSaqIndex::PersistToDisk() {
	auto lock = rwlock.GetExclusiveLock();
	if (!is_dirty) {
		return;
	}
	WriteStateStream();
	is_dirty = false;
}

IndexStorageInfo AiSaqIndex::SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) {
	PersistToDisk();

	// Convert transient blocks to persistent DuckDB blocks.
	block_store_->ConvertToPersistent(context);

	// Re-write state stream now that block IDs are persistent.
	WriteStateStream();

	// Serialize the state store's LinkedBlock chain to metadata blocks.
	auto &block_manager = table_io_manager.GetIndexBlockManager();
	PartialBlockManager partial_block_manager(context, block_manager, PartialBlockType::FULL_CHECKPOINT);
	state_store_->SerializeBuffers(partial_block_manager);
	partial_block_manager.FlushPartialBlocks();

	IndexStorageInfo info = state_store_->GetInfo();
	info.name = name;
	info.root = state_root_.Get();
	info.options = stored_options_;
	return info;
}

IndexStorageInfo AiSaqIndex::SerializeToWAL(const case_insensitive_map_t<Value> & /*options*/) {
	PersistToDisk();
	IndexStorageInfo info = state_store_->GetInfo();
	info.name = name;
	info.root = state_root_.Get();
	info.options = stored_options_;
	info.buffers = state_store_->InitSerializationToWAL();
	return info;
}

idx_t AiSaqIndex::GetInMemorySize(IndexLock &state) {
	return block_store_ ? block_store_->GraphNodeCount() * block_store_->NodeSize() : 0;
}

bool AiSaqIndex::MergeIndexes(IndexLock &state, BoundIndex &other_index) {
	throw NotImplementedException("AiSaqIndex::MergeIndexes() not implemented");
}

void AiSaqIndex::Vacuum(IndexLock &state) {
}

void AiSaqIndex::Verify(IndexLock &state) {
}

string AiSaqIndex::ToString(IndexLock &state, bool display_ascii) {
	throw NotImplementedException("AiSaqIndex::ToString() not implemented");
}

void AiSaqIndex::VerifyAllocations(IndexLock &state) {
}

void AiSaqIndex::VerifyBuffers(IndexLock &lock) {
}

//------------------------------------------------------------------------------
// Registration
//------------------------------------------------------------------------------

void RegisterIndex(DatabaseInstance &db) {
	IndexType index_type;
	index_type.name = AiSaqIndex::TYPE_NAME;
	index_type.create_instance = [](CreateIndexInput &input) -> unique_ptr<BoundIndex> {
		// DuckDB v1.5.x: IndexCatalogEntry::GetInfo() does not copy the options
		// map, so on checkpoint restart input.options is empty. Fall back to
		// storage_info.options (populated in SerializeToDisk/SerializeToWAL).
		const auto &opts = input.options.empty() ? input.storage_info.options : input.options;
		auto res = make_uniq<AiSaqIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
		                                 input.unbound_expressions, input.db, opts, input.storage_info);
		return std::move(res);
	};
	index_type.create_plan = AiSaqIndex::CreatePlan;

	if (!db.config.GetOptionByName("vindex_enable_experimental_persistence")) {
		db.config.AddExtensionOption("vindex_enable_experimental_persistence",
		                             "experimental: enable creating vector indexes in persistent databases",
		                             LogicalType::BOOLEAN, Value::BOOLEAN(false));
	}
	if (!db.config.GetOptionByName("vindex_rerank_multiple")) {
		db.config.AddExtensionOption("vindex_rerank_multiple",
		                             "override the rerank over-fetch factor when scanning vector indexes",
		                             LogicalType::BIGINT);
	}
	if (!db.config.GetOptionByName("vindex_aisaq_l_search")) {
		db.config.AddExtensionOption("vindex_aisaq_l_search", "override the AiSAQ beam width (L_search) when scanning",
		                             LogicalType::BIGINT);
	}
	if (!db.config.GetOptionByName("vindex_aisaq_build_strategy")) {
		db.config.AddExtensionOption(
		    "vindex_aisaq_build_strategy",
		    "AiSAQ build acceleration: 'auto' (default), 'paged', 'pq_buffer', 'exact_prune' (slow, best quality)",
		    LogicalType::VARCHAR, Value("auto"));
	}
	if (!db.config.GetOptionByName("vindex_aisaq_ram_budget_mb")) {
		db.config.AddExtensionOption("vindex_aisaq_ram_budget_mb",
		                             "RAM budget in MB for AiSAQ build-time buffers (0 = auto = 25%% of memory_limit)",
		                             LogicalType::BIGINT, Value::BIGINT(0));
	}

	db.config.GetIndexTypes().RegisterIndexType(index_type);

	VectorIndexRegistry::Instance().RegisterTypeName(AiSaqIndex::TYPE_NAME);
}

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
