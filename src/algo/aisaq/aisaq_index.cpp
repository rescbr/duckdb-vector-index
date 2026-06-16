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
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/storage/storage_info.hpp"
#include "duckdb/storage/table_io_manager.hpp"

#include "algo/aisaq/aisaq_module.hpp"
#include "vindex/vector_index_registry.hpp"

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

	auto lock = rwlock.GetExclusiveLock();
	(void)info; // persistence is experimental — see SerializeToDisk.
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

unique_ptr<IndexScanState> AiSaqIndex::InitializeScan(float *query_vector, idx_t limit, ClientContext &context) {
	auto state = make_uniq<AiSaqIndexScanState>();
	const idx_t L_search = ResolveLSearch(context, params_.L);

	vector<float> qpre(quantizer_->QueryWorkspaceSize());
	quantizer_->PreprocessQuery(query_vector, qpre.data());
	vector<float> lut(quantizer_->LUTSize());
	quantizer_->PopulateDistanceLUT(qpre.data(), lut.data());

	auto lock = rwlock.GetSharedLock();
	const idx_t oversample = limit + tombstones_.size();
	auto cands = core_->Search(lut.data(), oversample, L_search, beam_width_, io_limit_);

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
	quantizer_->Train(buf.data(), n, dim_);
}

void AiSaqIndex::EncodePqCodes(ColumnDataCollection &collection) {
	auto lock = rwlock.GetExclusiveLock();
	const idx_t code_size = quantizer_->CodeSize();
	const idx_t codes_per_page = block_store_->CodesPerPage();

	vector<data_t> page(codes_per_page * code_size, 0);
	uint32_t page_idx = 0;
	idx_t slot = 0;

	DataChunk scan_chunk;
	collection.InitializeScanChunk(scan_chunk);
	ColumnDataScanState scan_state;
	collection.InitializeScan(scan_state, {0}, ColumnDataScanProperties::ALLOW_ZERO_COPY);

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
		for (idx_t i = 0; i < chunk_size; i++) {
			if (FlatVector::IsNull(vec_vec, i)) {
				std::memset(page.data() + slot * code_size, 0, code_size);
			} else {
				quantizer_->Encode(vec_child_data + i * array_size, page.data() + slot * code_size);
			}
			slot++;
			if (slot >= codes_per_page) {
				block_store_->WritePqPage(page_idx, page.data());
				page_idx++;
				slot = 0;
				std::memset(page.data(), 0, page.size());
			}
		}
	}
	if (slot > 0) {
		block_store_->WritePqPage(page_idx, page.data());
	}
}

void AiSaqIndex::Construct(DataChunk &input, Vector &row_ids, idx_t /*thread_idx*/) {
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
		const uint32_t internal_id = core_->Insert(int64_t(rowid), vec_child_data + (i * array_size));
		row_to_internal_[rowid] = internal_id;
	}
	index_size = core_->Size();
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

IndexStorageInfo AiSaqIndex::SerializeToDisk(QueryContext context, const case_insensitive_map_t<Value> &options) {
	throw NotImplementedException("AiSAQ persistence is not yet implemented (experimental)");
}

IndexStorageInfo AiSaqIndex::SerializeToWAL(const case_insensitive_map_t<Value> &options) {
	throw NotImplementedException("AiSAQ persistence is not yet implemented (experimental)");
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
		auto res = make_uniq<AiSaqIndex>(input.name, input.constraint_type, input.column_ids, input.table_io_manager,
		                                 input.unbound_expressions, input.db, input.options, input.storage_info);
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

	db.config.GetIndexTypes().RegisterIndexType(index_type);

	VectorIndexRegistry::Instance().RegisterTypeName(AiSaqIndex::TYPE_NAME);
}

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
