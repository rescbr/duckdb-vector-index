#include "algo/aisaq/aisaq_build.hpp"

#include "duckdb/catalog/catalog_entry/duck_index_entry.hpp"
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception/transaction_exception.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/parallel/base_pipeline_event.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/data_table.hpp"
#include "duckdb/storage/index.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/storage_manager.hpp"
#include "duckdb/storage/table_io_manager.hpp"

#include "algo/aisaq/aisaq_index.hpp"
#include "vindex/logging.hpp"
#include "vindex/vamana.hpp"

#include <algorithm>
#include <chrono>

namespace duckdb {
namespace vindex {
namespace aisaq {

PhysicalCreateAiSaqIndex::PhysicalCreateAiSaqIndex(PhysicalPlan &physical_plan, const vector<LogicalType> &types_p,
                                                   TableCatalogEntry &table_p, const vector<column_t> &column_ids,
                                                   unique_ptr<CreateIndexInfo> info,
                                                   vector<unique_ptr<Expression>> unbound_expressions,
                                                   idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, types_p, estimated_cardinality),
      table(table_p.Cast<DuckTableEntry>()), info(std::move(info)), unbound_expressions(std::move(unbound_expressions)),
      sorted(false) {
	for (auto &column_id : column_ids) {
		storage_ids.push_back(table.GetColumns().LogicalToPhysical(LogicalIndex(column_id)).index);
	}
}

//-------------------------------------------------------------
// Global / Local State
//-------------------------------------------------------------
class CreateAiSaqIndexGlobalState final : public GlobalSinkState {
  public:
	explicit CreateAiSaqIndexGlobalState(const PhysicalOperator &op_p) : op(op_p) {
	}

	const PhysicalOperator &op;
	unique_ptr<AiSaqIndex> global_index;

	mutex glock;
	unique_ptr<ColumnDataCollection> collection;
	shared_ptr<ClientContext> context;

	ColumnDataParallelScanState scan_state;
	bool has_labels = false;

	atomic<bool> is_building{false};
	atomic<idx_t> loaded_count{0};
	atomic<idx_t> pq_encoded_count{0};
	atomic<idx_t> built_count{0};
};

static bool OptionsHaveLabelColumn(const case_insensitive_map_t<Value> &options) {
	auto it = options.find("label_column");
	return it != options.end() && !it->second.IsNull() && !it->second.GetValue<string>().empty();
}

unique_ptr<GlobalSinkState> PhysicalCreateAiSaqIndex::GetGlobalSinkState(ClientContext &context) const {
	auto gstate = make_uniq<CreateAiSaqIndexGlobalState>(*this);

	gstate->has_labels = OptionsHaveLabelColumn(info->options);

	vector<LogicalType> data_types = {unbound_expressions[0]->return_type};
	if (gstate->has_labels) {
		data_types.push_back(LogicalType::BIGINT);
	}
	data_types.emplace_back(LogicalType::ROW_TYPE);
	gstate->collection = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context), data_types);
	gstate->context = context.shared_from_this();

	auto &storage = table.GetStorage();
	auto &table_manager = TableIOManager::Get(storage);
	auto &constraint_type = info->constraint_type;
	auto &db = storage.db;
	gstate->global_index =
	    make_uniq<AiSaqIndex>(info->index_name, constraint_type, storage_ids, table_manager, unbound_expressions, db,
	                          info->options, IndexStorageInfo(), estimated_cardinality);

	return std::move(gstate);
}

class CreateAiSaqIndexLocalState final : public LocalSinkState {
  public:
	unique_ptr<ColumnDataCollection> collection;
	ColumnDataAppendState append_state;
};

unique_ptr<LocalSinkState> PhysicalCreateAiSaqIndex::GetLocalSinkState(ExecutionContext &context) const {
	auto state = make_uniq<CreateAiSaqIndexLocalState>();
	vector<LogicalType> data_types = {unbound_expressions[0]->return_type};
	if (OptionsHaveLabelColumn(info->options)) {
		data_types.push_back(LogicalType::BIGINT);
	}
	data_types.emplace_back(LogicalType::ROW_TYPE);
	state->collection = make_uniq<ColumnDataCollection>(BufferManager::GetBufferManager(context.client), data_types);
	state->collection->InitializeAppend(state->append_state);
	return std::move(state);
}

SinkResultType PhysicalCreateAiSaqIndex::Sink(ExecutionContext &context, DataChunk &chunk,
                                              OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<CreateAiSaqIndexLocalState>();
	auto &gstate = input.global_state.Cast<CreateAiSaqIndexGlobalState>();
	lstate.collection->Append(lstate.append_state, chunk);
	gstate.loaded_count += chunk.size();
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalCreateAiSaqIndex::Combine(ExecutionContext &context,
                                                        OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<CreateAiSaqIndexGlobalState>();
	auto &lstate = input.local_state.Cast<CreateAiSaqIndexLocalState>();

	if (lstate.collection->Count() == 0) {
		return SinkCombineResultType::FINISHED;
	}

	lock_guard<mutex> l(gstate.glock);
	if (!gstate.collection) {
		gstate.collection = std::move(lstate.collection);
	} else {
		gstate.collection->Combine(*lstate.collection);
	}
	return SinkCombineResultType::FINISHED;
}

//-------------------------------------------------------------
// Finalize — two-pass build
//-------------------------------------------------------------

// Parallel Vamana construct task (Phase 9 Task 5.5). Each task:
//   - Owns a disjoint row range [row_start, row_end).
//   - Owns its VamanaTLS (per-thread scratch) + local maps.
//   - Scans its row range via a private ColumnDataScanState (NOT the shared
//     parallel scan state — that interleaves chunks and would break the
//     internal_id <-> PQ-slot mapping).
//   - Calls AiSaqIndex::InsertBuildRange which forwards to
//     AiSaqCore::InsertBuild with internal_id = row_ordinal deterministically.
//
// On completion, hands its local maps to AiSaqIndex's parallel_slots_ via
// PushParallelConstructResults. FinishEvent merges them serially.
//
// No rwlock is held during ExecuteTask. Graph-state safety is handled by
// per-node spinlocks inside AiSaqCore::ConnectAndPrune — back-edge writes
// take only the target node's lock (sub-µs critical section). Forward edges
// are written to task-exclusive new internal_ids with no lock. No per-task
// deferred accumulator, no periodic-apply lock acquisition.
class AiSaqIndexConstructTask final : public ExecutorTask {
  public:
	AiSaqIndexConstructTask(shared_ptr<Event> event_p, ClientContext &context, CreateAiSaqIndexGlobalState &gstate_p,
	                        size_t thread_id_p, idx_t row_start_p, idx_t row_end_p, idx_t skip_row_p,
	                        const PhysicalCreateAiSaqIndex &op_p)
	    : ExecutorTask(context, std::move(event_p), op_p), gstate(gstate_p), thread_id(thread_id_p),
	      row_start(row_start_p), row_end(row_end_p), skip_row(skip_row_p),
	      tls_(gstate_p.global_index->CoreParams().seed + thread_id_p + 1) {
	}

	TaskExecutionResult ExecuteTask(TaskExecutionMode /*mode*/) override {
		auto &collection = gstate.collection;
		auto &global_index = gstate.global_index;

		const idx_t total_n = collection->Count();
		const LogLevel log_level = global_index->GetBuildLogLevel();
		auto last_log = std::chrono::steady_clock::now();

		global_index->InsertBuildRange(executor.context, *collection, row_start, row_end, skip_row, tls_,
		                               local_row_to_internal_, local_id2label_, local_label2ids_, gstate.built_count,
		                               thread_id);

		// Log throttled progress (built_count is shared across tasks).
		if (LogInfo(log_level)) {
			auto now = std::chrono::steady_clock::now();
			if (now - last_log > std::chrono::seconds(2)) {
				const auto built = gstate.built_count.load();
				fprintf(stderr, "[vindex] graph construction: %llu/%llu (%.0f%%)\n", (unsigned long long)built,
				        (unsigned long long)total_n, 100.0 * double(built) / double(total_n));
				last_log = now;
			}
		}

		// Hand off local state to the event's collector.
		global_index->PushParallelConstructResults(thread_id, std::move(local_row_to_internal_),
		                                           std::move(local_id2label_), std::move(local_label2ids_));

		event->FinishTask();
		return TaskExecutionResult::TASK_FINISHED;
	}

  private:
	CreateAiSaqIndexGlobalState &gstate;
	size_t thread_id;
	idx_t row_start;
	idx_t row_end;
	idx_t skip_row;
	vamana::VamanaTLS tls_;
	unordered_map<row_t, uint32_t> local_row_to_internal_;
	unordered_map<uint32_t, int64_t> local_id2label_;
	unordered_map<int64_t, vector<uint32_t>> local_label2ids_;
};

class AiSaqIndexConstructionEvent final : public BasePipelineEvent {
  public:
	AiSaqIndexConstructionEvent(const PhysicalCreateAiSaqIndex &op_p, CreateAiSaqIndexGlobalState &gstate_p,
	                            Pipeline &pipeline_p, CreateIndexInfo &info_p, const vector<column_t> &storage_ids_p,
	                            DuckTableEntry &table_p)
	    : BasePipelineEvent(pipeline_p), op(op_p), gstate(gstate_p), info(info_p), storage_ids(storage_ids_p),
	      table(table_p) {
	}

	const PhysicalCreateAiSaqIndex &op;
	CreateAiSaqIndexGlobalState &gstate;
	CreateIndexInfo &info;
	const vector<column_t> &storage_ids;
	DuckTableEntry &table;
	std::chrono::steady_clock::time_point schedule_ts;

	void Schedule() override {
		schedule_ts = std::chrono::steady_clock::now();
		auto &context = pipeline->GetClientContext();
		auto &collection = gstate.collection;
		const idx_t total_n = collection->Count();

		if (total_n == 0) {
			// Nothing to construct; FinishEvent still runs the post-build
			// phases (they no-op on empty state).
			return;
		}

		auto &global_index = gstate.global_index;

		// Pre-reserve [0, total_n) in one atomic step. This bumps
		// graph_node_count_ to total_n so every internal_id in [0, total_n)
		// passes PinGraphNode's D_ASSERT, and each task can use
		// internal_id = row_ordinal deterministically.
		global_index->PreAllocateGraphRange(total_n);

		// Phase 9 Task 5.5: pre-size the per-node spinlock array. MUST happen
		// before any parallel InsertBuild task spawns — the array is indexed
		// by internal_id and must not be reallocated concurrently with task
		// references into it.
		global_index->ResizeNodeLocks(total_n);

		// Parallel-construct collector: +1 slot for the leader (slot 0).
		auto &scheduler = TaskScheduler::GetScheduler(context);
		const size_t num_threads = std::max<size_t>(1, NumericCast<size_t>(scheduler.NumberOfThreads()));
		const size_t num_tasks = std::min<size_t>(num_threads, std::max<idx_t>(1, total_n - 1));
		global_index->InitParallelConstructCollector(num_tasks + 1);

		// Leader: serially insert the first non-NULL row as the entry point.
		// This sets entry_internal_ + size_=1 before any task spawns, so every
		// parallel BeamSearch has a valid entry point.
		const idx_t entry_row = global_index->LeaderInsertEntry(context, *collection);

		if (entry_row == DConstants::INVALID_INDEX) {
			// All rows NULL — nothing to construct.
			return;
		}

		// Spawn num_tasks tasks for rows [1, total_n) \ {entry_row}.
		// entry_row is always 0 in practice (first non-NULL row), but we pass
		// it explicitly so tasks can skip it even if partitioning puts it
		// inside their range.
		vector<shared_ptr<Task>> construct_tasks;
		const idx_t work_n = total_n - 1; // rows [1, total_n)
		for (size_t t = 0; t < num_tasks; t++) {
			const idx_t start = 1 + idx_t(t) * work_n / num_tasks;
			const idx_t end = 1 + idx_t(t + 1) * work_n / num_tasks;
			if (start >= end) {
				continue;
			}
			construct_tasks.push_back(make_shared_ptr<AiSaqIndexConstructTask>(shared_from_this(), context, gstate,
			                                                                   t + 1, start, end, entry_row, op));
		}
		SetTasks(std::move(construct_tasks));
	}

	void FinishEvent() override {
		auto &global_index = gstate.global_index;
		auto &collection = gstate.collection;
		const idx_t total_n = collection->Count();
		const LogLevel ll = global_index->GetBuildLogLevel();
		const bool timing = LogInfo(ll);
		auto phase_ts = std::chrono::steady_clock::now();
		if (timing) {
			const double construct_ms =
			    std::chrono::duration<double, std::milli>(phase_ts - schedule_ts).count();
			fprintf(stderr, "[vindex] pass2_construct (%.0fms)\n", construct_ms);
		}
		auto mark = [&](const char *what) {
			if (!timing) {
				return;
			}
			const auto now = std::chrono::steady_clock::now();
			const double ms = std::chrono::duration<double, std::milli>(now - phase_ts).count();
			fprintf(stderr, "[vindex] %s (%.0fms)\n", what, ms);
			phase_ts = now;
		};

		// (1) Merge per-task maps, set final size. This is the serial tail of
		// parallel Pass 2 — runs under rwlock. (Task 5.5: no reciprocity apply
		// here — ConnectAndPrune applied every back-edge inline during the
		// parallel phase via per-node spinlocks.)
		if (total_n > 0) {
			global_index->FinalizeParallelConstruct(total_n);
		}
		mark("merge_parallel_construct");

		if (timing) {
			fprintf(stderr, "[vindex] finalizing inline codes...\n");
		}
		global_index->FinalizeInlineCodes();
		mark("finalize_inline_codes");
		global_index->ComputeEntryPoints();
		mark("compute_entry_points");
		global_index->ComputeLabelMedoids();
		mark("compute_label_medoids");
		global_index->FlushBuildNodes();
		mark("flush_graph_nodes");
		global_index->ClearBuildBuffers();
		global_index->SetDirty();
		global_index->SyncSize();

		auto &storage = table.GetStorage();
		if (!storage.db.GetStorageManager().InMemory()) {
			// Persistence is experimental; AiSAQ SerializeToDisk throws until
			// implemented. Skip for in-memory databases (the default test path).
		}

		if (!storage.IsRoot()) {
			throw TransactionException("Cannot create index on non-root transaction");
		}

		auto &schema = table.schema;
		info.column_ids = storage_ids;

		if (schema.GetEntry(schema.GetCatalogTransaction(*gstate.context), CatalogType::INDEX_ENTRY, info.index_name)) {
			if (info.on_conflict != OnCreateConflict::IGNORE_ON_CONFLICT) {
				throw CatalogException("Index with name \"%s\" already exists", info.index_name);
			}
		}

		const auto index_entry = schema.CreateIndex(schema.GetCatalogTransaction(*gstate.context), info, table).get();
		D_ASSERT(index_entry);
		auto &duck_index = index_entry->Cast<DuckIndexEntry>();
		duck_index.initial_index_size = global_index->Cast<BoundIndex>().GetInMemorySize();

		storage.AddIndex(std::move(gstate.global_index));
	}
};

SinkFinalizeType PhysicalCreateAiSaqIndex::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                                    OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<CreateAiSaqIndexGlobalState>();
	auto &collection = gstate.collection;

	gstate.is_building = true;

	// Resolve log level (env var overrides session option).
	auto log_level = GetLogLevel(context);
	gstate.global_index->SetBuildLogLevel(log_level);

	// Resolve build strategy (Tier 1/2/3) before encoding.
	gstate.global_index->ResolveBuildStrategy(context, collection->Count());

	// Pass 1 prerequisite: train the quantizer before encoding any codes.
	if (LogInfo(log_level)) {
		fprintf(stderr, "[vindex] training quantizer on %llu vectors...\n", (unsigned long long)collection->Count());
	}
	auto t_train_start = std::chrono::steady_clock::now();
	gstate.global_index->TrainQuantizer(*collection);
	auto t_train_end = std::chrono::steady_clock::now();
	if (LogInfo(log_level)) {
		fprintf(stderr, "[vindex] train_quantizer (%lldms)\n",
		        (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t_train_end - t_train_start).count());
	}

	// Pass 1: PQ encoding — write all codes to block-store pages. Also
	// populates flat build buffers if Tier 2/3 is active.
	gstate.global_index->SetPqEncodeProgress(&gstate.pq_encoded_count);
	auto t_pq_start = std::chrono::steady_clock::now();
	gstate.global_index->EncodePqCodes(*collection);
	auto t_pq_end = std::chrono::steady_clock::now();
	gstate.global_index->SetPqEncodeProgress(nullptr);
	if (LogInfo(log_level)) {
		fprintf(stderr, "[vindex] PQ encoding complete: %llu vectors in %lldms\n",
		        (unsigned long long)gstate.pq_encoded_count.load(),
		        (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t_pq_end - t_pq_start).count());
	}

	// Pre-size the graph block vectors once, single-threaded, so the
	// per-node EnsureGraphCapacity inside AllocGraphNode is a no-op during
	// Pass 2 (and any future parallel construct). This removes the
	// lazy-block-allocation race in paged mode (Phase 9 Task 4).
	gstate.global_index->PreAllocateGraphCapacity(collection->Count());

	// Activate flat buffers on the core (zero-op for Tier 1).
	gstate.global_index->ActivateBuildBuffers();

	// Pass 2: graph construction — Vamana online inserts. Re-scan the same
	// collection in the same order so internal_ids match the PQ code layout.
	collection->InitializeScan(gstate.scan_state, ColumnDataScanProperties::ALLOW_ZERO_COPY);

	auto new_event = make_shared_ptr<AiSaqIndexConstructionEvent>(*this, gstate, pipeline, *info, storage_ids, table);
	event.InsertEvent(std::move(new_event));
	return SinkFinalizeType::READY;
}

ProgressData PhysicalCreateAiSaqIndex::GetSinkProgress(ClientContext &context, GlobalSinkState &gstate,
                                                       ProgressData source_progress) const {
	ProgressData res;
	const auto &state = gstate.Cast<CreateAiSaqIndexGlobalState>();
	// 3-phase tracking: load + PQ encode + graph construction.
	const idx_t total = state.loaded_count.load() + state.loaded_count.load() + state.loaded_count.load();
	if (!state.is_building) {
		// Phase 1: loading vectors into the collection.
		res.done = state.loaded_count.load() + 0.0;
		res.total = double(total);
	} else {
		// Phase 2+3: PQ encode + construction (both contribute to the back 2/3).
		res.done = double(state.loaded_count.load() + state.pq_encoded_count.load() + state.built_count.load());
		res.total = double(total);
	}
	return res;
}

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
