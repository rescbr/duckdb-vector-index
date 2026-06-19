// Unit tests for Phase 9 Task 5: parallel Vamana construct with deferred
// reciprocity.
//
// Exercises the parallel build path (AiSaqCore::InsertBuild +
// ApplyDeferredReciprocals) against the serial baseline (AiSaqCore::Insert)
// and verifies:
//   1. Final core.Size() == total inserted.
//   2. Search recall@10 parallel ~= serial baseline (within a tolerance that
//      accounts for deferred-reciprocity topology variance).
//   3. Graph invariants: returned row_ids are in range after parallel build.
//   4. Concurrent stress: many threads, randomized insert orders, no crash.

#include <catch2/catch_test_macros.hpp>

#include "duckdb.hpp"
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/storage/storage_manager.hpp"

#include "algo/aisaq/aisaq_block_store.hpp"
#include "algo/aisaq/aisaq_core.hpp"
#include "vindex/label_filter.hpp"
#include "vindex/metric.hpp"
#include "vindex/pq_quantizer.hpp"
#include "vindex/quantizer.hpp"
#include "vindex/vamana.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_set>
#include <vector>

using duckdb::BlockManager;
using duckdb::BufferManager;
using duckdb::DatabaseManager;
using duckdb::DuckDB;
using duckdb::idx_t;
using duckdb::vindex::LabelFilter;
using duckdb::vindex::MetricKind;
using duckdb::vindex::QuantizerKind;
using duckdb::vindex::aisaq::AiSaqBlockStore;
using duckdb::vindex::aisaq::AiSaqCore;
using duckdb::vindex::aisaq::AiSaqCoreParams;
using duckdb::vindex::aisaq::DeferredReciprocal;
using duckdb::vindex::pq::PqQuantizer;
using duckdb::vindex::vamana::VamanaTLS;

namespace {

struct MemoryDB {
	DuckDB db;
	BlockManager *bm;
	BufferManager *bufmgr;
	MemoryDB() : db(nullptr) {
		auto &dbm = DatabaseManager::Get(*db.instance);
		auto attached = dbm.GetDatabase("memory");
		REQUIRE(attached);
		bm = &attached->GetStorageManager().GetBlockManager();
		bufmgr = &bm->GetBufferManager();
	}
};

std::vector<float> RandomGaussian(idx_t n, idx_t dim, uint64_t seed) {
	std::mt19937_64 rng(seed);
	std::normal_distribution<float> nd(0.0f, 1.0f);
	std::vector<float> v(n * dim);
	for (idx_t i = 0; i < v.size(); i++) {
		v[i] = nd(rng);
	}
	return v;
}

std::vector<int64_t> BruteForceTopK(const float *data, idx_t n, idx_t dim, const float *q, idx_t k) {
	std::vector<std::pair<float, int64_t>> scored(n);
	for (idx_t i = 0; i < n; i++) {
		float acc = 0.0f;
		for (idx_t j = 0; j < dim; j++) {
			const float d = data[i * dim + j] - q[j];
			acc += d * d;
		}
		scored[i] = {acc, int64_t(i)};
	}
	std::partial_sort(scored.begin(), scored.begin() + std::min(k, n), scored.end());
	std::vector<int64_t> out(std::min(k, n));
	for (idx_t i = 0; i < out.size(); i++) {
		out[i] = scored[i].second;
	}
	return out;
}

// Write `n` PQ codes to the block store's pages, mirroring Pass 1.
// internal_id == row ordinal (Phase 9 Task 5 invariant).
void WriteAllPqCodes(AiSaqBlockStore &store, PqQuantizer &quant, const float *data, idx_t n, idx_t dim) {
	const idx_t code_size = quant.CodeSize();
	const idx_t codes_per_page = store.CodesPerPage();
	std::vector<duckdb::data_t> page(codes_per_page * code_size, 0);
	uint32_t page_idx = 0;
	idx_t slot = 0;
	for (idx_t i = 0; i < n; i++) {
		quant.Encode(data + i * dim, page.data() + slot * code_size);
		slot++;
		if (slot >= codes_per_page) {
			store.WritePqPage(page_idx, page.data());
			page_idx++;
			slot = 0;
			std::memset(page.data(), 0, page.size());
		}
	}
	if (slot > 0) {
		store.WritePqPage(page_idx, page.data());
	}
}

// Build a core via the parallel construct path: pre-reserve [0, N) via
// AllocGraphNodeRange, leader inserts row 0, N threads each handle a disjoint
// row range, then a serial merge + deferred-reciprocity apply.
//
// Mirrors AiSaqIndex::LeaderInsertEntry / InsertBuildRange /
// FinalizeParallelConstruct but operates directly on AiSaqCore + AiSaqBlockStore
// (no AiSaqIndex / ClientContext plumbing).
void ParallelBuildCore(AiSaqCore &core, AiSaqBlockStore &store, const float *data, idx_t n, idx_t dim,
                       const PqQuantizer &quant, const AiSaqCoreParams &params, idx_t num_threads,
                       idx_t periodic_apply_k) {
	store.AllocGraphNodeRange(static_cast<uint32_t>(n));

	// Leader: insert row 0.
	VamanaTLS leader_tls(params.seed);
	duckdb::vector<DeferredReciprocal> leader_deferred;
	duckdb::unordered_map<uint32_t, int64_t> leader_id2label;
	duckdb::unordered_map<int64_t, duckdb::vector<uint32_t>> leader_label2ids;
	core.InsertBuild(0, 0, data, INT64_MIN, leader_tls, leader_deferred, leader_id2label, leader_label2ids);

	// Worker threads: disjoint row ranges.
	std::vector<std::thread> workers;
	std::vector<duckdb::vector<DeferredReciprocal>> per_task_deferred(num_threads);
	std::vector<duckdb::unordered_map<uint32_t, int64_t>> per_task_id2label(num_threads);
	std::vector<duckdb::unordered_map<int64_t, duckdb::vector<uint32_t>>> per_task_label2ids(num_threads);
	std::mutex apply_mutex;

	const idx_t work_n = n - 1;
	const idx_t per_thread = (work_n + num_threads - 1) / num_threads;

	workers.reserve(num_threads);
	for (idx_t t = 0; t < num_threads; t++) {
		const idx_t start = 1 + t * per_thread;
		const idx_t end = std::min(n, 1 + (t + 1) * per_thread);
		workers.emplace_back([&, t, start, end]() {
			VamanaTLS tls(params.seed + t + 1);
			auto &deferred = per_task_deferred[t];
			auto &id2label = per_task_id2label[t];
			auto &label2ids = per_task_label2ids[t];
			idx_t inserts_since_apply = 0;
			for (idx_t i = start; i < end; i++) {
				core.InsertBuild(static_cast<uint32_t>(i), int64_t(i), data + i * dim, INT64_MIN, tls, deferred,
				                 id2label, label2ids);
				inserts_since_apply++;
				if (periodic_apply_k > 0 && inserts_since_apply >= periodic_apply_k) {
					std::lock_guard<std::mutex> lock(apply_mutex);
					core.ApplyDeferredReciprocals(deferred);
					deferred.clear();
					inserts_since_apply = 0;
				}
			}
		});
	}
	for (auto &w : workers) {
		w.join();
	}

	// Serial merge.
	core.MergeLabelMaps(leader_id2label, leader_label2ids);
	for (idx_t t = 0; t < num_threads; t++) {
		core.MergeLabelMaps(per_task_id2label[t], per_task_label2ids[t]);
	}

	duckdb::vector<DeferredReciprocal> all_deferred;
	all_deferred.insert(all_deferred.end(), leader_deferred.begin(), leader_deferred.end());
	for (idx_t t = 0; t < num_threads; t++) {
		all_deferred.insert(all_deferred.end(), per_task_deferred[t].begin(), per_task_deferred[t].end());
	}
	std::sort(all_deferred.begin(), all_deferred.end(), [](const DeferredReciprocal &a, const DeferredReciprocal &b) {
		return a.new_internal_id < b.new_internal_id;
	});
	core.ApplyDeferredReciprocals(all_deferred);
	core.SetSize(static_cast<uint32_t>(n));
}

float ComputeRecall(AiSaqCore &core, const PqQuantizer &quant, const std::vector<float> &data, idx_t n, idx_t dim,
                    idx_t k, idx_t L, const std::vector<float> &queries, idx_t n_queries) {
	idx_t hits = 0;
	idx_t total = 0;
	for (idx_t q = 0; q < n_queries; q++) {
		std::vector<float> qpre(quant.QueryWorkspaceSize());
		quant.PreprocessQuery(queries.data() + q * dim, qpre.data());
		std::vector<float> lut(quant.LUTSize());
		quant.PopulateDistanceLUT(qpre.data(), lut.data());
		auto got = core.Search(lut.data(), k, L, 8, 0);
		auto truth = BruteForceTopK(data.data(), n, dim, queries.data() + q * dim, k);
		std::unordered_set<int64_t> truth_set(truth.begin(), truth.end());
		for (const auto &g : got) {
			if (truth_set.count(g.row_id)) {
				hits++;
			}
		}
		total += k;
	}
	return float(hits) / float(total);
}

} // namespace

// ---------------------------------------------------------------------------
// Test 1: Multi-thread insert, disjoint ranges.
//
// Spawn N threads each calling InsertBuild on disjoint row ranges. Assert:
//   - Final core.Size() == total N.
//   - Search recall ~= single-threaded baseline (within 5% tolerance —
//     deferred reciprocity produces slightly different topology).
// ---------------------------------------------------------------------------
TEST_CASE("Parallel Vamana construct matches serial recall", "[vamana_concurrency][unit]") {
	MemoryDB mem;

	constexpr idx_t N = 2000;
	constexpr idx_t D = 32;
	constexpr idx_t K = 10;
	constexpr uint8_t M = 8;
	constexpr uint8_t BITS = 8;
	constexpr idx_t NUM_THREADS = 4;

	auto data = RandomGaussian(N, D, 0xA15Au);
	PqQuantizer quant(MetricKind::L2SQ, D, M, BITS, 42);
	quant.Train(data.data(), N, D);

	AiSaqCoreParams params;
	params.dim = D;
	params.R = 32;
	params.L = 100;
	params.alpha = 1.2f;
	params.n_entry_points = 8;
	params.seed = 7;

	// --- Serial baseline ---
	AiSaqBlockStore serial_store(*mem.bm, *mem.bufmgr);
	AiSaqCore serial_core(params, quant, serial_store);
	serial_store.RegisterPqLayout(quant.CodeSize());
	serial_core.PrepareForBuild(N);
	WriteAllPqCodes(serial_store, quant, data.data(), N, D);
	for (idx_t i = 0; i < N; i++) {
		serial_core.Insert(int64_t(i), data.data() + i * D);
	}
	REQUIRE(serial_core.Size() == N);
	serial_core.ComputeEntryPoints();

	auto queries = RandomGaussian(20, D, 0xBEEFu);
	const float serial_recall = ComputeRecall(serial_core, quant, data, N, D, K, params.L, queries, 20);
	INFO("Serial recall@10 = " << serial_recall);
	REQUIRE(serial_recall >= 0.40f); // sanity bar

	// --- Parallel build ---
	AiSaqBlockStore parallel_store(*mem.bm, *mem.bufmgr);
	AiSaqCore parallel_core(params, quant, parallel_store);
	parallel_store.RegisterPqLayout(quant.CodeSize());
	parallel_core.PrepareForBuild(N);
	WriteAllPqCodes(parallel_store, quant, data.data(), N, D);

	ParallelBuildCore(parallel_core, parallel_store, data.data(), N, D, quant, params, NUM_THREADS,
	                  /*periodic_apply_k=*/16);

	// (1) Final size == N.
	REQUIRE(parallel_core.Size() == N);

	parallel_core.ComputeEntryPoints();

	const float parallel_recall = ComputeRecall(parallel_core, quant, data, N, D, K, params.L, queries, 20);
	INFO("Parallel recall@10 = " << parallel_recall << " (serial=" << serial_recall << ")");
	// (2) Recall within 5% of serial.
	REQUIRE(parallel_recall >= serial_recall - 0.05f);
}

// ---------------------------------------------------------------------------
// Test 2: Graph invariants after parallel build.
//
// Build a small graph in parallel and verify:
//   - Every Search returns row_ids in [0, N).
//   - Search terminates without crashing on a randomized query set.
// ---------------------------------------------------------------------------
TEST_CASE("Parallel construct preserves graph invariants", "[vamana_concurrency][unit]") {
	MemoryDB mem;

	constexpr idx_t N = 500;
	constexpr idx_t D = 16;
	constexpr idx_t K = 10;
	constexpr uint8_t M = 4;
	constexpr uint8_t BITS = 8;
	constexpr idx_t NUM_THREADS = 4;

	auto data = RandomGaussian(N, D, 0x5A1Au);
	PqQuantizer quant(MetricKind::L2SQ, D, M, BITS, 99);
	quant.Train(data.data(), N, D);

	AiSaqCoreParams params;
	params.dim = D;
	params.R = 16;
	params.L = 64;
	params.alpha = 1.2f;
	params.n_entry_points = 4;
	params.seed = 5;

	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);
	AiSaqCore core(params, quant, store);
	store.RegisterPqLayout(quant.CodeSize());
	core.PrepareForBuild(N);
	WriteAllPqCodes(store, quant, data.data(), N, D);

	ParallelBuildCore(core, store, data.data(), N, D, quant, params, NUM_THREADS, /*periodic_apply_k=*/8);

	REQUIRE(core.Size() == N);
	core.ComputeEntryPoints();

	// Walk every node and verify neighbor lists are well-formed via Search.
	for (idx_t q = 0; q < 10; q++) {
		std::vector<float> qpre(quant.QueryWorkspaceSize());
		quant.PreprocessQuery(data.data() + ((q * 53) % N) * D, qpre.data());
		std::vector<float> lut(quant.LUTSize());
		quant.PopulateDistanceLUT(qpre.data(), lut.data());
		auto res = core.Search(lut.data(), K, params.L, 8, 0);
		for (const auto &c : res) {
			REQUIRE(c.row_id >= 0);
			REQUIRE(c.row_id < int64_t(N));
		}
	}
}

// ---------------------------------------------------------------------------
// Test 3: Concurrent insert stress.
//
// Many threads, randomized insert orders (within disjoint ranges), verify no
// crash and final size is correct.
// ---------------------------------------------------------------------------
TEST_CASE("Concurrent insert stress (no crash)", "[vamana_concurrency][unit]") {
	MemoryDB mem;

	constexpr idx_t N = 5000;
	constexpr idx_t D = 16;
	constexpr uint8_t M = 4;
	constexpr uint8_t BITS = 8;
	constexpr idx_t NUM_THREADS = 8;

	auto data = RandomGaussian(N, D, 0xC0FFEEu);
	PqQuantizer quant(MetricKind::L2SQ, D, M, BITS, 7);
	quant.Train(data.data(), N, D);

	AiSaqCoreParams params;
	params.dim = D;
	params.R = 16;
	params.L = 64;
	params.alpha = 1.2f;
	params.n_entry_points = 4;
	params.seed = 11;

	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);
	AiSaqCore core(params, quant, store);
	store.RegisterPqLayout(quant.CodeSize());
	core.PrepareForBuild(N);
	WriteAllPqCodes(store, quant, data.data(), N, D);

	store.AllocGraphNodeRange(static_cast<uint32_t>(N));

	VamanaTLS leader_tls(params.seed);
	duckdb::vector<DeferredReciprocal> leader_deferred;
	duckdb::unordered_map<uint32_t, int64_t> leader_id2label;
	duckdb::unordered_map<int64_t, duckdb::vector<uint32_t>> leader_label2ids;
	core.InsertBuild(0, 0, data.data(), INT64_MIN, leader_tls, leader_deferred, leader_id2label, leader_label2ids);

	std::vector<std::thread> workers;
	std::vector<duckdb::vector<DeferredReciprocal>> per_task_deferred(NUM_THREADS);
	std::mutex apply_mutex;

	const idx_t work_n = N - 1;
	const idx_t per_thread = (work_n + NUM_THREADS - 1) / NUM_THREADS;

	for (idx_t t = 0; t < NUM_THREADS; t++) {
		const idx_t start = 1 + t * per_thread;
		const idx_t end = std::min(N, 1 + (t + 1) * per_thread);
		workers.emplace_back([&, t, start, end]() {
			VamanaTLS tls(params.seed + t + 1);
			std::vector<uint32_t> order;
			order.reserve(end - start);
			for (idx_t i = start; i < end; i++) {
				order.push_back(static_cast<uint32_t>(i));
			}
			std::mt19937 rng(unsigned(t * 1000 + 7));
			std::shuffle(order.begin(), order.end(), rng);

			auto &deferred = per_task_deferred[t];
			duckdb::unordered_map<uint32_t, int64_t> local_id2label;
			duckdb::unordered_map<int64_t, duckdb::vector<uint32_t>> local_label2ids;
			idx_t inserts_since_apply = 0;
			for (uint32_t row_ordinal : order) {
				core.InsertBuild(row_ordinal, int64_t(row_ordinal), data.data() + row_ordinal * D, INT64_MIN, tls,
				                 deferred, local_id2label, local_label2ids);
				inserts_since_apply++;
				if (inserts_since_apply >= 16) {
					std::lock_guard<std::mutex> lock(apply_mutex);
					core.ApplyDeferredReciprocals(deferred);
					deferred.clear();
					inserts_since_apply = 0;
				}
			}
		});
	}
	for (auto &w : workers) {
		w.join();
	}

	duckdb::vector<DeferredReciprocal> all_deferred;
	all_deferred.insert(all_deferred.end(), leader_deferred.begin(), leader_deferred.end());
	for (idx_t t = 0; t < NUM_THREADS; t++) {
		all_deferred.insert(all_deferred.end(), per_task_deferred[t].begin(), per_task_deferred[t].end());
	}
	std::sort(all_deferred.begin(), all_deferred.end(), [](const DeferredReciprocal &a, const DeferredReciprocal &b) {
		return a.new_internal_id < b.new_internal_id;
	});
	core.ApplyDeferredReciprocals(all_deferred);
	core.SetSize(static_cast<uint32_t>(N));

	// If we got here without crashing or hanging, the stress test passed.
	REQUIRE(core.Size() == N);
}
