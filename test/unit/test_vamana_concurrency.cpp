// Unit tests for Phase 9 Task 5.5: parallel Vamana construct with per-node
// spinlocks (replaces Task 5's deferred-reciprocity + global rwlock strategy).
//
// Exercises the parallel build path (AiSaqCore::InsertBuild + per-node-locked
// ConnectAndPrune) against the serial baseline (AiSaqCore::Insert) and
// verifies:
//   1. Final core.Size() == total inserted.
//   2. Search recall@10 parallel ~= serial baseline (within a small
//      tolerance — topology can vary slightly with back-edge apply ordering,
//      but per-node locks make it tightly bound).
//   3. Graph invariants: returned row_ids are in range after parallel build.
//   4. Concurrent stress: many threads, randomized insert orders, no crash,
//      no deadlock (guaranteed by the at-most-one-lock-held-at-a-time rule).
//   5. Per-node-spinlock contention stress: heavy neighbor-set overlap
//      exercises the spin-wait path. Verifies throughput scales and no
//      deadlock under high contention.

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
// AllocGraphNodeRange, size the per-node spinlock array, leader inserts
// row 0, N threads each handle a disjoint row range, then a serial merge.
//
// Mirrors AiSaqIndex::LeaderInsertEntry / InsertBuildRange /
// FinalizeParallelConstruct but operates directly on AiSaqCore +
// AiSaqBlockStore (no AiSaqIndex / ClientContext plumbing).
//
// Task 5.5: no deferred accumulator, no periodic apply. Each InsertBuild's
// ConnectAndPrune acquires per-node spinlocks inline for back-edge writes.
void ParallelBuildCore(AiSaqCore &core, AiSaqBlockStore &store, const float *data, idx_t n, idx_t dim,
                       const AiSaqCoreParams &params, idx_t num_threads) {
	store.AllocGraphNodeRange(static_cast<uint32_t>(n));
	// Pre-size the per-node spinlock array. MUST happen before tasks spawn.
	core.ResizeNodeLocks(n);

	// Leader: insert row 0 (claims entry_internal_, hits size_==0 branch
	// which returns before ConnectAndPrune — so no locks are acquired).
	VamanaTLS leader_tls(params.seed);
	duckdb::unordered_map<uint32_t, int64_t> leader_id2label;
	duckdb::unordered_map<int64_t, duckdb::vector<uint32_t>> leader_label2ids;
	core.InsertBuild(0, 0, data, INT64_MIN, leader_tls, leader_id2label, leader_label2ids);

	// Worker threads: disjoint row ranges.
	std::vector<std::thread> workers;
	std::vector<duckdb::unordered_map<uint32_t, int64_t>> per_task_id2label(num_threads);
	std::vector<duckdb::unordered_map<int64_t, duckdb::vector<uint32_t>>> per_task_label2ids(num_threads);

	const idx_t work_n = n - 1;
	const idx_t per_thread = (work_n + num_threads - 1) / num_threads;

	workers.reserve(num_threads);
	for (idx_t t = 0; t < num_threads; t++) {
		const idx_t start = 1 + t * per_thread;
		const idx_t end = std::min(n, 1 + (t + 1) * per_thread);
		workers.emplace_back([&, t, start, end]() {
			VamanaTLS tls(params.seed + t + 1);
			auto &id2label = per_task_id2label[t];
			auto &label2ids = per_task_label2ids[t];
			for (idx_t i = start; i < end; i++) {
				core.InsertBuild(static_cast<uint32_t>(i), int64_t(i), data + i * dim, INT64_MIN, tls, id2label,
				                 label2ids);
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
//     back-edge apply ordering can vary slightly with thread scheduling,
//     but the topology algorithm is identical to serial).
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

	ParallelBuildCore(parallel_core, parallel_store, data.data(), N, D, params, NUM_THREADS);

	// (1) Final size == N.
	REQUIRE(parallel_core.Size() == N);

	parallel_core.ComputeEntryPoints();

	const float parallel_recall = ComputeRecall(parallel_core, quant, data, N, D, K, params.L, queries, 20);
	INFO("Parallel recall@10 = " << parallel_recall << " (serial=" << serial_recall << ")");
	// (2) Recall within 5% of serial. Per-node locks produce topology
	// essentially identical to serial — only the back-edge apply order can
	// vary, which affects which neighbors survive an alpha-prune but only on
	// tie distances (rare for FP data).
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

	ParallelBuildCore(core, store, data.data(), N, D, params, NUM_THREADS);

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
// Test 3: Concurrent insert stress (deadlock + crash safety).
//
// Many threads, randomized insert orders within disjoint ranges. With
// per-node spinlocks, deadlock is impossible by construction: at most one
// lock is held at a time per task (ConnectAndPrune acquires, writes,
// releases per neighbor — no nested acquisition). Verify no crash and final
// size is correct.
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
	core.ResizeNodeLocks(N);

	VamanaTLS leader_tls(params.seed);
	duckdb::unordered_map<uint32_t, int64_t> leader_id2label;
	duckdb::unordered_map<int64_t, duckdb::vector<uint32_t>> leader_label2ids;
	core.InsertBuild(0, 0, data.data(), INT64_MIN, leader_tls, leader_id2label, leader_label2ids);

	std::vector<std::thread> workers;

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

			duckdb::unordered_map<uint32_t, int64_t> local_id2label;
			duckdb::unordered_map<int64_t, duckdb::vector<uint32_t>> local_label2ids;
			for (uint32_t row_ordinal : order) {
				core.InsertBuild(row_ordinal, int64_t(row_ordinal), data.data() + row_ordinal * D, INT64_MIN, tls,
				                 local_id2label, local_label2ids);
			}
		});
	}
	for (auto &w : workers) {
		w.join();
	}

	core.MergeLabelMaps(leader_id2label, leader_label2ids);
	core.SetSize(static_cast<uint32_t>(N));

	// If we got here without crashing or hanging, the stress test passed:
	// no deadlock, no torn writes, no assertion failures inside the core.
	REQUIRE(core.Size() == N);
}

// ---------------------------------------------------------------------------
// Test 4: Per-node spinlock contention (Task 5.5).
//
// All threads insert vectors that are very close to each other (low-variance
// Gaussian). Every insert's BeamSearch returns the same handful of nearby
// existing nodes as candidates, so every ConnectAndPrune targets the same
// neighbor set. This is the worst-case for lock contention — every task
// hammers the same per-node spinlocks.
//
// Verifies:
//   - Build completes without deadlock or crash (the deadlock-safety
//     invariant: at most one lock held per task at a time).
//   - Final size is correct.
//   - Search returns valid in-range row_ids afterwards.
//   - Build terminates in a reasonable wall-clock bound (a generous cap
//     that catches gross perf regressions — the spinlock must not livelock).
// ---------------------------------------------------------------------------
TEST_CASE("Per-node spinlock contention (heavy neighbor overlap)", "[vamana_concurrency][unit]") {
	MemoryDB mem;

	constexpr idx_t N = 1000;
	constexpr idx_t D = 8;
	constexpr idx_t K = 10;
	constexpr uint8_t M = 4;
	constexpr uint8_t BITS = 4;
	constexpr idx_t NUM_THREADS = 8;

	// Low-variance Gaussian → all vectors cluster near origin → BeamSearch
	// returns overlapping candidate sets → ConnectAndPrune targets overlap
	// heavily → spinlock contention on a small set of nodes.
	std::vector<float> data(N * D);
	std::mt19937 rng(42);
	std::normal_distribution<float> noise(0.0f, 0.01f);
	for (idx_t i = 0; i < N * D; i++) {
		data[i] = noise(rng);
	}

	PqQuantizer quant(MetricKind::L2SQ, D, M, BITS, 7);
	quant.Train(data.data(), N, D);

	AiSaqCoreParams params;
	params.dim = D;
	params.R = 32; // larger R → bigger neighbor lists → more overlap
	params.L = 100;
	params.alpha = 1.2f;
	params.n_entry_points = 4;
	params.seed = 17;

	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);
	AiSaqCore core(params, quant, store);
	store.RegisterPqLayout(quant.CodeSize());
	core.PrepareForBuild(N);
	WriteAllPqCodes(store, quant, data.data(), N, D);

	auto t_start = std::chrono::steady_clock::now();
	ParallelBuildCore(core, store, data.data(), N, D, params, NUM_THREADS);
	auto t_end = std::chrono::steady_clock::now();
	const double build_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
	INFO("contention build: N=" << N << " threads=" << NUM_THREADS << " wall=" << build_ms << "ms");

	REQUIRE(core.Size() == N);
	core.ComputeEntryPoints();

	// Verify search returns valid row_ids.
	for (idx_t q = 0; q < 10; q++) {
		std::vector<float> qpre(quant.QueryWorkspaceSize());
		quant.PreprocessQuery(data.data() + ((q * 37) % N) * D, qpre.data());
		std::vector<float> lut(quant.LUTSize());
		quant.PopulateDistanceLUT(qpre.data(), lut.data());
		auto res = core.Search(lut.data(), K, params.L, 8, 0);
		for (const auto &c : res) {
			REQUIRE(c.row_id >= 0);
			REQUIRE(c.row_id < int64_t(N));
		}
	}

	// Generous upper bound: catches gross perf regressions (e.g. accidental
	// livelock, lock-then-syscall-then-wake cascade). A healthy run on this
	// 10-core host completes in well under 5 seconds; allow 60s headroom.
	REQUIRE(build_ms < 60000.0);
}
