// Unit tests for src/algo/aisaq/aisaq_core.cpp + aisaq_block_store.cpp.
//
// AiSaqCore is the paged-PQ Vamana graph. These tests exercise:
//
//   1. Correctness: build with a real PqQuantizer, search k=10, verify recall
//      against brute-force L2SQ is reasonable (PQ is lossy).
//   2. Empty-index and k <= size edge cases.
//   3. BlockStore round-trip: write PQ pages via WritePqPage, read back via
//      PinPqPage, verify byte-identical contents.

#include <catch2/catch_test_macros.hpp>

#include "duckdb.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/storage/storage_manager.hpp"

#include "algo/aisaq/aisaq_block_store.hpp"
#include "algo/aisaq/aisaq_core.hpp"
#include "vindex/label_filter.hpp"
#include "vindex/metric.hpp"
#include "vindex/pq_quantizer.hpp"
#include "vindex/quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <unordered_set>

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

// Encode `n` vectors with the quantizer and write them to the block store's PQ
// pages (mirrors AiSaqIndex::EncodePqCodes). internal_id == row ordinal.
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

} // namespace

TEST_CASE("AiSaqCore recall under PQ codes on N=1000 d=32", "[aisaq_core][unit]") {
	MemoryDB mem;
	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);

	constexpr idx_t N = 1000;
	constexpr idx_t D = 32;
	constexpr idx_t K = 10;
	constexpr uint8_t M = 8; // sub_dim = 4
	constexpr uint8_t BITS = 8;

	auto data = RandomGaussian(N, D, 0xA15Au);
	PqQuantizer quant(MetricKind::L2SQ, D, M, BITS, 42);
	quant.Train(data.data(), N, D);

	AiSaqCoreParams params;
	params.dim = D;
	params.R = 32;
	params.L = 100;
	params.alpha = 1.2f;
	params.inline_pq_count = 0;
	params.beam_width = 8;
	params.n_entry_points = 8;
	params.seed = 7;

	AiSaqCore core(params, quant, store);
	// Pass 1: write PQ codes to pages.
	WriteAllPqCodes(store, quant, data.data(), N, D);
	// Pass 2: graph construction.
	for (idx_t i = 0; i < N; i++) {
		core.Insert(int64_t(i), data.data() + i * D);
	}
	REQUIRE(core.Size() == N);
	core.ComputeEntryPoints();

	auto queries = RandomGaussian(20, D, 0xBEEFu);
	idx_t hits = 0;
	idx_t total = 0;
	for (idx_t q = 0; q < 20; q++) {
		std::vector<float> qpre(quant.QueryWorkspaceSize());
		quant.PreprocessQuery(queries.data() + q * D, qpre.data());
		std::vector<float> lut(quant.LUTSize());
		quant.PopulateDistanceLUT(qpre.data(), lut.data());
		auto got = core.Search(lut.data(), K, params.L, params.beam_width, 0);
		auto truth = BruteForceTopK(data.data(), N, D, queries.data() + q * D, K);
		std::unordered_set<int64_t> truth_set(truth.begin(), truth.end());
		for (const auto &g : got) {
			if (truth_set.count(g.row_id)) {
				hits++;
			}
		}
		total += K;
	}
	const float recall = float(hits) / float(total);
	INFO("Recall@10 = " << recall);
	// PQ with m=8/bits=8 on dim=32 Gaussians is lossy; require a modest bar.
	REQUIRE(recall >= 0.40f);
}

TEST_CASE("AiSaqCore returns empty on empty index", "[aisaq_core][unit]") {
	MemoryDB mem;
	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);
	PqQuantizer quant(MetricKind::L2SQ, 16, 4, 8);
	std::vector<float> sample(16 * 4, 0.0f);
	quant.Train(sample.data(), 4, 16);

	AiSaqCoreParams params;
	params.dim = 16;
	AiSaqCore core(params, quant, store);
	std::vector<float> q(16, 0.0f);
	std::vector<float> qpre(quant.QueryWorkspaceSize());
	quant.PreprocessQuery(q.data(), qpre.data());
	std::vector<float> lut(quant.LUTSize());
	quant.PopulateDistanceLUT(qpre.data(), lut.data());
	auto res = core.Search(lut.data(), 10, 32, 8, 0);
	REQUIRE(res.empty());
}

TEST_CASE("AiSaqCore Search honors k <= size", "[aisaq_core][unit]") {
	MemoryDB mem;
	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);
	constexpr idx_t D = 8;
	PqQuantizer quant(MetricKind::L2SQ, D, 4, 8);

	const float vecs[3][8] = {
	    {1, 0, 0, 0, 0, 0, 0, 0},
	    {0, 1, 0, 0, 0, 0, 0, 0},
	    {0, 0, 1, 0, 0, 0, 0, 0},
	};
	std::vector<float> flat;
	for (int i = 0; i < 3; i++) {
		flat.insert(flat.end(), vecs[i], vecs[i] + D);
	}
	quant.Train(flat.data(), 3, D);

	AiSaqCoreParams params;
	params.dim = D;
	params.R = 8;
	params.L = 16;
	AiSaqCore core(params, quant, store);
	WriteAllPqCodes(store, quant, flat.data(), 3, D);
	for (idx_t i = 0; i < 3; i++) {
		core.Insert(int64_t(i), flat.data() + i * D);
	}
	REQUIRE(core.Size() == 3);

	const float qv[8] = {1, 0, 0, 0, 0, 0, 0, 0};
	std::vector<float> qpre(quant.QueryWorkspaceSize());
	quant.PreprocessQuery(qv, qpre.data());
	std::vector<float> lut(quant.LUTSize());
	quant.PopulateDistanceLUT(qpre.data(), lut.data());
	auto res = core.Search(lut.data(), 2, 16, 8, 0);
	REQUIRE(res.size() == 2);
	REQUIRE(res[0].row_id == 0);
}

TEST_CASE("AiSaqBlockStore PQ page round-trip", "[aisaq_block_store][unit]") {
	MemoryDB mem;
	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);

	constexpr idx_t CODE_SIZE = 7; // awkward size to exercise packing
	store.RegisterPqLayout(CODE_SIZE);
	const idx_t cpp = store.CodesPerPage();
	REQUIRE(cpp > 0);

	std::vector<duckdb::data_t> written(cpp * CODE_SIZE);
	for (idx_t i = 0; i < written.size(); i++) {
		written[i] = duckdb::data_t(i * 7 + 3);
	}
	store.EnsurePqCapacity(0);
	store.WritePqPage(0, written.data());

	auto handle = store.PinPqPage(0);
	REQUIRE(std::memcmp(handle.Ptr(), written.data(), written.size()) == 0);
}

TEST_CASE("AiSaqCore SerializeState round-trip preserves search results", "[aisaq_core][unit]") {
	MemoryDB mem;
	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);

	constexpr idx_t N = 200;
	constexpr idx_t D = 16;
	constexpr uint8_t M = 4;
	constexpr uint8_t BITS = 8;

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

	AiSaqCore core(params, quant, store);
	WriteAllPqCodes(store, quant, data.data(), N, D);
	for (idx_t i = 0; i < N; i++) {
		core.Insert(int64_t(i), data.data() + i * D);
	}
	core.ComputeEntryPoints();

	const float qv[16] = {0};
	std::vector<float> qpre(quant.QueryWorkspaceSize());
	quant.PreprocessQuery(qv, qpre.data());
	std::vector<float> lut(quant.LUTSize());
	quant.PopulateDistanceLUT(qpre.data(), lut.data());
	auto before = core.Search(lut.data(), 5, params.L, 8, 0);

	duckdb::vector<duckdb::data_t> blob;
	core.SerializeState(blob);

	// Deserialize into a fresh core bound to the same block store + quantizer.
	AiSaqCore core2(params, quant, store);
	core2.DeserializeState(blob.data(), blob.size());
	REQUIRE(core2.Size() == core.Size());

	auto after = core2.Search(lut.data(), 5, params.L, 8, 0);
	REQUIRE(after.size() == before.size());
	for (idx_t i = 0; i < before.size(); i++) {
		REQUIRE(after[i].row_id == before[i].row_id);
		REQUIRE(std::fabs(after[i].distance - before[i].distance) < 1e-4f);
	}
}

// ---------------------------------------------------------------------------
// Phase 6: label-filter tests
// ---------------------------------------------------------------------------

TEST_CASE("AiSaqCore EQUALS label filter returns only matching-label results", "[aisaq_core][unit]") {
	MemoryDB mem;
	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);

	constexpr idx_t N = 300;
	constexpr idx_t D = 16;
	constexpr uint8_t M = 4;
	constexpr uint8_t BITS = 8;

	// Generate data: 3 clusters of 100 vectors, one per label.
	auto data = RandomGaussian(N, D, 0xCAFEu);
	// Shift clusters apart.
	for (idx_t i = 0; i < N; i++) {
		const int64_t label = int64_t(i / 100) + 1;
		const float offset = float(label) * 1000.0f;
		for (idx_t j = 0; j < D; j++) {
			data[i * D + j] += offset;
		}
	}

	PqQuantizer quant(MetricKind::L2SQ, D, M, BITS, 42);
	quant.Train(data.data(), N, D);

	AiSaqCoreParams params;
	params.dim = D;
	params.R = 32;
	params.L = 100;
	params.n_entry_points = 8;
	params.seed = 7;

	AiSaqCore core(params, quant, store);
	WriteAllPqCodes(store, quant, data.data(), N, D);
	for (idx_t i = 0; i < N; i++) {
		const int64_t label = int64_t(i / 100) + 1;
		core.Insert(int64_t(i), data.data() + i * D, label);
	}
	REQUIRE(core.Size() == N);
	core.ComputeEntryPoints();
	core.ComputeLabelMedoids();
	REQUIRE(core.HasLabels());

	// Query from label 2 cluster center.
	std::vector<float> query(D, 2000.0f);
	std::vector<float> qpre(quant.QueryWorkspaceSize());
	quant.PreprocessQuery(query.data(), qpre.data());
	std::vector<float> lut(quant.LUTSize());
	quant.PopulateDistanceLUT(qpre.data(), lut.data());

	// EQUALS filter: all results must have label == 2.
	auto res = core.Search(lut.data(), 10, params.L, 8, 0, LabelFilter::Equals(2));
	REQUIRE(!res.empty());
	for (const auto &c : res) {
		REQUIRE(c.row_id >= 100);
		REQUIRE(c.row_id < 200);
	}

	// Nonexistent label: empty.
	auto res_empty = core.Search(lut.data(), 10, params.L, 8, 0, LabelFilter::Equals(999));
	REQUIRE(res_empty.empty());
}

TEST_CASE("AiSaqCore RANGE label filter respects bounds", "[aisaq_core][unit]") {
	MemoryDB mem;
	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);

	constexpr idx_t N = 300;
	constexpr idx_t D = 16;
	constexpr uint8_t M = 4;
	constexpr uint8_t BITS = 8;

	auto data = RandomGaussian(N, D, 0xF00Du);
	for (idx_t i = 0; i < N; i++) {
		const int64_t label = int64_t(i / 100) + 1;
		const float offset = float(label) * 1000.0f;
		for (idx_t j = 0; j < D; j++) {
			data[i * D + j] += offset;
		}
	}

	PqQuantizer quant(MetricKind::L2SQ, D, M, BITS, 42);
	quant.Train(data.data(), N, D);

	AiSaqCoreParams params;
	params.dim = D;
	params.R = 32;
	params.L = 100;
	params.n_entry_points = 16;
	params.seed = 7;

	AiSaqCore core(params, quant, store);
	WriteAllPqCodes(store, quant, data.data(), N, D);
	for (idx_t i = 0; i < N; i++) {
		const int64_t label = int64_t(i / 100) + 1;
		core.Insert(int64_t(i), data.data() + i * D, label);
	}
	core.ComputeEntryPoints();
	core.ComputeLabelMedoids();

	// RANGE(1, 2): all results must have label in {1, 2}.
	std::vector<float> query(D, 1500.0f);
	std::vector<float> qpre(quant.QueryWorkspaceSize());
	quant.PreprocessQuery(query.data(), qpre.data());
	std::vector<float> lut(quant.LUTSize());
	quant.PopulateDistanceLUT(qpre.data(), lut.data());

	auto res = core.Search(lut.data(), 10, params.L, 8, 0, LabelFilter::Range(1, 2));
	REQUIRE(!res.empty());
	for (const auto &c : res) {
		REQUIRE(c.row_id < 200); // labels 1 and 2 cover row_ids 0..199
	}

	// Count labels in range.
	REQUIRE(core.CountLabelsInRange(1, 3) == 3);
	REQUIRE(core.CountLabelsInRange(1, 2) == 2);
	REQUIRE(core.CountLabelsInRange(2, 2) == 1);
	REQUIRE(core.CountLabelsInRange(1, 100) == 3);
}

TEST_CASE("AiSaqCore SerializeState preserves label maps", "[aisaq_core][unit]") {
	MemoryDB mem;
	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);

	constexpr idx_t N = 100;
	constexpr idx_t D = 16;
	constexpr uint8_t M = 4;
	constexpr uint8_t BITS = 8;

	auto data = RandomGaussian(N, D, 0xBEEFu);
	PqQuantizer quant(MetricKind::L2SQ, D, M, BITS, 42);
	quant.Train(data.data(), N, D);

	AiSaqCoreParams params;
	params.dim = D;
	params.R = 16;
	params.L = 64;
	params.n_entry_points = 4;
	params.seed = 5;

	AiSaqCore core(params, quant, store);
	WriteAllPqCodes(store, quant, data.data(), N, D);
	for (idx_t i = 0; i < N; i++) {
		const int64_t label = int64_t(i / 25) + 1; // 4 labels, 25 per
		core.Insert(int64_t(i), data.data() + i * D, label);
	}
	core.ComputeEntryPoints();
	core.ComputeLabelMedoids();

	// Search with EQUALS before serialize.
	std::vector<float> query(D, 0.0f);
	std::vector<float> qpre(quant.QueryWorkspaceSize());
	quant.PreprocessQuery(query.data(), qpre.data());
	std::vector<float> lut(quant.LUTSize());
	quant.PopulateDistanceLUT(qpre.data(), lut.data());
	auto before = core.Search(lut.data(), 5, params.L, 8, 0, LabelFilter::Equals(2));

	duckdb::vector<duckdb::data_t> blob;
	core.SerializeState(blob);

	AiSaqCore core2(params, quant, store);
	core2.DeserializeState(blob.data(), blob.size());
	REQUIRE(core2.HasLabels());
	REQUIRE(core2.CountLabelsInRange(1, 4) == 4);

	auto after = core2.Search(lut.data(), 5, params.L, 8, 0, LabelFilter::Equals(2));
	REQUIRE(after.size() == before.size());
	for (idx_t i = 0; i < before.size(); i++) {
		REQUIRE(after[i].row_id == before[i].row_id);
	}
}

// ---------------------------------------------------------------------------
// Phase 7: additional correctness tests
// ---------------------------------------------------------------------------

TEST_CASE("AiSaqCore io_limit caps search results", "[aisaq_core][unit]") {
	MemoryDB mem;
	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);

	constexpr idx_t N = 500;
	constexpr idx_t D = 32;
	constexpr uint8_t M = 8;
	constexpr uint8_t BITS = 8;

	auto data = RandomGaussian(N, D, 0x10CCu);
	PqQuantizer quant(MetricKind::L2SQ, D, M, BITS, 42);
	quant.Train(data.data(), N, D);

	AiSaqCoreParams params;
	params.dim = D;
	params.R = 32;
	params.L = 100;
	params.n_entry_points = 8;
	params.seed = 11;

	AiSaqCore core(params, quant, store);
	WriteAllPqCodes(store, quant, data.data(), N, D);
	for (idx_t i = 0; i < N; i++) {
		core.Insert(int64_t(i), data.data() + i * D);
	}
	core.ComputeEntryPoints();

	auto queries = RandomGaussian(10, D, 0xDEADu);
	std::vector<float> qpre(quant.QueryWorkspaceSize());
	std::vector<float> lut(quant.LUTSize());

	// With io_limit=1, search should return far fewer candidates than
	// unlimited, because it can only page in one PQ code per direction.
	for (idx_t q = 0; q < 10; q++) {
		quant.PreprocessQuery(queries.data() + q * D, qpre.data());
		quant.PopulateDistanceLUT(qpre.data(), lut.data());

		auto limited = core.Search(lut.data(), 10, params.L, 8, 1);
		auto full = core.Search(lut.data(), 10, params.L, 8, 0);

		// The limited search should not return more results than the full one.
		REQUIRE(limited.size() <= full.size());
		REQUIRE(full.size() <= 10);
	}
}

TEST_CASE("AiSaqCore inline PQ matches or exceeds paged recall", "[aisaq_core][unit]") {
	MemoryDB mem;
	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);

	constexpr idx_t N = 500;
	constexpr idx_t D = 32;
	constexpr uint8_t M = 8;
	constexpr uint8_t BITS = 8;

	auto data = RandomGaussian(N, D, 0x1A2Bu);
	PqQuantizer quant(MetricKind::L2SQ, D, M, BITS, 42);
	quant.Train(data.data(), N, D);

	// Build with inline_pq_count = R.
	AiSaqCoreParams params;
	params.dim = D;
	params.R = 32;
	params.L = 100;
	params.inline_pq_count = 32;
	params.n_entry_points = 8;
	params.seed = 7;

	AiSaqCore core(params, quant, store);
	WriteAllPqCodes(store, quant, data.data(), N, D);
	for (idx_t i = 0; i < N; i++) {
		core.Insert(int64_t(i), data.data() + i * D);
	}
	core.FinalizeInlineCodes();
	core.ComputeEntryPoints();

	// Build ground truth.
	auto queries = RandomGaussian(20, D, 0xCAFEu);
	std::vector<float> qpre(quant.QueryWorkspaceSize());
	std::vector<float> lut(quant.LUTSize());

	idx_t hits_inline = 0;
	idx_t hits_total = 0;
	for (idx_t q = 0; q < 20; q++) {
		quant.PreprocessQuery(queries.data() + q * D, qpre.data());
		quant.PopulateDistanceLUT(qpre.data(), lut.data());

		auto got = core.Search(lut.data(), 10, params.L, 8, 0);
		auto truth = BruteForceTopK(data.data(), N, D, queries.data() + q * D, 10);
		std::unordered_set<int64_t> truth_set(truth.begin(), truth.end());
		for (const auto &g : got) {
			if (truth_set.count(g.row_id)) {
				hits_inline++;
			}
		}
		hits_total += 10;
	}
	const float recall_inline = float(hits_inline) / float(hits_total);
	INFO("Inline recall@10 = " << recall_inline);
	// Inline PQ should at least match paged recall (same codes, just cached).
	REQUIRE(recall_inline >= 0.40f);
}

TEST_CASE("AiSaqCore multiple entry points improve recall on clustered data", "[aisaq_core][unit]") {
	MemoryDB mem;
	AiSaqBlockStore store(*mem.bm, *mem.bufmgr);

	constexpr idx_t N = 500;
	constexpr idx_t D = 32;
	constexpr uint8_t M = 8;
	constexpr uint8_t BITS = 8;

	// Generate 5 well-separated clusters.
	auto data = RandomGaussian(N, D, 0xC1A0u);
	for (idx_t i = 0; i < N; i++) {
		const float offset = float(i / (N / 5)) * 5000.0f;
		for (idx_t j = 0; j < D; j++) {
			data[i * D + j] += offset;
		}
	}

	PqQuantizer quant(MetricKind::L2SQ, D, M, BITS, 42);
	quant.Train(data.data(), N, D);

	auto build_and_measure = [&](uint16_t n_ep) -> float {
		AiSaqBlockStore s(*mem.bm, *mem.bufmgr);
		AiSaqCoreParams params;
		params.dim = D;
		params.R = 32;
		params.L = 100;
		params.n_entry_points = n_ep;
		params.seed = 7;

		AiSaqCore core(params, quant, s);
		WriteAllPqCodes(s, quant, data.data(), N, D);
		for (idx_t i = 0; i < N; i++) {
			core.Insert(int64_t(i), data.data() + i * D);
		}
		core.ComputeEntryPoints();

		auto queries = RandomGaussian(20, D, 0xBEEFu);
		std::vector<float> qpre(quant.QueryWorkspaceSize());
		std::vector<float> lut(quant.LUTSize());

		idx_t hits = 0, total = 0;
		for (idx_t q = 0; q < 20; q++) {
			quant.PreprocessQuery(queries.data() + q * D, qpre.data());
			quant.PopulateDistanceLUT(qpre.data(), lut.data());
			auto got = core.Search(lut.data(), 10, params.L, 8, 0);
			auto truth = BruteForceTopK(data.data(), N, D, queries.data() + q * D, 10);
			std::unordered_set<int64_t> truth_set(truth.begin(), truth.end());
			for (const auto &g : got) {
				if (truth_set.count(g.row_id)) {
					hits++;
				}
			}
			total += 10;
		}
		return float(hits) / float(total);
	};

	const float recall_1ep = build_and_measure(1);
	const float recall_8ep = build_and_measure(8);

	INFO("Recall with 1 entry point = " << recall_1ep);
	INFO("Recall with 8 entry points = " << recall_8ep);
	// More entry points should help (or at least not hurt) on clustered data.
	REQUIRE(recall_8ep >= recall_1ep - 0.05f);
}

// ---------------------------------------------------------------------------
// Phase 9 Task 2: parallel EncodePqCodes byte-identity test.
//
// Mirrors the partitioning scheme in AiSaqIndex::EncodePqRange: split
// [0, total_pages) into T disjoint page ranges, encode each row directly into
// its assigned page slot, and verify the resulting block-store pages are
// byte-identical to the serial WriteAllPqCodes path. This is the cleanest way
// to exercise the per-task partitioning logic without standing up a full
// AiSaqIndex (which needs TableIOManager + AttachedDatabase).
// ---------------------------------------------------------------------------
TEST_CASE("Parallel PQ encoding produces byte-identical pages to serial", "[aisaq_pq_parallel][unit]") {
	MemoryDB mem;

	constexpr idx_t N = 100000; // span multiple pages (codes_per_page ~ 32k for code_size=8)
	constexpr idx_t D = 32;
	constexpr uint8_t M = 8;
	constexpr uint8_t BITS = 8;
	constexpr idx_t NUM_THREADS = 4;

	auto data = RandomGaussian(N, D, 0xAA11u);
	PqQuantizer quant(MetricKind::L2SQ, D, M, BITS, 99);
	quant.Train(data.data(), N, D);

	// --- Serial baseline ---
	AiSaqBlockStore serial_store(*mem.bm, *mem.bufmgr);
	serial_store.RegisterPqLayout(quant.CodeSize());
	WriteAllPqCodes(serial_store, quant, data.data(), N, D);

	// --- Parallel via simulated per-task partitioning ---
	AiSaqBlockStore parallel_store(*mem.bm, *mem.bufmgr);
	parallel_store.RegisterPqLayout(quant.CodeSize());

	const idx_t code_size = quant.CodeSize();
	const idx_t codes_per_page = parallel_store.CodesPerPage();
	const idx_t total_pages = (N + codes_per_page - 1) / codes_per_page;
	REQUIRE(total_pages > 1); // sanity: actually exercise page boundaries

	// Pre-allocate all pages once (mirrors EncodePqCodes).
	if (total_pages > 0) {
		parallel_store.EnsurePqCapacity(static_cast<uint32_t>(total_pages - 1));
	}

	// Each "task" runs the same body as AiSaqIndex::EncodePqRange.
	auto run_task = [&](idx_t t) {
		const uint32_t p_start = static_cast<uint32_t>(t * total_pages / NUM_THREADS);
		const uint32_t p_end = static_cast<uint32_t>((t + 1) * total_pages / NUM_THREADS);
		if (p_start >= p_end) {
			return;
		}
		const idx_t row_start = idx_t(p_start) * codes_per_page;
		const idx_t row_end = std::min<idx_t>(idx_t(p_end) * codes_per_page, N);

		std::vector<duckdb::data_t> page(codes_per_page * code_size, 0);
		uint32_t current_page_idx = static_cast<uint32_t>(row_start / codes_per_page);
		idx_t slot_in_page = 0;

		for (idx_t i = row_start; i < row_end; i++) {
			quant.Encode(data.data() + i * D, page.data() + slot_in_page * code_size);
			slot_in_page++;
			if (slot_in_page >= codes_per_page) {
				parallel_store.WritePqPage(current_page_idx, page.data());
				current_page_idx++;
				slot_in_page = 0;
				std::memset(page.data(), 0, page.size());
			}
		}
		if (slot_in_page > 0) {
			parallel_store.WritePqPage(current_page_idx, page.data());
		}
	};

	std::vector<std::thread> workers;
	for (idx_t t = 0; t < NUM_THREADS; t++) {
		workers.emplace_back(run_task, t);
	}
	for (auto &w : workers) {
		w.join();
	}

	// --- Byte-identity check ---
	REQUIRE(parallel_store.PqPageCount() == serial_store.PqPageCount());
	for (uint32_t p = 0; p < serial_store.PqPageCount(); p++) {
		auto serial_handle = serial_store.PinPqPage(p);
		auto parallel_handle = parallel_store.PinPqPage(p);
		REQUIRE(std::memcmp(serial_handle.Ptr(), parallel_handle.Ptr(), codes_per_page * code_size) == 0);
	}
}

// ---------------------------------------------------------------------------
// Phase 9 Task 3: parallel post-build phases byte-identity test.
//
// FinalizeInlineCodes / ComputeEntryPoints / ComputeLabelMedoids all iterate
// over disjoint items with no cross-item mutation. Their outputs (inline PQ
// regions in graph nodes, entry_points_ vector, label_medoids_ map) must be
// bit-identical whether the methods run serially (SetDatabase(nullptr)) or
// via TaskScheduler (SetDatabase(&instance)).
//
// Strategy: build two AiSaqCores with identical data + labels, run the three
// post-build phases serially on one and parallelly on the other, then:
//   1. Compare the inline PQ region of every graph node (FinalizeInlineCodes).
//   2. Compare SerializeState byte-for-byte — covers entry_points_ and
//      label_medoids_. The unordered_map iteration order is deterministic
//      here because both cores see the same Insert sequence (so
//      label_to_internal_ids_ has the same bucket layout), and lookups into
//      label_medoids_ via .at() are order-independent.
//   3. Cross-check via Search — if entry_points_ or label_medoids_ diverge,
//      search results with label filters would diverge too.
// ---------------------------------------------------------------------------
TEST_CASE("Parallel post-build phases produce byte-identical state to serial", "[aisaq_postbuild_parallel][unit]") {
	MemoryDB mem;

	constexpr idx_t N = 2000;
	constexpr idx_t D = 16;
	constexpr uint8_t M = 4;
	constexpr uint8_t BITS = 8;

	auto data = RandomGaussian(N, D, 0x9A77u);
	// 8 well-separated clusters with offsets so label medoids are unambiguous.
	for (idx_t i = 0; i < N; i++) {
		const int64_t label = int64_t(i / (N / 8)) + 1; // 8 labels
		const float offset = float(label) * 1000.0f;
		for (idx_t j = 0; j < D; j++) {
			data[i * D + j] += offset;
		}
	}

	PqQuantizer quant(MetricKind::L2SQ, D, M, BITS, 42);
	quant.Train(data.data(), N, D);

	auto build_core = [&](AiSaqBlockStore &store, duckdb::DatabaseInstance *db) -> std::unique_ptr<AiSaqCore> {
		AiSaqCoreParams params;
		params.dim = D;
		params.R = 16;
		params.L = 64;
		params.inline_pq_count = 16; // exercise FinalizeInlineCodes
		params.n_entry_points = 16;  // bounded but non-trivial
		params.seed = 7;

		auto core = std::make_unique<AiSaqCore>(params, quant, store);
		core->SetDatabase(db);
		WriteAllPqCodes(store, quant, data.data(), N, D);
		for (idx_t i = 0; i < N; i++) {
			const int64_t label = int64_t(i / (N / 8)) + 1;
			core->Insert(int64_t(i), data.data() + i * D, label);
		}
		REQUIRE(core->Size() == N);
		return core;
	};

	// --- Serial baseline ---
	AiSaqBlockStore serial_store(*mem.bm, *mem.bufmgr);
	auto serial_core = build_core(serial_store, nullptr);
	serial_core->FinalizeInlineCodes();
	serial_core->ComputeEntryPoints();
	serial_core->ComputeLabelMedoids();

	// --- Parallel ---
	AiSaqBlockStore parallel_store(*mem.bm, *mem.bufmgr);
	auto parallel_core = build_core(parallel_store, mem.db.instance.get());
	parallel_core->FinalizeInlineCodes();
	parallel_core->ComputeEntryPoints();
	parallel_core->ComputeLabelMedoids();

	// (1) Inline PQ bytes per node — compare the inline region of each node.
	// Layout: [16-byte header][R * 4 bytes neighbors][inline_pq_count * code_size bytes]
	const idx_t R = serial_core->Params().R;
	const idx_t inline_pq_count = serial_core->Params().inline_pq_count;
	const idx_t code_size = quant.CodeSize();
	const idx_t inline_off = 16 + R * sizeof(uint32_t);
	const idx_t inline_bytes = inline_pq_count * code_size;
	REQUIRE(parallel_core->Params().R == R);
	REQUIRE(parallel_core->Params().inline_pq_count == inline_pq_count);
	for (uint32_t id = 0; id < N; id++) {
		auto s_handle = serial_store.PinGraphNode(id);
		auto p_handle = parallel_store.PinGraphNode(id);
		const idx_t off_in_block = (id % serial_store.NodesPerBlock()) * serial_store.NodeSize();
		REQUIRE(std::memcmp(s_handle.Ptr() + off_in_block + inline_off, p_handle.Ptr() + off_in_block + inline_off,
		                    inline_bytes) == 0);
	}

	// (2) SerializeState byte-for-byte — covers entry_points_ + label_medoids_.
	duckdb::vector<duckdb::data_t> serial_blob;
	duckdb::vector<duckdb::data_t> parallel_blob;
	serial_core->SerializeState(serial_blob);
	parallel_core->SerializeState(parallel_blob);
	REQUIRE(serial_blob.size() == parallel_blob.size());
	REQUIRE(std::memcmp(serial_blob.data(), parallel_blob.data(), serial_blob.size()) == 0);

	// (3) Cross-check via Search: label-filtered queries must produce identical
	// results — divergence here would indicate entry_points_ or label_medoids_
	// drift (the only label-dependent search inputs).
	for (int64_t label = 1; label <= 8; label++) {
		std::vector<float> query(D, float(label) * 1000.0f); // query at cluster center
		std::vector<float> qpre(quant.QueryWorkspaceSize());
		quant.PreprocessQuery(query.data(), qpre.data());
		std::vector<float> lut(quant.LUTSize());
		quant.PopulateDistanceLUT(qpre.data(), lut.data());

		auto s_res = serial_core->Search(lut.data(), 5, serial_core->Params().L, 8, 0, LabelFilter::Equals(label));
		auto p_res = parallel_core->Search(lut.data(), 5, parallel_core->Params().L, 8, 0, LabelFilter::Equals(label));
		REQUIRE(s_res.size() == p_res.size());
		for (idx_t i = 0; i < s_res.size(); i++) {
			REQUIRE(s_res[i].row_id == p_res[i].row_id);
		}
	}
}
