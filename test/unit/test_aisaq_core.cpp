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
#include "vindex/metric.hpp"
#include "vindex/pq_quantizer.hpp"
#include "vindex/quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <unordered_set>

using duckdb::BlockManager;
using duckdb::BufferManager;
using duckdb::DatabaseManager;
using duckdb::DuckDB;
using duckdb::idx_t;
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
