#include "algo/aisaq/aisaq_core.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cstring>
#include <queue>

namespace duckdb {
namespace vindex {
namespace aisaq {

namespace {

constexpr uint64_t kStateMagicV1 = 0x3156534141494756ULL; // "VGIASV1"

// Min-heap frontier (pop closest first).
struct FrontierItem {
	float dist;
	uint32_t internal_id;
};
struct FrontierCmp {
	bool operator()(const FrontierItem &a, const FrontierItem &b) const {
		return a.dist > b.dist;
	}
};

// Max-heap working set W (pop farthest first).
struct WorkingItem {
	float dist;
	uint32_t internal_id;
};
struct WorkingCmp {
	bool operator()(const WorkingItem &a, const WorkingItem &b) const {
		return a.dist < b.dist;
	}
};

} // namespace

AiSaqCore::AiSaqCore(AiSaqCoreParams params, Quantizer &quantizer, AiSaqBlockStore &store)
    : params_(params), quantizer_(quantizer), store_(store), rng_(params.seed) {
	if (params_.R == 0) {
		throw InternalException("AiSaqCore: R must be >= 1");
	}
	if (params_.L < params_.R) {
		params_.L = params_.R;
	}
	if (params_.alpha < 1.0f) {
		params_.alpha = 1.0f;
	}
	code_size_ = quantizer_.CodeSize();
	store_.RegisterGraphLayout(StaticNodeSize());
	store_.RegisterPqLayout(code_size_);
}

idx_t AiSaqCore::StaticNodeSize() const {
	idx_t sz = kNeighborArrayOffset + idx_t(params_.R) * sizeof(uint32_t);
	if (params_.inline_pq_count > 0) {
		sz += idx_t(params_.inline_pq_count) * code_size_;
	}
	// Pad to the 8-byte alignment invariant (see AGENTS.md).
	sz = (sz + idx_t(7)) & ~idx_t(7);
	return sz;
}

AiSaqCore::PqCodeRef AiSaqCore::ReadPqCode(uint32_t internal_id) const {
	const idx_t codes_per_page = store_.CodesPerPage();
	const uint32_t page_idx = internal_id / static_cast<uint32_t>(codes_per_page);
	const idx_t offset = (idx_t(internal_id) % codes_per_page) * code_size_;
	auto handle = store_.PinPqPage(page_idx);
	const_data_ptr_t ptr = handle.Ptr() + offset;
	return {std::move(handle), ptr};
}

uint32_t AiSaqCore::PickEntryPoint(const float *query_lut) const {
	if (entry_points_.empty()) {
		return entry_internal_;
	}
	uint32_t best = entry_points_[0].internal_id;
	float best_d = quantizer_.LUTDistance(entry_points_[0].code.data(), query_lut);
	for (idx_t i = 1; i < entry_points_.size(); i++) {
		const float d = quantizer_.LUTDistance(entry_points_[i].code.data(), query_lut);
		if (d < best_d) {
			best_d = d;
			best = entry_points_[i].internal_id;
		}
	}
	return best;
}

// ---------------------------------------------------------------------------
// BeamSearch
// ---------------------------------------------------------------------------

vector<AiSaqCore::Candidate> AiSaqCore::BeamSearch(const float *query_lut, idx_t L, idx_t io_limit) const {
	vector<Candidate> out;
	if (size_ == 0 || L == 0) {
		return out;
	}
	L = std::max<idx_t>(L, 1);

	visit_counter_++;
	if (visit_counter_ == 0) {
		std::fill(visit_marks_.begin(), visit_marks_.end(), 0);
		visit_counter_ = 1;
	}
	const uint32_t vc = visit_counter_;

	std::priority_queue<FrontierItem, std::vector<FrontierItem>, FrontierCmp> frontier;
	std::priority_queue<WorkingItem, std::vector<WorkingItem>, WorkingCmp> W;

	uint32_t io_count = 0;

	const uint32_t entry_internal = PickEntryPoint(query_lut);
	if (entry_internal >= visit_marks_.size()) {
		visit_marks_.resize(std::max<size_t>(visit_marks_.size() * 2, size_t(entry_internal) + 1), 0);
	}
	visit_marks_[entry_internal] = vc;
	if (io_limit > 0 && io_count >= io_limit) {
		// Even under a 0-budget cap we must seed the search from the entry.
		out.push_back({int64_t(entry_internal), DistanceToCode(query_lut, entry_internal)});
		return out;
	}
	io_count++;
	const float entry_dist = DistanceToCode(query_lut, entry_internal);
	frontier.push({entry_dist, entry_internal});
	W.push({entry_dist, entry_internal});

	while (!frontier.empty()) {
		const auto best = frontier.top();
		frontier.pop();
		if (W.size() >= L && best.dist > W.top().dist) {
			break;
		}
		// Expand neighbors of `best`.
		auto nref = PinNode(best.internal_id);
		const uint16_t n = GetNeighborCount(nref.ptr);
		for (uint16_t i = 0; i < n; i++) {
			const uint32_t nb_internal = GetNeighbor(nref.ptr, i);
			if (nb_internal >= visit_marks_.size()) {
				visit_marks_.resize(std::max<size_t>(visit_marks_.size() * 2, size_t(nb_internal) + 1), 0);
			}
			if (visit_marks_[nb_internal] == vc) {
				continue;
			}
			visit_marks_[nb_internal] = vc;
			if (io_limit > 0 && io_count >= io_limit) {
				continue; // out of I/O budget: mark visited, skip distance.
			}
			io_count++;
			const float d = DistanceToCode(query_lut, nb_internal);
			if (W.size() < L || d < W.top().dist) {
				frontier.push({d, nb_internal});
				W.push({d, nb_internal});
				if (W.size() > L) {
					W.pop();
				}
			}
		}
		// nref handle released here → block eligible for eviction.
	}

	out.reserve(W.size());
	while (!W.empty()) {
		const auto &t = W.top();
		out.push_back({int64_t(t.internal_id), t.dist});
		W.pop();
	}
	std::reverse(out.begin(), out.end()); // ascending
	return out;
}

// ---------------------------------------------------------------------------
// RobustPrune
// ---------------------------------------------------------------------------

vector<AiSaqCore::Candidate> AiSaqCore::RobustPrune(const float * /*query_lut*/, vector<Candidate> candidates, idx_t R,
                                                    float alpha) const {
	std::sort(candidates.begin(), candidates.end(),
	          [](const Candidate &a, const Candidate &b) { return a.distance < b.distance; });
	vector<Candidate> kept;
	kept.reserve(R);
	while (!candidates.empty() && kept.size() < R) {
		Candidate p = candidates.front();
		kept.push_back(p);
		vector<Candidate> remaining;
		remaining.reserve(candidates.size());
		for (idx_t i = 1; i < candidates.size(); i++) {
			const auto &pp = candidates[i];
			const float d_pp = CodeDistance(uint32_t(p.row_id), uint32_t(pp.row_id));
			if (alpha * d_pp > pp.distance) {
				remaining.push_back(pp);
			}
		}
		candidates = std::move(remaining);
	}
	return kept;
}

// ---------------------------------------------------------------------------
// ConnectAndPrune
// ---------------------------------------------------------------------------

void AiSaqCore::ConnectAndPrune(uint32_t new_internal_id, const float *new_lut, const vector<Candidate> &selected) {
	// Forward edges on the new node.
	{
		auto nref = PinNode(new_internal_id);
		const uint16_t cnt = uint16_t(std::min<idx_t>(selected.size(), params_.R));
		SetNeighborCount(nref.ptr, cnt);
		for (idx_t i = 0; i < cnt; i++) {
			SetNeighbor(nref.ptr, i, uint32_t(selected[i].row_id));
		}
	}

	// Reciprocal edges.
	for (const auto &s : selected) {
		const uint32_t s_internal = uint32_t(s.row_id);
		auto sref = PinNode(s_internal);
		uint16_t cnt = GetNeighborCount(sref.ptr);
		if (cnt < params_.R) {
			SetNeighbor(sref.ptr, cnt, new_internal_id);
			SetNeighborCount(sref.ptr, cnt + 1);
			continue;
		}
		// Overflow: re-prune s's neighbor set.
		vector<Candidate> cand;
		cand.reserve(idx_t(cnt) + 1);
		cand.push_back({int64_t(new_internal_id), CodeDistance(s_internal, new_internal_id)});
		for (uint16_t i = 0; i < cnt; i++) {
			const uint32_t nb = GetNeighbor(sref.ptr, i);
			cand.push_back({int64_t(nb), CodeDistance(s_internal, nb)});
		}
		auto kept = RobustPrune(new_lut, std::move(cand), params_.R, params_.alpha);
		SetNeighborCount(sref.ptr, uint16_t(kept.size()));
		for (idx_t i = 0; i < kept.size(); i++) {
			SetNeighbor(sref.ptr, i, uint32_t(kept[i].row_id));
		}
	}
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

uint32_t AiSaqCore::Insert(int64_t row_id, const float *vec) {
	const uint32_t internal_id = store_.AllocGraphNode();

	{
		auto nref = PinNode(internal_id);
		// Zero the fixed header + neighbor array (the inline PQ region, if
		// any, is filled by FinalizeInlineCodes).
		std::memset(nref.ptr, 0, kNeighborArrayOffset + idx_t(params_.R) * sizeof(uint32_t));
		SetRowId(nref.ptr, row_id);
		SetInternalId(nref.ptr, internal_id);
		SetNeighborCount(nref.ptr, 0);
		SetInlinePqCount(nref.ptr, params_.inline_pq_count);
	}

	if (internal_id >= visit_marks_.size()) {
		visit_marks_.resize(std::max<size_t>(visit_marks_.size() * 2, size_t(internal_id) + 1), 0);
	}

	if (size_ == 0) {
		entry_internal_ = internal_id;
		size_ = 1;
		return internal_id;
	}

	// Build a LUT from the vector for distance estimates during the search.
	vector<float> qpre(quantizer_.QueryWorkspaceSize());
	quantizer_.PreprocessQuery(vec, qpre.data());
	vector<float> lut(quantizer_.LUTSize() > 0 ? quantizer_.LUTSize() : quantizer_.QueryWorkspaceSize());
	if (quantizer_.LUTSize() > 0) {
		quantizer_.PopulateDistanceLUT(qpre.data(), lut.data());
	} else {
		// Fallback (shouldn't happen: AiSAQ requires PQ/ScaNN). Use the
		// preprocessed query directly as the "LUT" — EstimateDistance-path
		// quantizers are rejected at the index layer.
		std::memcpy(lut.data(), qpre.data(), qpre.size() * sizeof(float));
	}

	auto cands = BeamSearch(lut.data(), params_.L, 0);
	auto selected = RobustPrune(lut.data(), std::move(cands), params_.R, params_.alpha);
	ConnectAndPrune(internal_id, lut.data(), selected);

	size_++;
	return internal_id;
}

// ---------------------------------------------------------------------------
// FinalizeInlineCodes
// ---------------------------------------------------------------------------

void AiSaqCore::FinalizeInlineCodes() {
	if (params_.inline_pq_count == 0 || size_ == 0) {
		return;
	}
	const idx_t inline_region_off = kNeighborArrayOffset + idx_t(params_.R) * sizeof(uint32_t);
	for (uint32_t id = 0; id < size_; id++) {
		auto nref = PinNode(id);
		const uint16_t n = GetNeighborCount(nref.ptr);
		const uint16_t nin = std::min<uint16_t>(n, params_.inline_pq_count);
		for (uint16_t i = 0; i < nin; i++) {
			const uint32_t nb = GetNeighbor(nref.ptr, i);
			auto ref = ReadPqCode(nb);
			std::memcpy(nref.ptr + inline_region_off + idx_t(i) * code_size_, ref.ptr, code_size_);
		}
	}
}

// ---------------------------------------------------------------------------
// ComputeEntryPoints
// ---------------------------------------------------------------------------

void AiSaqCore::ComputeEntryPoints() {
	entry_points_.clear();
	if (size_ == 0) {
		return;
	}
	const uint16_t want = std::min<uint32_t>(params_.n_entry_points, size_);
	for (uint16_t i = 0; i < want; i++) {
		// Evenly-spaced internal_ids across the dataset.
		const uint32_t id = (uint32_t(i) * uint32_t(size_)) / uint32_t(want);
		auto ref = ReadPqCode(id);
		EntryPoint ep;
		ep.internal_id = id;
		ep.code.assign(ref.ptr, ref.ptr + code_size_);
		entry_points_.push_back(std::move(ep));
	}
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

vector<AiSaqCore::Candidate> AiSaqCore::Search(const float *query_lut, idx_t k, idx_t L_search, idx_t /*beam_width*/,
                                               idx_t io_limit) const {
	if (size_ == 0 || k == 0) {
		return {};
	}
	if (L_search == 0) {
		L_search = params_.L;
	}
	if (L_search < k) {
		L_search = k;
	}
	auto cands = BeamSearch(query_lut, L_search, io_limit);
	if (cands.size() > k) {
		cands.resize(k);
	}
	// Resolve internal_id → row_id.
	for (auto &c : cands) {
		auto nref = PinNode(uint32_t(c.row_id));
		c.row_id = GetRowId(nref.ptr);
	}
	return cands;
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

namespace {

template <typename T> void Append(vector<data_t> &out, const T &v) {
	const auto *p = reinterpret_cast<const data_t *>(&v);
	out.insert(out.end(), p, p + sizeof(T));
}

template <typename T> T Consume(const_data_ptr_t &cur, const_data_ptr_t end) {
	if (cur + sizeof(T) > end) {
		throw InternalException("AiSaqCore: state stream truncated");
	}
	T v;
	std::memcpy(&v, cur, sizeof(T));
	cur += sizeof(T);
	return v;
}

} // namespace

void AiSaqCore::SerializeState(vector<data_t> &out) const {
	Append<uint64_t>(out, kStateMagicV1);
	Append<uint64_t>(out, uint64_t(params_.dim));
	Append<uint16_t>(out, params_.R);
	Append<uint16_t>(out, params_.L);
	Append<float>(out, params_.alpha);
	Append<uint16_t>(out, params_.inline_pq_count);
	Append<uint16_t>(out, params_.beam_width);
	Append<uint16_t>(out, params_.n_entry_points);
	Append<uint64_t>(out, params_.seed);
	Append<uint64_t>(out, uint64_t(size_));
	Append<uint64_t>(out, uint64_t(entry_internal_));
	Append<uint64_t>(out, uint64_t(code_size_));

	Append<uint64_t>(out, uint64_t(entry_points_.size()));
	for (const auto &ep : entry_points_) {
		Append<uint32_t>(out, ep.internal_id);
		Append<uint64_t>(out, uint64_t(ep.code.size()));
		if (!ep.code.empty()) {
			out.insert(out.end(), ep.code.begin(), ep.code.end());
		}
	}
}

void AiSaqCore::DeserializeState(const_data_ptr_t in, idx_t size) {
	const_data_ptr_t cur = in;
	const_data_ptr_t end = in + size;

	const auto magic = Consume<uint64_t>(cur, end);
	if (magic != kStateMagicV1) {
		throw InternalException("AiSaqCore: unrecognized state stream (magic mismatch)");
	}
	const auto dim = Consume<uint64_t>(cur, end);
	const auto R = Consume<uint16_t>(cur, end);
	const auto L = Consume<uint16_t>(cur, end);
	const auto alpha = Consume<float>(cur, end);
	const auto inline_pq_count = Consume<uint16_t>(cur, end);
	const auto beam_width = Consume<uint16_t>(cur, end);
	const auto n_entry_points = Consume<uint16_t>(cur, end);
	const auto seed = Consume<uint64_t>(cur, end);
	if (dim != params_.dim) {
		throw InternalException("AiSaqCore: dim mismatch (stored=%llu runtime=%llu)", (unsigned long long)dim,
		                        (unsigned long long)params_.dim);
	}
	params_.R = R;
	params_.L = L;
	params_.alpha = alpha;
	params_.inline_pq_count = inline_pq_count;
	params_.beam_width = beam_width;
	params_.n_entry_points = n_entry_points;
	params_.seed = seed;

	size_ = uint32_t(Consume<uint64_t>(cur, end));
	entry_internal_ = uint32_t(Consume<uint64_t>(cur, end));
	code_size_ = Consume<uint64_t>(cur, end);

	const auto n_ep = Consume<uint64_t>(cur, end);
	entry_points_.clear();
	entry_points_.reserve(n_ep);
	for (uint64_t i = 0; i < n_ep; i++) {
		EntryPoint ep;
		ep.internal_id = Consume<uint32_t>(cur, end);
		const auto clen = Consume<uint64_t>(cur, end);
		if (clen > 0) {
			if (cur + clen > end) {
				throw InternalException("AiSaqCore: entry-point code stream truncated");
			}
			ep.code.assign(cur, cur + clen);
			cur += clen;
		}
		entry_points_.push_back(std::move(ep));
	}

	visit_marks_.assign(size_, 0);
	visit_counter_ = 0;
}

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
