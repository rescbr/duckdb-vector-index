#include "algo/aisaq/aisaq_core.hpp"

#include "duckdb/common/exception.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <queue>

namespace duckdb {
namespace vindex {
namespace aisaq {

namespace {

constexpr uint64_t kStateMagicV1 = 0x3156534141494756ULL; // "VGIASV1"
constexpr uint64_t kStateMagicV2 = 0x3256534141494756ULL; // "VGIASV2" — adds label maps

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
    : params_(params), quantizer_(quantizer), store_(store), tls_(params.seed) {
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

vector<AiSaqCore::Candidate> AiSaqCore::BeamSearch(const float *query_lut, idx_t L, idx_t io_limit,
                                                    const vector<uint32_t> *forced_entry_points,
                                                    const LabelFilter *label_filter) const {
	vector<Candidate> out;
	if (size_ == 0 || L == 0) {
		return out;
	}
	L = std::max<idx_t>(L, 1);

	const uint32_t vc = tls_.NextVisitEpoch();

	std::priority_queue<FrontierItem, std::vector<FrontierItem>, FrontierCmp> frontier;
	std::priority_queue<WorkingItem, std::vector<WorkingItem>, WorkingCmp> W;

	uint32_t io_count = 0;

	// Seed entry points.
	if (forced_entry_points && !forced_entry_points->empty()) {
		for (uint32_t ep_id : *forced_entry_points) {
			if (ep_id >= tls_.visit_marks.size()) {
				tls_.visit_marks.resize(std::max<size_t>(tls_.visit_marks.size() * 2, size_t(ep_id) + 1), 0);
			}
			if (tls_.visit_marks[ep_id] == vc) {
				continue;
			}
			tls_.visit_marks[ep_id] = vc;
			io_count++;
			const float d = DistanceToCode(query_lut, ep_id);
			frontier.push({d, ep_id});
			W.push({d, ep_id});
			if (W.size() > L) {
				W.pop();
			}
		}
	} else {
		const uint32_t entry_internal = PickEntryPoint(query_lut);
		if (entry_internal >= tls_.visit_marks.size()) {
			tls_.visit_marks.resize(
			    std::max<size_t>(tls_.visit_marks.size() * 2, size_t(entry_internal) + 1), 0);
		}
		tls_.visit_marks[entry_internal] = vc;
		io_count++;
		const float entry_dist = DistanceToCode(query_lut, entry_internal);
		frontier.push({entry_dist, entry_internal});
		W.push({entry_dist, entry_internal});
	}

	if (frontier.empty()) {
		return out;
	}

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
			if (nb_internal >= tls_.visit_marks.size()) {
				tls_.visit_marks.resize(std::max<size_t>(tls_.visit_marks.size() * 2, size_t(nb_internal) + 1), 0);
			}
			if (tls_.visit_marks[nb_internal] == vc) {
				continue;
			}
			tls_.visit_marks[nb_internal] = vc;

			// Label filtering: skip neighbors whose label doesn't match.
			if (label_filter && label_filter->IsActive()) {
				auto it = internal_id_to_label_.find(nb_internal);
				if (it != internal_id_to_label_.end()) {
					if (!label_filter->Matches(it->second)) {
						continue;
					}
				}
			}

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
// RawDistance — full-precision pairwise distance for Tier 3 build
// ---------------------------------------------------------------------------

float AiSaqCore::RawDistance(const float *a, const float *b) const {
	const idx_t dim = params_.dim;
	switch (build_metric_) {
	case MetricKind::L2SQ: {
		float acc = 0.0f;
		for (idx_t i = 0; i < dim; i++) {
			const float d = a[i] - b[i];
			acc += d * d;
		}
		return acc;
	}
	case MetricKind::IP: {
		float acc = 0.0f;
		for (idx_t i = 0; i < dim; i++) {
			acc += a[i] * b[i];
		}
		return -acc; // negate so min-distance = max-similarity
	}
	case MetricKind::COSINE: {
		float dot = 0.0f, na = 0.0f, nb = 0.0f;
		for (idx_t i = 0; i < dim; i++) {
			dot += a[i] * b[i];
			na += a[i] * a[i];
			nb += b[i] * b[i];
		}
		if (na == 0.0f || nb == 0.0f) {
			return std::numeric_limits<float>::max();
		}
		return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
	}
	}
	return std::numeric_limits<float>::max();
}

// ---------------------------------------------------------------------------
// RobustPrune — with per-prune gather (Tier 1) and flat-buffer fast paths
// (Tier 2/3). The bitmap approach avoids vector rebuilding.
// ---------------------------------------------------------------------------

vector<AiSaqCore::Candidate> AiSaqCore::RobustPrune(const float * /*query_lut*/, vector<Candidate> candidates, idx_t R,
                                                    float alpha) const {
	std::sort(candidates.begin(), candidates.end(),
	          [](const Candidate &a, const Candidate &b) { return a.distance < b.distance; });
	const idx_t n = candidates.size();
	if (n == 0) {
		return {};
	}

	// Tier 1: gather all candidate PQ codes into a flat local buffer.
	// O(n) Pin calls instead of O(n²) from per-pair ReadPqCode.
	// Tier 2/3: CodeDistance uses flat buffers directly, no gather needed.
	vector<uint8_t> local_codes;
	const bool need_gather = !build_codes_ && !build_vectors_;
	if (need_gather) {
		tls_.prune_scratch.resize(n * code_size_);
		for (idx_t i = 0; i < n; i++) {
			auto ref = ReadPqCode(uint32_t(candidates[i].row_id));
			std::memcpy(tls_.prune_scratch.data() + i * code_size_, ref.ptr, code_size_);
		}
	}

	vector<bool> removed(n, false);
	vector<Candidate> kept;
	kept.reserve(std::min(n, idx_t(R)));

	for (idx_t p_idx = 0; p_idx < n && kept.size() < R; p_idx++) {
		if (removed[p_idx]) {
			continue;
		}
		kept.push_back(candidates[p_idx]);

		for (idx_t pp_idx = p_idx + 1; pp_idx < n; pp_idx++) {
			if (removed[pp_idx]) {
				continue;
			}
			float d_pp;
			if (need_gather) {
				d_pp = quantizer_.CodeDistance(tls_.prune_scratch.data() + p_idx * code_size_,
				                                tls_.prune_scratch.data() + pp_idx * code_size_);
			} else {
				d_pp = CodeDistance(uint32_t(candidates[p_idx].row_id), uint32_t(candidates[pp_idx].row_id));
			}
			if (alpha * d_pp <= candidates[pp_idx].distance) {
				removed[pp_idx] = true;
			}
		}
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
		// Tier 1 optimization: gather s + new + all neighbor codes into a
		// local buffer to avoid O(R²) per-pair Pin calls.
		vector<uint8_t> local_codes;
		const bool need_gather = !build_codes_ && !build_vectors_;
		if (need_gather) {
			// Layout: [0]=s_internal, [1]=new_internal_id, [2..cnt+1]=neighbors
			local_codes.resize((idx_t(cnt) + 2) * code_size_);
			CopyBuildCode(s_internal, local_codes.data());
			CopyBuildCode(new_internal_id, local_codes.data() + code_size_);
			for (uint16_t i = 0; i < cnt; i++) {
				const uint32_t nb = GetNeighbor(sref.ptr, i);
				CopyBuildCode(nb, local_codes.data() + idx_t(i + 2) * code_size_);
			}
		}

		vector<Candidate> cand;
		cand.reserve(idx_t(cnt) + 1);
		if (need_gather) {
			const_data_ptr_t s_ptr = local_codes.data();
			cand.push_back({int64_t(new_internal_id),
			                quantizer_.CodeDistance(s_ptr, local_codes.data() + code_size_)});
			for (uint16_t i = 0; i < cnt; i++) {
				cand.push_back({int64_t(GetNeighbor(sref.ptr, i)),
				                quantizer_.CodeDistance(s_ptr, local_codes.data() + idx_t(i + 2) * code_size_)});
			}
		} else {
			cand.push_back({int64_t(new_internal_id), CodeDistance(s_internal, new_internal_id)});
			for (uint16_t i = 0; i < cnt; i++) {
				const uint32_t nb = GetNeighbor(sref.ptr, i);
				cand.push_back({int64_t(nb), CodeDistance(s_internal, nb)});
			}
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

uint32_t AiSaqCore::Insert(int64_t row_id, const float *vec, int64_t label) {
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

	if (internal_id >= tls_.visit_marks.size()) {
		tls_.visit_marks.resize(std::max<size_t>(tls_.visit_marks.size() * 2, size_t(internal_id) + 1), 0);
	}

	// Label bookkeeping.
	if (label != INT64_MIN) {
		internal_id_to_label_[internal_id] = label;
		label_to_internal_ids_[label].push_back(internal_id);
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

	const idx_t L_build = params_.L_build > 0 ? params_.L_build : params_.R;
	auto cands = BeamSearch(lut.data(), L_build, 0);
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
// ComputeLabelMedoids
// ---------------------------------------------------------------------------

void AiSaqCore::ComputeLabelMedoids() {
	if (label_to_internal_ids_.empty()) {
		return;
	}

	// Build sorted_labels_ for range queries.
	sorted_labels_.clear();
	sorted_labels_.reserve(label_to_internal_ids_.size());
	for (const auto &kv : label_to_internal_ids_) {
		sorted_labels_.push_back(kv.first);
	}
	std::sort(sorted_labels_.begin(), sorted_labels_.end());

	// For each label, find the medoid: the member whose PQ code is closest
	// to the label's centroid (approximated by averaging member codes).
	label_medoids_.clear();
	for (const auto &kv : label_to_internal_ids_) {
		const auto &members = kv.second;
		if (members.empty()) {
			continue;
		}
		if (members.size() == 1) {
			label_medoids_[kv.first] = members[0];
			continue;
		}

		// Compute centroid by averaging PQ codes.
		vector<float> centroid(code_size_, 0.0f);
		for (uint32_t mid : members) {
			auto ref = ReadPqCode(mid);
			for (idx_t j = 0; j < code_size_; j++) {
				centroid[j] += float(ref.ptr[j]);
			}
		}
		const float inv = 1.0f / float(members.size());
		for (idx_t j = 0; j < code_size_; j++) {
			centroid[j] *= inv;
		}

		// Find the member closest to the centroid (by L2 on PQ codes).
		uint32_t best_id = members[0];
		float best_dist = std::numeric_limits<float>::max();
		for (uint32_t mid : members) {
			auto ref = ReadPqCode(mid);
			float d = 0.0f;
			for (idx_t j = 0; j < code_size_; j++) {
				const float diff = float(ref.ptr[j]) - centroid[j];
				d += diff * diff;
			}
			if (d < best_dist) {
				best_dist = d;
				best_id = mid;
			}
		}
		label_medoids_[kv.first] = best_id;
	}
}

// ---------------------------------------------------------------------------
// Range-query helpers
// ---------------------------------------------------------------------------

idx_t AiSaqCore::CountLabelsInRange(int64_t lo, int64_t hi) const {
	if (sorted_labels_.empty()) {
		return 0;
	}
	auto lower = std::lower_bound(sorted_labels_.begin(), sorted_labels_.end(), lo);
	auto upper = std::upper_bound(sorted_labels_.begin(), sorted_labels_.end(), hi);
	return idx_t(upper - lower);
}

vector<uint32_t> AiSaqCore::GetMedoidsInRange(int64_t lo, int64_t hi, idx_t max_count) const {
	vector<uint32_t> result;
	auto lower = std::lower_bound(sorted_labels_.begin(), sorted_labels_.end(), lo);
	auto upper = std::upper_bound(sorted_labels_.begin(), sorted_labels_.end(), hi);
	for (auto it = lower; it != upper; ++it) {
		auto mit = label_medoids_.find(*it);
		if (mit != label_medoids_.end()) {
			result.push_back(mit->second);
		}
		if (result.size() >= max_count) {
			break;
		}
	}
	return result;
}
// ---------------------------------------------------------------------------

vector<AiSaqCore::Candidate> AiSaqCore::Search(const float *query_lut, idx_t k, idx_t L_search, idx_t /*beam_width*/,
                                               idx_t io_limit, const LabelFilter &label_filter) const {
	if (size_ == 0 || k == 0) {
		return {};
	}
	if (L_search == 0) {
		L_search = params_.L;
	}
	if (L_search < k) {
		L_search = k;
	}

	vector<Candidate> cands;
	const LabelFilter *lf = label_filter.IsActive() ? &label_filter : nullptr;

	if (lf && lf->kind == LabelFilter::Kind::EQUALS) {
		// EQUALS: use the label's medoid as the single entry point.
		auto it = label_medoids_.find(lf->value);
		if (it == label_medoids_.end()) {
			return {}; // no vectors with this label
		}
		vector<uint32_t> ep = {it->second};
		cands = BeamSearch(query_lut, L_search, io_limit, &ep, lf);
	} else if (lf && lf->kind == LabelFilter::Kind::RANGE) {
		// RANGE: adaptive strategy.
		const idx_t match_count = CountLabelsInRange(lf->lo, lf->hi);
		if (match_count > 0 && match_count <= params_.n_entry_points) {
			// Multi-medoid: use all matching medoids as entry points.
			auto eps = GetMedoidsInRange(lf->lo, lf->hi, params_.n_entry_points);
			cands = BeamSearch(query_lut, L_search, io_limit, &eps, lf);
		} else {
			// Too many labels or none: fall back to global entry points.
			cands = BeamSearch(query_lut, L_search, io_limit, nullptr, lf);
		}
	} else {
		// No label filter or unsupported kind: normal search.
		cands = BeamSearch(query_lut, L_search, io_limit, nullptr, lf);
	}

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
	Append<uint64_t>(out, kStateMagicV2);
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

	// Label maps (V2).
	Append<uint64_t>(out, uint64_t(label_to_internal_ids_.size()));
	for (const auto &kv : label_to_internal_ids_) {
		Append<int64_t>(out, kv.first);
		Append<uint32_t>(out, label_medoids_.count(kv.first) ? label_medoids_.at(kv.first) : 0);
		Append<uint32_t>(out, uint32_t(kv.second.size()));
		for (uint32_t mid : kv.second) {
			Append<uint32_t>(out, mid);
		}
	}
}

void AiSaqCore::DeserializeState(const_data_ptr_t in, idx_t size) {
	const_data_ptr_t cur = in;
	const_data_ptr_t end = in + size;

	const auto magic = Consume<uint64_t>(cur, end);
	if (magic != kStateMagicV1 && magic != kStateMagicV2) {
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

	// Label maps (V2 only).
	label_to_internal_ids_.clear();
	internal_id_to_label_.clear();
	label_medoids_.clear();
	sorted_labels_.clear();
	if (magic == kStateMagicV2) {
		const auto n_labels = Consume<uint64_t>(cur, end);
		for (uint64_t i = 0; i < n_labels; i++) {
			const auto label = Consume<int64_t>(cur, end);
			const auto medoid = Consume<uint32_t>(cur, end);
			const auto n_members = Consume<uint32_t>(cur, end);
			vector<uint32_t> members(n_members);
			for (uint32_t j = 0; j < n_members; j++) {
				members[j] = Consume<uint32_t>(cur, end);
			}
			for (uint32_t mid : members) {
				internal_id_to_label_[mid] = label;
			}
			label_to_internal_ids_[label] = std::move(members);
			if (medoid != 0 || n_members > 0) {
				label_medoids_[label] = medoid;
			}
		}
		// Rebuild sorted_labels_.
		sorted_labels_.reserve(label_to_internal_ids_.size());
		for (const auto &kv : label_to_internal_ids_) {
			sorted_labels_.push_back(kv.first);
		}
		std::sort(sorted_labels_.begin(), sorted_labels_.end());
	}

	tls_.visit_marks.assign(size_, 0);
	tls_.visit_counter = 0;
}

} // namespace aisaq
} // namespace vindex
} // namespace duckdb
