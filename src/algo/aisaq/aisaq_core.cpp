#include "algo/aisaq/aisaq_core.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/numeric_utils.hpp"
#include "duckdb/parallel/task_executor.hpp"
#include "duckdb/parallel/task_scheduler.hpp"

#include <algorithm>
#include <chrono>
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

// Per-task workers for the parallel post-build phases (Phase 9 Task 3).
// Mirror the AiSaqPqEncodeTask pattern: subclass BaseExecutorTask, hold
// references to the core + the partition range, invoke the *_Range worker.
class AiSaqFinalizeInlineTask final : public BaseExecutorTask {
  public:
	AiSaqFinalizeInlineTask(TaskExecutor &executor, const AiSaqCore &core, idx_t id_start, idx_t id_end)
	    : BaseExecutorTask(executor), core_(core), id_start_(id_start), id_end_(id_end) {
	}

	void ExecuteTask() override {
		core_.FinalizeInlineCodesRange(id_start_, id_end_);
	}

	string TaskType() const override {
		return "AiSaqFinalizeInlineTask";
	}

  private:
	const AiSaqCore &core_;
	idx_t id_start_;
	idx_t id_end_;
};

class AiSaqComputeMedoidsTask final : public BaseExecutorTask {
  public:
	AiSaqComputeMedoidsTask(TaskExecutor &executor, const AiSaqCore &core, const vector<int64_t> &labels, idx_t start,
	                        idx_t end, unordered_map<int64_t, uint32_t> &local_out)
	    : BaseExecutorTask(executor), core_(core), labels_(labels), start_(start), end_(end), local_out_(local_out) {
	}

	void ExecuteTask() override {
		core_.ComputeLabelMedoidsRange(labels_, start_, end_, local_out_);
	}

	string TaskType() const override {
		return "AiSaqComputeMedoidsTask";
	}

  private:
	const AiSaqCore &core_;
	const vector<int64_t> &labels_;
	idx_t start_;
	idx_t end_;
	unordered_map<int64_t, uint32_t> &local_out_;
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
	// Tier 2/3 fast path: when the flat build_codes_ buffer is active, skip
	// BufferManager entirely. The bytes are identical to what PinPqPage would
	// return (Pass 1 wrote both buffers from the same encoder). This is the
	// same shortcut DistanceToCode / CopyBuildCode already take; without it,
	// parallel post-build phases serialise on BufferManager::Pin's pool mutex.
	if (build_codes_) {
		return {BufferHandle(), build_codes_ + idx_t(internal_id) * code_size_};
	}
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
                                                   vamana::VamanaTLS &tls, const vector<uint32_t> *forced_entry_points,
                                                   const LabelFilter *label_filter) const {
	vector<Candidate> out;
	if (size_ == 0 || L == 0) {
		return out;
	}
	L = std::max<idx_t>(L, 1);

	const uint32_t vc = tls.NextVisitEpoch();

	std::priority_queue<FrontierItem, std::vector<FrontierItem>, FrontierCmp> frontier;
	std::priority_queue<WorkingItem, std::vector<WorkingItem>, WorkingCmp> W;

	uint32_t io_count = 0;

	// Seed entry points.
	if (forced_entry_points && !forced_entry_points->empty()) {
		for (uint32_t ep_id : *forced_entry_points) {
			if (ep_id >= tls.visit_marks.size()) {
				tls.visit_marks.resize(std::max<size_t>(tls.visit_marks.size() * 2, size_t(ep_id) + 1), 0);
			}
			if (tls.visit_marks[ep_id] == vc) {
				continue;
			}
			tls.visit_marks[ep_id] = vc;
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
		if (entry_internal >= tls.visit_marks.size()) {
			tls.visit_marks.resize(std::max<size_t>(tls.visit_marks.size() * 2, size_t(entry_internal) + 1), 0);
		}
		tls.visit_marks[entry_internal] = vc;
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
			if (nb_internal >= tls.visit_marks.size()) {
				tls.visit_marks.resize(std::max<size_t>(tls.visit_marks.size() * 2, size_t(nb_internal) + 1), 0);
			}
			if (tls.visit_marks[nb_internal] == vc) {
				continue;
			}
			tls.visit_marks[nb_internal] = vc;

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
		// nref handle released here -> block eligible for eviction.
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
                                                    float alpha, vamana::VamanaTLS &tls) const {
	std::sort(candidates.begin(), candidates.end(),
	          [](const Candidate &a, const Candidate &b) { return a.distance < b.distance; });
	const idx_t n = candidates.size();
	if (n == 0) {
		return {};
	}

	// Tier 1: gather all candidate PQ codes into a flat local buffer.
	// O(n) Pin calls instead of O(n^2) from per-pair ReadPqCode.
	// Tier 2/3: CodeDistance uses flat buffers directly, no gather needed.
	const bool need_gather = !build_codes_ && !build_vectors_;
	if (need_gather) {
		tls.prune_scratch.resize(n * code_size_);
		for (idx_t i = 0; i < n; i++) {
			auto ref = ReadPqCode(uint32_t(candidates[i].row_id));
			std::memcpy(tls.prune_scratch.data() + i * code_size_, ref.ptr, code_size_);
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
				d_pp = quantizer_.CodeDistance(tls.prune_scratch.data() + p_idx * code_size_,
				                               tls.prune_scratch.data() + pp_idx * code_size_);
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
// ConnectAndPrune — per-node spinlock strategy (Phase 9 Task 5.5).
//
// Forward edges: written to new_internal_id without a lock. Only one task
// can be inserting a given new_internal_id (caller contract: tasks run on
// disjoint internal_id ranges), so the write is race-free.
//
// Reciprocal edges: each target's per-node spinlock is acquired individually
// while we append the back-edge (or re-prune on overflow). At most one lock
// is held at a time per task — deadlock is impossible without nested lock
// acquisition, regardless of acquisition order.
//
// Critical section: append-or-re-prune of a ≤R-neighbor list, sub-µs. The
// spinlock keeps the contended thread on-core via PAUSE/yield; this is the
// same pattern hnswlib and DiskANN use (see NodeSpinlock's class comment).
//
// On overflow re-prune, distances come from CodeDistance (Tier 2/3 flat
// buffer fast path) or a per-prune gather (Tier 1, paged). The LUT
// parameter to RobustPrune is explicitly unused; new_lut is taken only for
// API symmetry and discarded.
//
// (Concurrency note on BeamSearch reads: BeamSearch reads neighbor lists
// WITHOUT a lock, so a concurrent ConnectAndPrune write to a node being
// expanded by another task's search is technically a data race under the
// C++ memory model. In practice this is benign: the cnt and neighbor[i]
// fields are uint16_t/uint32_t, and aligned loads/stores of those widths
// are atomic at the hardware level on every target we ship, so BeamSearch
// observes either the pre- or post-update view, never a torn value. Vamana
// search tolerates slightly-stale topology — it's a best-effort greedy
// traversal. TSan will flag this race; it's a known acceptable one. The
// alternative — locking during search — would add lock overhead to the
// build hot path with no recall benefit.)
// ---------------------------------------------------------------------------

void AiSaqCore::ConnectAndPrune(uint32_t new_internal_id, const float * /*new_lut*/, const vector<Candidate> &selected,
                                vamana::VamanaTLS &tls) {
	// Forward edges on the new node — race-free (disjoint internal_id per task).
	{
		auto nref = PinNode(new_internal_id);
		const uint16_t cnt = uint16_t(std::min<idx_t>(selected.size(), params_.R));
		SetNeighborCount(nref.ptr, cnt);
		for (idx_t i = 0; i < cnt; i++) {
			SetNeighbor(nref.ptr, i, uint32_t(selected[i].row_id));
		}
	}

	// Reciprocal edges. Each target's lock is acquired and released in
	// isolation — no nested lock acquisition anywhere in this loop.
	for (const auto &s : selected) {
		const uint32_t s_internal = uint32_t(s.row_id);
		// Defensive: skip out-of-range targets. node_locks_ is sized to N
		// by ResizeNodeLocks before tasks spawn; selected row_ids come from
		// BeamSearch which only visits already-inserted nodes [0, N), so
		// this branch should never fire in practice.
		if (s_internal >= node_locks_count_) {
			continue;
		}

		auto &lock = node_locks_[s_internal];
		lock.lock();
		auto sref = PinNode(s_internal);
		uint16_t cnt = GetNeighborCount(sref.ptr);
		if (cnt < params_.R) {
			// Common case: room left in s's neighbor list. Append.
			SetNeighbor(sref.ptr, cnt, new_internal_id);
			SetNeighborCount(sref.ptr, cnt + 1);
			lock.unlock();
			continue;
		}
		// Overflow: re-prune s's neighbor list. The body mirrors the old
		// ApplyDeferredReciprocals re-prune path bit-for-bit.
		const bool need_gather = !build_codes_ && !build_vectors_;
		if (need_gather) {
			// Layout: [0]=s_internal, [1]=new_internal_id, [2..cnt+1]=neighbors
			tls.prune_scratch.resize((idx_t(cnt) + 2) * code_size_);
			CopyBuildCode(s_internal, tls.prune_scratch.data());
			CopyBuildCode(new_internal_id, tls.prune_scratch.data() + code_size_);
			for (uint16_t i = 0; i < cnt; i++) {
				const uint32_t nb = GetNeighbor(sref.ptr, i);
				CopyBuildCode(nb, tls.prune_scratch.data() + idx_t(i + 2) * code_size_);
			}
		}

		vector<Candidate> cand;
		cand.reserve(idx_t(cnt) + 1);
		if (need_gather) {
			const_data_ptr_t s_ptr = tls.prune_scratch.data();
			cand.push_back(
			    {int64_t(new_internal_id), quantizer_.CodeDistance(s_ptr, tls.prune_scratch.data() + code_size_)});
			for (uint16_t i = 0; i < cnt; i++) {
				cand.push_back({int64_t(GetNeighbor(sref.ptr, i)),
				                quantizer_.CodeDistance(s_ptr, tls.prune_scratch.data() + idx_t(i + 2) * code_size_)});
			}
		} else {
			cand.push_back({int64_t(new_internal_id), CodeDistance(s_internal, new_internal_id)});
			for (uint16_t i = 0; i < cnt; i++) {
				const uint32_t nb = GetNeighbor(sref.ptr, i);
				cand.push_back({int64_t(nb), CodeDistance(s_internal, nb)});
			}
		}
		auto kept = RobustPrune(nullptr, std::move(cand), params_.R, params_.alpha, tls);
		SetNeighborCount(sref.ptr, uint16_t(kept.size()));
		for (idx_t i = 0; i < kept.size(); i++) {
			SetNeighbor(sref.ptr, i, uint32_t(kept[i].row_id));
		}
		lock.unlock();
	}
}

void AiSaqCore::ResizeNodeLocks(idx_t count) {
	if (count <= node_locks_count_) {
		return;
	}
	// Amortized doubling keeps the serial live-insert path O(1) amortized
	// per call; the parallel build calls this once with the exact N before
	// tasks spawn, so the doubling never kicks in for it.
	idx_t new_count = std::max<idx_t>(count, node_locks_count_ * 2);
	if (new_count < 64) {
		new_count = 64;
	}
	// `new NodeSpinlock[new_count]` default-initializes each element: the
	// implicit default ctor runs, applying the NSDMI `flag{0}` initializer,
	// so every lock starts in the unlocked state.
	//
	// MUST be called only from single-threaded context. Reallocating while
	// a parallel build is in flight would dangle per-task references into
	// the old array — see the caller contract on ResizeNodeLocks in the
	// header.
	node_locks_.reset(new NodeSpinlock[new_count]);
	node_locks_count_ = new_count;
}

void AiSaqCore::MergeLabelMaps(const unordered_map<uint32_t, int64_t> &id2label,
                               const unordered_map<int64_t, vector<uint32_t>> &label2ids) {
	for (const auto &kv : id2label) {
		internal_id_to_label_[kv.first] = kv.second;
	}
	for (const auto &kv : label2ids) {
		auto &target = label_to_internal_ids_[kv.first];
		target.insert(target.end(), kv.second.begin(), kv.second.end());
	}
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

uint32_t AiSaqCore::Insert(int64_t row_id, const float *vec, int64_t label) {
	// Serial / test path. Allocates the internal_id via the block store
	// (per-call), runs the insert with the core's own tls_, and
	// synchronously applies reciprocals via the per-node-locked
	// ConnectAndPrune — preserving the old immediate-back-edge semantics.
	const uint32_t internal_id = store_.AllocGraphNode();
	// Live-insert path: lazily grow node_locks_ past the build-time size.
	// Safe — Insert runs serially (caller holds the rwlock via Construct).
	ResizeNodeLocks(idx_t(internal_id) + 1);
	const uint32_t prev_size = size_;
	InsertBuild(internal_id, row_id, vec, label, tls_, internal_id_to_label_, label_to_internal_ids_);
	// InsertBuild's first-call branch (size_==0) sets size_=1 itself; for
	// every subsequent call it leaves size_ untouched (parallel path owns
	// the final SetSize). Bump here for non-first serial inserts.
	if (prev_size > 0) {
		size_++;
	}
	return internal_id;
}

uint32_t AiSaqCore::InsertBuild(uint32_t internal_id, int64_t row_id, const float *vec, int64_t label,
                                vamana::VamanaTLS &tls, unordered_map<uint32_t, int64_t> &local_id2label,
                                unordered_map<int64_t, vector<uint32_t>> &local_label2ids) {
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

	if (internal_id >= tls.visit_marks.size()) {
		tls.visit_marks.resize(std::max<size_t>(tls.visit_marks.size() * 2, size_t(internal_id) + 1), 0);
	}

	// Per-task label bookkeeping (merged serially into the core's global
	// maps after all tasks finish). BeamSearch during build never reads
	// label maps (label_filter is null at build time), so per-task isolation
	// is safe.
	if (label != INT64_MIN) {
		local_id2label[internal_id] = label;
		local_label2ids[label].push_back(internal_id);
	}

	// First-insert claim of the entry point. The parallel coordinator MUST
	// serialize the first InsertBuild across all tasks (typically by running
	// it once in Schedule() before spawning tasks). Concurrent first-calls
	// would race on entry_internal_ / size_. Subsequent calls are safe to
	// run concurrently on disjoint internal_id ranges.
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
		// Fallback (shouldn't happen: AiSAQ requires PQ/ScaNN).
		std::memcpy(lut.data(), qpre.data(), qpre.size() * sizeof(float));
	}

	const idx_t L_build = params_.L_build > 0 ? params_.L_build : params_.R;
	auto cands = BeamSearch(lut.data(), L_build, 0, tls);
	auto selected = RobustPrune(lut.data(), std::move(cands), params_.R, params_.alpha, tls);
	ConnectAndPrune(internal_id, lut.data(), selected, tls);

	return internal_id;
}

// ---------------------------------------------------------------------------
// FinalizeInlineCodes
// ---------------------------------------------------------------------------

void AiSaqCore::FinalizeInlineCodesRange(idx_t id_start, idx_t id_end) const {
	const idx_t inline_region_off = kNeighborArrayOffset + idx_t(params_.R) * sizeof(uint32_t);
	for (idx_t id = id_start; id < id_end; id++) {
		auto nref = PinNode(uint32_t(id));
		const uint16_t n = GetNeighborCount(nref.ptr);
		const uint16_t nin = std::min<uint16_t>(n, params_.inline_pq_count);
		for (uint16_t i = 0; i < nin; i++) {
			const uint32_t nb = GetNeighbor(nref.ptr, i);
			auto ref = ReadPqCode(nb);
			std::memcpy(nref.ptr + inline_region_off + idx_t(i) * code_size_, ref.ptr, code_size_);
		}
	}
}

void AiSaqCore::FinalizeInlineCodes() {
	if (params_.inline_pq_count == 0 || size_ == 0) {
		return;
	}
	// Serial fallback for unit tests without a DatabaseInstance.
	if (db_ == nullptr) {
		FinalizeInlineCodesRange(0, size_);
		return;
	}
	auto &scheduler = TaskScheduler::GetScheduler(*db_);
	const size_t num_threads = std::max<size_t>(1, NumericCast<size_t>(scheduler.NumberOfThreads()));
	TaskExecutor executor(scheduler);
	for (size_t t = 0; t < num_threads; t++) {
		const idx_t id_start = idx_t(t) * size_ / num_threads;
		const idx_t id_end = idx_t(t + 1) * size_ / num_threads;
		if (id_start >= id_end) {
			continue;
		}
		executor.ScheduleTask(make_uniq<AiSaqFinalizeInlineTask>(executor, *this, id_start, id_end));
	}
	executor.WorkOnTasks();
}

// ---------------------------------------------------------------------------
// ComputeEntryPoints
//
// Stays serial: at most params_.n_entry_points (≤16 by default, ≤64 max)
// iterations of a single ReadPqCode Pin + small code copy. The TaskExecutor
// scheduling overhead (~tens of µs per task) exceeds the work even at the
// upper bound. entry_points_ is pre-sized so per-iteration indexed writes
// are concurrency-safe regardless of caller threading model.
// ---------------------------------------------------------------------------

void AiSaqCore::ComputeEntryPoints() {
	entry_points_.clear();
	if (size_ == 0) {
		return;
	}
	const uint16_t want = std::min<uint32_t>(params_.n_entry_points, size_);
	entry_points_.resize(want);
	for (uint16_t i = 0; i < want; i++) {
		// Evenly-spaced internal_ids across the dataset.
		const uint32_t id = (uint32_t(i) * uint32_t(size_)) / uint32_t(want);
		auto ref = ReadPqCode(id);
		entry_points_[i].internal_id = id;
		entry_points_[i].code.assign(ref.ptr, ref.ptr + code_size_);
	}
}

// ---------------------------------------------------------------------------
// ComputeLabelMedoids
// ---------------------------------------------------------------------------

void AiSaqCore::ComputeLabelMedoidsRange(const vector<int64_t> &labels, idx_t start, idx_t end,
                                         unordered_map<int64_t, uint32_t> &local_out) const {
	for (idx_t li = start; li < end; li++) {
		const int64_t label = labels[li];
		auto it = label_to_internal_ids_.find(label);
		if (it == label_to_internal_ids_.end()) {
			continue;
		}
		const auto &members = it->second;
		if (members.empty()) {
			continue;
		}
		if (members.size() == 1) {
			local_out[label] = members[0];
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
		local_out[label] = best_id;
	}
}

void AiSaqCore::ComputeLabelMedoids() {
	if (label_to_internal_ids_.empty()) {
		return;
	}

	// Build sorted_labels_ for range queries (serial — small relative to the
	// per-label work below, and the sort needs to happen anyway).
	sorted_labels_.clear();
	sorted_labels_.reserve(label_to_internal_ids_.size());
	for (const auto &kv : label_to_internal_ids_) {
		sorted_labels_.push_back(kv.first);
	}
	std::sort(sorted_labels_.begin(), sorted_labels_.end());

	label_medoids_.clear();
	label_medoids_.reserve(sorted_labels_.size());

	// Serial fallback for unit tests without a DatabaseInstance, or when the
	// label count is too small to amortise TaskExecutor scheduling overhead.
	const size_t num_labels = sorted_labels_.size();
	const bool serial = db_ == nullptr || num_labels < 8;
	if (serial) {
		ComputeLabelMedoidsRange(sorted_labels_, 0, num_labels, label_medoids_);
		return;
	}

	auto &scheduler = TaskScheduler::GetScheduler(*db_);
	const size_t num_threads = std::max<size_t>(1, NumericCast<size_t>(scheduler.NumberOfThreads()));
	const size_t actual_threads = std::min(num_threads, num_labels);

	// Per-task local maps avoid concurrent unordered_map writes (which would
	// race on rehashing). Merged into label_medoids_ serially after all tasks
	// complete.
	vector<unordered_map<int64_t, uint32_t>> locals(actual_threads);
	TaskExecutor executor(scheduler);
	for (size_t t = 0; t < actual_threads; t++) {
		const idx_t start = idx_t(t) * num_labels / actual_threads;
		const idx_t end = idx_t(t + 1) * num_labels / actual_threads;
		if (start >= end) {
			continue;
		}
		executor.ScheduleTask(
		    make_uniq<AiSaqComputeMedoidsTask>(executor, *this, sorted_labels_, start, end, locals[t]));
	}
	executor.WorkOnTasks();

	// Serial merge — bucket positions are stable (label_medoids_ was reserved
	// above so no rehash occurs during insertion).
	for (auto &local : locals) {
		for (auto &kv : local) {
			label_medoids_[kv.first] = kv.second;
		}
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
		cands = BeamSearch(query_lut, L_search, io_limit, tls_, &ep, lf);
	} else if (lf && lf->kind == LabelFilter::Kind::RANGE) {
		// RANGE: adaptive strategy.
		const idx_t match_count = CountLabelsInRange(lf->lo, lf->hi);
		if (match_count > 0 && match_count <= params_.n_entry_points) {
			// Multi-medoid: use all matching medoids as entry points.
			auto eps = GetMedoidsInRange(lf->lo, lf->hi, params_.n_entry_points);
			cands = BeamSearch(query_lut, L_search, io_limit, tls_, &eps, lf);
		} else {
			// Too many labels or none: fall back to global entry points.
			cands = BeamSearch(query_lut, L_search, io_limit, tls_, nullptr, lf);
		}
	} else {
		// No label filter or unsupported kind: normal search.
		cands = BeamSearch(query_lut, L_search, io_limit, tls_, nullptr, lf);
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
