#!/usr/bin/env python3
"""Analyze a samply profile (Firefox Profiler JSON format) with .syms.json sidecar.

Resolves raw frame addresses to function names using the symbol sidecar
produced by `samply record --unstable-presymbolicate`, then computes
self-time and inclusive-time breakdowns per function.

Usage:
    python3 scripts/analyze_profile.py profile.json[.gz]

Requires the profile to have been recorded with --unstable-presymbolicate
so that a profile.json.syms.json sidecar exists alongside it.

Example:
    samply record --unstable-presymbolicate -r 10000 -s -o profile.json.gz -- ./build/release/duckdb -f script.sql
    python3 scripts/analyze_profile.py profile.json.gz
"""
from __future__ import annotations

import bisect
import collections
import gzip
import json
import sys
from pathlib import Path


def _open_maybe_gzip(path: str):
	if path.endswith(".gz"):
		return gzip.open(path, "rt")
	return open(path, "rt")


def load_profile(profile_path: str) -> dict:
	with _open_maybe_gzip(profile_path) as f:
		return json.load(f)


def load_symbols(profile_path: str) -> dict | None:
	"""Load the .syms.json sidecar if it exists."""
	for p in [
		profile_path + ".syms.json",
		profile_path[:-3] + ".syms.json" if profile_path.endswith(".gz") else None,
	]:
		if p is None:
			continue
		try:
			with open(p) as f:
				return json.load(f)
		except FileNotFoundError:
			continue
	return None


def build_syms_resolver(syms: dict):
	"""Build resolver from samply .syms.json known_addresses.

	known_addresses maps profile frame addresses to symbol indices.
	Each entry is [frame_address, symbol_index_into_string_table].
	"""
	st = syms["string_table"]

	# Build address -> (name, lib) from known_addresses
	addr_map: dict[int, tuple[str, str]] = {}
	for entry in syms["data"]:
		debug_name = entry.get("debug_name", "?")
		for pair in entry.get("known_addresses", []):
			addr, sym_idx = pair[0], pair[1]
			name = st[sym_idx] if 0 <= sym_idx < len(st) else f"sym_{sym_idx}"
			addr_map[addr] = (name, debug_name)

		# Also add symbol_table entries as fallback
		for sym in entry.get("symbol_table", []):
			rva, sym_idx = sym["rva"], sym["symbol"]
			name = st[sym_idx] if 0 <= sym_idx < len(st) else f"sym_{sym_idx}"
			if rva not in addr_map:
				addr_map[rva] = (name, debug_name)

	sorted_addrs = sorted(addr_map.keys())

	def resolve(addr: int) -> tuple[str, str]:
		if addr in addr_map:
			return addr_map[addr]
		idx = bisect.bisect_right(sorted_addrs, addr) - 1
		if idx >= 0 and addr - sorted_addrs[idx] < 0x10000:
			return addr_map[sorted_addrs[idx]]
		return (f"0x{addr:x}", "unknown")

	return resolve


def analyze_thread(thread: dict, resolve) -> tuple[collections.Counter, collections.Counter, int]:
	frame_table = thread.get("frameTable", {})
	stack_table = thread.get("stackTable", {})
	samples = thread.get("samples", {})

	sample_stacks = samples.get("stack", [])
	sample_weights = samples.get("weight", [])
	frame_addresses = frame_table.get("address", [])
	stack_frame = stack_table.get("frame", [])
	stack_prefix = stack_table.get("prefix", [])

	if not sample_stacks or not frame_addresses:
		return collections.Counter(), collections.Counter(), 0

	# Pre-resolve all frames
	frame_names = []
	frame_libs = []
	for addr in frame_addresses:
		name, lib = resolve(addr)
		frame_names.append(name)
		frame_libs.append(lib)

	self_time: collections.Counter = collections.Counter()
	incl_time: collections.Counter = collections.Counter()

	for i, sidx in enumerate(sample_stacks):
		w = abs(sample_weights[i]) if i < len(sample_weights) else 1.0

		if sidx is not None and 0 <= sidx < len(stack_frame):
			fidx = stack_frame[sidx]
			if 0 <= fidx < len(frame_names):
				self_time[(frame_names[fidx], frame_libs[fidx])] += w

		cur = sidx
		visited: set[int] = set()
		while cur is not None and 0 <= cur < len(stack_frame) and cur not in visited:
			visited.add(cur)
			fidx = stack_frame[cur]
			if 0 <= fidx < len(frame_names):
				incl_time[(frame_names[fidx], frame_libs[fidx])] += w
			cur = stack_prefix[cur] if cur < len(stack_prefix) else None

	total = sum(self_time.values())
	return self_time, incl_time, total


KEYWORDS = [
	"aisaq", "vindex", "quantizer", "pq_", "beam", "prune", "insert",
	"construct", "encode", "search", "block_store", "buffermanager",
	"pin", "block", "createindex", "finalize", "train", "rawdistance",
	"codedistance", "lut", "robust", "connect", "alloc", "write",
]


def is_vindex_func(name: str) -> bool:
	lower = name.lower()
	return any(kw in lower for kw in KEYWORDS)


def main():
	if len(sys.argv) < 2:
		print(f"Usage: {sys.argv[0]} <profile.json[.gz]>")
		sys.exit(1)

	profile_path = sys.argv[1]
	profile = load_profile(profile_path)
	syms = load_symbols(profile_path)

	if syms is None:
		print("ERROR: No .syms.json sidecar found.", file=sys.stderr)
		print("Re-record with: samply record --unstable-presymbolicate ...", file=sys.stderr)
		sys.exit(1)

	resolve = build_syms_resolver(syms)

	threads = profile.get("threads", [])

	# Analyze ALL threads
	results = []
	for ti, thread in enumerate(threads):
		st, it, tot = analyze_thread(thread, resolve)
		vindex_w = sum(w for (n, _), w in st.items() if is_vindex_func(n))
		results.append((ti, thread, st, it, tot, vindex_w))

	# Sort by total samples descending
	results.sort(key=lambda r: r[4], reverse=True)

	for rank, (ti, thread, self_time, incl_time, total, vindex_self) in enumerate(results):
		if total == 0:
			continue

		thread_name = thread.get("name", "?")
		vindex_pct = 100.0 * vindex_self / total if total > 0 else 0

		print(f"\n{'='*120}")
		print(f"Thread {ti}: name={thread_name}, samples={total:.0f}, vindex_self={vindex_pct:.1f}%")
		print(f"{'='*120}")

		if vindex_pct < 1.0 and rank > 0:
			continue

		# Top self time — no truncation
		print(f"\n--- TOP 30 SELF TIME ---")
		for (name, lib), w in self_time.most_common(30):
			pct = 100.0 * w / total
			lib_short = Path(lib).name if lib else ""
			print(f"  {pct:>6.1f}%  [{lib_short}]  {name}")

		# Top inclusive time for vindex functions
		print(f"\n--- TOP 30 INCLUSIVE TIME (vindex + DuckDB hot paths) ---")
		filtered = {k: v for k, v in incl_time.items() if is_vindex_func(k[0])}
		if not filtered:
			filtered = dict(incl_time)
		for (name, lib), w in sorted(filtered.items(), key=lambda x: -x[1])[:30]:
			pct = 100.0 * w / total
			lib_short = Path(lib).name if lib else ""
			print(f"  {pct:>6.1f}%  [{lib_short}]  {name}")

	# Summary across all threads
	print(f"\n{'='*120}")
	print(f"SUMMARY")
	print(f"{'='*120}")
	for ti, thread, self_time, incl_time, total, vindex_self in results:
		if total == 0:
			continue
		name = thread.get("name", "?")
		pct = 100.0 * vindex_self / total if total > 0 else 0
		print(f"  Thread {ti} ({name}): {total:.0f} samples, vindex={pct:.1f}%")


if __name__ == "__main__":
	main()
