#!/usr/bin/env python3
"""Sample instantaneous CPU% of a process via repeated cpu_times() reads.

Uses os.wait4-style rusage via subprocess resource module to get cumulative
CPU. Polls at ~50ms cadence; reports delta-cpu/delta-wall as instantaneous %.
This avoids macOS ps's 60s decaying-average behavior.

Usage: python3 peak_cpu.py <pid>
"""
import os
import sys
import time
import resource

pid = int(sys.argv[1])
# Use ru usage of the PROCESS GROUP via getrusage(RUSAGE_CHILDREN) if we're
# the parent; but we can't getrusage(RUSAGE_CHILDREN) without being parent.
# Instead, read /dev/null — we need proc info. macOS doesn't expose /proc.
#
# Fallback: use os.times() — but that's only for THIS process, not the child.
#
# Truly portable approach: use the mach task_info API via ctypes.
import ctypes
import ctypes.util

libc = ctypes.CDLL(ctypes.util.find_library("c"))

LIBRARY = ctypes.CDLL("/usr/lib/system/libsystem_kernel.dylib", use_errno=True)


class TaskThreadTimesInfo(ctypes.Structure):
    _fields_ = [
        ("user_time_seconds", ctypes.c_uint),
        ("user_time_microseconds", ctypes.c_int),
        ("system_time_seconds", ctypes.c_uint),
        ("system_time_microseconds", ctypes.c_int),
    ]


# task_info(for_task_t, flavor, out, count) is too involved for a quick
# script. Use the higher-level: psutil is missing — fall back to running
# `ps` repeatedly and computing (cpu_delta) / (wall_delta) * 100 from the
# cputime field (which is cumulative, NOT decayed).
#
# ps -o time= gives elapsed-format H:MM:SS, no good.
# ps -o cputime= not on mac.
# Use `ps -o utime=,stime=` which gives H:MM:SS or M:SS.SS cumulative.

def parse_time(s):
    """Parse H:MM:SS or MM:SS.SS into seconds."""
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    return float(s)


def get_cpu_seconds(pid):
    """Returns (user_sec, sys_sec) cumulative for the process."""
    out = os.popen(f"ps -p {pid} -o utime=,stime=").read().strip()
    parts = out.split()
    if len(parts) < 2:
        return None
    return parse_time(parts[0]), parse_time(parts[1])


def main():
    pid = int(sys.argv[1])
    interval = 0.1
    peak_pct = 0.0
    samples = []

    prev = get_cpu_seconds(pid)
    if prev is None:
        print(f"process {pid} not found", file=sys.stderr)
        return 1
    prev_t = time.time()

    while True:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(interval)
        cur = get_cpu_seconds(pid)
        cur_t = time.time()
        if cur is None:
            continue
        dt = cur_t - prev_t
        if dt <= 0:
            continue
        cpu = (cur[0] - prev[0]) + (cur[1] - prev[1])
        pct = 100.0 * cpu / dt
        samples.append(pct)
        if pct > peak_pct:
            peak_pct = pct
        prev = cur
        prev_t = cur_t

    samples.sort(reverse=True)
    top5 = samples[:5] if samples else []
    avg_top10pct = sum(samples[: max(1, len(samples) // 10)]) / max(1, len(samples) // 10) if samples else 0.0
    overall_avg = sum(samples) / len(samples) if samples else 0.0
    print(f"PEAK_CPU_PCT={peak_pct:.1f}")
    print(f"AVG_TOP_10PCT={avg_top10pct:.1f}")
    print(f"OVERALL_AVG={overall_avg:.1f}")
    print(f"TOP5={top5}")
    print(f"NUM_SAMPLES={len(samples)}")


if __name__ == "__main__":
    sys.exit(main() or 0)
