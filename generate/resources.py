"""Worker sizing from the resources this process is actually allowed to use.

os.cpu_count() reports the machine's cores and psutil.virtual_memory() reports
the machine's RAM, but a run may be confined to fewer cores by CPU affinity and
to less memory by a cgroup (a container, a systemd slice, a Slurm job). Sizing a
process pool from the machine's totals is how a run ends up killed by the OOM
killer on a box that looks like it had plenty free.

Everything here reads the *effective* limit and falls back to the machine value
when no limit is set.
"""

import os

_CGROUP_V2_MAX = "/sys/fs/cgroup/memory.max"
_CGROUP_V2_CUR = "/sys/fs/cgroup/memory.current"
_CGROUP_V1_MAX = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
_CGROUP_V1_CUR = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
_CGROUP_V2_CPU = "/sys/fs/cgroup/cpu.max"

# A forked child begins as a copy-on-write view of the parent, but CPython
# writes to object headers when it refcounts, so a child that walks the graph
# and its task arrays ends up resident for most of the parent's footprint.
# Measured on Miami-Dade: 31 children against a ~7 GB parent reached ~154 GB,
# i.e. essentially 1.0. Kept at 1.0 rather than a hopeful fraction.
COW_RESIDENT_FRACTION = 1.0

# Never plan to use the last slice of what is available.
MEMORY_HEADROOM = 0.8


def _read_int(path):
    try:
        with open(path) as fh:
            value = fh.read().strip()
        if value in ("max", "-1"):
            return None
        return int(value)
    except (OSError, ValueError):
        return None


def usable_cpus():
    """Cores this process may actually run on."""
    try:
        n = len(os.sched_getaffinity(0))  # respects taskset / cpuset cgroups
    except (AttributeError, OSError):
        n = os.cpu_count() or 1

    # A CPU quota caps throughput even when many cores are visible.
    try:
        with open(_CGROUP_V2_CPU) as fh:
            quota, period = fh.read().split()
        if quota != "max":
            n = min(n, max(1, int(float(quota) / float(period))))
    except (OSError, ValueError):
        pass

    return max(1, n)


def available_memory_bytes():
    """Memory this process may still allocate, honouring any cgroup limit."""
    try:
        import psutil

        avail = psutil.virtual_memory().available
    except Exception:
        return None

    for limit_path, usage_path in (
        (_CGROUP_V2_MAX, _CGROUP_V2_CUR),
        (_CGROUP_V1_MAX, _CGROUP_V1_CUR),
    ):
        limit = _read_int(limit_path)
        usage = _read_int(usage_path)
        if limit is not None and usage is not None and limit < (1 << 62):
            avail = min(avail, max(0, limit - usage))

    return avail


def current_rss_bytes():
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except Exception:
        return None


def plan_workers(requested, per_worker_bytes=None, label="workers"):
    """Largest worker count that fits in the memory this process may use.

    per_worker_bytes defaults to the parent's own RSS, which is the right
    estimate for a forked pool: each child starts from the parent's image.
    """
    requested = max(1, min(requested, usable_cpus()))

    avail = available_memory_bytes()
    if avail is None:
        return requested

    if per_worker_bytes is None:
        rss = current_rss_bytes()
        if rss is None:
            return requested
        per_worker_bytes = rss * COW_RESIDENT_FRACTION

    per_worker_bytes = max(per_worker_bytes, 512 << 20)  # floor at 0.5 GB
    budget = avail * MEMORY_HEADROOM
    safe = max(1, int(budget // per_worker_bytes))
    chosen = max(1, min(requested, safe))

    if chosen < requested:
        print(
            f"[mem-guard] Throttling {label} {requested} -> {chosen} "
            f"({avail / 1e9:.1f} GB usable, ~{per_worker_bytes / 1e9:.1f} GB/worker)"
        )
    return chosen
