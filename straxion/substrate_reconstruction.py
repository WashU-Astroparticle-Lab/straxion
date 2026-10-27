"""Acquisition-isolated single-link groups and independent nonextending veto views.

No energy or intactness cut is applied here. Membership indices reference the
input table. Stable IDs reference the producer identities, independent of sorting.
"""

import hashlib
import numpy as np

GROUP_DTYPE = [
    ("group_id", "S64"),
    ("first_time", "i8"),
    ("last_time", "i8"),
    ("brightest_time", "i8"),
    ("brightest_hit_id", "S64"),
    ("n_hits", "i4"),
    ("n_channels", "i4"),
    ("n_repeated_channels", "i4"),
    ("max_hits_per_channel", "i4"),
    ("qualifies", "?"),
    ("run_id", "S64"),
    ("record_time", "i8"),
]
MEMBER_DTYPE = [("group_id", "S64"), ("hit_id", "S64"), ("hit_index", "i8")]
VETO_DTYPE = [("accepted", "?"), ("blocked_by_group_id", "S64"), ("veto_end", "i8")]


def timing(hits, basis):
    if basis == "start":
        return hits["time"]
    if basis not in ("raw", "ma", "convolved"):
        raise ValueError("time basis must be start, raw, ma, or convolved")
    if not np.all(hits[f"peak_{basis}_valid"]):
        raise ValueError("Selected hits contain invalid peaks for requested time basis")
    return hits[f"peak_{basis}_time"]


def identities(hits):
    if "hit_id" in hits.dtype.names:
        ids = hits["hit_id"]
    else:
        # Legacy tables lack producer identity; identical window/channel rows are ambiguous.
        ids = np.array(
            [
                hashlib.sha256(f"{h['time']}:{h['endtime']}:{h['channel']}".encode())
                .hexdigest()
                .encode()
                for h in hits
            ],
            dtype="S64",
        )
    if len(np.unique(ids)) != len(ids) or np.any(ids == b""):
        raise ValueError("Hit identities must be nonempty and unique within the input")
    return ids


def acquisitions(hits):
    if "record_time" in hits.dtype.names:
        return list(zip(hits["run_id"], hits["record_time"]))
    return [(b"", 0)] * len(hits)


def prompt_groups(hits, mask, *, basis="ma", gap_ns=1_000_000, min_hits=5):
    if gap_ns < 0 or int(gap_ns) != gap_ns or min_hits < 1 or int(min_hits) != min_hits:
        raise ValueError("gap_ns and min_hits must be nonnegative/positive integers")
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (len(hits),):
        raise ValueError("mask must be a boolean vector aligned with hits")
    ids = identities(hits)
    ix = np.flatnonzero(mask)
    times = timing(hits[ix], basis)
    if times.dtype != np.dtype("int64"):
        raise ValueError("Times must be int64 nanoseconds")
    acq = acquisitions(hits)
    groups, members = [], []
    for key in sorted(set(acq)):
        local = np.array([j for j, i in enumerate(ix) if acq[i] == key], dtype=int)
        # Explicit legacy start-time comparison retains stable input tie order.
        order = (
            local[np.argsort(times[local], kind="stable")]
            if basis == "start"
            else local[np.lexsort((ids[ix[local]], times[local]))]
        )
        if not len(order):
            continue
        boundaries = np.r_[0, np.flatnonzero(np.diff(times[order]) > gap_ns) + 1, len(order)]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            selected = order[a:b]
            indices = ix[selected]
            amp = hits["amplitude"][indices]
            if not np.all(np.isfinite(amp)):
                raise ValueError("Seed amplitudes must be finite")
            bright = np.argmax(amp)
            counts = np.unique(hits["channel"][indices], return_counts=True)[1]
            gid = hashlib.sha256(b"|".join(sorted(ids[indices]))).hexdigest().encode()
            groups.append(
                (
                    gid,
                    times[selected[0]],
                    times[selected[-1]],
                    times[selected[bright]],
                    ids[indices[bright]],
                    len(indices),
                    len(counts),
                    np.sum(counts > 1),
                    counts.max(),
                    len(indices) >= min_hits,
                    key[0],
                    key[1],
                )
            )
            members.extend((gid, ids[i], i) for i in indices)
    return np.array(groups, dtype=GROUP_DTYPE), np.array(members, dtype=MEMBER_DTYPE)


def apply_veto(groups, veto_ns):
    """Accept first >= last accepted member + veto; rejected groups never extend it."""
    if veto_ns < 0 or int(veto_ns) != veto_ns:
        raise ValueError("veto_ns must be a nonnegative integer")
    out = np.zeros(len(groups), dtype=VETO_DTYPE)
    out["veto_end"] = -1
    state = {}
    previous = {}
    for i, group in enumerate(groups):
        key = (group["run_id"], group["record_time"])
        first, last = int(group["first_time"]), int(group["last_time"])
        if key in previous and first < previous[key]:
            raise ValueError("Groups must be chronological and nonoverlapping per acquisition")
        previous[key] = last
        until, blocker = state.get(key, (None, b""))
        if until is not None and first < until:
            out[i] = (False, blocker, until)
        elif group["qualifies"]:
            out["accepted"][i] = True
            state[key] = (last + int(veto_ns), group["group_id"])
    return out


def hit_coincidences(hits, window_ns=131_000, *, with_links=False):
    """MA-peak neighborhoods of ALL detected hits, independently of quality flags.

    Includes self and repeated same-channel hits. Invalid MA peaks have no
    neighborhood and cannot contribute. Endpoints are inclusive. Membership is
    directed (reference_hit_id, member_hit_id); overlaps are expected. Call on a
    complete acquisition, as the records-based hit producer does.
    """
    if window_ns < 0 or int(window_ns) != window_ns:
        raise ValueError("window_ns must be a nonnegative integer")
    ids = identities(hits)
    counts = np.zeros(len(hits), dtype=[("n_hits", "i4"), ("n_channels", "i4")])
    links = []
    acq = acquisitions(hits)
    t, valid = hits["peak_ma_time"], hits["peak_ma_valid"]
    for key in sorted(set(acq)):
        ix = np.array([i for i, k in enumerate(acq) if k == key and valid[i]], dtype=int)
        ix = ix[np.lexsort((ids[ix], t[ix]))]
        sorted_t = t[ix]
        for i in ix:
            a = np.searchsorted(sorted_t, int(t[i]) - window_ns, side="left")
            b = np.searchsorted(sorted_t, int(t[i]) + window_ns, side="right")
            members = ix[a:b]
            counts[i] = (len(members), len(np.unique(hits["channel"][members])))
            if with_links:
                links.extend((ids[i], ids[j]) for j in members)
    return counts, np.array(links, dtype=[("reference_hit_id", "S64"), ("member_hit_id", "S64")])
