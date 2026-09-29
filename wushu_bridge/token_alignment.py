"""Validate native text-token supervision and optional monotone target transport."""
from __future__ import annotations

import hashlib
from difflib import SequenceMatcher

import numpy as np


METHOD = "equal_length_edit_blocks_v1"
TRANSPORT_METHOD = "token_id_monotone_transport_v1"


def transport_edges(bad_ids, good_ids):
    """Return row-stochastic (source, target, weight) edges from native IDs.

    Equal blocks stay one-to-one. Unequal replacements overlap uniform token
    cells; insertions/deletions attach at the left boundary when available.
    The correspondence is symbolic and never changes the deployed token count.
    """
    if not bad_ids or not good_ids:
        raise ValueError("Transport requires nonempty native token ID lists")
    mass = {}
    def add(i, j, weight):
        mass[i, j] = mass.get((i, j), 0.) + weight
    for tag, a, b, c, d in SequenceMatcher(None, bad_ids, good_ids, autojunk=False).get_opcodes():
        p, q = b-a, d-c
        if tag == "equal":
            for k in range(p):
                add(a+k, c+k, 1.)
        elif p and q:
            i = j = 0
            while i < p and j < q:
                source_end, target_end = (i+1)*q, (j+1)*p
                overlap = min(source_end, target_end) - max(i*q, j*p)
                if overlap > 0:
                    add(a+i, c+j, overlap / q)
                if source_end <= target_end:
                    i += 1
                if target_end <= source_end:
                    j += 1
        elif q:
            source = a-1 if a else a
            if source >= len(bad_ids):
                raise ValueError("Insertion has no source boundary")
            for j in range(c, d):
                add(source, j, 1.)
        elif p:
            target = c-1 if c else c
            if target >= len(good_ids):
                raise ValueError("Deletion has no target boundary")
            for i in range(a, b):
                add(i, target, 1.)
    rows = [0.] * len(bad_ids)
    columns = [0.] * len(good_ids)
    for (i, j), value in mass.items():
        rows[i] += value
        columns[j] += value
    if any(v <= 0 for v in rows) or any(v <= 0 for v in columns):
        raise ValueError("Transport has an uncovered source row or target column")
    edges = [(i, j, value / rows[i]) for (i, j), value in sorted(mass.items())]
    # Later source rows may share a boundary target, but cannot reach behind
    # an earlier row's target support.
    if any(j > l for (i, j, _), (k, l, _) in zip(edges, edges[1:]) if i != k):
        raise ValueError("Transport is not monotone")
    return edges


def validate_token_alignment(ds, max_tokens, method=METHOD):
    if method not in (METHOD, TRANSPORT_METHOD):
        raise ValueError("Unknown token alignment method")
    if not (len(ds.x) == len(ds.y) == len(ds.records)):
        raise ValueError("Every token pair requires an alignment record")
    encoder_sha = hashlib.sha256(ds.meta.encoder.encode("utf-8")).hexdigest()
    for i, (x, y, record) in enumerate(zip(ds.x, ds.y, ds.records)):
        info = record.get("token_alignment", {})
        if info.get("method") != method or info.get("encoder_identity_sha256") != encoder_sha:
            raise ValueError(f"Pair {i}: verified token alignment/encoder identity is required")
        if not len(x) or not len(y) or max(len(x), len(y)) > max_tokens or (method == METHOD and len(x) != len(y)):
            raise ValueError(f"Pair {i}: token targets require nonempty native lengths within the limit; equal-length mode requires equal lengths (no truncation)")
        if method == TRANSPORT_METHOD and (not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError(f"Pair {i}: nonfinite native token embedding")
        ids = []
        for side, seq in (("bad", x), ("good", y)):
            values = info.get(side + "_ids")
            text = record.get(side)
            if (not isinstance(values, list) or len(values) != len(seq)
                    or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values)):
                raise ValueError(f"Pair {i}: {side} token IDs do not match embedding length")
            if not isinstance(text, str) or info.get(side + "_text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
                raise ValueError(f"Pair {i}: {side} text hash differs from token alignment")
            if method == TRANSPORT_METHOD:
                native_sha = hashlib.sha256(np.ascontiguousarray(seq).tobytes()).hexdigest()
                if info.get(side + "_embedding_sha256") != native_sha:
                    raise ValueError(f"Pair {i}: {side} embedding hash differs from token alignment")
            ids.append(values)
        blocks = SequenceMatcher(None, *ids, autojunk=False).get_opcodes()
        if not any(tag != "equal" for tag, *_ in blocks):
            raise ValueError(f"Pair {i}: token correction needs a nonempty edit")
        if method == METHOD and any(b-a != d-c for _, a, b, c, d in blocks):
            raise ValueError(f"Pair {i}: token edit blocks are not position aligned")
        if method == TRANSPORT_METHOD:
            if info.get("opcodes") != [list(b) for b in blocks]:
                raise ValueError(f"Pair {i}: stale transport opcodes")
            transport_edges(*ids)
