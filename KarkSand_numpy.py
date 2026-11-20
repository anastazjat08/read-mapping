"""Kärkkäinen–Sanders (DC3 / skew) suffix array construction, vectorised with NumPy.

All per-element Python loops of the original are replaced by array operations:
  * the three radix passes  -> one np.lexsort over the triples
  * the naming loop         -> diff + cumsum
  * the merge loop          -> two np.searchsorted calls that find, for every
                               mod-0 suffix, how many mod-1/2 suffixes precede it
"""
import numpy as np


def direct_kark_sort(s):
    """Suffix array of a string / sequence of hashable symbols."""
    alphabet = sorted(set(s))
    t = {c: i + 1 for i, c in enumerate(alphabet)}   # symbols start at 1; 0 is the sentinel
    a = np.fromiter((t[c] for c in s), dtype=np.int64, count=len(s))
    return kark_sort(a)


def kark_sort(s):
    """Suffix array of an integer array `s` whose values are all >= 1."""
    s = np.asarray(s, dtype=np.int64)
    n = len(s)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    n0 = (n + 2) // 3
    n1 = (n + 1) // 3
    n02 = n0 + n // 3
    sp = np.concatenate([s, np.zeros(3, dtype=np.int64)])   # padded copy

    # positions i mod 3 != 0 (plus a dummy mod-1 position n when n % 3 == 1)
    p12 = np.arange(n + (n0 - n1))
    p12 = p12[p12 % 3 != 0]

    # sort the triples (s[i], s[i+1], s[i+2]) — replaces the 3 radix passes
    order = np.lexsort((sp[p12 + 2], sp[p12 + 1], sp[p12]))
    p12 = p12[order]

    # name the triples: a new name each time the triple changes
    tri = np.stack([sp[p12], sp[p12 + 1], sp[p12 + 2]], axis=1)
    new = np.ones(n02, dtype=bool)
    new[1:] = np.any(tri[1:] != tri[:-1], axis=1)
    names = np.cumsum(new)
    n_names = names[-1]

    # reduced string: names of all mod-1 positions, then all mod-2 positions
    idx = np.where(p12 % 3 == 1, p12 // 3, p12 // 3 + n0)
    s12 = np.empty(n02, dtype=np.int64)
    s12[idx] = names

    if n_names < n02:                       # names not unique -> recurse
        SA12 = kark_sort(s12)
    else:                                   # already unique -> invert directly
        SA12 = np.empty(n02, dtype=np.int64)
        SA12[s12 - 1] = np.arange(n02)

    # sorted mod-1/2 positions in the original string, and their ranks (1-based)
    P12 = np.where(SA12 < n0, SA12 * 3 + 1, (SA12 - n0) * 3 + 2)
    R = np.zeros(n + 3, dtype=np.int64)
    R[P12] = np.arange(1, n02 + 1)

    # sort mod-0 suffixes: by rank of i+1 (given by SA12 order), then stably by s[i]
    s0 = 3 * SA12[SA12 < n0]
    SA0 = s0[np.argsort(sp[s0], kind="stable")]

    # ---- merge -------------------------------------------------------------
    P12 = P12[n0 - n1:]                     # drop the dummy suffix
    P1 = P12[P12 % 3 == 1]                  # still sorted
    P2 = P12[P12 % 3 == 2]
    B = n02 + 2                             # > any rank

    # mod-0 vs mod-1: compare (s[i], rank[i+1])
    k1_P = sp[P1] * B + R[P1 + 1]
    k1_0 = sp[SA0] * B + R[SA0 + 1]
    c1 = np.searchsorted(k1_P, k1_0, side="right")

    # mod-0 vs mod-2: compare (s[i], s[i+1], rank[i+2]); compress the char pair first
    K = int(sp.max()) + 1
    pairs = np.concatenate([sp[P2] * K + sp[P2 + 1], sp[SA0] * K + sp[SA0 + 1]])
    _, pr = np.unique(pairs, return_inverse=True)
    pr = pr.reshape(-1)
    k2_P = pr[:len(P2)] * B + R[P2 + 2]
    k2_0 = pr[len(P2):] * B + R[SA0 + 2]
    c2 = np.searchsorted(k2_P, k2_0, side="right")

    # each mod-0 suffix lands at (its index in SA0) + (# mod-1/2 suffixes before it);
    # mod-1/2 suffixes fill the remaining slots in their sorted order
    SA = np.empty(n, dtype=np.int64)
    pos0 = np.arange(n0) + c1 + c2
    is0 = np.zeros(n, dtype=bool)
    is0[pos0] = True
    SA[pos0] = SA0
    SA[~is0] = P12
    return SA


if __name__ == "__main__":
    print(direct_kark_sort("banana"))   # [5 3 1 0 4 2]