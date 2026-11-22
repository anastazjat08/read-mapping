"""Karkkainen-Sanders (DC3) suffix array construction with NumPy."""

import os

# setting thread limits before importing NumPy
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"

import numpy as np


def direct_kark_sort(s):
    """ Build the suffix array of a sequence of sortable, hashable symbols.
        s(str/bytes): reference sequence"""
    alphabet = sorted(set(s))
    # symbols start at 1, zero is for padding
    t = {c: i + 1 for i, c in enumerate(alphabet)}
    a = np.fromiter((t[c] for c in s), dtype=np.int32, count=len(s))
    return kark_sort(a)


def kark_sort(s):
    """ Build the suffix array
        s(str/bytes): reference sequence"""
    s = np.asarray(s, dtype=np.int32)
    n = len(s)
    # when in recurrence we get to the end
    if n == 0:
        return np.empty(0, dtype=np.int32)
    if n == 1:
        return np.zeros(1, dtype=np.int32)

    n0 = (n + 2) // 3 # number of positions for group n % 3 == 0
    n1 = (n + 1) // 3 # number of positions for group n % 3 == 1
    n02 = n0 + n // 3 # number of positions for group n % 3 == 1 or 2 (with sometimes additional pos)
    sp = np.concatenate([s, np.zeros(3, dtype=np.int32)]) # copy of sequence with pad at the end
    del s

    # positions for mod 3 != 0 groups (with dummy when n % 3 == 1)
    p12 = np.empty(n02, dtype=np.int32)
    p12[:n0] = np.arange(1, n + (n0 - n1), 3, dtype=np.int32)
    p12[n0:] = np.arange(2, n, 3, dtype=np.int32)

    # sort triples by their first, second, and third symbols
    order = np.lexsort((sp[p12 + 2], sp[p12 + 1], sp[p12]))
    p12 = p12[order]
    del order

    # assign a new rank(name) whenever adjacent triples differ
    # each True means - start a new rank
    new = np.zeros(n02, dtype=bool)
    new[0] = True
    for offset in range(3):
        column = sp[p12 + offset]
        # mark a new group if the current symbol differs from the previous one
        new[1:] |= column[1:] != column[:-1]
        del column

    names = np.cumsum(new, dtype=np.int32) # assigning
    n_names = int(names[-1]) # how many names
    del new

    # mod-1 names followed by mod-2 names
    idx = np.where(p12 % 3 == 1, p12 // 3, p12 // 3 + n0)
    s12 = np.empty(n02, dtype=np.int32)
    s12[idx] = names
    del p12, idx, names

    if n_names < n02: # some triples identical
        SA12 = kark_sort(s12)
    else:
        SA12 = np.empty(n02, dtype=np.int32)
        # s12-1 because indx in array starts from 0
        # positions sorted by ranks
        SA12[s12 - 1] = np.arange(n02, dtype=np.int32)
    del s12

    # Convert reduced indices back to positions in reference and assign suffix ranks
    # n0 sets the threshold between mod-1 and mod-2
    P12 = np.where(SA12 < n0, SA12 * 3 + 1, (SA12 - n0) * 3 + 2)
    R = np.zeros(n + 3, dtype=np.int32)
    R[P12] = np.arange(1, n02 + 1, dtype=np.int32) # sorted positions (at whole reference scale) by ranks 

    # mod-0 suffixes are ordered by the rank of their tails, then
    # stably sorted by their first symbol
    s0 = 3 * SA12[SA12 < n0] # we have pairs mod-0 mod-1; positions for mod-0
    del SA12
    SA0 = s0[np.argsort(sp[s0], kind="stable")]
    del s0

    # drop the dummy suffix from the merge, if it was included
    P12 = P12[n0 - n1:]
    B = n02 + 2  # base for suffix ranks

    # Compare mod-0 and mod-1 suffixes using keys - (first symbol, tail rank)
    P1 = P12[P12 % 3 == 1]
    k1_P = sp[P1].astype(np.int64) * B + R[P1 + 1] # *B so it is still lexicographic order
    del P1
    k1_0 = sp[SA0].astype(np.int64) * B + R[SA0 + 1]

    pos0 = np.searchsorted(k1_P, k1_0, side="right").astype(np.int32)
    del k1_P, k1_0

    # Compare mod-0 and mod-2 suffixes using (two symbols, tail rank)
    P2 = P12[P12 % 3 == 2]
    K = int(sp.max()) + 1 # base for nucleotide codes
    pair2 = sp[P2].astype(np.int64) * K + sp[P2 + 1]
    pair0 = sp[SA0].astype(np.int64) * K + sp[SA0 + 1]
    del sp

    if K * K * B - 1 > np.iinfo(np.int64).max:
        pairs = np.concatenate([pair2, pair0])
        del pair2, pair0
        _, pair_ranks = np.unique(pairs, return_inverse=True)
        del pairs, _
        pair_ranks = pair_ranks.reshape(-1)
        pair2 = pair_ranks[:len(P2)].astype(np.int64, copy=True)
        pair0 = pair_ranks[len(P2):].astype(np.int64, copy=True)
        del pair_ranks


    # reuse the pair arrays as the final keys
    pair2 *= B
    pair2 += R[P2 + 2]
    pair0 *= B
    pair0 += R[SA0 + 2]
    del P2, R

    # how many mod-2 suffixises is before mod-0
    pos0 += np.searchsorted(pair2, pair0, side="right")
    del pair2, pair0

    # each mod-0 suffix follows the preceding suffixes from all groups
    pos0 += np.arange(n0, dtype=np.int32)
    SA = np.empty(n, dtype=np.int32)
    is0 = np.zeros(n, dtype=bool)
    is0[pos0] = True
    SA[pos0] = SA0
    SA[~is0] = P12
    return SA
