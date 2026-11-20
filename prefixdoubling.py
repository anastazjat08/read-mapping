import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"


import numpy as np
from array import array

def build_suffix_array(s: str):
    """
    https://gist.github.com/lucaswiman/5cc6584a7c6316998357af1cbdf300ba

    Implements the "prefix doubling" algorithm.
    See https://en.wikipedia.org/wiki/Suffix_array
    """
    n = len(s)
    if n == 0:
        return []
    suffix_arr = list(range(n))

    rank = [ord(c) for c in s]
    smallest = min(rank)
    rank = [r - smallest for r in rank]

    k = 1

    def key(i):
        return (rank[i], rank[i + k] if i + k < n else -1)

    while k < n:
        suffix_arr.sort(key=key)
        new_rank = [0] * n

        # Note that suffix_arr[0] already has the correct rank in new_rank (0).
        for i in range(1, n):
            if key(suffix_arr[i-1]) != key(suffix_arr[i]):
                new_rank[suffix_arr[i]] = new_rank[suffix_arr[i-1]] + 1
            else:
                new_rank[suffix_arr[i]] = new_rank[suffix_arr[i-1]]
        rank = new_rank

        k *= 2

    return suffix_arr


def build_suffix_array_numpy(s: str):
    """
    https://gist.github.com/lucaswiman/5cc6584a7c6316998357af1cbdf300ba

    Implements the "prefix doubling" algorithm.
    See https://en.wikipedia.org/wiki/Suffix_array

    NumPy version
    """
    n = len(s)
    if n == 0:
        return np.array([], dtype=np.int64)

    # suffix_arr = list(range(n))
    suffix_arr = np.arange(n)

    # rank = [ord(c) for c in s]; rank = [r - smallest for r in rank]
    if isinstance(s, str):
        rank = np.frombuffer(s.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
    else:
        rank = np.frombuffer(s, dtype=np.uint8).astype(np.int64)
    rank -= rank.min()


    k = 1

    # def key(i):
    #    return (rank[i], rank[i + k] if i + k < n else -1)

    while k < n:
        # key(i) = (rank[i], rank[i + k] if i + k < n else -1), packed into
        # one integer: rank[i] * m + (second + 1). The +1 turns -1 into 0,
        # and m is larger than any second + 1, so integer order = tuple order.
        second = np.zeros(n, dtype=np.int64)          # the "-1" case, after +1
        second[:n - k] = rank[k:] + 1
        m = int(rank.max()) + 2
        key = rank * m + second

        # suffix_arr.sort(key=key)
        suffix_arr = np.argsort(key)

        # The for loop: new_rank goes up by 1 whenever the key differs from
        # the previous key in sorted order, and stays the same otherwise.
        sorted_key = key[suffix_arr]
        new_rank = np.empty(n, dtype=np.int64)
        new_rank[suffix_arr[0]] = 0
        new_rank[suffix_arr[1:]] = np.cumsum(sorted_key[1:] != sorted_key[:-1])
        rank = new_rank

        k *= 2

        # Optional, not in the original: stop once all ranks are distinct.
        # Later rounds wouldn't change the order, so the result is the same.
        if rank[suffix_arr[-1]] == n - 1:
            break

    return suffix_arr