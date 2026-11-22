from KarkSand_numpy_final import direct_kark_sort
from array import array
import numpy as np
from collections import defaultdict
from Bio import SeqIO
import argparse
import time
import os


import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"


# 1. Preparing reference genome - indexing

def bwt_from_sa(text, sa):
    """Build BWT from text and suffix array."""
    bwt = bytearray(len(text))

    for i, pos in enumerate(sa):
        bwt[i] = text[pos - 1] if pos > 0 else ord('$')

    return bytes(bwt)


class FmCheckpoints:
    """Rank checkpoints with exclusive-end indexing"""

    def __init__(self, bwt, cpIval):
        self.bwt = bwt
        self.cpIval = cpIval #space between checkpoints
        self.alphabet = sorted(set(bwt))
        # array for each letter
        # 'I' means integer, here we represent letters as integers (ASCII - bytes)
        self.cps = {c: array('I', [0]) for c in self.alphabet}

        counts = {c: 0 for c in self.alphabet} # for saving/updating counts in cps after cpIval positions

        for i, c in enumerate(bwt, 1):
            counts[c] += 1

            # update the checkpoints dictionary
            if i % cpIval == 0:
                for letter in self.alphabet:
                    self.cps[letter].append(counts[letter])


    def rank(self, c, pos):
        """Count occurrences of c in BWT[0:pos]"""
        if c not in self.cps:
            return 0

        block = pos // self.cpIval # how many checkpoints we have before/untill pos
        start = block * self.cpIval # last checkpoint position before/at our pos

        return self.cps[c][block] + self.bwt.count(c, start, pos)

class FmIndex:

    @staticmethod
    def downsample_sa(sa, n):
        """ Take only the suffix-array entries for every nth suffix.
            Sample SA entries by reference position.
            To save memory."""
        return {
            row: pos
            for row, pos in enumerate(sa)
            if pos % n == 0
        }

    def __init__(self, text, cpIval, ssaIval):
        text = (text.upper() + '$').encode('ascii')

        print("Building suffix array...", flush=True)
        sa = direct_kark_sort(text)
        print("Suffix array constructed.", flush=True)

        self.bwt = bwt_from_sa(text, sa)
        self.slen = len(text)

        self.ssa = self.downsample_sa(sa, ssaIval)
        self.cps = FmCheckpoints(self.bwt, cpIval)

        self.first = {}
        counts = {c: self.bwt.count(c) for c in set(self.bwt)}

        total = 0
        for c in sorted(counts):
            self.first[c] = total
            total += counts[c]


    def count(self, c):
        """Number of characters lexicographically smaller than c."""
        return self.first[c]

    def range(self, pattern):
        """Return half-open SA interval for query [left, right)."""
        pattern = pattern.upper().encode('ascii')

        left, right = 0, self.slen

        for c in reversed(pattern):
            if c not in self.first:
                return 0, 0

            left = self.count(c) + self.cps.rank(c, left)
            right = self.count(c) + self.cps.rank(c, right)

            if left >= right:
                return 0, 0

        return left, right

    def resolve(self, row):
        """Recover reference position using LF-mapping."""
        steps = 0

        while row not in self.ssa:
            c = self.bwt[row]

            # count - number of occurences of all letters smaller than c
            # rank - number of occurences of letter c before row position
            row = self.count(c) + self.cps.rank(c, row)
            steps += 1

        return (self.ssa[row] + steps) % self.slen

    def has_substring(self, pattern):
        left, right = self.range(pattern)
        return left < right

    def has_suffix(self, pattern):
        left, right = self.range(pattern + '$')
        return left < right

    def occurrences(self, pattern):
        left, right = self.range(pattern)
        return [self.resolve(row) for row in range(left, right)]


# 2. Finding seeds

def extract_seeds(read, seed_len, step):
    """Extract regularly spaced seeds and their positions in read."""
    seeds = []

    for pos in range(0, len(read) - seed_len + 1, step):
        seed = read[pos:pos + seed_len]
        seeds.append((pos, seed))

        # if 'N' not in seed:
        #     seeds.append((pos, seed))

    return seeds

def find_seed_hits(fm, seeds, max_hits):
    """Find exact seed matches and estimate read start positions."""
    hits = []

    for read_pos, seed in seeds:
        left, right = fm.range(seed)

        # Skip seeds occurring too often in the reference
        if right - left > max_hits:
            continue

        for row in range(left, right):
            ref_pos = fm.resolve(row)
            cand_start = ref_pos - read_pos # potential start of the read in reference

            hits.append((cand_start, read_pos, ref_pos))

    return hits

def group_seed_hits(hits, max_spread, max_cands):
    """Group seed hits by approximate read start with sliding window."""
    if not hits:
        return []

    # Sort hits by estimated read start
    hits = sorted(hits, key=lambda x: x[0])

    counts = defaultdict(int)
    windows = []
    left = 0 # index of first hit in the window; right last hit in the window

    for right in range(len(hits)):
        start, read_pos, ref_pos = hits[right]
        counts[read_pos] += 1

        # Keep hits within max_spread
        # Delete previous analysed hits that are too far away already from
        # the newest hit position
        # they fall out of sliding window
        while start - hits[left][0] > max_spread:
            old_read_pos = hits[left][1]
            counts[old_read_pos] -= 1

            # removing seed from the dict that
            # does not have now any hits left in window
            if counts[old_read_pos] == 0:
                del counts[old_read_pos] 

            left += 1

        # state of the window
        # len(counts) - number of different seeds that have hits in current window
        windows.append((len(counts), left, right))

    # Highest seed support first
    windows.sort(reverse=True)

    cands = []

    for support, left, right in windows:
        # Median estimated start
        mid = (left + right) // 2
        cand_start = hits[mid][0]

        # Avoid reporting the same location repeatedly
        if any(abs(cand_start - pos) <= max_spread for pos, _ in cands):
            continue

        cands.append((cand_start, support))
        if len(cands) >= max_cands:
            break

    return cands

# 3. Assessing the reads' hits

def banded_align(read, max_err_rate, ref_win, expected_start, band_width):
    """Align entire read to a substring of reference window.
    Return (start, end, editDistance) relative to refWindow.
    refWindow: fragment of the reference where read has hits.

    One NumPy pass per read position; costs and starts are kept in separate arrays.
    """

    r = np.frombuffer(read.encode("ascii"), dtype=np.uint8) # sequence to table of ints; without copying
    g = np.frombuffer(ref_win.encode("ascii"), dtype=np.uint8)

    m = len(read)
    n = len(ref_win)

    inf = np.int64(m + n + 1)
    num_of_errors = int(max_err_rate * m)

    num_of_errors = int(max_err_rate * m)

    # First DP row: free reference prefix
    prev_costs = np.zeros(n+1, dtype=np.int32)  # cost of aligning read prefix to empty reference
    prev_starts = np.arange(n+1, dtype=np.int32)  # Starts of the subalignments
    prev_first_col = 0  # First column of the previous DP row

    for i in range(1, m + 1):
        # i - read positions
        # j - reference positions

        # Restrict DP to a band around expected alignment
        first_col = max(0, i + expected_start - band_width)
        last_col = min(n, i + expected_start + band_width)

        if first_col > last_col:
            return None

        # Costs and starting positions for the current DP row
        cur_costs = np.full(last_col - first_col + 1, inf, dtype=np.int32)
        cur_starts = np.full(last_col - first_col + 1, -1, dtype=np.int32)

        # Match or mismatch (diagonal)
        j = max(first_col, prev_first_col + 1) # first column that has a diagonal
        k = j - first_col # index in current row
        p = j - 1 - prev_first_col # index in previous row
        size = last_col - j + 1

        cur_costs[k:] = prev_costs[p:p + size] + (r[i - 1] != g[j - 1:last_col])
        cur_starts[k:] = prev_starts[p:p+size]

        # gap in reference (move down from previous row)
        p = first_col - prev_first_col
        cost = prev_costs[p:p+len(cur_costs)] + 1
        size = len(cost)

        better = cost < cur_costs[:size]
        cur_costs[:size][better] = cost[better]
        cur_starts[:size][better] = prev_starts[p:p+size][better]

        # Gap in read (move right from previous column)
        k = np.arange(len(cur_costs), dtype=np.int32)
        cost = cur_costs - k
        best_cost = np.minimum.accumulate(cost)
        best_k = np.maximum.accumulate(np.where(cost == best_cost, k, 0)) # where the gap begins

        cur_costs = best_cost + k
        cur_starts = cur_starts[best_k]

        # early stop if the best cost in this row exceeds the allowed number of errors
        if cur_costs.min() > num_of_errors:
            return None

        # current row becomes previous row for the next iteration
        prev_costs = cur_costs
        prev_starts = cur_starts
        prev_first_col = first_col

    # Best alignment endpoint: free reference suffix
    best_idx = int(prev_costs.argmin())

    if prev_costs[best_idx] >= inf:
        return None

    start = int(prev_starts[best_idx])
    end = prev_first_col + best_idx
    editDistance = int(prev_costs[best_idx])   

    return start, end, editDistance


# 4. Mapping a single read

def map_read(fm, reference, read, seed_len, step, max_hits,
             max_spread, max_cands, margin, max_err_rate, band_width):

    """Return (start, end) in reference for best alignment of read, or None if the read cannot be mapped."""

    seeds = extract_seeds(read, seed_len, step)
    hits = find_seed_hits(fm, seeds, max_hits)
    candidates = group_seed_hits(hits, max_spread, max_cands)

    if not candidates:
        return None

    best_support = candidates[0][1] # the highest number of seeds supporting a candidate
    best_alignment = None

    for cand_start, support in candidates:

        # Skip candidates with too few supporting seeds
        if support < best_support / 2:
            continue

        ref_win_start = max(0, cand_start - margin)
        ref_win_end = min(len(reference), cand_start + len(read) + margin)

        # If clamped reference window is smaller than the read, skip this candidate
        if ref_win_end - ref_win_start < len(read):
            continue

        ref_window = reference[ref_win_start:ref_win_end]
        alignment = banded_align(read, max_err_rate, ref_window, expected_start=cand_start - ref_win_start, band_width=band_width)

        if alignment is None:
            continue

        start, end, dist = alignment

        # moving into reference coordinates
        start += ref_win_start
        end += ref_win_start

        if best_alignment is None or dist < best_alignment[2]:
            best_alignment = (start, end, dist)

    if best_alignment is None:
        return None

    start, end, dist = best_alignment

    if dist > max_err_rate * len(read): # too many errors
        return None

    return start, end

def main():
    start = time.perf_counter()

    parser = argparse.ArgumentParser(description="FM-index read mapper")
    parser.add_argument("reference", help="Reference genome FASTA")
    parser.add_argument("reads", help="Reads FASTA file")
    parser.add_argument("output", help="Output file for read mapping results")
    args = parser.parse_args()

    # Mapping parameters - step20 configuration
    seed_len = 15
    step = 20
    max_hits = 100
    max_spread = 100
    max_cands = 10
    margin = 150
    max_err_rate = 0.2
    band_width = 150

    # 1. Read reference
    record = SeqIO.read(args.reference, "fasta")
    reference = str(record.seq).upper()

    # 2. Build FM-index
    fm = FmIndex(reference, cpIval=128, ssaIval=32)

    # 3. Map reads and save output
    n_reads = 0
    n_mapped = 0

    with open(args.output, "w") as out_file:
        for record in SeqIO.parse(args.reads, "fasta"):
            n_reads += 1
            read = str(record.seq).upper()

            result = map_read(
                fm, reference, read,
                seed_len=seed_len,
                step=step,
                max_hits=max_hits,
                max_spread=max_spread,
                max_cands=max_cands,
                margin=margin,
                max_err_rate=max_err_rate,
                band_width=band_width
            )

            if result is not None:
                n_mapped += 1
                res_start, res_end = result
                out_file.write(f"{record.id}\t{res_start}\t{res_end}\n")

    elapsed_time = time.perf_counter() - start

    print(
        f"Mapped {n_mapped}/{n_reads} reads in {elapsed_time:.2f} s "
        f"({elapsed_time / n_reads:.2f} s/read)."
        if n_reads else "No reads found."
    )
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()