from KarkSand_numpy_final import direct_kark_sort
from array import array
from collections import defaultdict
from Bio import SeqIO
import argparse
import time
import psutil
import os

process = psutil.Process(os.getpid())

def report(stage, start):
    elapsed = time.perf_counter() - start
    ram = process.memory_info().rss / 1024**2
    print(f"{stage}: {elapsed:.2f} s | RAM: {ram:.1f} MB", flush=True)


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

def banded_align(read, max_err_rate, ref_win, expected_start, band_width=150):
    """Align entire read to a substring of reference window.
    Return (start, end, editDistance) relative to refWindow.
    refWindow: fragment of the reference where read has hits.
    """
    m = len(read)
    n = len(ref_win)

    inf = m + n + 1

    num_of_errors = int(max_err_rate * m)

    # First DP row: free reference prefix
    prev_costs = [0] * (n + 1)
    prev_starts = list(range(n + 1))  # Starts of the subalignments
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
        cur_costs = [inf] * (last_col - first_col + 1)
        cur_starts = [-1] * (last_col - first_col + 1)

        for j in range(first_col, last_col + 1):
            k = j - first_col  # Index in cur_costs and cur_starts

            # Align read prefix to empty reference
            if j == 0:
                cur_costs[k] = i  # Number of deletions
                cur_starts[k] = 0
                continue

            best_cost = inf
            best_start = -1

            # Match or mismatch (diagonal)
            p = j - 1 - prev_first_col  # Index in previous row lists

            if 0 <= p < len(prev_costs):
                cost = prev_costs[p] + (read[i - 1] != ref_win[j - 1])

                if cost < best_cost:
                    best_cost = cost
                    best_start = prev_starts[p]

            # Gap in reference (move down from previous row)
            p = j - prev_first_col

            if 0 <= p < len(prev_costs):
                cost = prev_costs[p] + 1

                if cost < best_cost:
                    best_cost = cost
                    best_start = prev_starts[p]

            # Gap in read (move right from previous column)
            if k > 0:
                cost = cur_costs[k - 1] + 1

                if cost < best_cost:
                    best_cost = cost
                    best_start = cur_starts[k - 1]

            cur_costs[k] = best_cost
            cur_starts[k] = best_start

        # early stop if the best cost exceeds the allowed number of errors
        if min(cur_costs) > num_of_errors:
            return None

        # Current row becomes previous row
        prev_costs = cur_costs
        prev_starts = cur_starts
        prev_first_col = first_col

    # Best alignment endpoint: free reference suffix
    best_idx = prev_costs.index(min(prev_costs))

    if prev_costs[best_idx] >= inf:
        return None

    start = prev_starts[best_idx]
    end = prev_first_col + best_idx
    editDistance = prev_costs[best_idx]

    return start, end, editDistance


# 4. Mapping a single read

def map_read(fm, reference, read, seed_len=15, step=10, max_hits=100, 
            max_spread=100, max_cands=10, margin=150, max_err_rate=0.2):

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
        alignment = banded_align(read, max_err_rate, ref_window, expected_start=cand_start - ref_win_start)

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
    parser = argparse.ArgumentParser(description="FM-index read mapper")
    parser.add_argument("reference", help="Reference genome FASTA")
    parser.add_argument("reads", help="Reads FASTA file")
    parser.add_argument("output", help="Output file for read mapping results")
    args = parser.parse_args()

    # 1. Read reference
    start = time.perf_counter()

    record = SeqIO.read(args.reference, "fasta")
    reference = str(record.seq).upper()

    report("Reference loaded", start)


    # 2. Build FM-index
    start = time.perf_counter()

    fm = FmIndex(reference, cpIval=128, ssaIval=32)

    report("FM-index constructed", start)


    # 3. Map reads, save output
    start = time.perf_counter()

    n_reads = 0
    n_mapped = 0

    with open(args.output, "w") as out_file:
        for record in SeqIO.parse(args.reads, "fasta"):
            n_reads += 1
            read = str(record.seq).upper()

            result = map_read(fm, reference, read)

            if result is not None:
                n_mapped += 1
                res_start, res_end = result
                out_file.write(f"{record.id}\t{res_start}\t{res_end}\n")

    report("Mapping completed", start)

    print(f"Mapped {n_mapped}/{n_reads} reads.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()