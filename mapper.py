from KarkSand import direct_kark_sort
from array import array
from collections import defaultdict
from Bio import SeqIO
import argparse
import time


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
        start = time.perf_counter()
        text = (text.upper() + '$').encode('ascii')
        print(f"Text preparation: {time.perf_counter() - start:.2f} s")

        start = time.perf_counter()
        sa = direct_kark_sort(text)
        print(f"Suffix array: {time.perf_counter() - start:.2f} s")

        start = time.perf_counter()
        self.bwt = bwt_from_sa(text, sa)
        self.slen = len(text)
        print(f"BWT: {time.perf_counter() - start:.2f} s")

        start = time.perf_counter()
        self.ssa = self.downsample_sa(sa, ssaIval)
        print(f"Sampled SA: {time.perf_counter() - start:.2f} s")

        start = time.perf_counter()
        self.cps = FmCheckpoints(self.bwt, cpIval)
        print(f"Checkpoints: {time.perf_counter() - start:.2f} s")

        start = time.perf_counter()
        self.first = {}
        counts = {c: self.bwt.count(c) for c in set(self.bwt)}

        total = 0
        for c in sorted(counts):
            self.first[c] = total
            total += counts[c]

        print(f"C array: {time.perf_counter() - start:.2f} s")


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
        start, readPos, refPos = hits[right]
        counts[readPos] += 1

        # Keep hits within max_spread
        # Delete previous analysed hits that are too far away already from
        # the newest hit position
        # they fall out of sliding window
        while start - hits[left][0] > max_spread:
            oldReadPos = hits[left][1]
            counts[oldReadPos] -= 1

            # removing seed from the dict that
            # does not have now any hits left in window
            if counts[oldReadPos] == 0:
                del counts[oldReadPos] 

            left += 1

        # state of the window
        # len(counts) - number of different seeds that have hits in current window
        windows.append((len(counts), left, right))

    # Highest seed support first
    windows.sort(reverse=True)

    candidates = []

    for support, left, right in windows:
        # Median estimated start
        mid = (left + right) // 2
        candidateStart = hits[mid][0]

        # Avoid reporting the same location repeatedly
        if any(abs(candidateStart - pos) <= max_spread for pos, _ in candidates):
            continue

        candidates.append((candidateStart, support))
        if len(candidates) >= max_cands:
            break

    return candidates

# 3. Assessing the reads' hits

def banded_align(read, ref_win, expected_start, band_width=150):
    """Align entire read to a substring of reference window.
    Return (start, end, editDistance) relative to refWindow.
    refWindow: fragment of the reference where read has hits.
    """
    m = len(read)
    n = len(ref_win)

    inf = m + n + 1

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


def main():
    parser = argparse.ArgumentParser(description="FM-index read mapper")
    parser.add_argument("reference", help="Reference genome FASTA")
    parser.add_argument("reads", help="Reads FASTA file")
    args = parser.parse_args()

    # 1. Read reference
    start = time.perf_counter()
    record = SeqIO.read(args.reference, "fasta")
    reference = str(record.seq).upper()
    print(f"Reading time: {time.perf_counter() - start:.2f} s")

    # 2. Build FM-index
    start = time.perf_counter()
    fm = FmIndex(reference, cpIval=128, ssaIval=32)
    print(f"FM-index construction time: {time.perf_counter() - start:.2f} s")

    # # 3. Process reads
    # print(f"\nProcessing: {args.reads}")

    # seedTime = 0
    # searchTime = 0
    # groupTime = 0

    # totalSeeds = 0
    # totalHits = 0
    # totalCandidates = 0
    # readsWithCandidates = 0
    # nReads = 0

    # startFile = time.perf_counter()

    # for record in SeqIO.parse(args.reads, "fasta"):

    #     nReads += 1
    #     read = str(record.seq).upper()

    #     # Extract seeds
    #     start = time.perf_counter()
    #     seeds = extract_seeds(read, seedLen=20, step=20)
    #     seedTime += time.perf_counter() - start

    #     # Search seed hits in FM-index
    #     start = time.perf_counter()
    #     hits = find_seed_hits(fm, seeds, maxHits=100)
    #     searchTime += time.perf_counter() - start

    #     # Group hits into candidate locations
    #     start = time.perf_counter()
    #     candidates = group_seed_hits(
    #         hits,
    #         maxSpread=100,
    #         maxCandidates=10
    #     )
    #     groupTime += time.perf_counter() - start

    #     # Statistics
    #     totalSeeds += len(seeds)
    #     totalHits += len(hits)
    #     totalCandidates += len(candidates)

    #     if candidates:
    #         readsWithCandidates += 1

    #     # Show examples
    #     if nReads <= 3:
    #         print(f"\nRead: {record.id}")
    #         print(f"  Seeds: {len(seeds)}")
    #         print(f"  Hits: {len(hits)}")
    #         print(f"  Candidates: {candidates[:5]}")

    #     # Progress
    #     if nReads % 100 == 0:
    #         print(f"Processed {nReads} reads")

    # totalTime = time.perf_counter() - startFile

    # # Summary
    # print(f"\n--- Results: {args.reads} ---")
    # print(f"Reads processed: {nReads}")
    # print(f"Reads with candidates: {readsWithCandidates}")
    # print(f"Total seeds: {totalSeeds}")
    # print(f"Total hits: {totalHits}")
    # print(f"Total candidates: {totalCandidates}")

    # print(f"\nSeed extraction: {seedTime:.3f} s")
    # print(f"FM-index searching: {searchTime:.3f} s")
    # print(f"Grouping: {groupTime:.3f} s")
    # print(f"Total read processing: {totalTime:.3f} s")

    # if nReads:
    #     print(f"Average per read: {totalTime / nReads:.4f} s")


if __name__ == "__main__":
    main()