#include "loom/core/asr_long_form.h"

#include <algorithm>

namespace loom {
namespace audio {
namespace {

// `_find_optimal_chunk_size`: the window length in [min, max], searched in `step`s, whose LAST window is
// longest -- i.e. the least padding on the final window. Ties keep the shorter length (the comparison is
// strict), and a length longer than the audio is skipped.
size_t choose_window(size_t total, const LongFormPolicy& p) {
    if (total < p.max_window_samples) return total;
    size_t best = p.min_window_samples;
    size_t best_last = 0;
    for (size_t len = p.min_window_samples; len <= p.max_window_samples; len += p.window_search_step_samples) {
        if (len <= p.overlap_samples) continue;
        if (len > total) continue;
        const size_t step = len - p.overlap_samples;
        const size_t n = (total + step - 1) / step;
        const size_t last = total - step * (n - 1);
        if (last > best_last) {
            best_last = last;
            best = len;
        }
    }
    return best;
}

// `longest_common_subsequence_merge`, returning (i, j, length): where the aligned run starts in `x` and
// in `y`, and how long it is. Despite NeMo's name this is a longest common SUBSTRING table (`lcs[i][j]`
// is the length of the common run ending at x[i-1], y[j-1]), followed by its two repair heuristics.
//
// Ported branch for branch, including the parts that read like accidents, because the point of this file
// is to return NeMo's transcript: in the complete-merge branch the backtracked length is NOT written back
// (the caller gets the run's full length with its backtracked start), and the partial branch's expansion
// counts skips the way NeMo does rather than the way one would design it.
struct Alignment {
    size_t i = 0;
    size_t j = 0;
    size_t length = 0;
};

Alignment longest_common_run(const std::vector<int32_t>& x, const std::vector<int32_t>& y) {
    const size_t m = x.size();
    const size_t n = y.size();
    std::vector<std::vector<size_t>> lcs(m + 1, std::vector<size_t>(n + 1, 0));
    size_t result = 0;
    size_t ri = 0, rj = 0, rlen = 0;
    for (size_t i = 1; i <= m; ++i) {
        for (size_t j = 1; j <= n; ++j) {
            if (x[i - 1] == y[j - 1]) {
                lcs[i][j] = lcs[i - 1][j - 1] + 1;
                // `<=`: a later run of equal length wins, so the alignment found is the LAST longest one.
                if (result <= lcs[i][j]) {
                    result = lcs[i][j];
                    ri = i;
                    rj = j;
                    rlen = result;
                }
            }
        }
    }

    // Signed from here on: the partial branch walks indices down past zero before it stops.
    long i = static_cast<long>(ri);
    long j = static_cast<long>(rj);
    const long M = static_cast<long>(m);
    const long N = static_cast<long>(n);
    const auto at = [&](long a, long b) -> size_t {
        // NeMo indexes a Python list here; an index past the table would raise there. It cannot be
        // reached by the walk below on any input we have found, and reading it as "no run" is the
        // conservative answer if it ever is.
        if (a < 0 || b < 0 || a > M || b > N) return 0;
        return lcs[static_cast<size_t>(a)][static_cast<size_t>(b)];
    };

    if (i == M) {
        // A run that reaches the end of the old buffer: walk it back to where it starts.
        long length = static_cast<long>(rlen);
        while (length >= 0 && i > 0 && j > 0) {
            if (at(i - 1, j - 1) > 0) {
                --length;
                --i;
                --j;
            } else {
                --i;
                --j;
                --length;
                break;
            }
        }
        // `length` is deliberately dropped: NeMo returns the run's full length with this start.
    } else {
        // (1) The leftmost longest run in the new data, scanning the old buffer from its end.
        size_t max_j = 0;
        long max_j_idx = N;
        long i_partial = M;
        long j_partial = -1;
        for (long ii = M; ii >= 0; --ii) {
            for (long jj = 0; jj <= N; ++jj) {
                if (at(ii, jj) > max_j && jj <= max_j_idx) {
                    max_j = at(ii, jj);
                    max_j_idx = jj;
                    i_partial = ii;
                    j_partial = jj;
                }
            }
        }

        // NeMo's MIN_MERGE_SUBSEQUENCE_LEN is 1: a run of one token is not trusted, so nothing is cut.
        if (max_j <= 1) {
            i = i_partial;
            j = 0;
            rlen = 0;
        } else {
            // (2) Extend the run diagonally towards the end of the old buffer, allowing one skipped
            // diagonal per row.
            long j_temp = j_partial + 1;
            long j_exp = 0;
            long j_skip = 0;
            for (long ii = i_partial + 1; ii <= M; ++ii) {
                long j_any_skip = 0;
                for (long jj = j_temp; jj < j_temp + j_skip + 1; ++jj) {
                    if (jj < N + 1) {
                        if (at(ii, jj) == 0) {
                            j_any_skip = 1;
                        } else {
                            j_exp = 1 + j_skip + j_any_skip;
                        }
                    }
                }
                j_skip += j_any_skip;
                ++j_temp;
            }
            j_skip = 0;
            j_partial += j_exp;

            // (3) Backtrack the extended run to where its slice starts, counting skipped diagonals.
            long slice_count = 0;
            while (i_partial > 0 && j_partial > 0) {
                if (at(i_partial, j_partial) == 0) {
                    --j_partial;
                    ++j_skip;
                }
                if (j_partial > 0) {
                    ++slice_count;
                    --i_partial;
                    --j_partial;
                }
            }
            i = std::max<long>(0, i_partial);
            j = std::max<long>(0, j_partial);
            rlen = static_cast<size_t>(slice_count + j_skip);
        }
    }
    return {static_cast<size_t>(std::max<long>(0, i)), static_cast<size_t>(std::max<long>(0, j)), rlen};
}

} // namespace

std::vector<Window> plan_windows(size_t total, const LongFormPolicy& policy) {
    if (!policy.declared() || total < policy.max_window_samples) return {{0, total, total}};
    const size_t len = choose_window(total, policy);
    if (len >= total) return {{0, total, total}};
    const size_t overlap = policy.overlap_samples;
    const size_t step = len - overlap;
    std::vector<Window> windows;
    // `_chunk_waveform`: a window starts wherever more than the overlap is left to cover, so the tail is
    // never a window made only of audio the previous one already heard.
    for (size_t start = 0; start + overlap < total; start += step) {
        const size_t end = std::min(start + len, total);
        windows.push_back({start, end - start, len});
    }
    return windows;
}

std::vector<int32_t> lcs_merge(const std::vector<int32_t>& buffer, const std::vector<int32_t>& data,
                               uint32_t search_tokens) {
    std::vector<int32_t> merged = buffer;
    if (search_tokens == 0 || buffer.empty()) {
        merged.insert(merged.end(), data.begin(), data.end());
        return merged;
    }
    const size_t slice = std::min<size_t>(search_tokens, buffer.size());
    const size_t base = buffer.size() - slice;
    const std::vector<int32_t> tail(buffer.begin() + static_cast<long>(base), buffer.end());
    const Alignment a = longest_common_run(tail, data);
    if (a.length < 1) {
        merged.insert(merged.end(), data.begin(), data.end());
        return merged;
    }
    // `buffer[:i_abs_end] + data[j_after:]`, with Python's clamping slices.
    const size_t keep = std::min(buffer.size(), base + a.i + a.length);
    const size_t skip = std::min(data.size(), a.j + a.length);
    merged.assign(buffer.begin(), buffer.begin() + static_cast<long>(keep));
    merged.insert(merged.end(), data.begin() + static_cast<long>(skip), data.end());
    return merged;
}

std::vector<int32_t> merge_windows(const std::vector<std::vector<int32_t>>& windows,
                                   const LongFormPolicy& policy) {
    if (windows.empty()) return {};
    std::vector<int32_t> merged = windows.front();
    for (size_t w = 1; w < windows.size(); ++w) {
        const std::vector<int32_t>& data = windows[w];
        const size_t head = std::min<size_t>(policy.merge_head_tokens, data.size());
        merged = lcs_merge(merged, std::vector<int32_t>(data.begin(), data.begin() + static_cast<long>(head)),
                           policy.merge_search_tokens);
        merged.insert(merged.end(), data.begin() + static_cast<long>(head), data.end());
    }
    return merged;
}

} // namespace audio
} // namespace loom
