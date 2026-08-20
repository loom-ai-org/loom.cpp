#include "loom/core/profile.h"

// ggml's own node-loop helper, `ggml_graph_view`, lives in a PRIVATE header -- it returns a
// `ggml_cgraph` by value, so the complete type is required and `ggml.h`'s opaque forward declaration is
// not enough. This is the one translation unit in the engine that reaches into ggml's `src/`, and the
// CMakeLists adds that directory to this target's include path for it alone.
//
// The intimacy is affordable here specifically because `cmake/GgmlPin.cmake` pins ggml to an exact
// revision: a private header cannot shift under us without somebody deliberately bumping that pin, at
// which point this file is a compile error rather than a silent behaviour change. The alternative --
// routing the CPU path through `ggml_backend_sched` just to borrow its eval callback -- would have made
// the profiler measure a DIFFERENT execution path from the one production uses (a scheduler, its own
// allocator, its own split plan), which defeats the point of profiling.
#include "ggml-impl.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>

namespace loom {
namespace profile {
namespace {

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

struct Bucket {
    double seconds = 0;
    uint64_t calls = 0;
    bool computes = false;   // false for ggml's no-op view ops -- see Totals::floor_seconds
};

// One key per (op, ne0, ne1). A `std::map` rather than an unordered one because the cost that matters
// here is the one paid per NODE, and both are far below a single ggml op; ordered iteration then makes
// the report deterministic without a second sort for equal times.
using Table = std::map<std::pair<std::string, std::pair<int64_t, int64_t>>, Bucket>;

// DELIBERATELY LEAKED, and this is not tidiness lost to laziness. An earlier draft held the table in a
// function-local `static Table` and dumped it from a static destructor; on a shared-library build that
// destructor ran AFTER the table's own, and iterating the corpse allocated a garbage-sized vector and
// threw `std::bad_alloc` out of `__cxa_finalize` -- a crash at exit, in the tool whose whole job is to
// print something at exit. A pointer that is never freed cannot be destroyed early, so `atexit` and any
// later host call both find a live object. The memory is a few kB and the process is ending.
Table& table() {
    static Table* t = new Table();
    return *t;
}

bool& atexit_registered() {
    static bool registered = false;
    return registered;
}

// Whether a report has already gone out for the numbers currently in the table. The atexit handler
// consults it so that a host calling write_report() itself -- which is what loom_cli does, and what any
// host should, since it is the only way to place the report AFTER its own output rather than wherever
// stdio buffering drops it -- does not then get a duplicate at exit. Cleared by reset(), so a host
// profiling two phases separately still gets a report for each.
bool& reported() {
    static bool value = false;
    return value;
}

void write_report_at_exit() {
    if (!reported()) write_report();
}

// `$LOOM_PROFILE`, read once. Empty means "not profiling"; see enabled().
const std::string& spec() {
    static const std::string value = [] {
        const char* raw = std::getenv("LOOM_PROFILE");
        if (raw == nullptr) return std::string();
        std::string v(raw);
        if (v == "0") return std::string();   // an explicit off, so LOOM_PROFILE=0 means what it looks like
        return v;
    }();
    return value;
}

void record(const ggml_tensor* node, double seconds) {
    if (!atexit_registered()) {
        atexit_registered() = true;
        std::atexit(&write_report_at_exit);
    }
    Bucket& b = table()[{ggml_op_name(node->op), {node->ne[0], node->ne[1]}}];
    b.seconds += seconds;
    b.calls += 1;
    // ggml's RESHAPE/VIEW/PERMUTE/TRANSPOSE produce a tensor without touching data, so their timings
    // are dispatch and nothing else. Asking ggml rather than restating the list keeps this true if ggml
    // adds a fifth.
    b.computes = !ggml_op_is_empty(node->op);
}

// The scheduler's eval callback needs somewhere to leave the start timestamp between its `ask` and its
// post-compute call. One per compute() call, on the stack.
struct SchedProbe {
    double started = 0;
};

bool sched_callback(ggml_tensor* t, bool ask, void* user_data) {
    SchedProbe* probe = static_cast<SchedProbe*>(user_data);
    if (ask) {
        // "Yes, I need this node" -- which is what makes the scheduler stop batching and run it alone.
        probe->started = now();
        return true;
    }
    record(t, now() - probe->started);
    return true;   // keep going; false would abandon the rest of the split
}

__attribute__((format(printf, 2, 3)))
void append(std::string& out, const char* fmt, ...) {
    char line[256];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(line, sizeof line, fmt, args);
    va_end(args);
    out += line;
}

} // namespace

bool enabled() { return !spec().empty(); }

ggml_status compute(ggml_backend_t backend, ggml_cgraph* graph) {
    ggml_status worst = GGML_STATUS_SUCCESS;
    const int n = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n; ++i) {
        // A VIEW of the real graph, not a copy of it: same node pointers, same allocated buffers, so
        // node i sees exactly the inputs it would have seen mid-`ggml_backend_graph_compute`.
        ggml_cgraph one = ggml_graph_view(graph, i, i + 1);
        const double t0 = now();
        const ggml_status status = ggml_backend_graph_compute(backend, &one);
        record(ggml_graph_node(graph, i), now() - t0);
        if (status != GGML_STATUS_SUCCESS) worst = status;
    }
    return worst;
}

ggml_status compute(ggml_backend_sched_t sched, ggml_cgraph* graph) {
    SchedProbe probe;
    ggml_backend_sched_set_eval_callback(sched, &sched_callback, &probe);
    const ggml_status status = ggml_backend_sched_graph_compute(sched, graph);
    // Cleared rather than left installed: the scheduler outlives this call (a GraphBuilder owns it for
    // its whole life), and a stale callback would make every later compute run node-by-node and write
    // into a SchedProbe that has gone out of scope.
    ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
    return status;
}

std::vector<Row> rows() {
    std::vector<Row> out;
    out.reserve(table().size());
    for (const auto& entry : table()) {
        Row row;
        row.op = entry.first.first;
        row.ne0 = entry.first.second.first;
        row.ne1 = entry.first.second.second;
        row.seconds = entry.second.seconds;
        row.calls = entry.second.calls;
        out.push_back(std::move(row));
    }
    std::sort(out.begin(), out.end(),
              [](const Row& a, const Row& b) { return a.seconds > b.seconds; });
    return out;
}

Totals totals() {
    Totals t;
    double floor = 0;
    for (const auto& entry : table()) {
        t.seconds += entry.second.seconds;
        t.nodes += entry.second.calls;
        if (!entry.second.computes || entry.second.calls == 0) continue;
        const double per_call = entry.second.seconds / static_cast<double>(entry.second.calls);
        if (floor == 0 || per_call < floor) floor = per_call;
    }
    t.floor_seconds = floor;
    return t;
}

void reset() {
    table().clear();
    reported() = false;
}

std::string report() {
    const std::vector<Row> all = rows();
    const Totals t = totals();
    std::string out;
    if (t.nodes == 0) return "==== loom profile: nothing recorded ====\n";

    append(out, "\n==== loom profile ====\n");
    append(out, "node executions   %10llu\n", static_cast<unsigned long long>(t.nodes));
    append(out, "sum of node time  %10.3f s\n", t.seconds);
    append(out, "per-node floor    %10.3f ms   (dispatch cost; see include/loom/core/profile.h --\n",
           t.floor_seconds * 1e3);
    append(out, "                               profile with ONE thread, or this dominates)\n\n");

    append(out, "%-22s %8s %8s %8s %10s %8s\n", "op", "ne0", "ne1", "calls", "ms", "%");
    double shown = 0;
    for (const Row& r : all) {
        // The tail of a real graph is hundreds of buckets worth microseconds each; printing them buries
        // the handful that matter. Everything down to 98.5% of the time, then stop.
        if (t.seconds > 0 && shown / t.seconds > 0.985) break;
        shown += r.seconds;
        append(out, "%-22s %8lld %8lld %8llu %10.2f %7.1f%%\n", r.op.c_str(),
               static_cast<long long>(r.ne0), static_cast<long long>(r.ne1),
               static_cast<unsigned long long>(r.calls), r.seconds * 1e3,
               t.seconds > 0 ? 100.0 * r.seconds / t.seconds : 0.0);
    }

    std::map<std::string, Bucket> by_op;
    for (const auto& entry : table()) {
        Bucket& b = by_op[entry.first.first];
        b.seconds += entry.second.seconds;
        b.calls += entry.second.calls;
        b.computes = b.computes || entry.second.computes;
    }
    std::vector<std::pair<std::string, Bucket>> ops(by_op.begin(), by_op.end());
    std::sort(ops.begin(), ops.end(), [](const auto& a, const auto& b) {
        return a.second.seconds > b.second.seconds;
    });

    append(out, "\n%-22s %8s %10s %8s %12s\n", "by op", "calls", "ms", "%", "ms - floor");
    for (const auto& op : ops) {
        // `ms - floor` removes this profiler's own per-node cost, which is the difference between a
        // four-thread profile that misleads and one that is merely approximate. Clamped at zero because
        // the floor is the minimum over all ops and a cheap op can sit below its own share of it.
        const double corrected =
            std::max(0.0, op.second.seconds - t.floor_seconds * static_cast<double>(op.second.calls));
        append(out, "%-22s %8llu %10.2f %7.1f%% %12.2f\n", op.first.c_str(),
               static_cast<unsigned long long>(op.second.calls), op.second.seconds * 1e3,
               t.seconds > 0 ? 100.0 * op.second.seconds / t.seconds : 0.0, corrected * 1e3);
    }
    return out;
}

void write_report() {
    if (!enabled()) return;
    reported() = true;
    const std::string text = report();
    // A spec that looks like a path gets a file; anything else ("1", "yes", "on") gets stderr. Deciding
    // by shape rather than by a second variable keeps the common case one word long.
    const std::string& where = spec();
    const bool is_path = where.find('/') != std::string::npos || where.find('.') != std::string::npos;
    if (is_path) {
        if (FILE* f = std::fopen(where.c_str(), "w")) {
            std::fwrite(text.data(), 1, text.size(), f);
            std::fclose(f);
            return;
        }
        // Falling through to stderr rather than failing: losing the profile because a directory does
        // not exist is a worse outcome than printing it somewhere unexpected.
    }
    std::fwrite(text.data(), 1, text.size(), stderr);
}

} // namespace profile
} // namespace loom
