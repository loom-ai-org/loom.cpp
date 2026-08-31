#pragma once

// Per-node timing of the graph the engine actually runs, switched on by `$LOOM_PROFILE` at run time.
//
// WHY THIS EXISTS. "Which op is the model spending its time in" was, until this, a question that could
// only be answered by rebuilding the engine with a hand-rolled hook -- which is exactly how the VITS
// vocaoder investigation was done, and the reason it is worth having permanently is what that
// investigation found: three plausible answers were wrong. The unfused elementwise chain, the C++<->Lua
// array boundary and `GGML_LLAMAFILE` being off were each argued for from the code and each measured at
// a few percent or less; the answer was MUL_MAT at 70%. A profiler is what makes that a five-minute
// question instead of a day.
//
// WHY IT RUNS THE GRAPH NODE BY NODE. ggml offers no per-node hook on the plain
// `ggml_backend_graph_compute` path -- the node loop lives inside the backend, below where a caller can
// reach. So `compute()` below walks the ALREADY-BUILT, ALREADY-ALLOCATED graph one node at a time
// through `ggml_graph_view`, which is the same device `ggml_backend_sched` uses for its own eval
// callback. Nothing about the graph, its allocation or its execution order changes: the profile
// describes the production graph rather than a re-derived model of it.
//
// WHAT IT COSTS, WHICH YOU MUST READ BEFORE BELIEVING A MULTI-THREADED PROFILE. One
// `ggml_backend_graph_compute` per node means one threadpool synchronisation per node. Measured on a
// Raspberry Pi 4 (Cortex-A72, VITS `flow_vocoder` + text encoder + duration predictor, ~2 990 node
// executions):
//
//     threads=1   sum of node times 5.939 s vs a 5.982 s un-profiled run  -- 0.7% overhead, exact
//     threads=4   sum of node times 5.53 s  vs a 2.35 s un-profiled run   -- ~1.4 ms floor PER NODE
//
// At four threads that floor is larger than most of the graph's real per-node work, so it lands mostly
// on whichever op has the MOST nodes rather than the one with the most work -- an attribution that is
// not merely noisy but actively misleading. **Profile with one thread.** `Totals::floor_seconds`
// reports the floor this run actually observed so the report can say how distorted it is, and
// `report()` prints a corrected column, but a corrected four-thread number is an estimate where a
// one-thread number is a measurement.
//
// AND THE SECOND THING IT CHANGES, WHICH IS WORSE THAN THE FLOOR BECAUSE IT IS NOT NOISE.
// A one-node graph does not merely cost a synchronisation -- it is PLANNED DIFFERENTLY. ggml chooses a
// thread count per GRAPH (`cplan.n_threads = MIN(max_tasks, n_threads)` in `ggml_graph_plan`, where
// `max_tasks` is the largest `ggml_get_n_tasks` over the graph's nodes) and there is no per-node
// thread count at all: every thread runs every node, and an op that must stay serial says so itself by
// returning early when `ith != 0`. So a node whose op declares `n_tasks = 1` -- `UNARY` (TANH,
// SIGMOID, EXP, RELU), `SUB`, `SCALE`, `SUM_ROWS`, `LEAKY_RELU`, `SQRT`, `CLAMP` and the rest of that
// list -- is planned at ONE thread when it is alone in a graph, while in the real graph it runs with
// all of them and splits over rows like anything else. In VITS at 4 threads that is 122 `UNARY`, 126
// `SUB`, 106 `SCALE`, 42 `SUM_ROWS` and 32 `LEAKY_RELU` nodes timed on one core.
//
// This is not hypothetical damage. P4.16 read the VITS gate as "30.4 ms per synthesis, identical at
// one thread and four, three cores idle", P4.25 was scoped and built on that reading, and P4.27 was
// opened to explain the 26 ms it did not deliver. The gate was threaded the whole time: forcing those
// unaries serial in the real graph costs a 285K 4.6% of the whole synthesis (p10 1.043, p90 1.060).
// A bucket for an `n_tasks = 1` op is a SINGLE-THREADED measurement sitting in a multi-threaded
// report; compare such an op against itself at one thread, or take it out of the graph and A/B the
// model. See Epic-05 P4.27 and `scripts/bench20.cpp`.
//
// THE ONE THING THIS CANNOT SEE. Only what runs inside a graph. Graph BUILDING, the driver script's own
// host-side loops, and marshalling across the Lua/Python boundaries are all outside it -- measure those
// by timing the call and subtracting this total. (For VITS on the Pi they came to 165 ms of a 2.4 s
// call, so "the profile does not add up to the wall clock" is expected, not a bug.)

#include <ggml.h>
#include <ggml-backend.h>

#include <cstdint>
#include <string>
#include <vector>

namespace loom {
namespace profile {

// Whether `$LOOM_PROFILE` asks for profiling: set, non-empty, and not "0". Read ONCE, on first call --
// so a process cannot half-profile itself by changing the environment mid-run, and so the check on the
// hot path is a load rather than a `getenv`.
//
// The value doubles as the report's destination: "1" (or any other non-path value) means stderr, and
// anything containing a '/' or ending in a filename is opened as a file. See `write_report()`.
bool enabled();

// Whether `$LOOM_PROFILE_NODES` additionally asks for the per-node table -- same "set, non-empty, not
// 0" rule, and read once for the same reasons. Independent of where the report goes; it only adds a
// section to it. Has no effect unless `enabled()`.
bool nodes_enabled();

// Runs `graph` node by node on `backend`, timing each node. Semantically identical to
// `ggml_backend_graph_compute(backend, graph)` -- same nodes, same order, same buffers -- and returns
// what that would have returned. Only meaningful for a graph that has already been allocated.
ggml_status compute(ggml_backend_t backend, ggml_cgraph* graph);

// The scheduled equivalent, for a hybrid (device + CPU fallback) builder, via ggml's own
// `ggml_backend_sched_set_eval_callback`. The scheduler already knows how to run a split one node at a
// time when a callback is installed, so this needs no private header and -- unlike the CPU path -- no
// second implementation of the node loop. The callback is installed per call and cleared after, so a
// non-profiled compute on the same scheduler is untouched.
//
// Costs a `ggml_backend_synchronize` per node on top of the CPU path's threadpool sync, which on a
// discrete device is a full round trip. Treat a device profile as ordering information, not timing.
ggml_status compute(ggml_backend_sched_t sched, ggml_cgraph* graph);

// One (op, shape) bucket. Nodes are bucketed by their op and their two leading dimensions rather than
// held individually: a graph runs the same op at the same shape dozens of times (60 CONV_1D in one VITS
// vocoder), and the useful question is always "what does this op cost me in total", never "what did
// node 337 cost".
struct Row {
    std::string op;      // `ggml_op_name` of the node
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    double seconds = 0;
    uint64_t calls = 0;
};

// Every bucket recorded since the last `reset()`, heaviest first.
std::vector<Row> rows();

// One (op, node name, full shape) bucket -- the finer table, recorded only when `$LOOM_PROFILE_NODES`
// is set. A `Row` cannot say WHICH GRAPH its time came from: it is keyed on `(op, ne0, ne1)` and
// nothing else, so a bucket whose leading dimensions do not happen to identify a phase gets attributed
// by eye. That is not a hypothetical -- whisper's largest layout bucket (`CONT 1500 x 64`) was read as
// the encoder's on exactly that reasoning and is 93% the decode loop's, which inverted the item built
// on it. A node NAME carries its graph (`xv_0 (reshaped) (permuted) (cont)` is a decoder input), and
// all four `ne` distinguish nodes that agree on the leading two.
//
// Off by default because it is a per-node table rather than a per-shape one -- a bigger map on the
// recording path, and a report long enough to bury the summary it sits under.
struct NodeRow {
    std::string op;        // `ggml_op_name` of the node
    std::string name;      // `node->name`, which ggml grows as `(reshaped) (permuted) ...`
    int64_t ne[4] = {0, 0, 0, 0};
    double seconds = 0;
    uint64_t calls = 0;
};

// Every node bucket recorded since the last `reset()`, heaviest first. Empty unless
// `$LOOM_PROFILE_NODES` was set when the first node was recorded.
std::vector<NodeRow> node_rows();

struct Totals {
    double seconds = 0;        // summed over every node execution
    uint64_t nodes = 0;        // node executions, not distinct nodes
    // The smallest per-execution time seen for any node that actually computes something (ggml's
    // RESHAPE/VIEW/PERMUTE/TRANSPOSE do not, and are excluded -- they cost tens of nanoseconds and
    // would report a floor of zero). This is the per-node dispatch cost described in the header
    // comment: at one thread it is microseconds, at four it is the dominant term.
    double floor_seconds = 0;
};
Totals totals();

// Drops everything recorded so far. Useful for excluding a warm-up call, which is worth doing: the
// first run of a graph pays first-touch page faults on every buffer the allocator just handed out.
void reset();

// The formatted table -- per (op, shape) buckets, then a rollup by op. Callers that want the numbers
// rather than the text use `rows()`/`totals()`.
std::string report();

// Writes `report()` where `$LOOM_PROFILE` asked for it: a path if it names one, stderr otherwise. Also
// registered with `atexit` the first time a node is recorded, so a host that never calls it still gets
// its profile -- which is what makes this usable against a SHIPPED wheel, where there is no main() to
// edit.
//
// Call it explicitly when you can, for ORDERING: a host writing results to a block-buffered stdout will
// otherwise see the report land ahead of its own output in a pipe, because the atexit write goes to
// unbuffered stderr after main has returned. The atexit handler skips when a report has already gone
// out for the current numbers, so calling this yourself costs no duplicate; `reset()` re-arms it.
void write_report();

} // namespace profile
} // namespace loom
