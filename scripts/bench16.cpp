// P4.21's FIRST EXPERIMENT: an outer-product GEMM tile, standalone, against `ggml_mul_mat` at
// whisper-small's `QK^T` shape. It is a MEASURING STICK, not a patch -- exactly what
// `scripts/bench6.cpp`'s hand-written 4x4 was, and that one shipped nothing.
//
// NOT part of the build.
//
// THE QUESTION. `tinyBLAS` puts the CONTRACTION in the vector lanes, so every output element ends in
// a horizontal reduction; P4.18 counted that epilogue at a k-independent **18.0 instructions per
// output** (`instructions/output = 0.2355k + 18.0`), which is 54% of everything the core retires at
// `k = 64`. An outer-product tile -- lanes on `m`, one operand broadcast -- issues the SAME number of
// FMAs and never reduces per output, so the 18.0 becomes a per-TILE store instead of a per-OUTPUT
// hsum. Ceiling if the intercept went entirely: 1.76x on the op, against the 1.96x gap to
// onnxruntime.
//
// THE GATE THIS EXISTS TO DECIDE, from Epic-05 §5: if the packed tile is not **~1.5x or better** at
// `m = n = 1500, k = 64, 12 heads`, **with the transpose counted**, the item stops and the number
// goes in Retro-012. So the headline line this prints is the WITH-PACK ratio, not the kernel-only
// one; the kernel-only ratio is printed beside it because the difference between the two IS the pack,
// and the design sketch's "fuse the pack into the producing matmul" is only worth writing if that
// difference is big.
//
// WHAT IT FOUND, 2026-08-29 -- P4.21 IS CLOSED, MEASURED OUT. Kept because it is the harness P4.22
// needs and because a negative result nobody can re-run is a rumour.
//
//   box                       roofline   tinyBLAS dot    outer product    ratio   any perfect kernel
//   Ryzen 3 3250U  (AVX2)      54.6 GF   23.9 (44%)      44.3 (81%)       1.85x   2.24x
//   Core Ultra 9 285K (AVX2)  177.2 GF   116.6 (66%)     161.3 (91%)      1.38x   1.52x
//   Raspberry Pi 4 (NEON)      14.3 GF   7.45 (52%)      9.18 (64%)       1.23x   1.93x
//
// The mechanism worked -- instructions per output 36.2 -> 19.7 on the 285K, the 18.0 intercept gone --
// and it bought 1.38x, because IPC fell 5.02 -> 3.63. The reductions were being issued in slots the
// FMA ports left idle. **An instruction ratio is not a time ratio**, and the last column is how to
// know before scoping: divide the incumbent by the roofline. `hsum(__m256)` is six instructions and
// `hsum(float32x4_t)` is one, which is the whole of the Pi row.
//
// It also found P4.22 on the way past, which is worth more: at `m = 1500` ggml's own arm uses
// 3.65 CPUs to deliver 1.02x, and `m = 1504` makes it 2.75x. Run `arm 1` at m in {1496,1500,1504}
// and threads in {1,4} to see it.
//
// WHY A PACK IS NEEDED AT ALL. `ggml_mul_mat(A, B)` contracts over `ne0` of both, so both operands
// arrive with `k` contiguous -- which is what a dot-product tile wants and the opposite of what an
// outer-product tile wants. Lanes on `m` means `A` transposed to `[m, k]`. `B` needs nothing: it is
// read one scalar at a time and broadcast, and for `QK^T` a 6-column block of it is 1536 contiguous
// bytes. So the pack is real work the current path does not do -- `K` is 4.6 MB per call here and,
// unlike P4.18 item B's `V`, is NOT constant across calls -- but it is 12 copies per encoder run
// rather than per token, which is why it can still pay.
//
// WHAT IS MEASURED, and how the two arms are kept honest:
//   * Both arms compute the same 12 head-slices of the same operands into their own output buffer.
//   * ABBA per rep (ggml, outer, outer, ggml) and the MEDIAN of all reps per arm, so a drifting clock
//     or a warming core lands on both arms equally. Same estimator on both sides -- see Retro-018.
//   * The pack is timed BOTH ways: inside the `outer` arm (which is the gate) and alone (which is the
//     split). The alone number is a separate set of reps, so it is not subtracted from anything.
//   * Correctness is checked against ggml's own output every run. A tile formulation that is 2x and
//     wrong is the failure mode this whole item could plausibly have.
//
// BUILD (same recipe as bench15):
//
//   g++ -O3 -std=c++17 -march=native \
//       -I <ggml-src>/include -I <ggml-src>/src -I <ggml-src>/src/ggml-cpu \
//       scripts/bench16.cpp -o bench16 \
//       -L <ggml-build>/src -L <ggml-build>/src/ggml-cpu -lggml -lggml-base -lggml-cpu -lpthread -lm
//
//   taskset -c 0 ./bench16                      # the gate: k=64 m=n=1500 heads=12 reps=7 threads=1
//   ./bench16 <k> <m> <n> <heads> <reps> <threads> <arm>
//
// `arm` is 0 (both, timed and checked -- the gate) or 1 / 2 to run ONLY ggml / ONLY the outer product
// in a loop, which is what a hardware profiler needs: `perf stat` counts a process, and an ABBA
// harness makes every counter a fifty-fifty average of two kernels. The single-arm modes skip the
// correctness pass as well, because it is a 108 MB sweep at this shape and would land in the counts.
//
//   taskset -c 0 perf stat -e cpu_core/instructions/,cpu_core/cycles/ ./bench16 64 1500 1500 12 7 1 1
//
// `-DNR_TILE=<n>` and `-DMR_VECS=<v>` change the register tile (defaults 6 and 2), and `-DNB_COLS=<c>`
// the column block (default 0 = no blocking, the whole of n at once). They exist because "the lanes
// are on the wrong axis" is the claim under test and the tile shape and loop order are NOT -- if the
// first thing tried is short of the gate, the next question is whether any of it clears it, and a
// naive loop order failing is not the formulation failing.
//
// PIN IT. On the 285K, `taskset -c 0` is a P-core; unpinned, the scheduler will put one arm on an
// E-core and the ratio means nothing. At `threads > 1` both arms are handed the same count, but note
// that ggml's `gemm` chunks its tiles DYNAMICALLY while the loop below partitions statically -- at
// high thread counts that difference is part of what is being measured, so the gate is the
// one-thread number.
//
// THE THRESHOLD SWEEP. `k` is an argument because the design sketch has to dispatch on it: the
// benefit is `18.0 / (0.2355k + 18.0)` -- 54% at k=64, 37% at k=128, 23% at k=256 -- but where it
// stops paying for the pack is exactly what this bench measures and the curve cannot say. Sweep `k`
// and read the crossing off the WITH-PACK column.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------------------------
// The one-vector abstraction the tile is written against. `VLEN` is the LANE COUNT of the machine's
// f32 vector, and here it counts OUTPUT ROWS rather than contraction steps -- which is the entire
// point of the item, and the reason the same source is meaningful on NEON and on AVX2 without being
// the same kernel.
#if defined(__AVX512F__)
  #include <immintrin.h>
  typedef __m512 vec_t;
  enum { VLEN = 16 };
  static inline vec_t vzero()                        { return _mm512_setzero_ps(); }
  static inline vec_t vload(const float * p)         { return _mm512_loadu_ps(p); }
  static inline void  vstore(float * p, vec_t v)     { _mm512_storeu_ps(p, v); }
  static inline vec_t vbcast(float x)                { return _mm512_set1_ps(x); }
  static inline vec_t vfma(vec_t a, vec_t b, vec_t c){ return _mm512_fmadd_ps(a, b, c); }
  static const char * isa_name = "AVX-512";
#elif defined(__AVX2__)
  #include <immintrin.h>
  typedef __m256 vec_t;
  enum { VLEN = 8 };
  static inline vec_t vzero()                        { return _mm256_setzero_ps(); }
  static inline vec_t vload(const float * p)         { return _mm256_loadu_ps(p); }
  static inline void  vstore(float * p, vec_t v)     { _mm256_storeu_ps(p, v); }
  static inline vec_t vbcast(float x)                { return _mm256_set1_ps(x); }
  static inline vec_t vfma(vec_t a, vec_t b, vec_t c){ return _mm256_fmadd_ps(a, b, c); }
  static const char * isa_name = "AVX2";
#elif defined(__ARM_NEON)
  #include <arm_neon.h>
  typedef float32x4_t vec_t;
  enum { VLEN = 4 };
  static inline vec_t vzero()                        { return vdupq_n_f32(0.f); }
  static inline vec_t vload(const float * p)         { return vld1q_f32(p); }
  static inline void  vstore(float * p, vec_t v)     { vst1q_f32(p, v); }
  static inline vec_t vbcast(float x)                { return vdupq_n_f32(x); }
  static inline vec_t vfma(vec_t a, vec_t b, vec_t c){ return vfmaq_f32(c, a, b); }
  static const char * isa_name = "NEON";
#else
  typedef float vec_t;
  enum { VLEN = 1 };
  static inline vec_t vzero()                        { return 0.f; }
  static inline vec_t vload(const float * p)         { return *p; }
  static inline void  vstore(float * p, vec_t v)     { *p = v; }
  static inline vec_t vbcast(float x)                { return x; }
  static inline vec_t vfma(vec_t a, vec_t b, vec_t c){ return a * b + c; }
  static const char * isa_name = "scalar";
#endif

// The register tile: MR output rows (two vectors' worth) by NR output columns. Two vectors rather
// than one because the A-load is then amortised over 2*NR FMAs instead of NR, and rather than four
// because 2*NR accumulators plus 2 A-vectors plus 1 broadcast is 15 live vectors at NR = 6, which is
// the whole AVX2 register file minus one. The NEON file is twice that, but the 4x6 spill P4.15
// measured is the standing warning against assuming a bigger tile is free there.
#ifndef MR_VECS
#define MR_VECS 2
#endif
#ifndef NR_TILE
#define NR_TILE 6
#endif
#ifndef NB_COLS
#define NB_COLS 0
#endif
enum { MR = MR_VECS * VLEN, NR_MAX = NR_TILE };

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------------------------
// THE ROOFLINE, measured rather than assumed. Epic-05's own P4.16 warning is to "rank by the
// machine's peak as well as by the competitor", and this item's 1.76x ceiling was derived from
// INSTRUCTION COUNTS -- which is only a time ceiling if IPC holds, and IPC is exactly what changes
// when the epilogue goes away and the tile becomes FMA-port-bound. So the run prints what one core
// does in dependency-free FMAs, and both arms are quoted as a fraction of it. If the dot-product
// kernel is already at 70% of this number, no formulation can be 1.76x, and that is a fact about the
// machine and not about either kernel.
//
// ACC independent accumulators so nothing is latency-bound; no loads in the loop, so nothing is
// port-bound anywhere but the FMA units.
// An optimisation barrier that emits NOTHING. Without it gcc 14.2 collapses the loop below into a
// handful of SCALAR `vfmadd132ss` -- verified in the disassembly -- and reports a "roofline" seven
// times the machine's, which would have made every percentage on this page a lie. Applied to the
// operands rather than the accumulators, so the 16 dependency chains stay independent.
#if defined(__x86_64__) || defined(__i386__)
  #define VBARRIER(v) asm volatile("" : "+x"(v))
#elif defined(__aarch64__) || defined(__arm__)
  #define VBARRIER(v) asm volatile("" : "+w"(v))
#else
  #define VBARRIER(v) asm volatile("" : "+r"(v))
#endif

__attribute__((noinline))
static double peak_fma_gflops() {
    // ACC + the two operands must FIT: 16 accumulators on AVX2 is 18 live vectors in a 16-register
    // file and the loop spills, which reported 108 GFLOP/s on a machine whose own GEMM does 163.
    // Twelve is comfortably past the ~8 in flight that FMA latency x throughput needs.
    enum { ACC = 12 };
    vec_t c[ACC], a = vbcast(1.000001f), b = vbcast(0.999999f);
    for (int i = 0; i < ACC; ++i) c[i] = vbcast((float) i);
    const int64_t iters = 4000000;
    const double t0 = now();
    for (int64_t l = 0; l < iters; ++l) {
        VBARRIER(a); VBARRIER(b);
        for (int i = 0; i < ACC; ++i) c[i] = vfma(a, b, c[i]);
    }
    const double dt = now() - t0;
    float sink = 0, tmp[VLEN];
    for (int i = 0; i < ACC; ++i) { vstore(tmp, c[i]); for (int v = 0; v < VLEN; ++v) sink += tmp[v]; }
    if (sink == 12345.678f) std::fprintf(stderr, " ");
    return 2.0 * VLEN * ACC * iters / dt / 1e9;
}

// ---------------------------------------------------------------------------------------------
// PACK. `A` arrives `[k, m]` with k contiguous (`A[lda*i + l]`, lda = k) and the tile wants m
// contiguous. Packed PANEL-MAJOR, not as a plain transpose: `Ap[(p*k + l)*MR + r]` puts everything
// one tile reads -- k*MR floats, 4 KB at k=64 on AVX2 -- in one contiguous run. A plain `[m, k]`
// transpose would make the tile stride by m*4 = 6000 bytes between k steps and touch 64 cache lines
// spread over 384 KB per tile, which is a different (and worse) experiment.
//
// The partial last panel is zero-padded, so the kernel never branches on m; the STORE is what has to
// know, below.
static void pack_A(const float * A, int64_t k, int64_t m, float * Ap) {
    const int64_t panels = (m + MR - 1) / MR;
    for (int64_t p = 0; p < panels; ++p) {
        float * dst = Ap + p * k * MR;
        const int64_t rows = std::min<int64_t>(MR, m - p * MR);
        if (rows < MR) std::memset(dst, 0, (size_t) k * MR * sizeof(float));
        for (int64_t r = 0; r < rows; ++r) {
            const float * src = A + (p * MR + r) * k;   // contiguous read, strided write
            for (int64_t l = 0; l < k; ++l) dst[l * MR + r] = src[l];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// THE TILE. No horizontal reduction anywhere in it: `Cv[j]` accumulates MR outputs at once, and the
// epilogue is 2*NR vector stores for MR*NR outputs instead of MR*NR hsums.
template <int NR>
static inline void tile(const float * Ap, const float * B, int64_t ldb, int64_t k,
                        float * C, int64_t ldc, int64_t ii, int64_t jj) {
    // THE OPERAND POINTERS ARE HOISTED, and that is not a style choice. Written against the members
    // as `b[ldb * j + l]`, gcc 14.2 re-derives all NR column addresses every iteration and the block
    // retires ~9 instructions an iteration it does not need: 19.7 instructions per output measured
    // against the ~13.5 the arithmetic asks for, and IPC 3.63 where the dot-product tile it is being
    // compared against gets 5.02. It is the same failure `ggml-0002` documents on aarch64, on the
    // other ISA, and here it is worth ~1.15x.
    const float * bp[NR];
    for (int j = 0; j < NR; ++j) bp[j] = B + ldb * (jj + j);
    const float * ap = Ap;

    vec_t c[NR][MR_VECS];
    for (int j = 0; j < NR; ++j)
        for (int v = 0; v < MR_VECS; ++v) c[j][v] = vzero();

    for (int64_t l = 0; l < k; ++l, ap += MR) {
        vec_t a[MR_VECS];
        for (int v = 0; v < MR_VECS; ++v) a[v] = vload(ap + v * VLEN);
        for (int j = 0; j < NR; ++j) {
            const vec_t bv = vbcast(bp[j][l]);
            for (int v = 0; v < MR_VECS; ++v) c[j][v] = vfma(a[v], bv, c[j][v]);
        }
    }
    for (int j = 0; j < NR; ++j)
        for (int v = 0; v < MR_VECS; ++v) vstore(C + ldc * (jj + j) + ii + v * VLEN, c[j][v]);
}

// One head-slice: C[m, n] = A[k, m]^T-packed x B[k, n], the same arithmetic `llamafile_sgemm` does
// for one `i12` of a batched `ggml_mul_mat`.
//
// Panel outer, columns inner: the 4 KB A panel stays in L1 while the whole of B (384 KB at this
// shape) streams past it once per panel. The other order was not chosen blind -- it keeps B in L1 and
// streams A instead, and A is the operand that was just packed, so this is the order that gets value
// out of the pack.
static void gemm_outer(const float * Ap, const float * B, int64_t ldb, int64_t k,
                       float * C, int64_t ldc, int64_t m, int64_t n,
                       int64_t p0, int64_t p1) {
    // COLUMN BLOCKING. With `NB_COLS = 0` the panel loop is outermost and the whole of B streams past
    // each A panel: at this shape that is 94 panels x 384 KB = 36 MB of B read per head, where the
    // operand is only 384 KB. Blocking n instead re-reads A once per block -- 6 MB at NB = 96 -- and
    // keeps the C band being written down to `m * NB * 4`. It is worth having as a knob rather than a
    // choice because the two boxes have 3 MB and 512 KB of L2 per core, and this is exactly the size
    // that decides between them.
    const int64_t NB = NB_COLS > 0 ? (int64_t) NB_COLS : n;
    for (int64_t jb = 0; jb < n; jb += NB) {
        const int64_t jend = std::min(jb + NB, n);
        for (int64_t p = p0; p < p1; ++p) {
            const float * panel = Ap + p * k * MR;
            const int64_t ii    = p * MR;
            const int64_t rows  = std::min<int64_t>(MR, m - ii);
            int64_t jj = jb;
            if (rows == MR) {
                for (; jj + NR_MAX <= jend; jj += NR_MAX) tile<NR_MAX>(panel, B, ldb, k, C, ldc, ii, jj);
                for (; jj < jend; ++jj) tile<1>(panel, B, ldb, k, C, ldc, ii, jj);
            } else {
                for (; jj < jend; ++jj) {
                    float buf[MR];
                    vec_t c[MR_VECS];
                    for (int v = 0; v < MR_VECS; ++v) c[v] = vzero();
                    const float * b = B + ldb * jj;
                    for (int64_t l = 0; l < k; ++l) {
                        const vec_t bv = vbcast(b[l]);
                        for (int v = 0; v < MR_VECS; ++v)
                            c[v] = vfma(vload(panel + l * MR + v * VLEN), bv, c[v]);
                    }
                    for (int v = 0; v < MR_VECS; ++v) vstore(buf + v * VLEN, c[v]);
                    for (int64_t r = 0; r < rows; ++r) C[ldc * jj + ii + r] = buf[r];
                }
            }
        }
    }
}

int main(int argc, char ** argv) {
    const int64_t k       = argc > 1 ? atoll(argv[1]) : 64;
    const int64_t m       = argc > 2 ? atoll(argv[2]) : 1500;
    const int64_t n       = argc > 3 ? atoll(argv[3]) : 1500;
    const int64_t heads   = argc > 4 ? atoll(argv[4]) : 12;
    const int     reps    = argc > 5 ? atoi(argv[5])  : 7;
    const int     threads = argc > 6 ? atoi(argv[6])  : 1;
    const int     arm     = argc > 7 ? atoi(argv[7])  : 0;   // 0 both, 1 ggml only, 2 outer only

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, threads);

    const size_t operand_bytes = (size_t)(k * m + k * n) * heads * sizeof(float);
    struct ggml_init_params dp = { operand_bytes + 64ull*1024*1024, nullptr, false };
    ggml_context * dctx = ggml_init(dp);
    if (!dctx) { std::fprintf(stderr, "ggml_init failed for %.1f MB\n", operand_bytes/1048576.0); return 1; }

    std::mt19937 rng(17);
    std::normal_distribution<float> dist(0.f, 1.f);
    auto mk = [&](int64_t a, int64_t b, int64_t c) {
        ggml_tensor * t = ggml_new_tensor_3d(dctx, GGML_TYPE_F32, a, b, c);
        float * p = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) p[i] = dist(rng);
        return t;
    };
    ggml_tensor * A = mk(k, m, heads);      // K in whisper's QK^T
    ggml_tensor * B = mk(k, n, heads);      // Q

    struct ggml_init_params gp = { (size_t) 16ull*1024*1024, nullptr, true };
    ggml_context * gc = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph(gc);
    ggml_tensor * Cg = ggml_mul_mat(gc, A, B);
    ggml_build_forward_expand(gf, Cg);
    ggml_gallocr_t al = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(al, gf);

    const int64_t panels = (m + MR - 1) / MR;
    std::vector<float> Ap((size_t) panels * MR * k * heads);
    std::vector<float> Co((size_t) m * n * heads);

    auto arm_pack = [&]() {
        if (threads <= 1) {
            for (int64_t h = 0; h < heads; ++h)
                pack_A((const float *) A->data + h*k*m, k, m, Ap.data() + (size_t) h*panels*MR*k);
            return;
        }
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t) ts.emplace_back([&, t] {
            for (int64_t h = t; h < heads; h += threads)
                pack_A((const float *) A->data + h*k*m, k, m, Ap.data() + (size_t) h*panels*MR*k);
        });
        for (auto & th : ts) th.join();
    };
    auto arm_kernel = [&]() {
        if (threads <= 1) {
            for (int64_t h = 0; h < heads; ++h)
                gemm_outer(Ap.data() + (size_t) h*panels*MR*k, (const float *) B->data + h*k*n, k, k,
                           Co.data() + (size_t) h*m*n, m, m, n, 0, panels);
            return;
        }
        // Static split over the (head, panel) job space. ggml chunks dynamically; see the header.
        const int64_t jobs = heads * panels;
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t) ts.emplace_back([&, t] {
            for (int64_t job = t; job < jobs; job += threads) {
                const int64_t h = job / panels, p = job % panels;
                gemm_outer(Ap.data() + (size_t) h*panels*MR*k, (const float *) B->data + h*k*n, k, k,
                           Co.data() + (size_t) h*m*n, m, m, n, p, p + 1);
            }
        });
        for (auto & th : ts) th.join();
    };
    auto arm_outer = [&]() { arm_pack(); arm_kernel(); };
    auto arm_ggml  = [&]() { ggml_backend_graph_compute(backend, gf); };

    arm_ggml(); arm_outer();                       // warm: first touch of both output buffers

    // Single-arm mode: no check, no ABBA, just the one kernel in a loop for a hardware profiler.
    if (arm != 0) {
        const int iters = reps * 4;
        const double t0 = now();
        for (int i = 0; i < iters; ++i) { if (arm == 1) arm_ggml(); else arm_outer(); }
        const double dt = now() - t0;
        const double fl = 2.0 * m * n * k * heads * iters;
        std::printf("arm=%d (%s)  %d iters  %.3f s  %.2f GFLOP/s  %.3f ms/iter\n",
                    arm, arm == 1 ? "ggml dot" : "outer+pack", iters, dt, fl/dt/1e9, dt/iters*1e3);
        std::printf("FMA floor = m*n*k*heads/%d * iters = %.4g instructions\n",
                    (int) VLEN, (double) m*n*k*heads/VLEN*iters);
        // FNV-1a over the raw output bytes. A SCHEDULING change -- which tile a thread takes, how
        // many rows a job owns -- must not move a single bit, because each output is still one dot
        // product accumulated in the same order; a change that DOES move bits is doing arithmetic
        // differently and every byte-identity gate baseline has to be re-recorded. Print it under
        // both builds and compare (P4.22).
        const unsigned char * bytes = (const unsigned char *)
            (arm == 1 ? (const void *) Cg->data : (const void *) Co.data());
        const size_t nbytes = (size_t) m * n * heads * sizeof(float);
        unsigned long long h = 1469598103934665603ULL;
        for (size_t i = 0; i < nbytes; ++i) { h ^= bytes[i]; h *= 1099511628211ULL; }
        std::printf("output FNV-1a = %016llx  (%zu bytes)\n", h, nbytes);
        return 0;
    }

    // Correctness, before any timing is believed.
    {
        // Scaled by the RMS of the reference, NOT element-wise relative: these outputs are sums of
        // k products of zero-mean normals, so a fair share of them sit near zero and an element-wise
        // ratio there reports 1e-2 for an absolute difference of 1e-5. The scale of the MATRIX is
        // what a dropped term would move.
        const float * ref = (const float *) Cg->data;
        double max_abs = 0, sum_sq = 0;
        for (size_t i = 0; i < Co.size(); ++i) {
            max_abs = std::max(max_abs, std::abs((double) ref[i] - (double) Co[i]));
            sum_sq += (double) ref[i] * (double) ref[i];
        }
        const double rms = std::sqrt(sum_sq / (double) Co.size());
        std::printf("check  max_abs %.3g  rms(ref) %.3g  max_abs/rms %.3g  (f32 order differs; ~1e-6 is fine)\n",
                    max_abs, rms, max_abs / rms);
        if (!(max_abs / rms < 1e-4)) { std::fprintf(stderr, "OUTER-PRODUCT TILE IS WRONG -- times are meaningless\n"); return 1; }
    }

    // ABBA per rep, median per arm.
    std::vector<double> tg, to, tp;
    for (int r = 0; r < reps; ++r) {
        double t0 = now(); arm_ggml();  tg.push_back(now() - t0);
        t0 = now();        arm_outer(); to.push_back(now() - t0);
        t0 = now();        arm_outer(); to.push_back(now() - t0);
        t0 = now();        arm_ggml();  tg.push_back(now() - t0);
    }
    for (int r = 0; r < reps; ++r) { double t0 = now(); arm_pack(); tp.push_back(now() - t0); }
    auto med = [](std::vector<double> & v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; };
    const double g = med(tg), o = med(to), p = med(tp);

    const double peak = peak_fma_gflops();
    const double flop = 2.0 * m * n * k * heads;
    std::printf("\n%s  MR=%d(%dv) NR=%d NB=%d  m=%lld n=%lld k=%lld heads=%lld  threads=%d  reps=%d (ABBA, median)\n",
                isa_name, (int) MR, (int) MR_VECS, (int) NR_MAX, (int) NB_COLS,
                (long long) m, (long long) n, (long long) k, (long long) heads, threads, reps);
    std::printf("  one-core FMA roofline                          %6.2f GFLOP/s   (dependency-free, no loads)\n", peak);
    std::printf("  ggml_mul_mat (tinyBLAS dot)   %8.3f ms   %6.2f GFLOP/s   %4.0f%% of roofline\n",
                g*1e3, flop/g/1e9, 100.0*(flop/g/1e9)/peak);
    std::printf("  outer-product, pack included  %8.3f ms   %6.2f GFLOP/s   %4.0f%% of roofline   ratio %.3fx   <-- THE GATE (>= 1.5x)\n",
                o*1e3, flop/o/1e9, 100.0*(flop/o/1e9)/peak, g/o);
    std::printf("  outer-product, kernel only    %8.3f ms   %6.2f GFLOP/s   %4.0f%% of roofline   ratio %.3fx\n",
                (o-p)*1e3, flop/(o-p)/1e9, 100.0*(flop/(o-p)/1e9)/peak, g/(o-p));
    std::printf("  pack alone                    %8.3f ms   %5.1f%% of the packed arm   (%.1f MB copied)\n",
                p*1e3, 100.0*p/o, (double)(k*m*heads*4)/1048576.0);
    std::printf("  ceiling on THIS machine: a perfect kernel is %.3fx of ggml here (roofline / ggml)\n",
                peak / (flop/g/1e9));
    std::printf("  VERDICT: %s\n", g/o >= 1.5 ? "PASS -- write the patch" : "FAIL -- stop, record in Retro-012");

    ggml_gallocr_free(al);
    ggml_free(gc);
    ggml_free(dctx);
    ggml_backend_free(backend);
    return 0;
}
