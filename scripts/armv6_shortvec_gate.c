// ADR-079 gate 1 (P4.32a): ARE THE VFP11'S SHORT VECTORS SAFE IN THE ENGINE? A kernel that holds
// FPSCR LEN = 4 (scripts/armv6_shortvec.py's harness) is interrupted by signals whose handlers do float
// arithmetic, and shares the one core with threads and a process that do scalar float math. Linux is
// expected to give a handler LEN = 0, restore the interrupted FPSCR on sigreturn, and save FPSCR per
// thread on a switch; this checks it instead of assuming it.
//
//   ./armv6_shortvec_gate runfast [seconds]   RunFast (FZ | DN), as the engine's compute threads run (ADR-078)
//   ./armv6_shortvec_gate ieee [seconds]      IEEE mode, with inputs whose products underflow: every vector
//                                             vmla bounces to the kernel's VFP support code, which must
//                                             emulate all four lanes (the GGML_CPU_RUNFAST=0 path)
//   SV_SABOTAGE=drop ...   the handler clears LEN in the sigframe's saved FPSCR, i.e. a kernel that loses
//                          it on a signal: the kernel's results must go wrong (exit 1)
//   SV_SABOTAGE=leak ...   the handler sets LEN = 4 in a scalar thread's saved FPSCR: its probes must fail
//
// Per thread, a CLOCK_MONOTONIC timer signals that thread every ~0.1 ms. Each worker checks its results
// bit for bit against a scalar reference computed at startup in the same FPSCR mode (every lane of a
// vector vmla is one scalar vmla, in the same k order). Each handler records the live FPSCR it runs with
// and the interrupted one (the VFP block of uc_regspace), runs a probe whose result depends on LEN, C
// float math, and the short-vector kernel itself. A pass needs zero mismatches AND evidence the case
// happened: signals that landed inside sv_kernel with LEN = 4 saved, and involuntary context switches.
//
// Build (loom-armv6-build:bookworm):
//   python3 scripts/armv6_shortvec.py shortvec_gate.S
//   gcc -O2 -march=armv6+fp -mfpu=vfp -mfloat-abi=hard -pthread scripts/armv6_shortvec_gate.c shortvec_gate.S
//       -o armv6_shortvec_gate
// The gate is the board's kernel. qemu-user emulates signals itself and FAILS this: it enters a handler with
// the interrupted LEN still set (2026-10-14, qemu in loom-armv6-build:bookworm), so the handler's own float
// math is vectorized. The interrupted kernel's results stay right there.
#define _GNU_SOURCE
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>

extern void sv_kernel(const float * x, const float * w, float * y, int n, int reps);
extern char sv_kernel_end[];

#define FIELD_MASK 0x370000u    // LEN | STRIDE
#define LEN4       0x030000u
#define RUNFAST    0x03000000u  // FZ | DN
#define VFP_MAGIC  0x56465001u

static unsigned fpscr_get(void) { unsigned f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); return f; }
static void fpscr_set(unsigned f) { __asm__ volatile("vmsr fpscr, %0" : : "r"(f) : "memory"); }

enum { N = 256, NH = 16 };
static int reps;                                 // per call: 32 (runfast) or 2 (ieee, every vmla bounces)
static unsigned mode_bits;                       // FZ | DN or 0
static int sabotage;                             // 0, 1 = drop, 2 = leak
static float X[N * 4] __attribute__((aligned(32))), W[N * 4] __attribute__((aligned(32)));
static float Y0[16], Yref[16];                   // the main problem: reps x N, from Y0
static float Xh[NH * 4], Wh[NH * 4], Yhref[16];  // the handler's: normal values, one rep
static float Cref;

static void reference(const float * x, const float * w, const float * y0, float * y, int n, int r) {
    memcpy(y, y0, 16 * sizeof(float));
    for (int i = 0; i < r; ++i)
        for (int k = 0; k < n; ++k)
            for (int c = 0; c < 4; ++c)
                for (int t = 0; t < 4; ++t) y[c * 4 + t] += x[k * 4 + t] * w[k * 4 + c];
}

// A destination in bank 1 with LEN = 4 would update s16-s19, not s16 alone.
static int probe(void) {
    static const float xs[4] = {2, 3, 4, 5}, ws[1] = {1.5f}, acc[4] = {10, 20, 30, 40};
    float out[4];
    __asm__ volatile(
        "vldmia %1, {s8-s11}\n vldr s0, [%2]\n vldmia %3, {s16-s19}\n"
        "vmla.f32 s16, s8, s0\n"
        "vstmia %0, {s16-s19}\n"
        : : "r"(out), "r"(xs), "r"(ws), "r"(acc)
        : "s0", "s8", "s9", "s10", "s11", "s16", "s17", "s18", "s19", "memory");
    return !(out[0] == 13 && out[1] == 20 && out[2] == 30 && out[3] == 40);
}

static float cmath(void) {
    volatile float a = 0.37f;
    float s = 1, p = 0.25f, q = 3;
    for (int i = 0; i < 64; ++i) { s = s * 0.5f + a * (float) i; p = p * 1.0625f - s * 1e-3f; q += p * s; }
    return s + p + q;
}

// Counters. Handlers run on whichever worker their timer targets; the atomics are lock-free on ARMv6.
static atomic_long n_sig, n_sig_vec, n_sig_in_kernel, n_sig_live_len, n_sig_no_vfp, n_sig_bad;
static atomic_long n_calls, n_bad, n_scalar_checks, n_scalar_bad, n_ivcsw;

static unsigned * saved_fpscr(ucontext_t * uc) {
    unsigned long * p = uc->uc_regspace;
    for (int guard = 0; guard < 8 && p[0]; ++guard) {    // {magic, size} blocks, ended by a zero magic
        if (p[0] == VFP_MAGIC) return (unsigned *) &p[2 + 64];   // after fpregs[32] (u64 each)
        p += p[1] / sizeof(*p);
    }
    return NULL;
}

static __thread int is_scalar_worker;

static void handler(int sig, siginfo_t * si, void * ctx) {
    (void) sig; (void) si;
    ucontext_t * uc = ctx;
    atomic_fetch_add(&n_sig, 1);
    if (fpscr_get() & FIELD_MASK) atomic_fetch_add(&n_sig_live_len, 1);
    unsigned * f = saved_fpscr(uc);
    if (!f) atomic_fetch_add(&n_sig_no_vfp, 1);
    else {
        const uintptr_t pc = uc->uc_mcontext.arm_pc;
        const int in_kernel = pc >= (uintptr_t) sv_kernel && pc < (uintptr_t) sv_kernel_end;
        if ((*f & FIELD_MASK) == LEN4) atomic_fetch_add(&n_sig_vec, 1);
        if ((*f & FIELD_MASK) == LEN4 && in_kernel) atomic_fetch_add(&n_sig_in_kernel, 1);
        if (sabotage == 1 && in_kernel) *f &= ~FIELD_MASK;
        if (sabotage == 2 && is_scalar_worker) *f = (*f & ~FIELD_MASK) | LEN4;
    }
    int bad = probe();
    bad |= cmath() != Cref;
    float y[16];
    memcpy(y, Y0, sizeof y);
    sv_kernel(Xh, Wh, y, NH, 1);
    bad |= memcmp(y, Yhref, sizeof y) != 0;
    bad |= probe();                              // and the kernel left LEN = 0 behind it
    if (bad) atomic_fetch_add(&n_sig_bad, 1);
}

static volatile int stop;

static timer_t arm_timer(long period_ns) {
    struct sigevent sev;
    memset(&sev, 0, sizeof sev);
    sev.sigev_notify = SIGEV_THREAD_ID;
    sev.sigev_signo = SIGRTMIN;
#ifdef sigev_notify_thread_id
    sev.sigev_notify_thread_id = gettid();
#else
    sev._sigev_un._tid = gettid();
#endif
    timer_t t;
    if (timer_create(CLOCK_MONOTONIC, &sev, &t)) { perror("timer_create"); exit(2); }
    struct itimerspec its = {{0, period_ns}, {0, period_ns}};
    timer_settime(t, 0, &its, NULL);
    return t;
}

struct job { int scalar; long period_ns; };

static void * worker(void * arg) {
    const struct job * j = arg;
    fpscr_set((fpscr_get() & ~(FIELD_MASK | RUNFAST)) | mode_bits);
    is_scalar_worker = j->scalar;
    timer_t t = arm_timer(j->period_ns);
    float y[16];
    while (!stop) {
        if (j->scalar) {
            int bad = probe() | (cmath() != Cref);
            atomic_fetch_add(&n_scalar_checks, 1);
            if (bad) atomic_fetch_add(&n_scalar_bad, 1);
        } else {
            memcpy(y, Y0, sizeof y);
            sv_kernel(X, W, y, N, reps);
            atomic_fetch_add(&n_calls, 1);
            if (memcmp(y, Yref, sizeof y) || probe() || (fpscr_get() & FIELD_MASK)) atomic_fetch_add(&n_bad, 1);
        }
    }
    timer_delete(t);
    struct rusage ru;
    getrusage(RUSAGE_THREAD, &ru);
    atomic_fetch_add(&n_ivcsw, ru.ru_nivcsw);
    return NULL;
}

// Two short-vector workers and one scalar one, on staggered periods so they do not fire in lockstep.
static int run(const char * who, int seconds) {
    struct job jobs[3] = {{0, 97000}, {0, 131000}, {1, 113000}};
    pthread_t th[3];
    for (int i = 0; i < 3; ++i) pthread_create(&th[i], NULL, worker, &jobs[i]);
    sleep(seconds);
    stop = 1;
    for (int i = 0; i < 3; ++i) pthread_join(th[i], NULL);
    const long sig = n_sig, vec = n_sig_vec, ink = n_sig_in_kernel, live = n_sig_live_len, novfp = n_sig_no_vfp,
               hbad = n_sig_bad, calls = n_calls, bad = n_bad, sc = n_scalar_checks, sbad = n_scalar_bad,
               ivcsw = n_ivcsw;
    printf("[%s] kernel calls %ld bad %ld | scalar checks %ld bad %ld | signals %ld, landed with LEN 4 %ld "
           "(in sv_kernel %ld), handler LEN != 0 %ld, no VFP frame %ld, handler bad %ld | involuntary switches %ld\n",
           who, calls, bad, sc, sbad, sig, vec, ink, live, novfp, hbad, ivcsw);
    const int ok = !bad && !sbad && !live && !novfp && !hbad && ink > 0 && ivcsw > 0 && calls > 0 && sc > 0;
    printf("[%s] %s\n", who, ok ? "PASS" : "FAIL");
    fflush(stdout);                          // the child leaves through _exit, which does not flush
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    const int ieee = argc > 1 && !strcmp(argv[1], "ieee");
    const int seconds = argc > 2 ? atoi(argv[2]) : 20;
    const char * sab = getenv("SV_SABOTAGE");
    sabotage = sab && !strcmp(sab, "drop") ? 1 : sab && !strcmp(sab, "leak") ? 2 : 0;
    mode_bits = ieee ? 0 : RUNFAST;
    reps = ieee ? 2 : 32;
    fpscr_set((fpscr_get() & ~(FIELD_MASK | RUNFAST)) | mode_bits);

    // ieee: x * w ~ 2^-130..2^-128, below the normal range, so every product is subnormal and accumulates
    // into subnormal sums: the VFP11 bounces each one. runfast: values in [-1, 1].
    uint32_t s = 12345;
    for (int i = 0; i < N * 4; ++i) {
        s = s * 1664525u + 1013904223u; const float a = ((float) (s >> 8) / 16777216.0f) * 2 - 1;
        s = s * 1664525u + 1013904223u; const float b = ((float) (s >> 8) / 16777216.0f) * 2 - 1;
        X[i] = ieee ? a * 0x1p-64f : a;
        W[i] = ieee ? b * 0x1p-64f : b;
    }
    for (int i = 0; i < 16; ++i) Y0[i] = ieee ? (float) (i - 8) * 0x1p-140f : (float) (i - 8) * 0.125f;
    for (int i = 0; i < NH * 4; ++i) { Xh[i] = (float) (i % 7 - 3) * 0.25f; Wh[i] = (float) (i % 5 - 2) * 0.5f; }
    reference(X, W, Y0, Yref, N, reps);
    reference(Xh, Wh, Y0, Yhref, NH, 1);
    Cref = cmath();
    if (probe()) { fprintf(stderr, "probe fails with LEN = 0: the probe is wrong\n"); return 2; }
    {   // the kernel without signals or competition first, so a FAIL below is about them
        float y[16];
        memcpy(y, Y0, sizeof y);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        sv_kernel(X, W, y, N, reps);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        const double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
        printf("mode %s: one call %.3f ms, %.2f ns per MAC; quiet result %s\n", ieee ? "ieee" : "runfast", dt * 1e3,
               dt / (16.0 * N * reps) * 1e9, memcmp(y, Yref, sizeof y) ? "WRONG" : "exact");
        if (memcmp(y, Yref, sizeof y)) return 1;
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGRTMIN, &sa, NULL);

    fflush(stdout);
    const pid_t child = fork();             // a second address space on the same core
    if (child == 0) _exit(run("child", seconds));
    int rc = run("parent", seconds), st = 0;
    waitpid(child, &st, 0);
    rc |= !WIFEXITED(st) || WEXITSTATUS(st);
    printf("%s\n", rc ? "GATE 1 FAIL" : "GATE 1 PASS");
    return rc;
}
