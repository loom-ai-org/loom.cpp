// Which unary ops are worth threading, as a measurement rather than an opinion (P4.25).
//
// ggml hands `n_tasks = 1` to its whole cheap-unary list -- ABS, SGN, NEG, STEP, TANH, ELU, RELU,
// SIGMOID, HARDSWISH, HARDSIGMOID, EXP, SOFTPLUS, EXPM1, FLOOR, CEIL, ROUND, TRUNC -- while GELU and
// SILU get `n_threads` (`ggml_get_n_tasks`, ggml-cpu.c). That list mixes two different kinds of op,
// and this says where the line is:
//
//   * a TRANSCENDENTAL unary (tanhf, expf) is COMPUTE-bound -- tens of cycles per element -- so it
//     scales with cores at any size that is not thrashing the bus;
//   * an ARITHMETIC unary (relu, neg) is MEMORY-bound at every size past L2, and on a Raspberry Pi 4
//     threading a streaming kernel makes it SLOWER, because one core already saturates the bus
//     (scripts/membw.c: 4.56 GB/s at one thread, 3.64 at four).
//
// So if the list is ever threaded, it is the transcendental half of it and not the whole thing.
//
// P4.25 DID BUILD THAT PATCH AND MEASURED IT OUT -- the op goes 3.92x and the model goes 0.5%, see
// Retro-012 -- so nothing in cmake/patches/ depends on this file today. It is kept because the class
// split above is a property of the machine rather than of that patch, and it is the thing to re-run
// before anyone proposes threading a unary op again.
//
//   gcc -O3 -fopenmp -march=native scripts/bench17.c -o bench17 -lm && ./bench17
//
// Both sizes matter and they are the two the models actually have: 220 KB is VITS's WN gate tensor
// ([286, 192], L2-resident on an A72), 9.4 MB is its vocoder's full-rate activation.
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

// Written as four separate loops rather than a function pointer so each one auto-vectorises (or does
// not) exactly as ggml's `vec_unary_op` template does -- a pointer would stop the compiler cold and
// this would measure the indirection instead of the op.
#define UNARY_LOOP(name, expr)                                                                     \
    static void run_##name(long n, float *y, const float *x, int nt) {                             \
        _Pragma("omp parallel for num_threads(nt) schedule(static)")                               \
        for (long i = 0; i < n; i++) { const float v = x[i]; y[i] = (expr); }                      \
    }

// Every op P4.25's patch moved gets a row here, and so do two it deliberately left behind -- an op
// nobody measured is how P4.26 happened.
UNARY_LOOP(tanh,     tanhf(v))                                   // would be threaded
UNARY_LOOP(sigmoid,  1.f / (1.f + expf(-v)))                     // would be threaded
UNARY_LOOP(exp,      expf(v))                                    // would be threaded
UNARY_LOOP(elu,      v > 0.f ? v : expm1f(v))                    // would be threaded
UNARY_LOOP(expm1,    expm1f(v))                                  // would be threaded
UNARY_LOOP(softplus, logf(1.f + expf(v)))                        // would be threaded
UNARY_LOOP(relu,     v > 0.f ? v : 0.f)                          // stays at n_tasks = 1
UNARY_LOOP(neg,      -v)                                         // stays at n_tasks = 1

int main(void) {
    const long sizes[]   = {286L * 192, 73216L * 32};      // VITS's gate tensor, and its full rate
    const char *labels[] = {"220 KB (L2)", "9.4 MB (bus)"};
    const int  threads[] = {1, 2, 4};

    printf("%-10s %-14s %10s %10s %10s %8s\n", "op", "size", "1t ms", "2t ms", "4t ms", "4t/1t");
    for (int s = 0; s < 2; s++) {
        const long n = sizes[s];
        float *x = aligned_alloc(64, n * 4), *y = aligned_alloc(64, n * 4);
        for (long i = 0; i < n; i++) { x[i] = (float) ((i % 2001) - 1000) / 250.f; y[i] = 0; }

        for (int op = 0; op < 8; op++) {
            const char *name = (const char *[]){"tanh", "sigmoid", "exp", "elu",
                                                "expm1", "softplus", "relu", "neg"}[op];
            double ms[3];
            for (int t = 0; t < 3; t++) {
                double best = 1e9;
                // One untimed pass first. Without it the first op at the first thread count pays for
                // faulting `y` in and for the frequency ramp, and reads ~30% slow -- which is a
                // scaling ratio's numerator, so it inflates exactly the number this bench reports.
                // Enough repetitions that the small size is not timing the clock, and best-of because
                // on this board thermal drift only ever makes a run slower.
                const int rep = n < 1000000 ? 200 : 20;
                for (int r = -1; r < rep; r++) {
                    if (r < 0) {   // the warm-up pass, run and discarded
                        switch (op) {
                            case 0: run_tanh(n, y, x, threads[t]); break;
                            case 1: run_sigmoid(n, y, x, threads[t]); break;
                            case 2: run_exp(n, y, x, threads[t]); break;
                            case 3: run_elu(n, y, x, threads[t]); break;
                            case 4: run_expm1(n, y, x, threads[t]); break;
                            case 5: run_softplus(n, y, x, threads[t]); break;
                            case 6: run_relu(n, y, x, threads[t]); break;
                            case 7: run_neg(n, y, x, threads[t]); break;
                        }
                        continue;
                    }
                    const double t0 = now();
                    switch (op) {
                        case 0: run_tanh(n, y, x, threads[t]); break;
                        case 1: run_sigmoid(n, y, x, threads[t]); break;
                        case 2: run_exp(n, y, x, threads[t]); break;
                        case 3: run_elu(n, y, x, threads[t]); break;
                        case 4: run_expm1(n, y, x, threads[t]); break;
                        case 5: run_softplus(n, y, x, threads[t]); break;
                        case 6: run_relu(n, y, x, threads[t]); break;
                        case 7: run_neg(n, y, x, threads[t]); break;
                    }
                    const double dt = now() - t0;
                    if (dt < best) best = dt;
                }
                ms[t] = best * 1e3;
            }
            printf("%-10s %-14s %10.4f %10.4f %10.4f %8.2f\n",
                   name, labels[s], ms[0], ms[1], ms[2], ms[0] / ms[2]);
        }
        free(x); free(y);
    }
    return 0;
}
