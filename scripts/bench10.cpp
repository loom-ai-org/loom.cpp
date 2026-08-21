// P4.15's last open question: is a DIRECT 1-D convolution -- no im2col at all -- faster than the
// im2col + GEMM this engine runs? NOT part of the build; a standalone measurement, kept because its
// answer is "in one regime, by a lot, and in the other it is four times slower", and the next person
// to have this idea should get both halves.
//
//   g++ -O3 -std=c++17 -fopenmp -march=armv8-a -I <ggml>/include -I <ggml>/src scripts/bench10.cpp \
//       -o bench10 -L<build>/_deps/ggml-build/src -lggml -lggml-base -lggml-cpu -lgomp -lpthread -lm
//   ./bench10 4
//
// WHY ASK. With the tinyBLAS patches the GEMM has nothing left in it -- 23.5 GFLOP/s in-model against
// 25.1 measured standalone -- but the im2col that feeds it is 37% of all convolution time (phase-timed
// inside ggml's conv: 396 ms of 1070 ms per VITS synthesis). That is not the gather's implementation.
// Making its inner copy branch-free and width-specialised is worth ~10% of it, twice measured. It is
// the gather's EXISTENCE: im2col writes every input element `kw` times, 137 M element-writes per
// synthesis. A direct convolution writes none of them -- it holds a tile of the OUTPUT in registers
// and sweeps the input in place, contiguous loads, contiguous stores, one lane-broadcast weight per
// (channel, tap). That is MLAS's kernel shape minus its assembly.
//
// THE ANSWER, on a Pi 4 at 4 threads over the eleven convolution shapes of a VITS vocoder, best tile
// (4 output channels x 16 positions), padded copy included, against ggml's cache-blocked CONV_2D:
//
//   long activation, few channels   32x32 kw7 L73472   37.7 ms vs 59.3   **1.57x faster**
//                                   32x32 kw5 L73472   30.9 ms vs 46.6     1.51x
//                                   64x64 kw7 L18368   41.0 ms vs 56.3     1.37x
//                                  128x128 kw7 L2296   24.7 ms vs 27.5     1.11x
//   weight-heavy, short activation 192x384 kw5 L287    40.8 ms vs 10.3     0.25x
//                                  768x768 kw3 L100   167   ms vs 24.8     0.15x
//
//   weighted by how often each appears in one synthesis: 0.37x overall -- and 1.15x if each shape
//   takes whichever is faster, worth ~174 ms of the 1156 ms this model spends in convolution.
//
// The split is not subtle and it is not about the kernel. A direct convolution re-reads the WEIGHTS
// once per position block; a GEMM blocks both operands. When the weights are 1.5 MB and the activation
// is 287 positions long, re-reading them 17 times is the whole cost, and no amount of register tuning
// touches it. When the activation is 73472 long and the weights are 28 KB, the direct form wins by not
// materialising 66 MB nobody needed.
//
// TWO THINGS THAT LOOKED LIKE THE ANSWER AND WERE NOT, both worth 3x on their own:
//   * The loop nesting. With the position block INSIDE the output-channel loop -- the obvious way to
//     write it -- every channel block re-streams the whole input from DRAM (8 passes over 9.4 MB for
//     the first shape), and the kernel then measures the SAME time whatever `kw` is, which is the tell:
//     it is not doing arithmetic, it is waiting for memory. Position block outermost: 0.12x -> 0.36x.
//   * The edges. Testing "is this block interior" and sending whole blocks to a scalar path costs a
//     third of a short convolution. Copying the input once into a zero-padded buffer makes every block
//     interior, and that copy is one pass against im2col's `kw`.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <arm_neon.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cstring>
#include <vector>
static double now(){using namespace std::chrono;return duration<double>(steady_clock::now().time_since_epoch()).count();}

// Weights packed [ic][kx][oc] so that the four output channels a tile covers are CONTIGUOUS, which is
// what lets the inner loop use one vector load and four lane-broadcast FMAs instead of four scalar
// loads. Constant per model, so a real engine packs it once at load time; timed separately below.
static void pack_weights(const float* w, float* wp, int64_t IC, int64_t OC, int64_t KW) {
    for (int64_t ic = 0; ic < IC; ++ic)
        for (int64_t kx = 0; kx < KW; ++kx)
            for (int64_t oc = 0; oc < OC; ++oc)
                wp[(ic*KW + kx)*OC + oc] = w[oc*(IC*KW) + ic*KW + kx];
}

// The input, copied once into a zero-padded buffer so that EVERY position block is interior and the
// inner loop needs no bounds test at all. One pass over the activations (~9 MB for the largest shape
// here) against im2col's kw passes -- this is the whole structural argument for a direct convolution.
static void pad_input(const float* x, float* xp, int64_t IC, int64_t L, int64_t pad, int nth) {
    const int64_t LP = L + 2*pad;
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t ic = 0; ic < IC; ++ic) {
        float* dst = xp + ic*LP;
        memset(dst, 0, (size_t)pad*sizeof(float));
        memcpy(dst + pad, x + ic*L, (size_t)L*sizeof(float));
        memset(dst + pad + L, 0, (size_t)pad*sizeof(float));
    }
}

// OCB output channels x (VEC*4) output positions held in registers, over the padded input.
template <int OCB, int VEC>
static void conv1d_direct(const float* xp, const float* wp, float* y,
                          int64_t IC, int64_t OC, int64_t KW, int64_t L, int64_t pad, int nth) {
    const int64_t OL = L, LP = L + 2*pad, P = VEC*4;
    const int64_t nblk = OL / P;                 // whole blocks; the ragged tail is handled after
    // POSITION BLOCK OUTERMOST, output channels inside it. The other way round -- which is the obvious
    // way to write it -- every output-channel block re-streams the entire input from DRAM: for a
    // 32-channel, 73472-long convolution that is 8 passes over 9.4 MB, and the kernel then measures
    // the same time whatever `kw` is, because it is not doing arithmetic, it is waiting for memory.
    // With the position block outside, the slice it needs (P + kw - 1 positions x IC channels, a few
    // KB) stays in L1 across every output channel.
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t b = 0; b < nblk; ++b) {
        for (int64_t oc0 = 0; oc0 < OC; oc0 += OCB) {
            const int64_t p0 = b * P;
            float32x4_t acc[OCB][VEC];
            for (int i = 0; i < OCB; ++i)
                for (int j = 0; j < VEC; ++j) acc[i][j] = vdupq_n_f32(0.0f);
            for (int64_t ic = 0; ic < IC; ++ic) {
                const float* xrow = xp + ic*LP + p0;
                for (int64_t kx = 0; kx < KW; ++kx) {
                    const float* q = xrow + kx;
                    float32x4_t xv[VEC];
                    for (int j = 0; j < VEC; ++j) xv[j] = vld1q_f32(q + j*4);
                    const float32x4_t wv = vld1q_f32(wp + (ic*KW + kx)*OC + oc0);
                    for (int j = 0; j < VEC; ++j) {
                        if (OCB > 0) acc[0][j] = vfmaq_laneq_f32(acc[0][j], xv[j], wv, 0);
                        if (OCB > 1) acc[1][j] = vfmaq_laneq_f32(acc[1][j], xv[j], wv, 1);
                        if (OCB > 2) acc[2][j] = vfmaq_laneq_f32(acc[2][j], xv[j], wv, 2);
                        if (OCB > 3) acc[3][j] = vfmaq_laneq_f32(acc[3][j], xv[j], wv, 3);
                    }
                }
            }
            for (int i = 0; i < OCB; ++i)
                for (int j = 0; j < VEC; ++j) vst1q_f32(y + (oc0+i)*OL + p0 + j*4, acc[i][j]);
        }
    }
    const int64_t done = nblk * P;
    if (done < OL) {                              // ragged tail, scalar: at most P-1 positions
#pragma omp parallel for num_threads(nth) schedule(static)
        for (int64_t oc = 0; oc < OC; ++oc)
            for (int64_t p = done; p < OL; ++p) {
                float acc = 0.0f;
                for (int64_t ic = 0; ic < IC; ++ic)
                    for (int64_t kx = 0; kx < KW; ++kx)
                        acc += wp[(ic*KW + kx)*OC + oc] * xp[ic*LP + p + kx];
                y[oc*OL + p] = acc;
            }
    }
}

struct Conf { int64_t IC, OC, kw, L; int calls; };

int main(int argc,char**argv){
    ggml_backend_load_all();
    ggml_backend_t B = ggml_backend_cpu_init();
    int nth = argc>1?atoi(argv[1]):4;
    ggml_backend_cpu_set_n_threads(B,nth);
    printf("threads=%d\n\n%-22s %5s %10s %9s %9s %9s %9s   %s\n", nth,
           "IC x OC x kw x L","calls","ggml conv","4x16","4x8","2x16","pad","max|rel|");

    Conf cs[] = {
        { 32,  32, 7, 73472, 3}, { 32,  32, 5, 73472, 2}, { 32,  32, 3, 73472, 2},
        { 64,  64, 7, 18368, 2}, { 64,  64, 5, 18368, 2}, { 64,  64, 3, 18368, 2},
        {128, 128, 7,  2296, 2}, {128, 128, 5,  2296, 2}, {128, 128, 3,  2296, 2},
        {192, 384, 5,   287,16}, {768, 768, 3,   100,12},
    };
    double tot_g=0, tot_a=0, tot_b=0, tot_c=0, tot_pack=0, tot_f=0;
    for (auto& c : cs) {
        const int64_t pad = (c.kw - 1) / 2;
        std::vector<float> K((size_t)c.kw*c.IC*c.OC), X((size_t)c.L*c.IC), WP(K.size());
        for (size_t i=0;i<K.size();++i) K[i] = 0.02f - 0.001f*(float)(i%53);
        for (size_t i=0;i<X.size();++i) X[i] = 0.01f + 0.001f*(float)(i%97);

        // ---- ggml's current lowering: GGML_OP_CONV_2D with KH=1 (cache-blocked, in-place write)
        ggml_init_params ip={(size_t)1024*1024*1024,nullptr,true};
        ggml_context* ctx=ggml_init(ip);
        ggml_cgraph* gf=ggml_new_graph(ctx);
        ggml_tensor* tk = ggml_new_tensor_4d(ctx,GGML_TYPE_F32,c.kw,1,c.IC,c.OC);
        ggml_tensor* tx = ggml_new_tensor_4d(ctx,GGML_TYPE_F32,c.L,1,c.IC,1);
        ggml_set_input(tk); ggml_set_input(tx);
        ggml_tensor* r = ggml_conv_2d_direct(ctx,tk,tx,1,1,(int)pad,0,1,1);
        ggml_build_forward_expand(gf,r);
        ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(B));
        if(!ggml_gallocr_alloc_graph(ga,gf)){fprintf(stderr,"alloc fail\n");exit(1);}
        ggml_backend_tensor_set(tk,K.data(),0,K.size()*sizeof(float));
        ggml_backend_tensor_set(tx,X.data(),0,X.size()*sizeof(float));
        ggml_backend_graph_compute(B,gf);
        std::vector<float> ref(ggml_nelements(r));
        ggml_backend_tensor_get(r,ref.data(),0,ref.size()*sizeof(float));
        double t0=now(); for(int i=0;i<3;i++) ggml_backend_graph_compute(B,gf);
        const double tg=(now()-t0)/3;
        ggml_gallocr_free(ga); ggml_free(ctx);

        t0=now(); for(int i=0;i<3;i++) pack_weights(K.data(), WP.data(), c.IC, c.OC, c.kw);
        const double tp=(now()-t0)/3;

        std::vector<float> Y(ref.size());
        std::vector<float> XP((size_t)c.IC*(c.L + 2*pad));
        double t1=now(); for(int i=0;i<3;i++) pad_input(X.data(), XP.data(), c.IC, c.L, pad, nth);
        const double tpad=(now()-t1)/3;
        double ta, tb, tc, md = 0;
        auto run = [&](auto fn) {
            std::fill(Y.begin(), Y.end(), 0.0f);
            fn();
            double t=now(); for(int i=0;i<3;i++) fn(); t=(now()-t)/3;
            for (size_t i=0;i<Y.size();++i) {
                const double d = std::fabs((double)Y[i]-(double)ref[i]);
                const double rr = std::fabs((double)ref[i]) + 1e-6;
                if (d/rr > md) md = d/rr;
            }
            return t;
        };
        ta = run([&]{ conv1d_direct<4,4>(XP.data(),WP.data(),Y.data(),c.IC,c.OC,c.kw,c.L,pad,nth); });
        tb = run([&]{ conv1d_direct<4,2>(XP.data(),WP.data(),Y.data(),c.IC,c.OC,c.kw,c.L,pad,nth); });
        tc = run([&]{ conv1d_direct<2,4>(XP.data(),WP.data(),Y.data(),c.IC,c.OC,c.kw,c.L,pad,nth); });

        const double gf_=2.0*c.kw*c.IC*c.OC*c.L/1e9;
        ta+=tpad; tb+=tpad; tc+=tpad;   // the padded copy is part of what a direct conv costs
        tot_g+=tg*c.calls; tot_a+=ta*c.calls; tot_b+=tb*c.calls; tot_c+=tc*c.calls;
        tot_pack+=tp; tot_f+=gf_*c.calls;
        char buf[64]; snprintf(buf,sizeof buf,"%lld x %lld x %lld x %lld",(long long)c.IC,(long long)c.OC,(long long)c.kw,(long long)c.L);
        printf("%-22s %5d %7.2f ms %6.2f ms %6.2f ms %6.2f ms %6.2f ms   %.1e\n",
               buf,c.calls,tg*1e3,ta*1e3,tb*1e3,tc*1e3,tpad*1e3,md);
    }
    printf("\nweighted total: ggml conv %.3f s | direct 4x16 %.3f s (%.2fx) | 4x8 %.3f s (%.2fx) | 2x16 %.3f s (%.2fx)\n",
           tot_g, tot_a, tot_g/tot_a, tot_b, tot_g/tot_b, tot_c, tot_g/tot_c);
    printf("arithmetic: %.3f GFLOP -> ggml %.1f GFLOP/s, best direct %.1f GFLOP/s   (weight packing, once per model: %.1f ms total)\n",
           tot_f, tot_f/tot_g, tot_f/std::min(std::min(tot_a,tot_b),tot_c), tot_pack*1e3);
    return 0;
}
