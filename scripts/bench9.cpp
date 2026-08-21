// Which lowering a 1-D convolution should get: im2col + one big mul_mat, or ggml's single fused
// convolution op a cache-sized batch at a time. NOT part of the build -- a standalone measurement,
// kept because it is what decides the `#if` in src/ops/primitives_conv.cpp and that decision is
// per-machine, so it has to stay re-runnable on the next machine.
//
//   g++ -O3 -std=c++17 -march=native -I <ggml-src>/include -I <ggml-src>/src scripts/bench9.cpp \
//       -o bench9 -L <build>/src -L <build>/src/ggml-cpu -lggml -lggml-base -lggml-cpu -lpthread -lm
//   ./bench9 4
//
// A is loom's own recipe (ggml_im2col into a full [IC*KW, OL] matrix, then one mul_mat); B is
// GGML_OP_CONV_2D with KH = 1. Both run the eleven convolution shapes of a VITS vocoder, weighted by
// how often each appears in one synthesis, and their outputs are compared -- they should be BIT
// identical, because batching splits the patch axis and never the reduction.
//
// P4.14 measured this at 0.98x and closed it. It is 1.18x on a Cortex-A72 once ggml's implementation
// of B is fixed to size its batch for a cache and to write its GEMM straight into the output
// (cmake/patches/ggml-0004-conv2d-cache-blocked.patch) -- and still 0.87x on an AVX2 Ryzen 3 3250U,
// which is why the lowering is chosen by architecture. Both of those are the LAST line of output;
// per-shape, the win on ARM is 1.10-1.63x on the long-activation convs and neutral on the short ones.
//
// If a number here is far from those, suspect the measurement first: check the box is idle, and see
// BACKLOG.md P4.14's three benchmarking traps, all of which produced a wrong answer that survived a
// write-up.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
static double now(){using namespace std::chrono;return duration<double>(steady_clock::now().time_since_epoch()).count();}

struct Conf { int64_t IC, OC, kw, L; int calls; };

int main(int argc,char**argv){
    ggml_backend_load_all();
    ggml_backend_t B = ggml_backend_cpu_init();
    int nth = argc>1?atoi(argv[1]):4;
    ggml_backend_cpu_set_n_threads(B,nth);
    printf("threads=%d\n\n", nth);
    printf("%-22s %5s %11s %11s %8s   %s\n","IC x OC x kw x L","calls","im2col+mm","conv2d_dir","ratio","max|diff|");

    Conf cs[] = {
        { 32,  32, 7, 73472, 3}, { 32,  32, 5, 73472, 2}, { 32,  32, 3, 73472, 2},
        { 64,  64, 7, 18368, 2}, { 64,  64, 5, 18368, 2}, { 64,  64, 3, 18368, 2},
        {128, 128, 7,  2296, 2}, {128, 128, 5,  2296, 2}, {128, 128, 3,  2296, 2},
        {192, 384, 5,   287,16}, {768, 768, 3,   100,12},
    };
    double tot_a=0, tot_b=0;
    for (auto& c : cs) {
        const int p0 = (int)(c.kw - 1) / 2;   // 'same' padding, dilation 1
        std::vector<float> K((size_t)c.kw*c.IC*c.OC), X((size_t)c.L*c.IC);
        for (size_t i=0;i<K.size();++i) K[i] = 0.02f - 0.001f*(float)(i%53);
        for (size_t i=0;i<X.size();++i) X[i] = 0.01f + 0.001f*(float)(i%97);

        std::vector<float> out_a, out_b;
        double ta=0, tb=0;
        for (int variant = 0; variant < 2; ++variant) {
            ggml_init_params ip={(size_t)1024*1024*1024,nullptr,true};
            ggml_context* ctx=ggml_init(ip);
            ggml_cgraph* gf=ggml_new_graph(ctx);
            ggml_tensor* tk; ggml_tensor* tx; ggml_tensor* r;
            if (variant==0) {                              // loom's im2col + mul_mat
                tk = ggml_new_tensor_3d(ctx,GGML_TYPE_F32,c.kw,c.IC,c.OC);
                tx = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,c.L,c.IC);
                ggml_set_input(tk); ggml_set_input(tx);
                ggml_tensor* im = ggml_im2col(ctx,tk,tx,1,0,p0,0,1,0,false,GGML_TYPE_F32);
                ggml_tensor* im2 = ggml_reshape_2d(ctx,im,im->ne[0],im->ne[2]*im->ne[1]);
                ggml_tensor* k2  = ggml_reshape_2d(ctx,tk,tk->ne[0]*tk->ne[1],tk->ne[2]);
                r = ggml_mul_mat(ctx,im2,k2);
            } else {                                       // GGML_OP_CONV_2D with KH = 1
                tk = ggml_new_tensor_4d(ctx,GGML_TYPE_F32,c.kw,1,c.IC,c.OC);
                tx = ggml_new_tensor_4d(ctx,GGML_TYPE_F32,c.L,1,c.IC,1);
                ggml_set_input(tk); ggml_set_input(tx);
                r = ggml_conv_2d_direct(ctx,tk,tx,1,1,p0,0,1,1);
            }
            ggml_build_forward_expand(gf,r);
            ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(B));
            if(!ggml_gallocr_alloc_graph(ga,gf)){fprintf(stderr,"alloc fail\n");exit(1);}
            ggml_backend_tensor_set(tk,K.data(),0,K.size()*sizeof(float));
            ggml_backend_tensor_set(tx,X.data(),0,X.size()*sizeof(float));
            ggml_backend_graph_compute(B,gf);
            std::vector<float>& out = variant==0 ? out_a : out_b;
            out.resize(ggml_nelements(r));
            ggml_backend_tensor_get(r,out.data(),0,out.size()*sizeof(float));
            double t0=now(); for(int i=0;i<3;i++) ggml_backend_graph_compute(B,gf);
            (variant==0?ta:tb)=(now()-t0)/3;
            ggml_gallocr_free(ga); ggml_free(ctx);
        }
        double md=0;
        if (out_a.size()==out_b.size())
            for(size_t i=0;i<out_a.size();++i){ double d=std::fabs((double)out_a[i]-(double)out_b[i]);
                double rr=std::fabs((double)out_a[i])+1e-6; if(d/rr>md) md=d/rr; }
        else md=-1;
        tot_a += ta*c.calls; tot_b += tb*c.calls;
        char buf[64]; snprintf(buf,sizeof buf,"%lld x %lld x %lld x %lld",(long long)c.IC,(long long)c.OC,(long long)c.kw,(long long)c.L);
        printf("%-22s %5d %8.2f ms %8.2f ms %7.2fx   %.1e\n",buf,c.calls,ta*1e3,tb*1e3,ta/tb,md);
    }
    printf("\nweighted total: im2col+mul_mat %.3f s   conv_2d_direct %.3f s   %.2fx\n", tot_a, tot_b, tot_a/tot_b);
    return 0;
}
