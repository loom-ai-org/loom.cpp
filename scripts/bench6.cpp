// Prototype for BACKLOG.md P4.15: is ggml's F32 mul_mat micro-kernel the loom-vs-onnxruntime gap,
// and can an MLAS-style register-blocked kernel close it? Yes and mostly yes -- ~60 lines gets 95% of
// MLAS's rate. NOT part of the build: a standalone measurement, kept because the number it produces is
// the justification for P4.15 and has to stay reproducible.
//
// NONE OF THIS KERNEL SHIPPED, and that is the item's result rather than a change of plan. What closed
// P4.15 was building ggml's own tinyBLAS in (`GGML_LLAMAFILE`, which a standalone ggml defaults OFF)
// and fixing the two things GCC does badly to it on ARM: a 24-accumulator tile it spills, and operand
// addresses it will not strength-reduce (cmake/patches/). `scripts/bench7.cpp` found the first; the
// per-shape columns below found the second. This file is the yardstick both were measured against,
// and the end state is that ggml's own kernel is FASTER than the one below -- 25.1 against 24.3.
//
//   g++ -O3 -std=c++17 -fopenmp -march=armv8-a
//       -I <ggml-src>/include -I <ggml-src>/src scripts/bench6.cpp -o bench6
//       -L <ggml-build>/src -L <ggml-build>/src/ggml-cpu -lggml -lggml-base -lggml-cpu
//       -lgomp -lpthread -lm       (one line; wrapped here for width)
//   ./bench6 4
//
// aarch64 only (NEON intrinsics). On a Pi 4 at 4 threads, expect ggml ~15.0 GFLOP/s against ~24.4 for
// the kernel below, a ratio of ~1.6x; onnxruntime/MLAS measures 25.7 in-model and the A72's fp32 peak
// is 57.6. Materially different numbers mean the measurement is wrong, not the machine -- read P4.15's
// "Picking this up from a cold start" for the three traps, the first of which is why every input here
// is explicitly filled.
//
// WHAT THE DIFFERENCE IS. ggml's F32 path computes ONE output element per `ggml_vec_dot_f32` call. On
// plain NEON that inner loop (GGML_F32_STEP=16, GGML_F32_ARR=4) issues TWO 128-bit loads per 128-bit
// FMA. A72 is load-issue limited well before FMA peak, so that ratio -- not cache, not bandwidth -- is
// the ceiling. That is 1x1 register blocking.
//
// MLAS (github.com/microsoft/MLAS, MIT, src/lib/aarch64/SgemmKernelNeon.S) instead holds a 4-row x
// 16-column accumulator tile in v16-v31 and uses a broadcast-lane FMA, `fmla v.4s, v4.4s, vA.s[lane]`:
// ~5 loads feed 16 FMAs, 0.31 loads/FMA against ggml's 2.0. It needs B pre-packed into 16-column
// panels, which is what MlasSgemmCopyPackB is for.
//
// WHY THIS PROTOTYPE NEEDS NO PACKING. loom's conv mul_mat has BOTH operands K-contiguous (im2col is
// [K, M], the kernel is [K, N]) -- i.e. C = A * B^T with both row-major over K -- so a 4x4 tile of
// dot-product accumulators works directly on the tensors as they already are: 8 loads feed 16 FMAs,
// 0.5 loads/FMA, four times ggml's ratio for no data movement at all. That is the cheapest possible
// test of whether the load/FMA ratio is the whole story, and it turns out to be.
//
// CAVEAT THAT DECIDES WHERE THIS CAN LAND: the result is NOT bit-identical to ggml's (~3e-7 relative,
// a different summation order), so a byte-identity gate over any conv-bearing model has to become a
// tolerance gate first. See CLAUDE.md's tensor-oracle rule and P4.15's own note.
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
#include <vector>
static double now(){using namespace std::chrono;return duration<double>(steady_clock::now().time_since_epoch()).count();}

// C[n*M + m] = sum_k A[m*K + k] * B[n*K + k]   (ggml's dst layout: ne0 = M)
static void sgemm_nt_4x4(int64_t M,int64_t N,int64_t K,
                         const float* A,const float* B,float* C,int nth){
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t m = 0; m < M; m += 4) {
        for (int64_t n = 0; n < N; n += 4) {
            float32x4_t c00=vdupq_n_f32(0),c01=vdupq_n_f32(0),c02=vdupq_n_f32(0),c03=vdupq_n_f32(0);
            float32x4_t c10=vdupq_n_f32(0),c11=vdupq_n_f32(0),c12=vdupq_n_f32(0),c13=vdupq_n_f32(0);
            float32x4_t c20=vdupq_n_f32(0),c21=vdupq_n_f32(0),c22=vdupq_n_f32(0),c23=vdupq_n_f32(0);
            float32x4_t c30=vdupq_n_f32(0),c31=vdupq_n_f32(0),c32=vdupq_n_f32(0),c33=vdupq_n_f32(0);
            const float* a0=A+(m+0)*K; const float* a1=A+(m+1)*K;
            const float* a2=A+(m+2)*K; const float* a3=A+(m+3)*K;
            const float* b0=B+(n+0)*K; const float* b1=B+(n+1)*K;
            const float* b2=B+(n+2)*K; const float* b3=B+(n+3)*K;
            for (int64_t k = 0; k < K; k += 4) {
                float32x4_t av0=vld1q_f32(a0+k), av1=vld1q_f32(a1+k),
                            av2=vld1q_f32(a2+k), av3=vld1q_f32(a3+k);
                float32x4_t bv0=vld1q_f32(b0+k), bv1=vld1q_f32(b1+k),
                            bv2=vld1q_f32(b2+k), bv3=vld1q_f32(b3+k);
                c00=vfmaq_f32(c00,av0,bv0); c01=vfmaq_f32(c01,av0,bv1);
                c02=vfmaq_f32(c02,av0,bv2); c03=vfmaq_f32(c03,av0,bv3);
                c10=vfmaq_f32(c10,av1,bv0); c11=vfmaq_f32(c11,av1,bv1);
                c12=vfmaq_f32(c12,av1,bv2); c13=vfmaq_f32(c13,av1,bv3);
                c20=vfmaq_f32(c20,av2,bv0); c21=vfmaq_f32(c21,av2,bv1);
                c22=vfmaq_f32(c22,av2,bv2); c23=vfmaq_f32(c23,av2,bv3);
                c30=vfmaq_f32(c30,av3,bv0); c31=vfmaq_f32(c31,av3,bv1);
                c32=vfmaq_f32(c32,av3,bv2); c33=vfmaq_f32(c33,av3,bv3);
            }
            float* c = C + n*M + m;
            c[0]=vaddvq_f32(c00); c[1]=vaddvq_f32(c10); c[2]=vaddvq_f32(c20); c[3]=vaddvq_f32(c30);
            c+=M; c[0]=vaddvq_f32(c01); c[1]=vaddvq_f32(c11); c[2]=vaddvq_f32(c21); c[3]=vaddvq_f32(c31);
            c+=M; c[0]=vaddvq_f32(c02); c[1]=vaddvq_f32(c12); c[2]=vaddvq_f32(c22); c[3]=vaddvq_f32(c32);
            c+=M; c[0]=vaddvq_f32(c03); c[1]=vaddvq_f32(c13); c[2]=vaddvq_f32(c23); c[3]=vaddvq_f32(c33);
        }
    }
}

int main(int argc,char**argv){
    ggml_backend_load_all();
    ggml_backend_t B = ggml_backend_cpu_init();
    int nth = argc>1?atoi(argv[1]):4;
    ggml_backend_cpu_set_n_threads(B,nth);
    printf("threads=%d  llamafile=%d\n\n",nth,ggml_cpu_has_llamafile());
    printf("%-26s %11s %11s %11s %10s %11s %10s %9s\n",
           "K x M x N","ggml ms","-floor ms","4x4 ms","ggml GF/s","-floor GF/s","4x4 GF/s","max|diff|");

    // The FIXED cost of one ggml_backend_graph_compute at this thread count: threadpool wake-up, the
    // two ggml_barrier calls tinyBLAS's gemm() brackets its work with, and the dispatch around them.
    // Measured on a node small enough that its arithmetic is nothing (4x16x4 = 512 FLOP), then
    // subtracted from every shape below -- because the question this file answers is how the KERNEL
    // compares, and a per-call constant is not the kernel. On a Pi 4 at 4 threads it is 0.009 ms,
    // i.e. 0.03% of these shapes, which is what retired "ggml's per-op overhead" as the explanation
    // for tinyBLAS trailing the kernel below (BACKLOG.md P4.15). Do not confuse it with P4.14's ~1.4
    // ms floor: that one is the PROFILING path's, one compute per node with its own graph view.
    double floor_s = 0.0;
    {
        ggml_init_params ip={64u*1024*1024,nullptr,true};
        ggml_context* c=ggml_init(ip);
        ggml_cgraph* gf=ggml_new_graph(c);
        ggml_tensor* ta=ggml_new_tensor_2d(c,GGML_TYPE_F32,4,16); ggml_set_input(ta);
        ggml_tensor* tb=ggml_new_tensor_2d(c,GGML_TYPE_F32,4,4);  ggml_set_input(tb);
        ggml_build_forward_expand(gf,ggml_mul_mat(c,ta,tb));
        ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(B));
        ggml_gallocr_alloc_graph(ga,gf);
        std::vector<float> z(64, 0.5f);                      // filled, always -- see the trap above
        ggml_backend_tensor_set(ta,z.data(),0,16*4*sizeof(float));
        ggml_backend_tensor_set(tb,z.data(),0,4*4*sizeof(float));
        for(int i=0;i<20;i++) ggml_backend_graph_compute(B,gf);
        double t0=now(); for(int i=0;i<200;i++) ggml_backend_graph_compute(B,gf);
        floor_s=(now()-t0)/200;
        ggml_gallocr_free(ga); ggml_free(c);
    }
    printf("per-compute floor: %.3f ms\n\n", floor_s*1e3);

    // the real flow_vocoder GEMMs (K=IC*kw, M=OL, N=OC), K/M/N all divisible by 4
    struct S{int64_t K,M,N;};
    S ss[]={{224,73472,32},{160,73472,32},{96,73472,32},{448,18368,64},{320,18368,64},
            {192,18368,64},{896,2296,128},{640,2296,128},{384,2296,128},{960,288,384},{1344,288,256}};
    double tot_g=0, tot_k=0, tot_f=0;
    for (auto& s : ss) {
        std::vector<float> A(s.M*s.K), Bm(s.N*s.K), C(s.M*s.N);
        for (size_t i=0;i<A.size();++i)  A[i]  = 0.01f+0.001f*(float)(i%97);
        for (size_t i=0;i<Bm.size();++i) Bm[i] = 0.02f-0.001f*(float)(i%53);

        // ggml: mul_mat(a=[K,M], b=[K,N]) -> [M,N]
        ggml_init_params ip={1024u*1024*1024,nullptr,true};
        ggml_context* c=ggml_init(ip);
        ggml_cgraph* gf=ggml_new_graph_custom(c,64,false);
        ggml_tensor* ta=ggml_new_tensor_2d(c,GGML_TYPE_F32,s.K,s.M); ggml_set_input(ta);
        ggml_tensor* tb=ggml_new_tensor_2d(c,GGML_TYPE_F32,s.K,s.N); ggml_set_input(tb);
        ggml_tensor* td=ggml_mul_mat(c,ta,tb);
        ggml_build_forward_expand(gf,td);
        ggml_gallocr_t ga=ggml_gallocr_new(ggml_backend_get_default_buffer_type(B));
        if(!ggml_gallocr_alloc_graph(ga,gf)){fprintf(stderr,"alloc fail\n");exit(1);}
        ggml_backend_tensor_set(ta,A.data(),0,A.size()*sizeof(float));
        ggml_backend_tensor_set(tb,Bm.data(),0,Bm.size()*sizeof(float));
        ggml_backend_graph_compute(B,gf);
        std::vector<float> ref(ggml_nelements(td));
        ggml_backend_tensor_get(td,ref.data(),0,ref.size()*sizeof(float));
        double t0=now(); for(int i=0;i<3;i++) ggml_backend_graph_compute(B,gf);
        double gms=(now()-t0)/3;
        ggml_gallocr_free(ga); ggml_free(c);

        sgemm_nt_4x4(s.M,s.N,s.K,A.data(),Bm.data(),C.data(),nth);
        t0=now(); for(int i=0;i<3;i++) sgemm_nt_4x4(s.M,s.N,s.K,A.data(),Bm.data(),C.data(),nth);
        double kms=(now()-t0)/3;

        double md=0;
        for(size_t i=0;i<C.size();++i){ double d=std::fabs((double)C[i]-(double)ref[i]);
                                        double r=std::fabs((double)ref[i])+1e-6;
                                        if(d/r>md) md=d/r; }
        double gf_=2.0*s.K*s.M*s.N/1e9;
        tot_g+=gms; tot_k+=kms; tot_f+=gf_;
        char buf[64]; snprintf(buf,sizeof buf,"%lld x %lld x %lld",(long long)s.K,(long long)s.M,(long long)s.N);
        printf("%-26s %8.2f ms %8.2f ms %8.2f ms %10.1f %11.1f %10.1f %9.1e\n",
               buf,gms*1e3,(gms-floor_s)*1e3,kms*1e3,gf_/gms,gf_/(gms-floor_s),gf_/kms,md);
    }
    const double tot_c = tot_g - 11*floor_s;
    printf("\ntotal: ggml %.3f s (%.1f GFLOP/s)  |  minus floor %.3f s (%.1f GFLOP/s)  |  4x4 kernel %.3f s (%.1f GFLOP/s)\n",
           tot_g, tot_f/tot_g, tot_c, tot_f/tot_c, tot_k, tot_f/tot_k);
    printf("A72 fp32 peak at 1.8 GHz x %d cores: %.1f GFLOP/s\n", nth, 1.8*4*2*nth);
    return 0;
}
