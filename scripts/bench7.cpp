// The bench that answered BACKLOG.md P4.15's open question: WHY does ggml's tinyBLAS run 1.6x slower
// than a naive 4x4 kernel when it accumulates identically and its 4x6 tile has the BETTER load/FMA
// ratio on paper (10 loads / 24 FMAs = 0.42, against 8 / 16 = 0.50)? NOT part of the build: a
// standalone measurement, kept because the answer it produced is the justification for the ggml patch
// in cmake/patches/ and has to stay reproducible.
//
//   g++ -O3 -std=c++17 -fopenmp -march=armv8-a scripts/bench7.cpp -o bench7 && ./bench7 4
//
// aarch64 only (NEON intrinsics), and it needs no ggml at all -- that is the point. `./bench6 1` had
// already shown the gap WIDENS at one thread (1.78x against 1.55x at four), which rules out ggml's
// threadpool, its barriers and its chunk scheduling; so this file lifts tinyBLAS's `gemm_bloc` out of
// `llamafile/sgemm.cpp` verbatim and runs it under the SAME OpenMP driver, over the same buffers, as
// bench6's 4x4 prototype. The tile is then the only variable left.
//
// THE ANSWER, on a Pi 4 at 4 threads over the eleven real flow_vocoder GEMM shapes (GFLOP/s):
//
//   4x3  24.2 | 4x4  24.5 | 4x5  18.4 | 4x6  16.3 | 8x4  16.3 | 4x8  14.6
//
// with `objdump -d` showing q-register spill stores of 0, 0, 8, 10, 32, 29 in the same order. gcc 14.2
// does not hold a 24-accumulator tile on aarch64 -- it keeps the `Cv[][]` array's canonical copy in
// memory and stores all 24 accumulators to the stack on EVERY k iteration, twelve `stp q` against
// twenty-four `fmla` -- and the A72's single store pipe then costs about what the arithmetic does. The
// tile that fits wins; the one with the better load/FMA ratio does not. tinyBLAS picks 4x6 on ARM
// because `VECTOR_REGISTERS == 32` says the register FILE is big enough, which it is; the allocator is
// the constraint, and that is a fact about the compiler, readable only out of the object file.
//
// Controls worth keeping, because each kills an explanation that sounds right:
//   * `tb 4x4` is tinyBLAS's own array-of-vectors code at a smaller tile and measures 24.4, identical
//     to bench6's 16 NAMED accumulators -- so the array form is not the problem, only its size is.
//   * `4x6 named` writes the same 24 accumulators as named variables and still spills (5 stores,
//     22.5 GFLOP/s) -- so it is not the array syntax either.
//   * `4x3 ggml-jobs` / `4x4 ggml-jobs` run the same tile under a copy of tinyBLAS's OWN work
//     partitioning -- 16-row jobs from a shared atomic instead of OpenMP's static split -- and measure
//     22.7 against 22.9, i.e. the scheduling is worth about 1%. That is what ruled partitioning out as
//     the reason ggml trailed this file, leaving the object code, which is where the answer was: 14
//     instructions an iteration of re-derived addresses (patch 0002).
//
// AND THE COMPILER IS A VARIABLE, not a constant. Everything above is gcc 14.2. Built with clang 19
// (`clang++ -O3 -fopenmp -march=armv8-a`) on the same Pi, NOTHING here spills below 32 accumulators
// and every tile from 4x3 to 4x6 lands within a few percent of 25 GFLOP/s -- so "which tile is fast"
// is a question about the compiler first and the core second. Check the object file before believing
// a tile argument; `objdump -d | grep 'stp.*\[sp'` inside the block is the whole test.
#include <arm_neon.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
static double now(){using namespace std::chrono;return duration<double>(steady_clock::now().time_since_epoch()).count();}

// ---------------------------------------------------------------- bench6's 4x4, named accumulators
static void k_4x4(int64_t M,int64_t N,int64_t K,const float*A,const float*B,float*C,int nth){
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

// ------------------------------------------ tinyBLAS's gemm_bloc, verbatim in structure (arrays)
template <int RM, int RN>
static inline void tb_bloc(int64_t M,int64_t K,const float*A,const float*B,float*C,
                           int64_t ii,int64_t jj){
    float32x4_t Cv[RN][RM] = {};
    for (int64_t l = 0; l < K; l += 4) {
        if constexpr (RM <= RN) {                       // the branch tinyBLAS takes for 4x6
            float32x4_t Av[RM];
            for (int i = 0; i < RM; ++i) Av[i] = vld1q_f32(A + K*(ii+i) + l);
            for (int j = 0; j < RN; ++j) {
                float32x4_t Bv = vld1q_f32(B + K*(jj+j) + l);
                for (int i = 0; i < RM; ++i) Cv[j][i] = vfmaq_f32(Cv[j][i], Av[i], Bv);
            }
        } else {
            float32x4_t Bv[RN];
            for (int j = 0; j < RN; ++j) Bv[j] = vld1q_f32(B + K*(jj+j) + l);
            for (int i = 0; i < RM; ++i) {
                float32x4_t Av = vld1q_f32(A + K*(ii+i) + l);
                for (int j = 0; j < RN; ++j) Cv[j][i] = vfmaq_f32(Cv[j][i], Av, Bv[j]);
            }
        }
    }
    for (int j = 0; j < RN; ++j)
        for (int i = 0; i < RM; ++i) C[M*(jj+j) + (ii+i)] = vaddvq_f32(Cv[j][i]);
}


// ------ 4x6 with 24 NAMED accumulators: is the spill the array, or just 24 live vectors?
static void k_4x6_named(int64_t M,int64_t N,int64_t K,const float*A,const float*B,float*C,int nth){
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t m = 0; m < M; m += 4) {
        for (int64_t n = 0; n + 6 <= N; n += 6) {
            float32x4_t c00=vdupq_n_f32(0),c01=vdupq_n_f32(0),c02=vdupq_n_f32(0),c03=vdupq_n_f32(0),c04=vdupq_n_f32(0),c05=vdupq_n_f32(0);
            float32x4_t c10=vdupq_n_f32(0),c11=vdupq_n_f32(0),c12=vdupq_n_f32(0),c13=vdupq_n_f32(0),c14=vdupq_n_f32(0),c15=vdupq_n_f32(0);
            float32x4_t c20=vdupq_n_f32(0),c21=vdupq_n_f32(0),c22=vdupq_n_f32(0),c23=vdupq_n_f32(0),c24=vdupq_n_f32(0),c25=vdupq_n_f32(0);
            float32x4_t c30=vdupq_n_f32(0),c31=vdupq_n_f32(0),c32=vdupq_n_f32(0),c33=vdupq_n_f32(0),c34=vdupq_n_f32(0),c35=vdupq_n_f32(0);
            const float* a0=A+(m+0)*K; const float* a1=A+(m+1)*K;
            const float* a2=A+(m+2)*K; const float* a3=A+(m+3)*K;
            const float* b0=B+(n+0)*K; const float* b1=B+(n+1)*K; const float* b2=B+(n+2)*K;
            const float* b3=B+(n+3)*K; const float* b4=B+(n+4)*K; const float* b5=B+(n+5)*K;
            for (int64_t k = 0; k < K; k += 4) {
                float32x4_t av0=vld1q_f32(a0+k), av1=vld1q_f32(a1+k),
                            av2=vld1q_f32(a2+k), av3=vld1q_f32(a3+k), bv;
                bv=vld1q_f32(b0+k); c00=vfmaq_f32(c00,av0,bv); c10=vfmaq_f32(c10,av1,bv); c20=vfmaq_f32(c20,av2,bv); c30=vfmaq_f32(c30,av3,bv);
                bv=vld1q_f32(b1+k); c01=vfmaq_f32(c01,av0,bv); c11=vfmaq_f32(c11,av1,bv); c21=vfmaq_f32(c21,av2,bv); c31=vfmaq_f32(c31,av3,bv);
                bv=vld1q_f32(b2+k); c02=vfmaq_f32(c02,av0,bv); c12=vfmaq_f32(c12,av1,bv); c22=vfmaq_f32(c22,av2,bv); c32=vfmaq_f32(c32,av3,bv);
                bv=vld1q_f32(b3+k); c03=vfmaq_f32(c03,av0,bv); c13=vfmaq_f32(c13,av1,bv); c23=vfmaq_f32(c23,av2,bv); c33=vfmaq_f32(c33,av3,bv);
                bv=vld1q_f32(b4+k); c04=vfmaq_f32(c04,av0,bv); c14=vfmaq_f32(c14,av1,bv); c24=vfmaq_f32(c24,av2,bv); c34=vfmaq_f32(c34,av3,bv);
                bv=vld1q_f32(b5+k); c05=vfmaq_f32(c05,av0,bv); c15=vfmaq_f32(c15,av1,bv); c25=vfmaq_f32(c25,av2,bv); c35=vfmaq_f32(c35,av3,bv);
            }
            float* c = C + n*M + m;
            #define ST(j,x0,x1,x2,x3) { float* q = c + (j)*M; q[0]=vaddvq_f32(x0); q[1]=vaddvq_f32(x1); q[2]=vaddvq_f32(x2); q[3]=vaddvq_f32(x3); }
            ST(0,c00,c10,c20,c30) ST(1,c01,c11,c21,c31) ST(2,c02,c12,c22,c32)
            ST(3,c03,c13,c23,c33) ST(4,c04,c14,c24,c34) ST(5,c05,c15,c25,c35)
            #undef ST
        }
        for (int64_t n = (N/6)*6; n < N; ++n) {          // scalar tail so N need not divide by 6
            for (int64_t i = 0; i < 4; ++i) {
                float32x4_t acc=vdupq_n_f32(0);
                for (int64_t k=0;k<K;k+=4) acc=vfmaq_f32(acc,vld1q_f32(A+(m+i)*K+k),vld1q_f32(B+n*K+k));
                C[n*M+m+i]=vaddvq_f32(acc);
            }
        }
    }
}

// tinyBLAS's n split: xtiles blocks, the first jj_RN of width RN and the rest of width RN-1.
template <int RM, int RN>
static void k_tiles(int64_t M,int64_t N,int64_t K,const float*A,const float*B,float*C,int nth){
    const int64_t xtiles = (N + RN - 1) / RN;
    const int64_t jj_RN  = xtiles - (xtiles * RN - N);
#pragma omp parallel for num_threads(nth) schedule(static)
    for (int64_t m = 0; m < M; m += RM) {
        int64_t jj = 0;
        for (int64_t b = 0; b < jj_RN; ++b, jj += RN)   tb_bloc<RM, RN  >(M,K,A,B,C,m,jj);
        if constexpr (RN > 1)
            for (int64_t b = jj_RN; b < xtiles; ++b, jj += RN-1) tb_bloc<RM, RN-1>(M,K,A,B,C,m,jj);
    }
}


// ------ the same tile under ggml's WORK PARTITIONING instead of OpenMP's static schedule.
// tinyBLAS's gemm() hands out jobs of BM*RM rows (16 here) from a shared atomic counter, and blocks
// the n range into BN column-tiles; this driver copies that, so the only thing left differing from
// k_tiles above is how the work is split. It is the last candidate for the ~8% that separates ggml's
// patched tinyBLAS (22.2 GFLOP/s) from the same tile run standalone (24.2).
#include <atomic>
#include <thread>
template <int RM, int RN, int BM>
static void k_tiles_ggmljobs(int64_t M,int64_t N,int64_t K,const float*A,const float*B,float*C,int nth){
    const int64_t xtiles = (N + RN - 1) / RN;
    const int64_t jj_RN  = xtiles - (xtiles * RN - N);
    const int64_t ytiles = M / (RM * BM);
    std::atomic<int64_t> next{(int64_t)nth};
    auto worker = [&](int ith) {
        for (int64_t job = ith; job < ytiles; job = next.fetch_add(1)) {
            const int64_t ii = job * RM * BM;
            for (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = 0;
                for (int64_t b = 0; b < jj_RN; ++b, jj += RN)   tb_bloc<RM, RN  >(M,K,A,B,C,ii+bi,jj);
                if constexpr (RN > 1)
                    for (int64_t b = jj_RN; b < xtiles; ++b, jj += RN-1) tb_bloc<RM, RN-1>(M,K,A,B,C,ii+bi,jj);
            }
        }
    };
    std::vector<std::thread> ts;
    for (int i = 1; i < nth; ++i) ts.emplace_back(worker, i);
    worker(0);
    for (auto& t : ts) t.join();
}

int main(int argc,char**argv){
    int nth = argc>1?atoi(argv[1]):4;
    struct S{int64_t K,M,N;};
    S ss[]={{224,73472,32},{160,73472,32},{96,73472,32},{448,18368,64},{320,18368,64},
            {192,18368,64},{896,2296,128},{640,2296,128},{384,2296,128},{960,288,384},{1344,288,256}};
    struct R{const char*name;void(*fn)(int64_t,int64_t,int64_t,const float*,const float*,float*,int);};
    R ks[]={{"4x4 (bench6)",k_4x4},
            {"tb 4x6",      k_tiles<4,6>},
            {"4x6 named",   k_4x6_named},
            {"tb 4x5",      k_tiles<4,5>},
            {"tb 4x4",      k_tiles<4,4>},
            {"tb 4x3",      k_tiles<4,3>},
            {"4x3 ggml-jobs", k_tiles_ggmljobs<4,3,4>},
            {"4x4 ggml-jobs", k_tiles_ggmljobs<4,4,4>},
            {"tb 8x4",      k_tiles<8,4>},
            {"tb 4x8",      k_tiles<4,8>},
            {"tb 8x6",      k_tiles<8,6>}};
    const int NK = sizeof(ks)/sizeof(ks[0]);
    printf("threads=%d   (GFLOP/s; ref = first kernel)\n\n", nth);
    printf("%-26s","K x M x N"); for(auto&r:ks) printf("%14s",r.name); printf("\n");
    std::vector<double> tot(NK,0.0); double tot_f=0;
    for (auto& s : ss) {
        std::vector<float> A(s.M*s.K), Bm(s.N*s.K), C(s.M*s.N), ref;
        for (size_t i=0;i<A.size();++i)  A[i]  = 0.01f+0.001f*(float)(i%97);
        for (size_t i=0;i<Bm.size();++i) Bm[i] = 0.02f-0.001f*(float)(i%53);
        double gf_=2.0*s.K*s.M*s.N/1e9; tot_f+=gf_;
        char buf[64]; snprintf(buf,sizeof buf,"%lld x %lld x %lld",(long long)s.K,(long long)s.M,(long long)s.N);
        printf("%-26s",buf); fflush(stdout);
        for (int i=0;i<NK;++i){
            ks[i].fn(s.M,s.N,s.K,A.data(),Bm.data(),C.data(),nth);
            if (i==0) ref=C; else { double md=0;
                for(size_t j=0;j<C.size();++j){double d=std::fabs((double)C[j]-(double)ref[j]);
                    double r=std::fabs((double)ref[j])+1e-6; if(d/r>md) md=d/r;}
                if (md>1e-5) printf("[MISMATCH %.1e]",md); }
            double t0=now(); for(int r=0;r<3;r++) ks[i].fn(s.M,s.N,s.K,A.data(),Bm.data(),C.data(),nth);
            double ms=(now()-t0)/3; tot[i]+=ms;
            printf("%14.1f",gf_/ms); fflush(stdout);
        }
        printf("\n");
    }
    printf("%-26s","TOTAL"); for(int i=0;i<NK;++i) printf("%14.1f",tot_f/tot[i]); printf("\n");
    printf("A72 fp32 peak at 1.8 GHz x %d cores: %.1f GFLOP/s\n", nth, 1.8*4*2*nth);
    return 0;
}
