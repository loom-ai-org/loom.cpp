// P4.30d: Metal's `CONV_2D` against the shapes VITS actually issues, one kernel variant per row.
//
// WHY A SEPARATE HARNESS. The end-to-end number for this op comes from `$LOOM_PROFILE` over a whole
// VITS synthesis, which costs a ggml rebuild (the Metal library is embedded) and a 280 ms run per
// idea. That is the wrong loop for choosing a tile. This file runs the SAME six shapes -- read off
// `scripts/conv_census.py`, which is validated against the profile's own bucket counts -- straight
// against a Metal device, so a variant is a recompile of one .mm and a few seconds.
//
//   clang++ -O3 -std=c++17 -fobjc-arc scripts/bench22.mm -o bench22 \
//       -framework Metal -framework Foundation
//   ./bench22 [iters]
//
// EVERY VARIANT IS CHECKED before it is timed, and a variant that disagrees prints its max abs error
// and is not reported as a rate. The oracle is two-level, because a scalar CPU reference for
// `L=73472 IC=32 K=7 OC=32` is half a billion MACs and there are six shapes that size: the STOCK
// kernel is the oracle for every shape, and the stock kernel is itself checked against the scalar CPU
// reference on every shape small enough to afford one. The stock kernel below is a VERBATIM copy of
// ggml v0.19.0's `kernel_conv_2d` (byte strides, uint64 offsets, grid-stride loop and all), so the
// baseline column is the thing actually shipping and not a re-derivation of it -- and `ctest`'s
// `test-backend-ops -o CONV_2D` is what stands behind the stock kernel in turn.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Mirrors `ggml_metal_kargs_conv_2d` field for field, because the whole point is that the winner
// ports into ggml-metal by copy.
struct conv_args {
    uint64_t nb00, nb01, nb02, nb03;
    uint64_t nb10, nb11, nb12, nb13;
    uint64_t nb0,  nb1,  nb2,  nb3;
    int32_t  IW, IH, KW, KH, IC, OC, OW, OH, N;
    int32_t  s0, s1, p0, p1, d0, d1;
};

static const char * kShaderSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

typedef struct {
    ulong nb00, nb01, nb02, nb03;
    ulong nb10, nb11, nb12, nb13;
    ulong nb0,  nb1,  nb2,  nb3;
    int   IW, IH, KW, KH, IC, OC, OW, OH, N;
    int   s0, s1, p0, p1, d0, d1;
} conv_args;

// ---------------------------------------------------------------------------------------------
// A: ggml v0.19.0 as shipped. One thread per output element, IC*KH*KW iterations of two global
// loads and one FMA, no reuse of either operand.
// ---------------------------------------------------------------------------------------------
kernel void conv_2d_stock(
        constant conv_args & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3    tgpg[[threadgroups_per_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]) {

    const uint threads_per_tg = ntg.x * ntg.y * ntg.z;
    const uint tg_index = (tgpig.z * tgpg.y + tgpig.y) * tgpg.x + tgpig.x;
    const uint local_thread = tpitg.z * (ntg.x * ntg.y) + tpitg.y * ntg.x + tpitg.x;
    const uint thread_index = tg_index * threads_per_tg + local_thread;
    const ulong total_threads = (ulong) threads_per_tg * tgpg.x * tgpg.y * tgpg.z;
    const ulong total_outputs = (ulong) args.N * args.OC * args.OH * args.OW;

    for (ulong index = thread_index; index < total_outputs; index += total_threads) {
        ulong tmp = index;

        const int ow = tmp % args.OW; tmp /= args.OW;
        const int oh = tmp % args.OH; tmp /= args.OH;
        const int oc = tmp % args.OC; tmp /= args.OC;
        const int  n = tmp;

        float acc = 0.0f;

        const int base_x = ow*args.s0 - args.p0;
        const int base_y = oh*args.s1 - args.p1;

        int ky_start = 0;
        if (base_y < 0) { ky_start = (-base_y + args.d1 - 1)/args.d1; }
        int ky_end = args.KH;
        const int y_max = args.IH - 1 - base_y;
        if (y_max < 0) { ky_end = ky_start; }
        else if (base_y + (args.KH - 1)*args.d1 >= args.IH) { ky_end = min(ky_end, y_max/args.d1 + 1); }

        int kx_start = 0;
        if (base_x < 0) { kx_start = (-base_x + args.d0 - 1)/args.d0; }
        int kx_end = args.KW;
        const int x_max = args.IW - 1 - base_x;
        if (x_max < 0) { kx_end = kx_start; }
        else if (base_x + (args.KW - 1)*args.d0 >= args.IW) { kx_end = min(kx_end, x_max/args.d0 + 1); }

        if (ky_start < ky_end && kx_start < kx_end) {
            const ulong src_base_n = (ulong) n  * args.nb13;
            const ulong w_base_oc  = (ulong) oc * args.nb03;

            for (int ic = 0; ic < args.IC; ++ic) {
                const ulong src_base_nc = src_base_n + (ulong) ic * args.nb12;
                const ulong w_base_ocic = w_base_oc  + (ulong) ic * args.nb02;

                for (int ky = ky_start; ky < ky_end; ++ky) {
                    const int iy = base_y + ky*args.d1;
                    const ulong src_base_row = src_base_nc + (ulong) iy * args.nb11;
                    const ulong w_base_row   = w_base_ocic + (ulong) ky * args.nb01;

                    for (int kx = kx_start; kx < kx_end; ++kx) {
                        const int ix = base_x + kx*args.d0;
                        const float x = *(device const float *)(src + src_base_row + (ulong) ix * args.nb10);
                        const float w = *(device const float *)(weights + w_base_row + (ulong) kx * args.nb00);
                        acc += x * w;
                    }
                }
            }
        }

        const ulong dst_offs = (ulong) n*args.nb3 + (ulong) oc*args.nb2 + (ulong) oh*args.nb1 + (ulong) ow*args.nb0;
        *(device float *)(dst + dst_offs) = acc;
    }
}

// ---------------------------------------------------------------------------------------------
// B: a register tile. One thread owns TW output positions x OC_T output channels, so each loaded
// activation feeds OC_T FMAs and each loaded weight feeds TW. Loads per FMA go from 2 to
// (TW + OC_T)/(TW*OC_T).
//
// The output positions a thread owns are ntg.x apart, NOT adjacent: adjacent lanes must keep
// reading adjacent `ix` or the coalescing that the stock kernel does get right is lost.
//
// Padding is handled by splitting, not by branching in the inner loop -- a threadgroup whose whole
// span is interior runs the unconditional loop, and only the one threadgroup at each edge takes the
// per-position path. `OW` is 73472 in the shape that matters, so "only at the edges" is exact.
// ---------------------------------------------------------------------------------------------
template <int OC_T, int TW>
kernel void conv_2d_tiled(
        constant conv_args & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]) {

    const int stride_w = (int) ntg.x;
    const int ow0 = (int)(tgpig.x * (uint) stride_w * TW + tpitg.x);
    const int oc0 = (int) tgpig.y * OC_T;
    const int oh  = (int)(tgpig.z % (uint) args.OH);
    const int n   = (int)(tgpig.z / (uint) args.OH);

    const int base_y = oh*args.s1 - args.p1;

    int ky_start = 0;
    if (base_y < 0) { ky_start = (-base_y + args.d1 - 1)/args.d1; }
    int ky_end = args.KH;
    const int y_max = args.IH - 1 - base_y;
    if (y_max < 0) { ky_end = ky_start; }
    else if (base_y + (args.KH - 1)*args.d1 >= args.IH) { ky_end = min(ky_end, y_max/args.d1 + 1); }

    // Out-of-range output channels read channel OC-1 instead of branching; the result is discarded
    // at the store. Every shape here has OC a multiple of the tile except the final OC=1 conv.
    device const char * wbase[OC_T];
    #pragma unroll
    for (int t = 0; t < OC_T; ++t) {
        wbase[t] = weights + (ulong) min(oc0 + t, args.OC - 1) * args.nb03;
    }

    const ulong src_n = (ulong) n * args.nb13;

    float acc[TW][OC_T];
    #pragma unroll
    for (int j = 0; j < TW; ++j) {
        #pragma unroll
        for (int t = 0; t < OC_T; ++t) { acc[j][t] = 0.0f; }
    }

    const int base_x0 = ow0*args.s0 - args.p0;
    const int base_xN = (ow0 + (TW - 1)*stride_w)*args.s0 - args.p0;
    const bool interior =
        (ow0 + (TW - 1)*stride_w) < args.OW &&
        base_x0 >= 0 && base_xN + (args.KW - 1)*args.d0 < args.IW &&
        ky_start == 0 && ky_end == args.KH;

    if (interior) {
        for (int ic = 0; ic < args.IC; ++ic) {
            const ulong w_ic  = (ulong) ic * args.nb02;
            const ulong x_ic  = src_n + (ulong) ic * args.nb12;
            for (int ky = 0; ky < args.KH; ++ky) {
                const int iy = base_y + ky*args.d1;
                const ulong w_row = w_ic + (ulong) ky * args.nb01;
                const ulong x_row = x_ic + (ulong) iy * args.nb11;
                for (int kx = 0; kx < args.KW; ++kx) {
                    const int ix = base_x0 + kx*args.d0;
                    const ulong x_off = x_row + (ulong) ix * args.nb10;

                    float xv[TW];
                    #pragma unroll
                    for (int j = 0; j < TW; ++j) {
                        xv[j] = *(device const float *)(src + x_off + (ulong)(j*stride_w*args.s0) * args.nb10);
                    }
                    #pragma unroll
                    for (int t = 0; t < OC_T; ++t) {
                        const float w = *(device const float *)(wbase[t] + w_row + (ulong) kx * args.nb00);
                        #pragma unroll
                        for (int j = 0; j < TW; ++j) { acc[j][t] += xv[j]*w; }
                    }
                }
            }
        }
    } else {
        #pragma unroll
        for (int j = 0; j < TW; ++j) {
            const int ow = ow0 + j*stride_w;
            if (ow >= args.OW) { continue; }
            const int base_x = ow*args.s0 - args.p0;

            int kx_start = 0;
            if (base_x < 0) { kx_start = (-base_x + args.d0 - 1)/args.d0; }
            int kx_end = args.KW;
            const int x_max = args.IW - 1 - base_x;
            if (x_max < 0) { kx_end = kx_start; }
            else if (base_x + (args.KW - 1)*args.d0 >= args.IW) { kx_end = min(kx_end, x_max/args.d0 + 1); }

            for (int ic = 0; ic < args.IC; ++ic) {
                const ulong w_ic = (ulong) ic * args.nb02;
                const ulong x_ic = src_n + (ulong) ic * args.nb12;
                for (int ky = ky_start; ky < ky_end; ++ky) {
                    const int iy = base_y + ky*args.d1;
                    const ulong w_row = w_ic + (ulong) ky * args.nb01;
                    const ulong x_row = x_ic + (ulong) iy * args.nb11;
                    for (int kx = kx_start; kx < kx_end; ++kx) {
                        const int ix = base_x + kx*args.d0;
                        const float x = *(device const float *)(src + x_row + (ulong) ix * args.nb10);
                        #pragma unroll
                        for (int t = 0; t < OC_T; ++t) {
                            const float w = *(device const float *)(wbase[t] + w_row + (ulong) kx * args.nb00);
                            acc[j][t] += x*w;
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < TW; ++j) {
        const int ow = ow0 + j*stride_w;
        if (ow >= args.OW) { continue; }
        #pragma unroll
        for (int t = 0; t < OC_T; ++t) {
            const int oc = oc0 + t;
            if (oc >= args.OC) { continue; }
            const ulong dst_offs = (ulong) n*args.nb3 + (ulong) oc*args.nb2 + (ulong) oh*args.nb1 + (ulong) ow*args.nb0;
            *(device float *)(dst + dst_offs) = acc[j][t];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// C: the same tile, with every 64-bit multiply hoisted out of the inner loop.
//
// B still computed `x_row + (ulong) ix * args.nb10` and `(ulong)(j*stride_w*s0) * args.nb10` per
// iteration -- a 64-bit integer multiply is several instructions on this GPU and there were twelve
// of them per thirty-two FMAs. Here every address is a running pointer plus a loop-invariant byte
// delta computed once.
//
// The weights get more than that. `ggml_metal_op_conv_2d` asserts `ggml_is_contiguous(src[0])`, and
// the (ic, ky, kx) loop nest walks a contiguous kernel in exactly memory order, so ONE pointer per
// output channel, post-incremented, replaces the whole address computation.
template <typename TK, int OC_T, int TW>
kernel void conv_2d_tiled2(
        constant conv_args & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]) {

    const int stride_w = (int) ntg.x;
    const int ow0 = (int)(tgpig.x * (uint) stride_w * TW + tpitg.x);
    const int oc0 = (int) tgpig.y * OC_T;
    const int oh  = (int)(tgpig.z % (uint) args.OH);
    const int n   = (int)(tgpig.z / (uint) args.OH);

    const int base_y = oh*args.s1 - args.p1;
    const int base_x0 = ow0*args.s0 - args.p0;

    int ky_start = 0;
    if (base_y < 0) { ky_start = (-base_y + args.d1 - 1)/args.d1; }
    int ky_end = args.KH;
    const int y_max = args.IH - 1 - base_y;
    if (y_max < 0) { ky_end = ky_start; }
    else if (base_y + (args.KH - 1)*args.d1 >= args.IH) { ky_end = min(ky_end, y_max/args.d1 + 1); }

    device const TK * wp[OC_T];
    #pragma unroll
    for (int t = 0; t < OC_T; ++t) {
        wp[t] = (device const TK *)(weights + (ulong) min(oc0 + t, args.OC - 1) * args.nb03);
    }

    float acc[TW][OC_T];
    #pragma unroll
    for (int j = 0; j < TW; ++j) {
        #pragma unroll
        for (int t = 0; t < OC_T; ++t) { acc[j][t] = 0.0f; }
    }

    const bool interior =
        (ow0 + (TW - 1)*stride_w) < args.OW &&
        base_x0 >= 0 &&
        (base_x0 + (TW - 1)*stride_w*args.s0) + (args.KW - 1)*args.d0 < args.IW &&
        ky_start == 0 && ky_end == args.KH;

    if (interior) {
        // loop-invariant byte deltas: one kernel tap, one row, one of this thread's positions
        const ulong dx_kx = (ulong)(args.d0) * args.nb10;
        const ulong dx_ky = (ulong)(args.d1) * args.nb11;
        ulong off_j[TW];
        #pragma unroll
        for (int j = 0; j < TW; ++j) { off_j[j] = (ulong)(j*stride_w*args.s0) * args.nb10; }

        device const char * x_ic = src
            + (ulong) n * args.nb13
            + (ulong) base_y * args.nb11
            + (ulong) base_x0 * args.nb10;

        for (int ic = 0; ic < args.IC; ++ic) {
            device const char * x_ky = x_ic;
            for (int ky = 0; ky < args.KH; ++ky) {
                device const char * x_kx = x_ky;
                for (int kx = 0; kx < args.KW; ++kx) {
                    float xv[TW];
                    #pragma unroll
                    for (int j = 0; j < TW; ++j) {
                        xv[j] = *(device const float *)(x_kx + off_j[j]);
                    }
                    #pragma unroll
                    for (int t = 0; t < OC_T; ++t) {
                        const float w = (float) *wp[t]++;
                        #pragma unroll
                        for (int j = 0; j < TW; ++j) { acc[j][t] += xv[j]*w; }
                    }
                    x_kx += dx_kx;
                }
                x_ky += dx_ky;
            }
            x_ic += args.nb12;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < TW; ++j) {
            const int ow = ow0 + j*stride_w;
            if (ow >= args.OW) { continue; }
            const int base_x = ow*args.s0 - args.p0;

            int kx_start = 0;
            if (base_x < 0) { kx_start = (-base_x + args.d0 - 1)/args.d0; }
            int kx_end = args.KW;
            const int x_max = args.IW - 1 - base_x;
            if (x_max < 0) { kx_end = kx_start; }
            else if (base_x + (args.KW - 1)*args.d0 >= args.IW) { kx_end = min(kx_end, x_max/args.d0 + 1); }

            for (int ic = 0; ic < args.IC; ++ic) {
                const ulong w_ic = (ulong) ic * args.nb02;
                const ulong x_ic = (ulong) n * args.nb13 + (ulong) ic * args.nb12;
                for (int ky = ky_start; ky < ky_end; ++ky) {
                    const int iy = base_y + ky*args.d1;
                    const ulong w_row = w_ic + (ulong) ky * args.nb01;
                    const ulong x_row = x_ic + (ulong) iy * args.nb11;
                    for (int kx = kx_start; kx < kx_end; ++kx) {
                        const int ix = base_x + kx*args.d0;
                        const float x = *(device const float *)(src + x_row + (ulong) ix * args.nb10);
                        #pragma unroll
                        for (int t = 0; t < OC_T; ++t) {
                            const float w = (float) *(device const TK *)((device const char *) wp[t] + w_row + (ulong) kx * args.nb00);
                            acc[j][t] += x*w;
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < TW; ++j) {
        const int ow = ow0 + j*stride_w;
        if (ow >= args.OW) { continue; }
        #pragma unroll
        for (int t = 0; t < OC_T; ++t) {
            const int oc = oc0 + t;
            if (oc >= args.OC) { continue; }
            const ulong dst_offs = (ulong) n*args.nb3 + (ulong) oc*args.nb2 + (ulong) oh*args.nb1 + (ulong) ow*args.nb0;
            *(device float *)(dst + dst_offs) = acc[j][t];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// D: the same tile again, with the ADDRESSES cut from 64-bit pointers to 32-bit element indices.
//
// C kept one `device const TK *` per output channel -- OC_T 64-bit pointers, sixteen registers of
// pure address at OC_T = 8 -- and `maxTotalThreadsPerThreadgroup` said what that cost: 512 for
// `oc8 tw4` and 704 for `oc8 tw1` against 1024 for a kernel with no pressure. On this GPU
// occupancy IS latency hiding, so those registers were being paid for twice.
//
// Here every operand is reached as `base[uint index]`. A conv's tensors are far below 4G elements,
// so a 32-bit index is enough, and `t*w_oc` for a compile-time `t` is one integer multiply rather
// than a live register. The interior path additionally requires a FULL output-channel tile, which
// is what lets the clamp for a partial tile leave the inner loop entirely.
// ---------------------------------------------------------------------------------------------
// E: D, plus the kernel width as a template parameter.
//
// `KW` is 1, 3, 5 or 7 in every convolution VITS issues and the host knows which before it picks a
// pipeline, so there is no reason for the tap loop to be a runtime loop. Unrolled, the whole of one
// input channel's loads is in front of the scheduler at once, which is the thing this kernel is
// short of -- at OC_T=8 it issues 17 instructions per 8 FMAs and still runs at 44% of what that
// mix allows, so it is waiting on memory, not on issue slots.
template <typename TK, int OC_T, int TW, int KW_C>
kernel void conv_2d_tiled4(
        constant conv_args & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]) {

    // KW_C == 0 means "read it from the args"; any other value is the compile-time kernel width,
    // which is what lets the tap loop unroll and puts every load of an input channel in front of the
    // scheduler at once instead of one per trip.
    const int KW = KW_C ? KW_C : args.KW;

    const int stride_w = (int) ntg.x;
    const int ow0 = (int)(tgpig.x * (uint) stride_w * TW + tpitg.x);
    const int oc0 = (int) tgpig.y * OC_T;
    const int oh  = (int)(tgpig.z % (uint) args.OH);
    const int n   = (int)(tgpig.z / (uint) args.OH);

    const int base_y  = oh*args.s1 - args.p1;
    const int base_x0 = ow0*args.s0 - args.p0;

    int ky_start = 0;
    if (base_y < 0) { ky_start = (-base_y + args.d1 - 1)/args.d1; }
    int ky_end = args.KH;
    const int y_max = args.IH - 1 - base_y;
    if (y_max < 0) { ky_end = ky_start; }
    else if (base_y + (args.KH - 1)*args.d1 >= args.IH) { ky_end = min(ky_end, y_max/args.d1 + 1); }

    float acc[TW][OC_T];
    #pragma unroll
    for (int j = 0; j < TW; ++j) {
        #pragma unroll
        for (int t = 0; t < OC_T; ++t) { acc[j][t] = 0.0f; }
    }

    // The fast path indexes with 32-bit element counts, so it has to decline a tensor that does not
    // fit one, or a stride that is not a whole number of elements. Both are uniform across the
    // threadgroup and cost four comparisons once; the slow path below is the general 64-bit one, so
    // declining is always correct rather than merely safe.
    const bool addressable =
        (args.nb10 % 4) == 0 && (args.nb11 % 4) == 0 &&
        (args.nb12 % 4) == 0 && (args.nb13 % 4) == 0 &&
        (args.nb03 % sizeof(TK)) == 0 &&
        (ulong) args.nb13*args.N/4        < 0xffffffffull &&
        (ulong) args.nb03*args.OC/sizeof(TK) < 0xffffffffull;

    const bool interior =
        addressable &&
        (ow0 + (TW - 1)*stride_w) < args.OW &&
        oc0 + OC_T <= args.OC &&
        base_x0 >= 0 &&
        (base_x0 + (TW - 1)*stride_w*args.s0) + (KW - 1)*args.d0 < args.IW &&
        ky_start == 0 && ky_end == args.KH;

    if (interior) {
        device const TK    * w0 = (device const TK    *) weights;
        device const float * x0 = (device const float *) src;

        // element strides; the kernel is contiguous by ggml's own assert, so its (ic, ky, kx) walk
        // is one running index
        const uint w_oc  = (uint)(args.nb03/sizeof(TK));
        const uint xs_ic = (uint)(args.nb12/4);
        const uint xs_ky = (uint)(args.nb11/4)*(uint) args.d1;
        const uint xs_kx = (uint)(args.nb10/4)*(uint) args.d0;
        const uint wt0   = (uint) oc0 * w_oc;

        uint off_j[TW];
        #pragma unroll
        for (int j = 0; j < TW; ++j) { off_j[j] = (uint)(j*stride_w*args.s0)*(uint)(args.nb10/4); }

        uint xi = (uint)((ulong) n*args.nb13/4)
                + (uint)(args.nb11/4)*(uint) base_y
                + (uint)(args.nb10/4)*(uint) base_x0;
        uint wi = 0;

        for (int ic = 0; ic < args.IC; ++ic) {
            uint xj = xi;
            for (int ky = 0; ky < args.KH; ++ky) {
                uint xk = xj;
                #pragma unroll
                for (int kx = 0; kx < KW; ++kx) {
                    float xv[TW];
                    #pragma unroll
                    for (int j = 0; j < TW; ++j) { xv[j] = x0[xk + off_j[j]]; }
                    #pragma unroll
                    for (int t = 0; t < OC_T; ++t) {
                        const float w = (float) w0[wt0 + (uint) t*w_oc + wi];
                        #pragma unroll
                        for (int j = 0; j < TW; ++j) { acc[j][t] += xv[j]*w; }
                    }
                    ++wi;
                    xk += xs_kx;
                }
                xj += xs_ky;
            }
            xi += xs_ic;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < TW; ++j) {
            const int ow = ow0 + j*stride_w;
            if (ow >= args.OW) { continue; }
            const int base_x = ow*args.s0 - args.p0;

            int kx_start = 0;
            if (base_x < 0) { kx_start = (-base_x + args.d0 - 1)/args.d0; }
            int kx_end = KW;
            const int x_max = args.IW - 1 - base_x;
            if (x_max < 0) { kx_end = kx_start; }
            else if (base_x + (KW - 1)*args.d0 >= args.IW) { kx_end = min(kx_end, x_max/args.d0 + 1); }

            for (int ic = 0; ic < args.IC; ++ic) {
                const ulong w_ic = (ulong) ic * args.nb02;
                const ulong x_ic = (ulong) n * args.nb13 + (ulong) ic * args.nb12;
                for (int ky = ky_start; ky < ky_end; ++ky) {
                    const int iy = base_y + ky*args.d1;
                    const ulong w_row = w_ic + (ulong) ky * args.nb01;
                    const ulong x_row = x_ic + (ulong) iy * args.nb11;
                    for (int kx = kx_start; kx < kx_end; ++kx) {
                        const int ix = base_x + kx*args.d0;
                        const float x = *(device const float *)(src + x_row + (ulong) ix * args.nb10);
                        #pragma unroll
                        for (int t = 0; t < OC_T; ++t) {
                            const int oc = min(oc0 + t, args.OC - 1);
                            const float w = (float) *(device const TK *)(
                                weights + (ulong) oc*args.nb03 + w_row + (ulong) kx * args.nb00);
                            acc[j][t] += x*w;
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < TW; ++j) {
        const int ow = ow0 + j*stride_w;
        if (ow >= args.OW) { continue; }
        #pragma unroll
        for (int t = 0; t < OC_T; ++t) {
            const int oc = oc0 + t;
            if (oc >= args.OC) { continue; }
            const ulong dst_offs = (ulong) n*args.nb3 + (ulong) oc*args.nb2 + (ulong) oh*args.nb1 + (ulong) ow*args.nb0;
            *(device float *)(dst + dst_offs) = acc[j][t];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// F: E, with the weight tile staged in threadgroup memory.
//
// At OC_T = 8 the inner loop issues 17 instructions per 8 FMAs and runs at 40% of what that mix
// allows, so what is left is latency, not issue slots. Eight of those seventeen are weight loads at
// an address every lane in the threadgroup shares. Staged once into threadgroup memory they become
// broadcasts out of a scratchpad instead of eight independent trips to L1, and the eight device
// pointers they needed stop competing for registers.
//
// The staging loop is coalesced by construction: `ggml_is_contiguous(src[0])` makes one output
// channel's (ic, ky, kx) block a contiguous run, so the copy is OC_T contiguous runs.
//
// THE INTERIOR TEST HAS TO BE THREADGROUP-UNIFORM HERE, unlike in D and E, because the fast path
// contains barriers and the slow one does not: it is computed from the threadgroup's own output
// range, not from the calling thread's.
template <typename TK, int OC_T, int TW, int KW_C>
kernel void conv_2d_tiled5(
        constant conv_args & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]) {

    constexpr int TG_W = 2048;          // 8 KB of staged weights
    threadgroup float w_tile[TG_W];

    const int KW = KW_C ? KW_C : args.KW;
    const int KHW = args.KH*KW;

    const int NT = (int) ntg.x;
    const int tg_ow0 = (int)(tgpig.x * (uint) NT * TW);
    const int ow0 = tg_ow0 + (int) tpitg.x;
    const int oc0 = (int) tgpig.y * OC_T;
    const int oh  = (int)(tgpig.z % (uint) args.OH);
    const int n   = (int)(tgpig.z / (uint) args.OH);

    const int base_y = oh*args.s1 - args.p1;

    int ky_start = 0;
    if (base_y < 0) { ky_start = (-base_y + args.d1 - 1)/args.d1; }
    int ky_end = args.KH;
    const int y_max = args.IH - 1 - base_y;
    if (y_max < 0) { ky_end = ky_start; }
    else if (base_y + (args.KH - 1)*args.d1 >= args.IH) { ky_end = min(ky_end, y_max/args.d1 + 1); }

    float acc[TW][OC_T];
    #pragma unroll
    for (int j = 0; j < TW; ++j) {
        #pragma unroll
        for (int t = 0; t < OC_T; ++t) { acc[j][t] = 0.0f; }
    }

    const int tg_owN  = tg_ow0 + NT*TW - 1;
    const int tg_bx0  = tg_ow0*args.s0 - args.p0;
    const int tg_bxN  = tg_owN*args.s0 - args.p0;
    const int ic_chunk = max(1, min(args.IC, TG_W/(OC_T*KHW)));

    const bool interior =
        tg_owN < args.OW &&
        oc0 + OC_T <= args.OC &&
        tg_bx0 >= 0 && tg_bxN + (KW - 1)*args.d0 < args.IW &&
        ky_start == 0 && ky_end == args.KH &&
        OC_T*KHW <= TG_W;

    if (interior) {
        device const TK    * w0 = (device const TK    *) weights;
        device const float * x0 = (device const float *) src;

        const uint w_oc  = (uint)(args.nb03/sizeof(TK));
        const uint xs_ic = (uint)(args.nb12/4);
        const uint xs_ky = (uint)(args.nb11/4)*(uint) args.d1;
        const uint xs_kx = (uint)(args.nb10/4)*(uint) args.d0;

        uint off_j[TW];
        #pragma unroll
        for (int j = 0; j < TW; ++j) { off_j[j] = (uint)(j*NT*args.s0)*(uint)(args.nb10/4); }

        const uint xi0 = (uint)((ulong) n*args.nb13/4)
                       + (uint)(args.nb11/4)*(uint) base_y
                       + (uint)(args.nb10/4)*(uint)(ow0*args.s0 - args.p0);

        for (int ic0 = 0; ic0 < args.IC; ic0 += ic_chunk) {
            const int icc = min(ic_chunk, args.IC - ic0);
            const int per_oc = icc*KHW;

            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int idx = (int) tpitg.x; idx < OC_T*per_oc; idx += NT) {
                const int t = idx/per_oc;
                const int m = idx - t*per_oc;
                w_tile[idx] = (float) w0[(uint)(oc0 + t)*w_oc + (uint)(ic0*KHW + m)];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint xi = xi0 + xs_ic*(uint) ic0;
            int wt = 0;
            for (int ic = 0; ic < icc; ++ic) {
                uint xj = xi;
                for (int ky = 0; ky < args.KH; ++ky) {
                    uint xk = xj;
                    #pragma unroll
                    for (int kx = 0; kx < KW; ++kx) {
                        float xv[TW];
                        #pragma unroll
                        for (int j = 0; j < TW; ++j) { xv[j] = x0[xk + off_j[j]]; }
                        #pragma unroll
                        for (int t = 0; t < OC_T; ++t) {
                            const float w = w_tile[t*per_oc + wt];
                            #pragma unroll
                            for (int j = 0; j < TW; ++j) { acc[j][t] += xv[j]*w; }
                        }
                        ++wt;
                        xk += xs_kx;
                    }
                    xj += xs_ky;
                }
                xi += xs_ic;
            }
        }
    } else {
        #pragma unroll
        for (int j = 0; j < TW; ++j) {
            const int ow = ow0 + j*NT;
            if (ow >= args.OW) { continue; }
            const int base_x = ow*args.s0 - args.p0;

            int kx_start = 0;
            if (base_x < 0) { kx_start = (-base_x + args.d0 - 1)/args.d0; }
            int kx_end = KW;
            const int x_max = args.IW - 1 - base_x;
            if (x_max < 0) { kx_end = kx_start; }
            else if (base_x + (KW - 1)*args.d0 >= args.IW) { kx_end = min(kx_end, x_max/args.d0 + 1); }

            for (int ic = 0; ic < args.IC; ++ic) {
                const ulong w_ic = (ulong) ic * args.nb02;
                const ulong x_ic = (ulong) n * args.nb13 + (ulong) ic * args.nb12;
                for (int ky = ky_start; ky < ky_end; ++ky) {
                    const int iy = base_y + ky*args.d1;
                    const ulong w_row = w_ic + (ulong) ky * args.nb01;
                    const ulong x_row = x_ic + (ulong) iy * args.nb11;
                    for (int kx = kx_start; kx < kx_end; ++kx) {
                        const int ix = base_x + kx*args.d0;
                        const float x = *(device const float *)(src + x_row + (ulong) ix * args.nb10);
                        #pragma unroll
                        for (int t = 0; t < OC_T; ++t) {
                            const int oc = min(oc0 + t, args.OC - 1);
                            const float w = (float) *(device const TK *)(
                                weights + (ulong) oc*args.nb03 + w_row + (ulong) kx * args.nb00);
                            acc[j][t] += x*w;
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < TW; ++j) {
        const int ow = ow0 + j*NT;
        if (ow >= args.OW) { continue; }
        #pragma unroll
        for (int t = 0; t < OC_T; ++t) {
            const int oc = oc0 + t;
            if (oc >= args.OC) { continue; }
            const ulong dst_offs = (ulong) n*args.nb3 + (ulong) oc*args.nb2 + (ulong) oh*args.nb1 + (ulong) ow*args.nb0;
            *(device float *)(dst + dst_offs) = acc[j][t];
        }
    }
}

#define INSTANTIATE_TILED5(OC_T, TW, KW_C)                              \
    template [[host_name("conv_2d_tiled5_" #OC_T "_" #TW "_" #KW_C)]]   \
    kernel void conv_2d_tiled5<float, OC_T, TW, KW_C>(                  \
        constant conv_args & args,                                      \
        device const char * weights,                                    \
        device const char * src,                                        \
        device       char * dst,                                        \
        uint3   tgpig[[threadgroup_position_in_grid]],                  \
        uint3   tpitg[[thread_position_in_threadgroup]],                \
        uint3     ntg[[threads_per_threadgroup]]);

#define INSTANTIATE_TILED4(OC_T, TW, KW_C)                              \
    template [[host_name("conv_2d_tiled4_" #OC_T "_" #TW "_" #KW_C)]]   \
    kernel void conv_2d_tiled4<float, OC_T, TW, KW_C>(                  \
        constant conv_args & args,                                      \
        device const char * weights,                                    \
        device const char * src,                                        \
        device       char * dst,                                        \
        uint3   tgpig[[threadgroup_position_in_grid]],                  \
        uint3   tpitg[[thread_position_in_threadgroup]],                \
        uint3     ntg[[threads_per_threadgroup]]);

template <typename TK, int OC_T, int TW>
kernel void conv_2d_tiled3(
        constant conv_args & args,
        device const char * weights,
        device const char * src,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3     ntg[[threads_per_threadgroup]]) {

    const int stride_w = (int) ntg.x;
    const int ow0 = (int)(tgpig.x * (uint) stride_w * TW + tpitg.x);
    const int oc0 = (int) tgpig.y * OC_T;
    const int oh  = (int)(tgpig.z % (uint) args.OH);
    const int n   = (int)(tgpig.z / (uint) args.OH);

    const int base_y  = oh*args.s1 - args.p1;
    const int base_x0 = ow0*args.s0 - args.p0;

    int ky_start = 0;
    if (base_y < 0) { ky_start = (-base_y + args.d1 - 1)/args.d1; }
    int ky_end = args.KH;
    const int y_max = args.IH - 1 - base_y;
    if (y_max < 0) { ky_end = ky_start; }
    else if (base_y + (args.KH - 1)*args.d1 >= args.IH) { ky_end = min(ky_end, y_max/args.d1 + 1); }

    float acc[TW][OC_T];
    #pragma unroll
    for (int j = 0; j < TW; ++j) {
        #pragma unroll
        for (int t = 0; t < OC_T; ++t) { acc[j][t] = 0.0f; }
    }

    const bool interior =
        (ow0 + (TW - 1)*stride_w) < args.OW &&
        oc0 + OC_T <= args.OC &&
        base_x0 >= 0 &&
        (base_x0 + (TW - 1)*stride_w*args.s0) + (args.KW - 1)*args.d0 < args.IW &&
        ky_start == 0 && ky_end == args.KH;

    if (interior) {
        device const TK    * w0 = (device const TK    *) weights;
        device const float * x0 = (device const float *) src;

        // element strides; the kernel is contiguous by ggml's own assert, so its (ic, ky, kx) walk
        // is one running index
        const uint w_oc  = (uint)(args.nb03/sizeof(TK));
        const uint xs_ic = (uint)(args.nb12/4);
        const uint xs_ky = (uint)(args.nb11/4)*(uint) args.d1;
        const uint xs_kx = (uint)(args.nb10/4)*(uint) args.d0;
        const uint wt0   = (uint) oc0 * w_oc;

        uint off_j[TW];
        #pragma unroll
        for (int j = 0; j < TW; ++j) { off_j[j] = (uint)(j*stride_w*args.s0)*(uint)(args.nb10/4); }

        uint xi = (uint)((ulong) n*args.nb13/4)
                + (uint)(args.nb11/4)*(uint) base_y
                + (uint)(args.nb10/4)*(uint) base_x0;
        uint wi = 0;

        for (int ic = 0; ic < args.IC; ++ic) {
            uint xj = xi;
            for (int ky = 0; ky < args.KH; ++ky) {
                uint xk = xj;
                for (int kx = 0; kx < args.KW; ++kx) {
                    float xv[TW];
                    #pragma unroll
                    for (int j = 0; j < TW; ++j) { xv[j] = x0[xk + off_j[j]]; }
                    #pragma unroll
                    for (int t = 0; t < OC_T; ++t) {
                        const float w = (float) w0[wt0 + (uint) t*w_oc + wi];
                        #pragma unroll
                        for (int j = 0; j < TW; ++j) { acc[j][t] += xv[j]*w; }
                    }
                    ++wi;
                    xk += xs_kx;
                }
                xj += xs_ky;
            }
            xi += xs_ic;
        }
    } else {
        #pragma unroll
        for (int j = 0; j < TW; ++j) {
            const int ow = ow0 + j*stride_w;
            if (ow >= args.OW) { continue; }
            const int base_x = ow*args.s0 - args.p0;

            int kx_start = 0;
            if (base_x < 0) { kx_start = (-base_x + args.d0 - 1)/args.d0; }
            int kx_end = args.KW;
            const int x_max = args.IW - 1 - base_x;
            if (x_max < 0) { kx_end = kx_start; }
            else if (base_x + (args.KW - 1)*args.d0 >= args.IW) { kx_end = min(kx_end, x_max/args.d0 + 1); }

            for (int ic = 0; ic < args.IC; ++ic) {
                const ulong w_ic = (ulong) ic * args.nb02;
                const ulong x_ic = (ulong) n * args.nb13 + (ulong) ic * args.nb12;
                for (int ky = ky_start; ky < ky_end; ++ky) {
                    const int iy = base_y + ky*args.d1;
                    const ulong w_row = w_ic + (ulong) ky * args.nb01;
                    const ulong x_row = x_ic + (ulong) iy * args.nb11;
                    for (int kx = kx_start; kx < kx_end; ++kx) {
                        const int ix = base_x + kx*args.d0;
                        const float x = *(device const float *)(src + x_row + (ulong) ix * args.nb10);
                        #pragma unroll
                        for (int t = 0; t < OC_T; ++t) {
                            const int oc = min(oc0 + t, args.OC - 1);
                            const float w = (float) *(device const TK *)(
                                weights + (ulong) oc*args.nb03 + w_row + (ulong) kx * args.nb00);
                            acc[j][t] += x*w;
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < TW; ++j) {
        const int ow = ow0 + j*stride_w;
        if (ow >= args.OW) { continue; }
        #pragma unroll
        for (int t = 0; t < OC_T; ++t) {
            const int oc = oc0 + t;
            if (oc >= args.OC) { continue; }
            const ulong dst_offs = (ulong) n*args.nb3 + (ulong) oc*args.nb2 + (ulong) oh*args.nb1 + (ulong) ow*args.nb0;
            *(device float *)(dst + dst_offs) = acc[j][t];
        }
    }
}

#define INSTANTIATE_TILED3(OC_T, TW)                                    \
    template [[host_name("conv_2d_tiled3_" #OC_T "_" #TW)]]             \
    kernel void conv_2d_tiled3<float, OC_T, TW>(                        \
        constant conv_args & args,                                      \
        device const char * weights,                                    \
        device const char * src,                                        \
        device       char * dst,                                        \
        uint3   tgpig[[threadgroup_position_in_grid]],                  \
        uint3   tpitg[[thread_position_in_threadgroup]],                \
        uint3     ntg[[threads_per_threadgroup]]);

#define INSTANTIATE_TILED2(OC_T, TW)                                    \
    template [[host_name("conv_2d_tiled2_" #OC_T "_" #TW)]]             \
    kernel void conv_2d_tiled2<float, OC_T, TW>(                        \
        constant conv_args & args,                                      \
        device const char * weights,                                    \
        device const char * src,                                        \
        device       char * dst,                                        \
        uint3   tgpig[[threadgroup_position_in_grid]],                  \
        uint3   tpitg[[thread_position_in_threadgroup]],                \
        uint3     ntg[[threads_per_threadgroup]]);

// ---------------------------------------------------------------------------------------------
// The two rooflines this part actually delivers, measured rather than quoted. A spec sheet's
// 5.31 TFLOP/s is not what a kernel can reach, and every ratio below is worthless without the
// achievable number to divide by (Retro-011, and `feedback_instruction_ratio_is_not_a_time_ratio`).
// ---------------------------------------------------------------------------------------------
kernel void roofline_fma(
        device float * out,
        constant int & reps,
        uint tpig[[thread_position_in_grid]]) {
    // Sixteen independent chains, so the number is an ISSUE rate and not an FMA-latency rate.
    float4 a[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) { a[i] = float4(tpig + i, tpig + i + 1, tpig + i + 2, tpig + i + 3); }
    const float k = 1.0000001f;
    for (int r = 0; r < reps; ++r) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) { a[i] = fma(a[i], k, 1.0f); }
    }
    float4 sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) { sum += a[i]; }
    const float t = sum.x + sum.y + sum.z + sum.w;
    if (t == 12345.678f) { out[tpig] = t; }
}

kernel void roofline_read(
        device const float4 * in,
        device       float  * out,
        constant int & n4,
        uint tpig[[thread_position_in_grid]],
        uint ntg [[threads_per_grid]]) {
    float4 acc = 0.0f;
    for (uint i = tpig; i < (uint) n4; i += ntg) { acc += in[i]; }
    const float sum = acc.x + acc.y + acc.z + acc.w;
    if (sum == 12345.678f) { out[tpig] = sum; }
}

#define INSTANTIATE_TILED(OC_T, TW)                                     \
    template [[host_name("conv_2d_tiled_" #OC_T "_" #TW)]]              \
    kernel void conv_2d_tiled<OC_T, TW>(                                \
        constant conv_args & args,                                      \
        device const char * weights,                                    \
        device const char * src,                                        \
        device       char * dst,                                        \
        uint3   tgpig[[threadgroup_position_in_grid]],                  \
        uint3   tpitg[[thread_position_in_threadgroup]],                \
        uint3     ntg[[threads_per_threadgroup]]);

INSTANTIATE_TILED(2, 1)
INSTANTIATE_TILED(4, 1)
INSTANTIATE_TILED(8, 1)
INSTANTIATE_TILED(4, 2)
INSTANTIATE_TILED(8, 2)
INSTANTIATE_TILED(4, 4)
INSTANTIATE_TILED(8, 4)
INSTANTIATE_TILED(16, 2)
INSTANTIATE_TILED(8, 8)
INSTANTIATE_TILED(16, 4)
INSTANTIATE_TILED(4, 8)
INSTANTIATE_TILED(32, 2)
INSTANTIATE_TILED2(4, 2)
INSTANTIATE_TILED2(8, 2)
INSTANTIATE_TILED2(4, 4)
INSTANTIATE_TILED2(8, 4)
INSTANTIATE_TILED2(16, 2)
INSTANTIATE_TILED2(8, 8)
INSTANTIATE_TILED2(16, 4)
INSTANTIATE_TILED2(4, 1)
INSTANTIATE_TILED2(8, 1)
INSTANTIATE_TILED2(16, 1)
INSTANTIATE_TILED2(32, 1)
INSTANTIATE_TILED3(4, 1)
INSTANTIATE_TILED3(8, 1)
INSTANTIATE_TILED3(8, 2)
INSTANTIATE_TILED3(8, 4)
INSTANTIATE_TILED3(4, 4)
INSTANTIATE_TILED3(16, 1)
INSTANTIATE_TILED3(16, 2)
INSTANTIATE_TILED3(8, 8)
INSTANTIATE_TILED4(8, 1, 0)
INSTANTIATE_TILED4(8, 1, 1)
INSTANTIATE_TILED4(8, 1, 3)
INSTANTIATE_TILED4(8, 1, 5)
INSTANTIATE_TILED4(8, 1, 7)
INSTANTIATE_TILED4(8, 2, 0)
INSTANTIATE_TILED4(8, 2, 1)
INSTANTIATE_TILED4(8, 2, 3)
INSTANTIATE_TILED4(8, 2, 5)
INSTANTIATE_TILED4(8, 2, 7)
INSTANTIATE_TILED4(16, 1, 0)
INSTANTIATE_TILED4(16, 1, 1)
INSTANTIATE_TILED4(16, 1, 3)
INSTANTIATE_TILED4(16, 1, 5)
INSTANTIATE_TILED4(16, 1, 7)
INSTANTIATE_TILED4(4, 4, 0)
INSTANTIATE_TILED4(4, 4, 1)
INSTANTIATE_TILED4(4, 4, 3)
INSTANTIATE_TILED4(4, 4, 5)
INSTANTIATE_TILED4(4, 4, 7)
INSTANTIATE_TILED5(8, 1, 0)
INSTANTIATE_TILED5(8, 1, 1)
INSTANTIATE_TILED5(8, 1, 3)
INSTANTIATE_TILED5(8, 1, 5)
INSTANTIATE_TILED5(8, 1, 7)
INSTANTIATE_TILED5(8, 2, 0)
INSTANTIATE_TILED5(8, 2, 1)
INSTANTIATE_TILED5(8, 2, 3)
INSTANTIATE_TILED5(8, 2, 5)
INSTANTIATE_TILED5(8, 2, 7)
INSTANTIATE_TILED5(16, 1, 0)
INSTANTIATE_TILED5(16, 1, 1)
INSTANTIATE_TILED5(16, 1, 3)
INSTANTIATE_TILED5(16, 1, 5)
INSTANTIATE_TILED5(16, 1, 7)
INSTANTIATE_TILED5(8, 4, 0)
INSTANTIATE_TILED5(8, 4, 1)
INSTANTIATE_TILED5(8, 4, 3)
INSTANTIATE_TILED5(8, 4, 5)
INSTANTIATE_TILED5(8, 4, 7)
INSTANTIATE_TILED5(16, 2, 0)
INSTANTIATE_TILED5(16, 2, 1)
INSTANTIATE_TILED5(16, 2, 3)
INSTANTIATE_TILED5(16, 2, 5)
INSTANTIATE_TILED5(16, 2, 7)
)METAL";

// ------------------------------------------------------------------------------------------------

struct Shape {
    const char * name;
    int IW, IC, KW, OC, s0, p0, d0;
    int calls;          // how many nodes of this shape a VITS synthesis issues
};

static int out_len(const Shape & s) {
    return (s.IW + 2*s.p0 - s.d0*(s.KW - 1) - 1)/s.s0 + 1;
}

static void cpu_reference(const Shape & s, const std::vector<float> & w,
                          const std::vector<float> & x, std::vector<float> & out) {
    const int OW = out_len(s);
    out.assign((size_t) OW * s.OC, 0.0f);
    for (int oc = 0; oc < s.OC; ++oc) {
        for (int ow = 0; ow < OW; ++ow) {
            float acc = 0.0f;
            const int base_x = ow*s.s0 - s.p0;
            for (int ic = 0; ic < s.IC; ++ic) {
                for (int kx = 0; kx < s.KW; ++kx) {
                    const int ix = base_x + kx*s.d0;
                    if (ix < 0 || ix >= s.IW) { continue; }
                    acc += x[(size_t) ic*s.IW + ix] * w[((size_t) oc*s.IC + ic)*s.KW + kx];
                }
            }
            out[(size_t) oc*OW + ow] = acc;
        }
    }
}

struct Variant {
    const char * label;
    const char * fn;        // "%d" in the name is replaced by the shape's KW
    int oc_t;               // 0 => stock, flat grid
    int tw;
    int nth;
};

int main(int argc, char ** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 10;

    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::fprintf(stderr, "no Metal device\n"); return 1; }
        std::printf("device: %s\n\n", [[dev name] UTF8String]);

        NSError * err = nil;
        MTLCompileOptions * opts = [MTLCompileOptions new];
        opts.fastMathEnabled = YES;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kShaderSrc]
                                               options:opts error:&err];
        if (!lib) { std::fprintf(stderr, "shader: %s\n", [[err description] UTF8String]); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        // The six shapes a VITS synthesis actually runs, from
        //   scripts/conv_census.py <vits.gguf> --syms n_tokens=100 --syms flow_vocoder:n_tokens=286
        // `calls` is that census's count, so the per-shape times below sum to the whole graph's
        // CONV_2D bucket and can be checked against $LOOM_PROFILE.
        // Every CONV_1D one VITS synthesis issues, at the multiplicity the graph runs it, from
        //   scripts/conv_census.py <vits.gguf> --syms n_tokens=100 --syms flow_vocoder:n_tokens=286
        // 117 nodes and 16.884 GFLOP -- the `CONV_2D` row of the $LOOM_PROFILE table in Epic-04 5.7
        // exactly (117 calls, 16.89 GFLOP). 286 is not a typo: it is the duration-expanded length
        // `bench_vits_loom.cpp`'s PINNED utterance produces, so the TOTAL below is comparable to that
        // row's ms and a ratio between two variants is a prediction about the model.
        const std::vector<Shape> shapes = {
            {"L=100    IC=192  K=1 OC=192  p=0  d=1 ",    100,  192, 1,  192, 1,  0,  1, 38},
            {"L=102    IC=192  K=3 OC=768  p=0  d=1 ",    102,  192, 3,  768, 1,  0,  1,  6},
            {"L=102    IC=768  K=3 OC=192  p=0  d=1 ",    102,  768, 3,  192, 1,  0,  1,  6},
            {"L=100    IC=192  K=1 OC=384  p=0  d=1 ",    100,  192, 1,  384, 1,  0,  1,  1},
            {"L=100    IC=1    K=1 OC=192  p=0  d=1 ",    100,    1, 1,  192, 1,  0,  1,  3},
            {"L=100    IC=192  K=1 OC=29   p=0  d=1 ",    100,  192, 1,   29, 1,  0,  1,  3},
            {"L=286    IC=96   K=1 OC=192  p=0  d=1 ",    286,   96, 1,  192, 1,  0,  1,  4},
            {"L=286    IC=192  K=5 OC=384  p=2  d=1 ",    286,  192, 5,  384, 1,  2,  1, 16},
            {"L=286    IC=192  K=1 OC=384  p=0  d=1 ",    286,  192, 1,  384, 1,  0,  1, 12},
            {"L=286    IC=192  K=1 OC=192  p=0  d=1 ",    286,  192, 1,  192, 1,  0,  1,  4},
            {"L=286    IC=192  K=1 OC=96   p=0  d=1 ",    286,  192, 1,   96, 1,  0,  1,  4},
            {"L=286    IC=192  K=7 OC=256  p=3  d=1 ",    286,  192, 7,  256, 1,  3,  1,  1},
            {"L=2288   IC=128  K=3 OC=128  p=1  d=1 ",   2288,  128, 3,  128, 1,  1,  1,  1},
            {"L=2288   IC=128  K=3 OC=128  p=2  d=2 ",   2288,  128, 3,  128, 1,  2,  2,  1},
            {"L=2288   IC=128  K=5 OC=128  p=4  d=2 ",   2288,  128, 5,  128, 1,  4,  2,  1},
            {"L=2288   IC=128  K=5 OC=128  p=12 d=6 ",   2288,  128, 5,  128, 1, 12,  6,  1},
            {"L=2288   IC=128  K=7 OC=128  p=9  d=3 ",   2288,  128, 7,  128, 1,  9,  3,  1},
            {"L=2288   IC=128  K=7 OC=128  p=36 d=12",   2288,  128, 7,  128, 1, 36, 12,  1},
            {"L=18304  IC=64   K=3 OC=64   p=1  d=1 ",  18304,   64, 3,   64, 1,  1,  1,  1},
            {"L=18304  IC=64   K=3 OC=64   p=2  d=2 ",  18304,   64, 3,   64, 1,  2,  2,  1},
            {"L=18304  IC=64   K=5 OC=64   p=4  d=2 ",  18304,   64, 5,   64, 1,  4,  2,  1},
            {"L=18304  IC=64   K=5 OC=64   p=12 d=6 ",  18304,   64, 5,   64, 1, 12,  6,  1},
            {"L=18304  IC=64   K=7 OC=64   p=9  d=3 ",  18304,   64, 7,   64, 1,  9,  3,  1},
            {"L=18304  IC=64   K=7 OC=64   p=36 d=12",  18304,   64, 7,   64, 1, 36, 12,  1},
            {"L=73216  IC=32   K=3 OC=32   p=1  d=1 ",  73216,   32, 3,   32, 1,  1,  1,  1},
            {"L=73216  IC=32   K=3 OC=32   p=2  d=2 ",  73216,   32, 3,   32, 1,  2,  2,  1},
            {"L=73216  IC=32   K=5 OC=32   p=4  d=2 ",  73216,   32, 5,   32, 1,  4,  2,  1},
            {"L=73216  IC=32   K=5 OC=32   p=12 d=6 ",  73216,   32, 5,   32, 1, 12,  6,  1},
            {"L=73216  IC=32   K=7 OC=32   p=9  d=3 ",  73216,   32, 7,   32, 1,  9,  3,  1},
            {"L=73216  IC=32   K=7 OC=32   p=36 d=12",  73216,   32, 7,   32, 1, 36, 12,  1},
            {"L=73216  IC=32   K=7 OC=1    p=3  d=1 ",  73216,   32, 7,    1, 1,  3,  1,  1},
        };

        // The staged sweep, in the order the wins landed -- every row from ONE run at the shape
        // table above, because a table assembled from several runs is not a table.
        const std::vector<Variant> variants = {
            {"stock",           "conv_2d_stock",         0, 0, 256},
            {"C 64-bit ptr",    "conv_2d_tiled2_8_1",    8, 1,  64},
            {"D 32-bit idx",    "conv_2d_tiled3_8_1",    8, 1,  64},
            {"E +KW const",     "conv_2d_tiled4_8_1_%d", 8, 1,  64},
            {"E +128 threads",  "conv_2d_tiled4_8_1_%d", 8, 1, 128},
            {"F tg-staged w",   "conv_2d_tiled5_8_1_%d", 8, 1, 128},
            {"E 4pos x 4ch",    "conv_2d_tiled4_4_4_%d", 4, 4, 128},
            {"E 2pos x 8ch",    "conv_2d_tiled4_8_2_%d", 8, 2,  64},
            {"D 8pos x 8ch",    "conv_2d_tiled3_8_8",    8, 8, 128},
        };

        // one pipeline per (variant, KW): a variant whose name carries a "%d" is compiled once for
        // each kernel width the shape table uses, and the shape loop picks the matching one.
        const int kws[] = {0, 1, 3, 5, 7};
        std::vector<std::vector<id<MTLComputePipelineState>>> pipes_kw(variants.size());
        std::vector<id<MTLComputePipelineState>> pipes;
        for (const auto & v : variants) {
            std::vector<id<MTLComputePipelineState>> row(8, nil);
            const bool templated = std::string(v.fn).find("%d") != std::string::npos;
            for (int kw : kws) {
                if (!templated && kw != 0) { continue; }
                char name[128];
                std::snprintf(name, sizeof(name), v.fn, kw);
                id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
                if (!fn) { std::fprintf(stderr, "missing kernel %s\n", name); return 1; }
                id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
                if (!p) { std::fprintf(stderr, "pipeline %s: %s\n", name, [[err description] UTF8String]); return 1; }
                row[kw] = p;
            }
            pipes_kw[&v - &variants[0]] = row;
            pipes.push_back(row[templated ? 7 : 0]);
        }

        // `maxTotalThreadsPerThreadgroup` is the only register-pressure signal Metal exposes without
        // a GPU capture: it falls below 1024 exactly when a kernel's register footprint stops a full
        // threadgroup fitting, so a variant that reads 256 here is one that has spilled.
        std::printf("occupancy limit (threads/threadgroup, 1024 = no register pressure):\n");
        for (size_t vi = 0; vi < variants.size(); ++vi) {
            std::printf("  %-16s %4d\n", variants[vi].label, (int)[pipes[vi] maxTotalThreadsPerThreadgroup]);
        }
        std::printf("\n");

        // The rooflines, first, because every number after them is read as a fraction of these.
        {
            id<MTLFunction> f1 = [lib newFunctionWithName:@"roofline_fma"];
            id<MTLComputePipelineState> p1 = [dev newComputePipelineStateWithFunction:f1 error:&err];
            id<MTLFunction> f2 = [lib newFunctionWithName:@"roofline_read"];
            id<MTLComputePipelineState> p2 = [dev newComputePipelineStateWithFunction:f2 error:&err];

            const int nthr = 65536, reps = 20000;
            id<MTLBuffer> o = [dev newBufferWithLength:nthr*4 options:MTLResourceStorageModePrivate];
            double best = 1e30;
            for (int r = 0; r < 3; ++r) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:p1];
                [e setBuffer:o offset:0 atIndex:0];
                [e setBytes:&reps length:4 atIndex:1];
                [e dispatchThreadgroups:MTLSizeMake(nthr/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
                [cb commit]; [cb waitUntilCompleted];
                best = std::min(best, [cb GPUEndTime] - [cb GPUStartTime]);
            }
            const double gflops = 2.0*(double) nthr*reps*16/best/1e9;

            const int n4 = 64*1024*1024/16;   // 64 MB, past any cache
            id<MTLBuffer> in = [dev newBufferWithLength:(size_t) n4*16 options:MTLResourceStorageModePrivate];
            double bbest = 1e30;
            for (int r = 0; r < 3; ++r) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:p2];
                [e setBuffer:in offset:0 atIndex:0];
                [e setBuffer:o  offset:0 atIndex:1];
                [e setBytes:&n4 length:4 atIndex:2];
                [e dispatchThreadgroups:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
                [e endEncoding];
                [cb commit]; [cb waitUntilCompleted];
                bbest = std::min(bbest, [cb GPUEndTime] - [cb GPUStartTime]);
            }
            std::printf("roofline: fma %.0f GFLOP/s   dram read %.0f GB/s\n\n",
                        gflops, (double) n4*16/bbest/1e9);
        }

        std::printf("%-38s %8s", "shape group", "GFLOP");
        for (const auto & v : variants) { std::printf(" %13s", v.label); }
        std::printf("\n");

        std::vector<double> total_ms(variants.size(), 0.0);
        std::vector<double> group_ms(variants.size(), 0.0);
        double total_gflop = 0.0, group_gflop = 0.0;
        int group_L = -1;
        std::string group_name;

        auto flush_group = [&]() {
            if (group_L < 0) { return; }
            std::printf("%-38s %8.3f", group_name.c_str(), group_gflop);
            for (size_t vi = 0; vi < variants.size(); ++vi) { std::printf(" %13.2f", group_ms[vi]); }
            std::printf("\n");
            std::fill(group_ms.begin(), group_ms.end(), 0.0);
            group_gflop = 0.0;
        };

        for (const auto & s : shapes) {
            if (s.IW != group_L) {
                flush_group();
                group_L = s.IW;
                char buf[64];
                std::snprintf(buf, sizeof(buf), "L=%d", s.IW);
                group_name = buf;
            }
            const int OW = out_len(s);
            const size_t n_w = (size_t) s.OC * s.IC * s.KW;
            const size_t n_x = (size_t) s.IC * s.IW;
            const size_t n_o = (size_t) s.OC * OW;

            std::vector<float> hw(n_w), hx(n_x), ref;
            std::srand(1234);
            for (auto & v : hw) { v = (float)(std::rand() % 2001 - 1000)/1000.0f; }
            for (auto & v : hx) { v = (float)(std::rand() % 2001 - 1000)/1000.0f; }
            const bool cpu_affordable = (double) n_o*s.IC*s.KW < 3e8;
            if (cpu_affordable) { cpu_reference(s, hw, hx, ref); }

            id<MTLBuffer> bw = [dev newBufferWithBytes:hw.data() length:n_w*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bx = [dev newBufferWithBytes:hx.data() length:n_x*4 options:MTLResourceStorageModeShared];
            id<MTLBuffer> bo = [dev newBufferWithLength:n_o*4 options:MTLResourceStorageModeShared];

            conv_args a = {};
            a.nb00 = 4;                       a.nb01 = 4*(uint64_t) s.KW;
            a.nb02 = a.nb01;                  a.nb03 = a.nb02*(uint64_t) s.IC;
            a.nb10 = 4;                       a.nb11 = 4*(uint64_t) s.IW;
            a.nb12 = a.nb11;                  a.nb13 = a.nb12*(uint64_t) s.IC;
            a.nb0  = 4;                       a.nb1  = 4*(uint64_t) OW;
            a.nb2  = a.nb1;                   a.nb3  = a.nb2*(uint64_t) s.OC;
            a.IW = s.IW; a.IH = 1; a.KW = s.KW; a.KH = 1;
            a.IC = s.IC; a.OC = s.OC; a.OW = OW; a.OH = 1; a.N = 1;
            a.s0 = s.s0; a.s1 = 1; a.p0 = s.p0; a.p1 = 0; a.d0 = s.d0; a.d1 = 1;

            const double gflop = 2.0*(double) OW*s.OC*s.IC*s.KW/1e9*s.calls;
            total_gflop += gflop;
            group_gflop += gflop;

            auto encode_for = [&](size_t vi, id<MTLCommandBuffer> cb, int reps) {
                const Variant & v = variants[vi];
                id<MTLComputePipelineState> pipe = pipes_kw[vi][s.KW] ? pipes_kw[vi][s.KW] : pipes_kw[vi][0];
                const int nth = std::min<int>(v.nth, (int)[pipe maxTotalThreadsPerThreadgroup]);
                MTLSize tg, tpt;
                if (v.oc_t == 0) {
                    tg  = MTLSizeMake(((uint64_t) n_o + nth - 1)/nth, 1, 1);
                    tpt = MTLSizeMake(nth, 1, 1);
                } else {
                    tg  = MTLSizeMake((OW + nth*v.tw - 1)/(nth*v.tw), (s.OC + v.oc_t - 1)/v.oc_t, 1);
                    tpt = MTLSizeMake(nth, 1, 1);
                }
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:pipe];
                [enc setBytes:&a length:sizeof(a) atIndex:0];
                [enc setBuffer:bw offset:0 atIndex:1];
                [enc setBuffer:bx offset:0 atIndex:2];
                [enc setBuffer:bo offset:0 atIndex:3];
                for (int i = 0; i < reps; ++i) { [enc dispatchThreadgroups:tg threadsPerThreadgroup:tpt]; }
                [enc endEncoding];
            };

            // the oracle: stock's output, itself checked against the CPU where that is affordable
            std::memset([bo contents], 0, n_o*4);
            { id<MTLCommandBuffer> cb = [q commandBuffer]; encode_for(0, cb, 1); [cb commit]; [cb waitUntilCompleted]; }
            std::vector<float> oracle((const float *)[bo contents], (const float *)[bo contents] + n_o);
            if (cpu_affordable) {
                double e = 0.0;
                for (size_t i = 0; i < n_o; ++i) { e = std::max(e, (double) std::fabs(oracle[i] - ref[i])); }
                if (e > 2e-3) { std::printf("\nSTOCK DISAGREES WITH CPU on %s (max abs err %.3g)\n", s.name, e); }
            }

            for (size_t vi = 0; vi < variants.size(); ++vi) {
                const Variant & v = variants[vi];

                std::memset([bo contents], 0, n_o*4);
                { id<MTLCommandBuffer> cb = [q commandBuffer]; encode_for(vi, cb, 1); [cb commit]; [cb waitUntilCompleted]; }
                const float * got = (const float *)[bo contents];
                double maxerr = 0.0;
                for (size_t i = 0; i < n_o; ++i) { maxerr = std::max(maxerr, (double) std::fabs(got[i] - oracle[i])); }
                if (maxerr > 2e-3) {
                    std::printf("\nWRONG: %s on %s (max abs err %.3g)\n", v.label, s.name, maxerr);
                    total_ms[vi] += 1e9; group_ms[vi] += 1e9;
                    continue;
                }

                { id<MTLCommandBuffer> cb = [q commandBuffer]; encode_for(vi, cb, 2); [cb commit]; [cb waitUntilCompleted]; }

                double best = 1e30;
                for (int r = 0; r < 3; ++r) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    encode_for(vi, cb, iters);
                    [cb commit];
                    [cb waitUntilCompleted];
                    best = std::min(best, ([cb GPUEndTime] - [cb GPUStartTime])*1000.0/iters);
                }
                total_ms[vi] += best*s.calls;
                group_ms[vi] += best*s.calls;
            }
        }
        flush_group();

        std::printf("%-38s %8.3f", "TOTAL ms", total_gflop);
        for (size_t vi = 0; vi < variants.size(); ++vi) { std::printf(" %13.2f", total_ms[vi]); }
        std::printf("\n%-38s %8s", "GFLOP/s", "");
        for (size_t vi = 0; vi < variants.size(); ++vi) {
            std::printf(" %13.0f", total_gflop/(total_ms[vi]/1000.0));
        }
        std::printf("\n%-38s %8s", "speedup over stock", "");
        for (size_t vi = 0; vi < variants.size(); ++vi) {
            std::printf(" %12.2fx", total_ms[0]/total_ms[vi]);
        }
        std::printf("\n");
    }
    return 0;
}
