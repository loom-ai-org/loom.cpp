// ADR-079 P4.32f f2 (ggml-0037): where a cold stream's time goes on the ARM1176 -- reads vs writes, VFP vs integer
// load/store multiples, one to three lines a step, and a 32-byte-aligned buffer against one 8 bytes off.
// Build (ARMv6, loom-armv6-build image): c++ -O3 -std=c++17 bench55_streams.cpp -o bench55_streams
// Run on the board: ./bench55_streams [reps]   (sets RunFast itself)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
#define NOINL __attribute__((noinline))
NOINL static void rd_vfp(int32_t b, const float * x) {
    __asm__ volatile("1:\n\tpld [%[x], #192]\n\tvldmia %[x]!, {s0-s7}\n\tsubs %[b], %[b], #1\n\tbne 1b\n\t"
                     : [x] "+r"(x), [b] "+r"(b) : : "s0","s1","s2","s3","s4","s5","s6","s7","cc","memory"); }
NOINL static void rd_int(int32_t b, const float * x) {
    __asm__ volatile("push {r4-r11}\n\t1:\n\tpld [%[x], #192]\n\tldm %[x]!, {r4-r11}\n\tsubs %[b], %[b], #1\n\tbne 1b\n\tpop {r4-r11}\n\t"
                     : [x] "+r"(x), [b] "+r"(b) : : "cc","memory"); }
NOINL static void wr_vfp(int32_t b, float * z) {
    __asm__ volatile("1:\n\tvstmia %[z]!, {s0-s7}\n\tsubs %[b], %[b], #1\n\tbne 1b\n\t"
                     : [z] "+r"(z), [b] "+r"(b) : : "cc","memory"); }
NOINL static void wr_int(int32_t b, float * z) {
    __asm__ volatile("push {r4-r11}\n\tmov r4,#0\n\tmov r5,#0\n\tmov r6,#0\n\tmov r7,#0\n\tmov r8,#0\n\tmov r9,#0\n\tmov r10,#0\n\tmov r11,#0\n\t"
                     "1:\n\tstm %[z]!, {r4-r11}\n\tsubs %[b], %[b], #1\n\tbne 1b\n\tpop {r4-r11}\n\t"
                     : [z] "+r"(z), [b] "+r"(b) : : "cc","memory"); }
NOINL static void cp_vfp(int32_t b, float * z, const float * x) {
    __asm__ volatile("1:\n\tpld [%[x], #192]\n\tvldmia %[x]!, {s0-s7}\n\tsubs %[b], %[b], #1\n\tvstmia %[z]!, {s0-s7}\n\tbne 1b\n\t"
                     : [x] "+r"(x), [z] "+r"(z), [b] "+r"(b) : : "s0","s1","s2","s3","s4","s5","s6","s7","cc","memory"); }
NOINL static void cp_int(int32_t b, float * z, const float * x) {
    __asm__ volatile("push {r4-r11}\n\t1:\n\tpld [%[x], #192]\n\tldm %[x]!, {r4-r11}\n\tsubs %[b], %[b], #1\n\tstm %[z]!, {r4-r11}\n\tbne 1b\n\tpop {r4-r11}\n\t"
                     : [x] "+r"(x), [z] "+r"(z), [b] "+r"(b) : : "cc","memory"); }
// integer loads, moved into the VFP for the multiply, VFP store
NOINL static void mul_intld(int32_t b, float * z, const float * x, float s) {
    __asm__ volatile("push {r4-r11}\n\tvmov s16, %[s]\n\t1:\n\tpld [%[x], #192]\n\tldm %[x]!, {r4-r11}\n\t"
                     "vmov s0, s1, r4, r5\n\tvmov s2, s3, r6, r7\n\tvmov s4, s5, r8, r9\n\tvmov s6, s7, r10, r11\n\t"
                     "vmul.f32 s0, s0, s16\n\tvmul.f32 s1, s1, s16\n\tvmul.f32 s2, s2, s16\n\tvmul.f32 s3, s3, s16\n\t"
                     "vmul.f32 s4, s4, s16\n\tvmul.f32 s5, s5, s16\n\tvmul.f32 s6, s6, s16\n\tvmul.f32 s7, s7, s16\n\t"
                     "subs %[b], %[b], #1\n\tvstmia %[z]!, {s0-s7}\n\tbne 1b\n\tpop {r4-r11}\n\t"
                     : [x] "+r"(x), [z] "+r"(z), [b] "+r"(b) : [s] "r"(s) : "s0","s1","s2","s3","s4","s5","s6","s7","s16","cc","memory"); }
// two blocks in flight: load the next 8 before storing the previous 8
NOINL static void mul_2x(int32_t b, float * z, const float * x, float s) {
    __asm__ volatile("vmov s16, %[s]\n\t1:\n\tpld [%[x], #192]\n\tpld [%[x], #224]\n\tvldmia %[x]!, {s0-s7}\n\tvldmia %[x]!, {s8-s15}\n\t"
                     "vmul.f32 s0, s0, s16\n\tvmul.f32 s1, s1, s16\n\tvmul.f32 s2, s2, s16\n\tvmul.f32 s3, s3, s16\n\t"
                     "vmul.f32 s4, s4, s16\n\tvmul.f32 s5, s5, s16\n\tvmul.f32 s6, s6, s16\n\tvmul.f32 s7, s7, s16\n\t"
                     "vmul.f32 s8, s8, s16\n\tvmul.f32 s9, s9, s16\n\tvmul.f32 s10, s10, s16\n\tvmul.f32 s11, s11, s16\n\t"
                     "vmul.f32 s12, s12, s16\n\tvmul.f32 s13, s13, s16\n\tvmul.f32 s14, s14, s16\n\tvmul.f32 s15, s15, s16\n\t"
                     "subs %[b], %[b], #2\n\tvstmia %[z]!, {s0-s15}\n\tbgt 1b\n\t"
                     : [x] "+r"(x), [z] "+r"(z), [b] "+r"(b) : [s] "r"(s) : "s0","s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","s12","s13","s14","s15","s16","cc","memory"); }
NOINL static void mul_3x(int32_t b, float * z, const float * x, float s) {
    __asm__ volatile("vmov s31, %[s]\n\t1:\n\tpld [%[x], #192]\n\tpld [%[x], #224]\n\tpld [%[x], #256]\n\t"
                     "vldmia %[x]!, {s0-s23}\n\t"
                     "vmul.f32 s0, s0, s31\n\tvmul.f32 s1, s1, s31\n\tvmul.f32 s2, s2, s31\n\tvmul.f32 s3, s3, s31\n\t"
                     "vmul.f32 s4, s4, s31\n\tvmul.f32 s5, s5, s31\n\tvmul.f32 s6, s6, s31\n\tvmul.f32 s7, s7, s31\n\t"
                     "vmul.f32 s8, s8, s31\n\tvmul.f32 s9, s9, s31\n\tvmul.f32 s10, s10, s31\n\tvmul.f32 s11, s11, s31\n\t"
                     "vmul.f32 s12, s12, s31\n\tvmul.f32 s13, s13, s31\n\tvmul.f32 s14, s14, s31\n\tvmul.f32 s15, s15, s31\n\t"
                     "vmul.f32 s16, s16, s31\n\tvmul.f32 s17, s17, s31\n\tvmul.f32 s18, s18, s31\n\tvmul.f32 s19, s19, s31\n\t"
                     "vmul.f32 s20, s20, s31\n\tvmul.f32 s21, s21, s31\n\tvmul.f32 s22, s22, s31\n\tvmul.f32 s23, s23, s31\n\t"
                     "subs %[b], %[b], #3\n\tvstmia %[z]!, {s0-s23}\n\tbgt 1b\n\t"
                     : [x] "+r"(x), [z] "+r"(z), [b] "+r"(b) : [s] "r"(s)
                     : "s0","s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21","s22","s23","s31","cc","memory"); }
NOINL static void mul_1x(int32_t b, float * z, const float * x, float s) {
    __asm__ volatile("vmov s16, %[s]\n\t1:\n\tpld [%[x], #192]\n\tvldmia %[x]!, {s0-s7}\n\t"
                     "vmul.f32 s0, s0, s16\n\tvmul.f32 s1, s1, s16\n\tvmul.f32 s2, s2, s16\n\tvmul.f32 s3, s3, s16\n\t"
                     "vmul.f32 s4, s4, s16\n\tvmul.f32 s5, s5, s16\n\tvmul.f32 s6, s6, s16\n\tvmul.f32 s7, s7, s16\n\t"
                     "subs %[b], %[b], #1\n\tvstmia %[z]!, {s0-s7}\n\tbne 1b\n\t"
                     : [x] "+r"(x), [z] "+r"(z), [b] "+r"(b) : [s] "r"(s) : "s0","s1","s2","s3","s4","s5","s6","s7","s16","cc","memory"); }
template <class F> static double ns_per(F f, int64_t n, int reps) {
    f(); std::vector<double> t;
    for (int i = 0; i < reps; ++i) { double t0 = now(); f(); t.push_back((now() - t0) * 1e9 / (double) n); }
    std::sort(t.begin(), t.end()); return t[t.size() / 2];
}
int main(int argc, char ** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 7;
    { uint32_t f; __asm__ volatile("vmrs %0, fpscr" : "=r"(f)); f |= (3u << 24); __asm__ volatile("vmsr fpscr, %0" : : "r"(f)); }
    const int64_t N = 130048 * 24; const int32_t B = (int32_t) (N / 8);
    // 32-byte aligned like ggml's CPU buffers, then the same at +8 bytes (std::vector's placement)
    float * XA = (float *) aligned_alloc(32, N * 4 + 64), * ZA = (float *) aligned_alloc(32, N * 4 + 64);
    for (int64_t i = 0; i < N + 16; ++i) { XA[i] = 1.5f; ZA[i] = 0.f; }
    auto row = [&](const char * n, double v) { printf("%-28s %6.1f\n", n, v); };
    for (int off = 0; off <= 2; off += 2) {
        float * X = XA + off, * Z = ZA + off;
        printf("-- offset %d bytes\n", off * 4);
        row("memcpy", ns_per([&] { memcpy(Z, X, N * 4); }, N, reps));
        row("read vldmia", ns_per([&] { rd_vfp(B, X); }, N, reps));
        row("copy vldmia/vstmia", ns_per([&] { cp_vfp(B, Z, X); }, N, reps));
        row("mul, 8 a step", ns_per([&] { mul_1x(B, Z, X, 0.5f); }, N, reps));
        row("mul, 16 a step", ns_per([&] { mul_2x(B, Z, X, 0.5f); }, N, reps));
        row("mul, 24 a step", ns_per([&] { mul_3x(B - B % 3, Z, X, 0.5f); }, N, reps));
    }
    return 0;
}
