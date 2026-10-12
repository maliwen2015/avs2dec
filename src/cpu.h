#ifndef AVS2DEC_SRC_CPU_H
#define AVS2DEC_SRC_CPU_H

#include <stdint.h>

/*
 * 是否编译 x86 SSE/SSSE3/SSE4.1 intrinsics 实现路径。
 *
 * wasm 构建下 Emscripten 提供 x86 intrinsics 兼容头 (tmmintrin.h/smmintrin.h),
 * 在 -msimd128 -msse4.1 下把 128 位 SSE 内建映射到 wasm SIMD 指令, 因此
 * wasm 也走同一份 SSE4.1 内核 (AVX2/AVX512 不支持, 由 cpu flags 关闭)。
 */
#if defined(__x86_64__) || defined(__i386__)
#define AVS2_HAVE_X86_SIMD 1
#elif defined(__wasm_simd128__)
#define AVS2_HAVE_X86_SIMD 1
#define AVS2_HAVE_X86_SIMD_WASM 1
#else
#define AVS2_HAVE_X86_SIMD 0
#endif

typedef struct {
    int sse2;
    int ssse3;
    int sse41;
    int avx2;
    int avx512;
    int neon;
    int dotprod;  /* ARM NEON dot product (SDOT/USDOT) */
} avs2_cpu_flags;

void avs2_cpu_detect(avs2_cpu_flags *flags);
int avs2_cpu_count(void);

#endif
