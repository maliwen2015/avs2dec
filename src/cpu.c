#include "cpu.h"

#if defined(_WIN32)
#include <windows.h>
#elif defined(__GNUC__)
#include <sched.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define AVS2_X86
#endif

#ifdef AVS2_X86

#if defined(_MSC_VER) || defined(_M_X64)
#include <intrin.h>
static void avs2_cpuid(int info[4], int leaf) { __cpuid(info, leaf); }
static void avs2_cpuidex(int info[4], int leaf, int subleaf) { __cpuidex(info, leaf, subleaf); }
#elif defined(__GNUC__)
#include <cpuid.h>
static void avs2_cpuid(int info[4], int leaf) {
    unsigned int a, b, c, d;
    __cpuid(leaf, a, b, c, d);
    info[0] = a; info[1] = b; info[2] = c; info[3] = d;
}
static void avs2_cpuidex(int info[4], int leaf, int subleaf) {
    unsigned int a, b, c, d;
    __cpuid_count(leaf, subleaf, a, b, c, d);
    info[0] = a; info[1] = b; info[2] = c; info[3] = d;
}
#endif

/* 检测 x86 CPU 的 SIMD 支持情况 */
static void avs2_cpu_detect_x86(avs2_cpu_flags *flags)
{
    int info[4];
    int max_leaf;

    flags->sse2 = 0;
    flags->ssse3 = 0;
    flags->sse41 = 0;
    flags->avx2 = 0;
    flags->avx512 = 0;
    flags->neon = 0;

    avs2_cpuid(info, 0);
    max_leaf = info[0];
    if (max_leaf < 1) return;

    /* leaf 1: EDX[26]=SSE2, ECX[0]=SSE3, ECX[9]=SSSE3, ECX[19]=SSE4.1, ECX[28]=AVX */
    avs2_cpuid(info, 1);
    if (info[3] & (1 << 26)) flags->sse2 = 1;
    if (info[2] & (1 << 9))  flags->ssse3 = 1;
    if (info[2] & (1 << 19)) flags->sse41 = 1;

    /* AVX 需要 OS 支持 (XGETBV) */
    int has_avx = (info[2] & (1 << 28)) != 0;
    if (has_avx) {
        unsigned long long xcr = 0;
        /* 检查 XGETBV 是否启用 YMM 寄存器 */
#if defined(_MSC_VER)
        xcr = _xgetbv(0);
#else
        /* GCC/MinGW: 用内联汇编避免 -mxsave 编译选项依赖 */
        unsigned int xcr0_hi, xcr0_lo;
        __asm__ volatile ("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
        xcr = ((unsigned long long)xcr0_hi << 32) | xcr0_lo;
#endif
        if ((xcr & 0x6) == 0x6) {
            /* leaf 7: EBX[5]=AVX2, EBX[16]=AVX512 */
            avs2_cpuidex(info, 7, 0);
            if (info[1] & (1 << 5))  flags->avx2 = 1;
            if (info[1] & (1 << 16)) flags->avx512 = 1;
        }
    }
}

#endif /* AVS2_X86 */

void avs2_cpu_detect(avs2_cpu_flags *flags)
{
    /* 初始化全部为 0 */
    flags->sse2 = 0;
    flags->ssse3 = 0;
    flags->sse41 = 0;
    flags->avx2 = 0;
    flags->avx512 = 0;
    flags->neon = 0;

#if defined(AVS2_X86)
    avs2_cpu_detect_x86(flags);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    flags->neon = 1;
#endif
}

int avs2_cpu_count(void)
{
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#elif defined(__GNUC__)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? (int)n : 1;
#else
    return 1;
#endif
}
