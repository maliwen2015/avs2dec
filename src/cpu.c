#include "cpu.h"

#if defined(__GNUC__) && defined(__linux__)
#include <stdio.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#elif defined(__GNUC__)
#include <sched.h>
#include <unistd.h>
#endif

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
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
    flags->dotprod = 0;

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
    flags->dotprod = 0;

#if defined(AVS2_X86)
    avs2_cpu_detect_x86(flags);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    flags->neon = 1;
  #if defined(__ARM_FEATURE_DOTPROD)
    /* 编译时已启用 dotprod 指令, 运行时确认 CPU 支持 */
    #if defined(__aarch64__) && defined(__linux__)
    flags->dotprod = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) ? 1 : 0;
    #else
    flags->dotprod = 1;
    #endif
  #endif
#endif
}

#if defined(__GNUC__) && defined(__linux__)
/* 容器/cgroup CPU 配额检测: 返回 cgroup 允许的 CPU 数 (向上取整), <=0 表示无限制.
 * 容器中逻辑 CPU 数可能远大于配额 (如 3 核机器配额 2.0 CPU),
 * 线程数超过配额会导致 cfs 配额节流 + 缓存抖动, 实测吞吐反而下降. */
static int avs2_cpu_quota_count(void)
{
    /* cgroup v2: /sys/fs/cgroup/cpu.max = "<quota> <period>" 或 "max <period>" */
    FILE *f = fopen("/sys/fs/cgroup/cpu.max", "r");
    if (f) {
        char quota[32];
        long period = 0;
        if (fscanf(f, "%31s %ld", quota, &period) == 2 && quota[0] != 'm' && period > 0) {
            long q = atol(quota);
            fclose(f);
            if (q > 0) {
                long n = (q + period - 1) / period;  /* 向上取整 */
                return (int)n;
            }
        } else {
            fclose(f);
        }
    }
    /* cgroup v1: /sys/fs/cgroup/cpu/cpu.cfs_quota_us (=-1 无限制) + cpu.cfs_period_us */
    f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r");
    if (f) {
        long q = 0, period = 0;
        int ok = (fscanf(f, "%ld", &q) == 1);
        fclose(f);
        if (ok && q > 0) {
            f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r");
            if (f) {
                if (fscanf(f, "%ld", &period) == 1 && period > 0) {
                    fclose(f);
                    long n = (q + period - 1) / period;
                    return (int)n;
                }
                fclose(f);
            }
        }
    }
    return 0;
}
#endif

int avs2_cpu_count(void)
{
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#elif defined(__GNUC__)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    int count = (n > 0) ? (int)n : 1;
#if defined(__linux__)
    /* 取逻辑 CPU 数与 cgroup 配额的较小值, 避免容器中线程数超配额 */
    int quota = avs2_cpu_quota_count();
    if (quota > 0 && quota < count) {
        count = quota;
    }
#endif
    return count;
#else
    return 1;
#endif
}
