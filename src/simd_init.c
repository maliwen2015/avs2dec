/*
 * simd_init.c - SIMD 实现注册总入口
 *
 * 根据编译期可用的指令集, 调用各模块的 SIMD init.
 * 每个 *_init_simd 函数在内部检查 cpu flags, 按优先级注册.
 *
 * 优先级从低到高调用: SSE4.1 -> AVX2 -> AVX512 (x86) / NEON (arm)
 * 后调用者覆盖先调用者, 确保最高优先级实现生效.
 */

#include "internal.h"

/* 反变换 SIMD init (由 itx_simd.c 提供) */
void avs2_itx_init_simd(const avs2_cpu_flags *cpu)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    extern void avs2_itx_init_sse41(const avs2_cpu_flags *flags);
    extern void avs2_itx_init_avx2(const avs2_cpu_flags *flags);
    extern void avs2_itx_init_avx512(const avs2_cpu_flags *flags);

    /* 低优先级先注册, 高优先级后注册并覆盖 */
    if (cpu->sse41)  avs2_itx_init_sse41(cpu);
    if (cpu->avx2)   avs2_itx_init_avx2(cpu);
    if (cpu->avx512) avs2_itx_init_avx512(cpu);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    extern void avs2_itx_init_neon(const avs2_cpu_flags *flags);
    if (cpu->neon) avs2_itx_init_neon(cpu);
#else
    (void)cpu;
#endif
}

void avs2_mc_init_simd(const avs2_cpu_flags *cpu)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    extern void avs2_mc_init_sse41(const avs2_cpu_flags *flags);
    extern void avs2_mc_init_avx2(const avs2_cpu_flags *flags);
    if (cpu->sse41)  avs2_mc_init_sse41(cpu);
    if (cpu->avx2)   avs2_mc_init_avx2(cpu);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    extern void avs2_mc_init_neon(const avs2_cpu_flags *flags);
    if (cpu->neon) avs2_mc_init_neon(cpu);
#else
    (void)cpu;
#endif
}

void avs2_ipred_init_simd(const avs2_cpu_flags *cpu)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    extern void avs2_ipred_init_sse41(const avs2_cpu_flags *flags);
    extern void avs2_ipred_init_avx2(const avs2_cpu_flags *flags);
    if (cpu->sse41)  avs2_ipred_init_sse41(cpu);
    if (cpu->avx2)   avs2_ipred_init_avx2(cpu);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    extern void avs2_ipred_init_neon(const avs2_cpu_flags *flags);
    if (cpu->neon) avs2_ipred_init_neon(cpu);
#else
    (void)cpu;
#endif
}

void avs2_lf_init_simd(const avs2_cpu_flags *cpu)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    extern void avs2_lf_init_sse41(const avs2_cpu_flags *flags);
    extern void avs2_lf_init_avx2(const avs2_cpu_flags *flags);
    if (cpu->sse41)  avs2_lf_init_sse41(cpu);
    if (cpu->avx2)   avs2_lf_init_avx2(cpu);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    extern void avs2_lf_init_neon(const avs2_cpu_flags *flags);
    if (cpu->neon) avs2_lf_init_neon(cpu);
#else
    (void)cpu;
#endif
}

void avs2_sao_init_simd(const avs2_cpu_flags *cpu)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    extern void avs2_sao_init_sse41(const avs2_cpu_flags *flags);
    extern void avs2_sao_init_avx2(const avs2_cpu_flags *flags);
    if (cpu->sse41)  avs2_sao_init_sse41(cpu);
    if (cpu->avx2)   avs2_sao_init_avx2(cpu);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    extern void avs2_sao_init_neon(const avs2_cpu_flags *flags);
    if (cpu->neon) avs2_sao_init_neon(cpu);
#else
    (void)cpu;
#endif
}

void avs2_alf_init_simd(const avs2_cpu_flags *cpu)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    extern void avs2_alf_init_sse41(const avs2_cpu_flags *flags);
    extern void avs2_alf_init_avx2(const avs2_cpu_flags *flags);
    if (cpu->sse41)  avs2_alf_init_sse41(cpu);
    if (cpu->avx2)   avs2_alf_init_avx2(cpu);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    extern void avs2_alf_init_neon(const avs2_cpu_flags *flags);
    if (cpu->neon) avs2_alf_init_neon(cpu);
#else
    (void)cpu;
#endif
}

void avs2_quant_init_simd(const avs2_cpu_flags *cpu)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    extern void avs2_quant_init_sse41(const avs2_cpu_flags *flags);
    extern void avs2_quant_init_avx2(const avs2_cpu_flags *flags);
    if (cpu->sse41)  avs2_quant_init_sse41(cpu);
    if (cpu->avx2)   avs2_quant_init_avx2(cpu);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)
    extern void avs2_quant_init_neon(const avs2_cpu_flags *flags);
    if (cpu->neon) avs2_quant_init_neon(cpu);
#else
    (void)cpu;
#endif
}
