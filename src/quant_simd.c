/*
 * quant_simd.c - 反量化 SIMD 实现 (x86 SSE4.1)
 *
 * 当前实现:
 *   - SSE4.1: 批量反量化, 一次处理 8 个 int16 系数
 *
 * 算法:
 *   coeff[i] = (coeff[i] * scale + add) >> shift, 饱和到 int16
 *   零系数保持零 (0*scale=0, 全零块跳过)
 *
 * SIMD 要点:
 *   - int16 扩展到 int32 做 32-bit 乘法 (避免溢出: level*scale 最大约 2^31)
 *   - _mm_mullo_epi32: 32-bit 乘法取低 32 位
 *   - _mm_packs_epi32: 有符号饱和打包 32→16 (等效 clip 到 [-32768,32767])
 *   - _mm_test_all_zeros: 全零检测, 跳过稀疏零块
 *
 * 对齐: coeff 来自 avs2_mem_allocz (32 字节对齐), 步进 8 (16 字节),
 *       使用对齐 load/store.
 */

#include "internal.h"
#include "quant.h"

#ifndef AVS2_CLIP3
#define AVS2_CLIP3(lo, hi, x) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <smmintrin.h>

/* ===========================================================================
 * SSE4.1: 一次处理 8 个 int16 系数
 * =========================================================================== */
static void dequant_block_sse41(int16_t *coeff, int n, int scale, int shift)
{
    int add = (shift > 0) ? (1 << (shift - 1)) : 0;
    __m128i v_scale = _mm_set1_epi32(scale);
    __m128i v_add = _mm_set1_epi32(add);
    __m128i v_shift = _mm_cvtsi32_si128(shift);
    int i;

    for (i = 0; i <= n - 8; i += 8) {
        __m128i v0 = _mm_load_si128((const __m128i*)(coeff + i));

        /* 全零检测 */
        if (_mm_test_all_zeros(v0, v0)) continue;

        /* int16 → int32 (低 4 + 高 4) */
        __m128i lo = _mm_cvtepi16_epi32(v0);
        __m128i hi = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(v0, v0));

        /* (coeff * scale + add) >> shift */
        lo = _mm_sra_epi32(_mm_add_epi32(_mm_mullo_epi32(lo, v_scale), v_add), v_shift);
        hi = _mm_sra_epi32(_mm_add_epi32(_mm_mullo_epi32(hi, v_scale), v_add), v_shift);

        /* 饱和打包回 int16 */
        _mm_store_si128((__m128i*)(coeff + i), _mm_packs_epi32(lo, hi));
    }

    /* 标量处理剩余 */
    for (; i < n; i++) {
        if (coeff[i]) {
            int c = (shift > 0) ? ((coeff[i] * scale + add) >> shift) : (coeff[i] * scale);
            coeff[i] = (int16_t)AVS2_CLIP3(-32768, 32767, c);
        }
    }
}

/* ===========================================================================
 * 注册函数
 * =========================================================================== */

void avs2_quant_init_sse41(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.dequant_block = dequant_block_sse41;
}

void avs2_quant_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }

void avs2_quant_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }

#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)

#include <arm_neon.h>

/* ===========================================================================
 * NEON 反量化: 一次处理 8 个 int16 系数 (由 SSE4.1 逐行移植).
 *
 *   coeff[i] = (coeff[i] * scale + add) >> shift, 饱和到 int16
 *   - vmulq_s32 与 _mm_mullo_epi32 同为 32-bit 低 32 位乘法 (回绕语义一致)
 *   - vshlq_s32(负数移位) 与 _mm_sra_epi32 同为算术右移
 *   - vqmovn_s32 与 _mm_packs_epi32 同为有符号饱和窄化
 * =========================================================================== */
static void dequant_block_neon(int16_t *coeff, int n, int scale, int shift)
{
    int add = (shift > 0) ? (1 << (shift - 1)) : 0;
    int32x4_t v_scale = vdupq_n_s32(scale);
    int32x4_t v_add = vdupq_n_s32(add);
    int i;

    for (i = 0; i <= n - 8; i += 8) {
        int16x8_t v0 = vld1q_s16(coeff + i);

        /* 全零检测 (等价于 _mm_test_all_zeros) */
        {
            int64x2_t v64 = vreinterpretq_s64_s16(v0);
            if ((vgetq_lane_s64(v64, 0) | vgetq_lane_s64(v64, 1)) == 0)
                continue;
        }

        /* int16 → int32 (低 4 + 高 4) */
        int32x4_t lo = vmovl_s16(vget_low_s16(v0));
        int32x4_t hi = vmovl_s16(vget_high_s16(v0));

        /* (coeff * scale + add) >> shift */
        lo = vaddq_s32(vmulq_s32(lo, v_scale), v_add);
        hi = vaddq_s32(vmulq_s32(hi, v_scale), v_add);
        if (shift > 0) {
            lo = vshlq_s32(lo, vdupq_n_s32(-shift));
            hi = vshlq_s32(hi, vdupq_n_s32(-shift));
        }

        /* 饱和打包回 int16 */
        vst1q_s16(coeff + i, vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi)));
    }

    /* 标量处理剩余 */
    for (; i < n; i++) {
        if (coeff[i]) {
            int c = (shift > 0) ? ((coeff[i] * scale + add) >> shift) : (coeff[i] * scale);
            coeff[i] = (int16_t)AVS2_CLIP3(-32768, 32767, c);
        }
    }
}

/* ===========================================================================
 * 注册函数
 * =========================================================================== */

void avs2_quant_init_neon(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.dequant_block = dequant_block_neon;
}

/* SSE/AVX 接口空实现以供链接 (非 x86 平台不调用) */
void avs2_quant_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_quant_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_quant_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }

#else /* 其它平台: 回退到 C */

/* NEON 路径回退到 C, 保留空实现以供链接 */
void avs2_quant_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_quant_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_quant_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_quant_init_neon(const avs2_cpu_flags *flags) { (void)flags; }

#endif
