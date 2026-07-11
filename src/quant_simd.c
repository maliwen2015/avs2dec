/*
 * quant_simd.c - 反量化 SIMD 实现 (x86)
 *
 * 当前实现:
 *   - AVX2: 批量反量化, 一次处理 16 个 int16 系数
 *
 * 算法:
 *   coeff[i] = (coeff[i] * scale + add) >> shift, 饱和到 int16
 *   零系数保持零 (0*scale=0, 全零块跳过)
 *
 * SIMD 要点:
 *   - int16 扩展到 int32 做 32-bit 乘法 (避免溢出: level*scale 最大约 2^31)
 *   - _mm256_mullo_epi32: 32-bit 乘法取低 32 位
 *   - _mm256_packs_epi32: 有符号饱和打包 32→16 (等效 clip 到 [-32768,32767])
 *   - _mm256_permute4x64_epi64: 重排 packs 后的 128-bit 通道顺序
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

#include <immintrin.h>

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
 * AVX2: 一次处理 16 个 int16 系数
 * =========================================================================== */
static void dequant_block_avx2_impl(int16_t *coeff, int n, int scale, int shift)
{
    int add = (shift > 0) ? (1 << (shift - 1)) : 0;
    __m256i v_scale = _mm256_set1_epi32(scale);
    __m256i v_add = _mm256_set1_epi32(add);
    __m128i v_shift = _mm_cvtsi32_si128(shift);
    int i;

    for (i = 0; i <= n - 16; i += 16) {
        __m128i v0 = _mm_load_si128((const __m128i*)(coeff + i));
        __m128i v1 = _mm_load_si128((const __m128i*)(coeff + i + 8));

        /* 全零检测 (16 个系数) */
        __m128i v_or = _mm_or_si128(v0, v1);
        if (_mm_test_all_zeros(v_or, v_or)) continue;

        /* int16 → int32 (256-bit, 8 个系数) */
        __m256i e0 = _mm256_cvtepi16_epi32(v0);
        __m256i e1 = _mm256_cvtepi16_epi32(v1);

        /* (coeff * scale + add) >> shift */
        e0 = _mm256_sra_epi32(_mm256_add_epi32(_mm256_mullo_epi32(e0, v_scale), v_add), v_shift);
        e1 = _mm256_sra_epi32(_mm256_add_epi32(_mm256_mullo_epi32(e1, v_scale), v_add), v_shift);

        /* 饱和打包回 int16 + 重排 128-bit 通道 */
        __m256i packed = _mm256_packs_epi32(e0, e1);
        packed = _mm256_permute4x64_epi64(packed, 0xD8);  /* 0,2,1,3 → [e0, e1] */

        _mm_store_si128((__m128i*)(coeff + i), _mm256_castsi256_si128(packed));
        _mm_store_si128((__m128i*)(coeff + i + 8), _mm256_extracti128_si256(packed, 1));
    }

    /* 8 像素块 (SSE4.1 路径) */
    if (i <= n - 8) {
        __m128i v0 = _mm_load_si128((const __m128i*)(coeff + i));
        if (!_mm_test_all_zeros(v0, v0)) {
            __m128i lo = _mm_cvtepi16_epi32(v0);
            __m128i hi = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(v0, v0));
            __m128i v_s = _mm256_castsi256_si128(v_scale);
            __m128i v_a = _mm256_castsi256_si128(v_add);
            lo = _mm_sra_epi32(_mm_add_epi32(_mm_mullo_epi32(lo, v_s), v_a), v_shift);
            hi = _mm_sra_epi32(_mm_add_epi32(_mm_mullo_epi32(hi, v_s), v_a), v_shift);
            _mm_store_si128((__m128i*)(coeff + i), _mm_packs_epi32(lo, hi));
        }
        i += 8;
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

void avs2_quant_init_avx2(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.dequant_block = dequant_block_avx2_impl;
}

void avs2_quant_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }

#else /* 非 x86 平台 */

void avs2_quant_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_quant_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_quant_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }

#endif
