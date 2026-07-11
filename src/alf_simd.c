/*
 * alf_simd.c - ALF SIMD 实现 (x86 AVX2, 10-bit)
 *
 * 从 davs2 intrinsic_alf_256.c 的 alf_flt_one_block_sse256_10bit 移植。
 * 9 抽头 7x7 对称滤波, 内层每次处理 16 个 16-bit 像素。
 *
 * block1: 主体 7x7 滤波 -> alf_filter_block1_avx2 (AVX2)
 * block2: 4 角点边界修正, 仅 4 像素, SIMD 无收益 -> 保留 C 实现 (不在此覆盖)
 *
 * 依赖帧边缘 padding (AVS2_PAD_LUMA=64, AVS2_PAD_CHROMA=32) 处理越界读取。
 */

#include "internal.h"

#ifndef AVS2_CLIP3
#define AVS2_CLIP3(lo, hi, x) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <immintrin.h>

/* ---------------------------------------------------------------------------
 * 尾部不足 16 像素时的 32 位掩码表 (配合 _mm256_maskstore_epi32 使用)。
 * 每行 16 个 int16 = 32 字节 = 8 个 int32; -1 表示对应 int32 车道需写入。
 * 索引 = (lcu_width & 15) - 1。假定 lcu_width 为 4 的倍数, 实际仅用到
 * 索引 3/7/11 (对应尾部 4/8/12 像素), 其余行保留以匹配参考实现。
 * ------------------------------------------------------------------------- */
static const int16_t alf_mask_10bit[15][16] = {
    { -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
    { -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
    { -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
    { -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
    { -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
    { -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
    { -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0 },
    { -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0 },
    { -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0 },
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0 },
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0 },
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0 },
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0 },
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0 },
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0 }
};

/* ---------------------------------------------------------------------------
 * block1: 主体 7x7 滤波 AVX2 (10-bit)
 * 对应 alf_filter_block1_c / davs2 alf_flt_one_block_sse256_10bit。
 *
 * 系数加权布局 (与 C 参考一致):
 *   coeff[0]: ±3 行同列            coeff[5]: 0 行 ±3 列
 *   coeff[1]: ±2 行同列            coeff[6]: 0 行 ±2 列
 *   coeff[2]: +1 行右1 + -1 行左1  coeff[7]: 0 行 ±1 列
 *   coeff[3]: ±1 行同列            coeff[8]: 0 行 0 列 (中心, 由重建得到)
 *   coeff[4]: +1 行左1 + -1 行右1
 *
 * 每次迭代将一对像素 (img_pad6[j]+img_pad5[j] 等) 交错打包后用 _mm256_madd_epi16
 * 完成 系数*(p1+p2) 的 int32 乘加, 9 项累加后 (+32)>>6, 饱和到 [0, max_pixel]。
 * ------------------------------------------------------------------------- */
static void alf_filter_block1_avx2(uint8_t *_dst, const uint8_t *_src, int stride,
                                   int lcu_pix_x, int lcu_pix_y,
                                   int lcu_width, int lcu_height,
                                   int *alf_coeff, int b_top_avail, int b_down_avail,
                                   int bit_depth)
{
    if (bit_depth <= 8) {
        /* 8-bit AVX2: 加载 16 字节→cvtepu8 扩展为 uint16, 复用 16-bit madd 逻辑,
         * packus 压缩回 16 字节存储. 尾部不足 16 像素用标量处理. */
        uint8_t *dst8 = (uint8_t *)_dst;
        const uint8_t *src8 = (const uint8_t *)_src;
        const uint8_t *img_pad1_8, *img_pad2_8, *img_pad3_8, *img_pad4_8, *img_pad5_8, *img_pad6_8;
        const int max_pel = (1 << bit_depth) - 1;
        int i, j;
        int start_pos = b_top_avail ? (lcu_pix_y - 4) : lcu_pix_y;
        int end_pos   = b_down_avail ? (lcu_pix_y + lcu_height - 4) : (lcu_pix_y + lcu_height);
        int x_pos_end = lcu_width;
        int x_pos_end16 = x_pos_end & ~15;

        __m256i t00, t01, t10, t11, t20, t21, t30, t31, t40, t41;
        __m256i e00, e01, e10, e11, e20, e21, e30, e31, e40, e41;
        __m256i c0, c1, c2, c3, c4, c5, c6, c7, c8;
        __m256i s00, s01, s10, s11, s20, s21, s30, s31, s40, s41;
        __m256i s50, s51, s60, s61, s70, s71, s80, s81;
        __m256i ss1, ss2, s, m_add_offset;
        __m256i zero = _mm256_setzero_si256();
        __m256i max_val = _mm256_set1_epi16((short)max_pel);

        src8 += (start_pos * stride) + lcu_pix_x;
        dst8 += (start_pos * stride) + lcu_pix_x;

        c0 = _mm256_set1_epi16((short)alf_coeff[0]);
        c1 = _mm256_set1_epi16((short)alf_coeff[1]);
        c2 = _mm256_set1_epi16((short)alf_coeff[2]);
        c3 = _mm256_set1_epi16((short)alf_coeff[3]);
        c4 = _mm256_set1_epi16((short)alf_coeff[4]);
        c5 = _mm256_set1_epi16((short)alf_coeff[5]);
        c6 = _mm256_set1_epi16((short)alf_coeff[6]);
        c7 = _mm256_set1_epi16((short)alf_coeff[7]);
        c8 = _mm256_set1_epi16((short)alf_coeff[8]);
        m_add_offset = _mm256_set1_epi32(32);

        for (i = start_pos; i < end_pos; i++) {
            int y_up     = AVS2_CLIP3(start_pos, end_pos - 1, i - 1);
            int y_bottom = AVS2_CLIP3(start_pos, end_pos - 1, i + 1);
            img_pad1_8 = src8 + (y_bottom - i) * stride;
            img_pad2_8 = src8 + (y_up     - i) * stride;

            y_up     = AVS2_CLIP3(start_pos, end_pos - 1, i - 2);
            y_bottom = AVS2_CLIP3(start_pos, end_pos - 1, i + 2);
            img_pad3_8 = src8 + (y_bottom - i) * stride;
            img_pad4_8 = src8 + (y_up     - i) * stride;

            y_up     = AVS2_CLIP3(start_pos, end_pos - 1, i - 3);
            y_bottom = AVS2_CLIP3(start_pos, end_pos - 1, i + 3);
            img_pad5_8 = src8 + (y_bottom - i) * stride;
            img_pad6_8 = src8 + (y_up     - i) * stride;

            /* SIMD 主循环: 每次 16 像素 */
            for (j = 0; j < x_pos_end16; j += 16) {
                /* coeff[0]: ±3 行同列 */
                t00 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&img_pad6_8[j]));
                t01 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&img_pad5_8[j]));
                e00 = _mm256_unpacklo_epi16(t00, t01);
                e01 = _mm256_unpackhi_epi16(t00, t01);
                s00 = _mm256_madd_epi16(e00, c0);
                s01 = _mm256_madd_epi16(e01, c0);

                /* coeff[1]: ±2 行同列 */
                t10 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&img_pad4_8[j]));
                t11 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&img_pad3_8[j]));
                e10 = _mm256_unpacklo_epi16(t10, t11);
                e11 = _mm256_unpackhi_epi16(t10, t11);
                s10 = _mm256_madd_epi16(e10, c1);
                s11 = _mm256_madd_epi16(e11, c1);

                /* coeff[2]: +1 行右1 + -1 行左1 (交叉) */
                t20 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&img_pad2_8[j - 1]));
                t21 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&img_pad1_8[j + 1]));
                e20 = _mm256_unpacklo_epi16(t20, t21);
                e21 = _mm256_unpackhi_epi16(t20, t21);
                s20 = _mm256_madd_epi16(e20, c2);
                s21 = _mm256_madd_epi16(e21, c2);

                /* coeff[3]: ±1 行同列 */
                t30 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&img_pad2_8[j]));
                t31 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&img_pad1_8[j]));
                e30 = _mm256_unpacklo_epi16(t30, t31);
                e31 = _mm256_unpackhi_epi16(t30, t31);
                s30 = _mm256_madd_epi16(e30, c3);
                s31 = _mm256_madd_epi16(e31, c3);

                /* coeff[4]: +1 行左1 + -1 行右1 (交叉) */
                t40 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&img_pad2_8[j + 1]));
                t41 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&img_pad1_8[j - 1]));
                e40 = _mm256_unpacklo_epi16(t40, t41);
                e41 = _mm256_unpackhi_epi16(t40, t41);
                s40 = _mm256_madd_epi16(e40, c4);
                s41 = _mm256_madd_epi16(e41, c4);

                /* coeff[5]: 0 行 ±3 列 */
                t40 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src8[j - 3]));
                t41 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src8[j + 3]));
                e40 = _mm256_unpacklo_epi16(t40, t41);
                e41 = _mm256_unpackhi_epi16(t40, t41);
                s50 = _mm256_madd_epi16(e40, c5);
                s51 = _mm256_madd_epi16(e41, c5);

                /* coeff[6]: 0 行 ±2 列 */
                t40 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src8[j - 2]));
                t41 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src8[j + 2]));
                e40 = _mm256_unpacklo_epi16(t40, t41);
                e41 = _mm256_unpackhi_epi16(t40, t41);
                s60 = _mm256_madd_epi16(e40, c6);
                s61 = _mm256_madd_epi16(e41, c6);

                /* coeff[7]: 0 行 ±1 列 */
                t40 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src8[j - 1]));
                t41 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src8[j + 1]));
                e40 = _mm256_unpacklo_epi16(t40, t41);
                e41 = _mm256_unpackhi_epi16(t40, t41);
                s70 = _mm256_madd_epi16(e40, c7);
                s71 = _mm256_madd_epi16(e41, c7);

                /* coeff[8]: 0 行 0 列 (中心, 与 0 交错使 madd 退化为单乘) */
                t40 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)&src8[j]));
                e40 = _mm256_unpacklo_epi16(t40, zero);
                e41 = _mm256_unpackhi_epi16(t40, zero);
                s80 = _mm256_madd_epi16(e40, c8);
                s81 = _mm256_madd_epi16(e41, c8);

                /* 累加前 8 个像素 */
                ss1 = _mm256_add_epi32(s00, s10);
                ss1 = _mm256_add_epi32(ss1, s20);
                ss1 = _mm256_add_epi32(ss1, s30);
                ss1 = _mm256_add_epi32(ss1, s40);
                ss1 = _mm256_add_epi32(ss1, s50);
                ss1 = _mm256_add_epi32(ss1, s60);
                ss1 = _mm256_add_epi32(ss1, s70);
                ss1 = _mm256_add_epi32(ss1, s80);

                /* 累加后 8 个像素 */
                ss2 = _mm256_add_epi32(s01, s11);
                ss2 = _mm256_add_epi32(ss2, s21);
                ss2 = _mm256_add_epi32(ss2, s31);
                ss2 = _mm256_add_epi32(ss2, s41);
                ss2 = _mm256_add_epi32(ss2, s51);
                ss2 = _mm256_add_epi32(ss2, s61);
                ss2 = _mm256_add_epi32(ss2, s71);
                ss2 = _mm256_add_epi32(ss2, s81);

                ss1 = _mm256_add_epi32(ss1, m_add_offset);
                ss1 = _mm256_srai_epi32(ss1, 6);
                ss2 = _mm256_add_epi32(ss2, m_add_offset);
                ss2 = _mm256_srai_epi32(ss2, 6);

                s = _mm256_packus_epi32(ss1, ss2);
                s = _mm256_min_epu16(s, max_val);
                /* packus 将 16 个 uint16 压为 16 字节, permute 修正通道顺序 */
                s = _mm256_permute4x64_epi64(_mm256_packus_epi16(s, zero), 0xD8);
                _mm_storeu_si128((__m128i*)&dst8[j], _mm256_castsi256_si128(s));
            }

            /* 标量尾部: 不足 16 像素逐个处理 */
            for (; j < x_pos_end; j++) {
                int sum = 0;
                sum += alf_coeff[0] * (img_pad6_8[j] + img_pad5_8[j]);
                sum += alf_coeff[1] * (img_pad4_8[j] + img_pad3_8[j]);
                sum += alf_coeff[2] * (img_pad2_8[j - 1] + img_pad1_8[j + 1]);
                sum += alf_coeff[3] * (img_pad2_8[j] + img_pad1_8[j]);
                sum += alf_coeff[4] * (img_pad2_8[j + 1] + img_pad1_8[j - 1]);
                sum += alf_coeff[5] * (src8[j - 3] + src8[j + 3]);
                sum += alf_coeff[6] * (src8[j - 2] + src8[j + 2]);
                sum += alf_coeff[7] * (src8[j - 1] + src8[j + 1]);
                sum += alf_coeff[8] * src8[j];
                dst8[j] = (uint8_t)AVS2_CLIP3(0, max_pel, (sum + 32) >> 6);
            }

            src8 += stride;
            dst8 += stride;
        }
        return;
    }

    const uint16_t *img_pad1, *img_pad2, *img_pad3, *img_pad4, *img_pad5, *img_pad6;
    uint16_t *dst;
    const uint16_t *src;
    dst = (uint16_t *)_dst;
    src = (const uint16_t *)_src;

    __m256i t00, t01, t10, t11, t20, t21, t30, t31, t40, t41;
    __m256i e00, e01, e10, e11, e20, e21, e30, e31, e40, e41;
    __m256i c0, c1, c2, c3, c4, c5, c6, c7, c8;
    __m256i s00, s01, s10, s11, s20, s21, s30, s31, s40, s41;
    __m256i s50, s51, s60, s61, s70, s71, s80, s81;
    __m256i ss1, ss2, s, m_add_offset, mask;
    __m256i zero = _mm256_setzero_si256();
    __m256i max_val = _mm256_set1_epi16((short)((1 << bit_depth) - 1));

    int i, j;
    int start_pos = b_top_avail ? (lcu_pix_y - 4) : lcu_pix_y;
    int end_pos   = b_down_avail ? (lcu_pix_y + lcu_height - 4) : (lcu_pix_y + lcu_height);
    int x_pos_end = lcu_width;  /* 相对索引: 指针已偏移到 LCU 水平起点 */

    /* 偏移到起始行 + LCU 水平起点 (相对寻址, 内层循环 j 从 0 开始) */
    src += (start_pos * stride) + lcu_pix_x;
    dst += (start_pos * stride) + lcu_pix_x;

    c0 = _mm256_set1_epi16((short)alf_coeff[0]);
    c1 = _mm256_set1_epi16((short)alf_coeff[1]);
    c2 = _mm256_set1_epi16((short)alf_coeff[2]);
    c3 = _mm256_set1_epi16((short)alf_coeff[3]);
    c4 = _mm256_set1_epi16((short)alf_coeff[4]);
    c5 = _mm256_set1_epi16((short)alf_coeff[5]);
    c6 = _mm256_set1_epi16((short)alf_coeff[6]);
    c7 = _mm256_set1_epi16((short)alf_coeff[7]);
    c8 = _mm256_set1_epi16((short)alf_coeff[8]);

    m_add_offset = _mm256_set1_epi32(32);

    if (lcu_width & 15) {
        /* lcu_width 非 16 的倍数: 尾部用 maskstore 处理剩余像素 */
        int x_pos_end15 = x_pos_end - (lcu_width & 15);
        mask = _mm256_loadu_si256((const __m256i*)alf_mask_10bit[(lcu_width & 15) - 1]);
        for (i = start_pos; i < end_pos; i++) {
            int y_up     = AVS2_CLIP3(start_pos, end_pos - 1, i - 1);
            int y_bottom = AVS2_CLIP3(start_pos, end_pos - 1, i + 1);
            img_pad1 = src + (y_bottom - i) * stride;
            img_pad2 = src + (y_up     - i) * stride;

            y_up     = AVS2_CLIP3(start_pos, end_pos - 1, i - 2);
            y_bottom = AVS2_CLIP3(start_pos, end_pos - 1, i + 2);
            img_pad3 = src + (y_bottom - i) * stride;
            img_pad4 = src + (y_up     - i) * stride;

            y_up     = AVS2_CLIP3(start_pos, end_pos - 1, i - 3);
            y_bottom = AVS2_CLIP3(start_pos, end_pos - 1, i + 3);
            img_pad5 = src + (y_bottom - i) * stride;
            img_pad6 = src + (y_up     - i) * stride;

            for (j = 0; j < x_pos_end; j += 16) {
                /* coeff[0]: ±3 行同列 */
                t00 = _mm256_loadu_si256((const __m256i*)&img_pad6[j]);
                t01 = _mm256_loadu_si256((const __m256i*)&img_pad5[j]);
                e00 = _mm256_unpacklo_epi16(t00, t01);
                e01 = _mm256_unpackhi_epi16(t00, t01);
                s00 = _mm256_madd_epi16(e00, c0);
                s01 = _mm256_madd_epi16(e01, c0);

                /* coeff[1]: ±2 行同列 */
                t10 = _mm256_loadu_si256((const __m256i*)&img_pad4[j]);
                t11 = _mm256_loadu_si256((const __m256i*)&img_pad3[j]);
                e10 = _mm256_unpacklo_epi16(t10, t11);
                e11 = _mm256_unpackhi_epi16(t10, t11);
                s10 = _mm256_madd_epi16(e10, c1);
                s11 = _mm256_madd_epi16(e11, c1);

                /* coeff[2]: +1 行右1 + -1 行左1 (交叉) */
                t20 = _mm256_loadu_si256((const __m256i*)&img_pad2[j - 1]);
                t21 = _mm256_loadu_si256((const __m256i*)&img_pad1[j + 1]);
                e20 = _mm256_unpacklo_epi16(t20, t21);
                e21 = _mm256_unpackhi_epi16(t20, t21);
                s20 = _mm256_madd_epi16(e20, c2);
                s21 = _mm256_madd_epi16(e21, c2);

                /* coeff[3]: ±1 行同列 */
                t30 = _mm256_loadu_si256((const __m256i*)&img_pad2[j]);
                t31 = _mm256_loadu_si256((const __m256i*)&img_pad1[j]);
                e30 = _mm256_unpacklo_epi16(t30, t31);
                e31 = _mm256_unpackhi_epi16(t30, t31);
                s30 = _mm256_madd_epi16(e30, c3);
                s31 = _mm256_madd_epi16(e31, c3);

                /* coeff[4]: +1 行左1 + -1 行右1 (交叉) */
                t40 = _mm256_loadu_si256((const __m256i*)&img_pad2[j + 1]);
                t41 = _mm256_loadu_si256((const __m256i*)&img_pad1[j - 1]);
                e40 = _mm256_unpacklo_epi16(t40, t41);
                e41 = _mm256_unpackhi_epi16(t40, t41);
                s40 = _mm256_madd_epi16(e40, c4);
                s41 = _mm256_madd_epi16(e41, c4);

                /* coeff[5]: 0 行 ±3 列 */
                t40 = _mm256_loadu_si256((const __m256i*)&src[j - 3]);
                t41 = _mm256_loadu_si256((const __m256i*)&src[j + 3]);
                e40 = _mm256_unpacklo_epi16(t40, t41);
                e41 = _mm256_unpackhi_epi16(t40, t41);
                s50 = _mm256_madd_epi16(e40, c5);
                s51 = _mm256_madd_epi16(e41, c5);

                /* coeff[6]: 0 行 ±2 列 */
                t40 = _mm256_loadu_si256((const __m256i*)&src[j - 2]);
                t41 = _mm256_loadu_si256((const __m256i*)&src[j + 2]);
                e40 = _mm256_unpacklo_epi16(t40, t41);
                e41 = _mm256_unpackhi_epi16(t40, t41);
                s60 = _mm256_madd_epi16(e40, c6);
                s61 = _mm256_madd_epi16(e41, c6);

                /* coeff[7]: 0 行 ±1 列 */
                t40 = _mm256_loadu_si256((const __m256i*)&src[j - 1]);
                t41 = _mm256_loadu_si256((const __m256i*)&src[j + 1]);
                e40 = _mm256_unpacklo_epi16(t40, t41);
                e41 = _mm256_unpackhi_epi16(t40, t41);
                s70 = _mm256_madd_epi16(e40, c7);
                s71 = _mm256_madd_epi16(e41, c7);

                /* coeff[8]: 0 行 0 列 (中心, 与 0 交错使 madd 退化为单乘) */
                t40 = _mm256_loadu_si256((const __m256i*)&src[j]);
                e40 = _mm256_unpacklo_epi16(t40, zero);
                e41 = _mm256_unpackhi_epi16(t40, zero);
                s80 = _mm256_madd_epi16(e40, c8);
                s81 = _mm256_madd_epi16(e41, c8);

                /* 累加前 8 个像素 */
                ss1 = _mm256_add_epi32(s00, s10);
                ss1 = _mm256_add_epi32(ss1, s20);
                ss1 = _mm256_add_epi32(ss1, s30);
                ss1 = _mm256_add_epi32(ss1, s40);
                ss1 = _mm256_add_epi32(ss1, s50);
                ss1 = _mm256_add_epi32(ss1, s60);
                ss1 = _mm256_add_epi32(ss1, s70);
                ss1 = _mm256_add_epi32(ss1, s80);

                /* 累加后 8 个像素 */
                ss2 = _mm256_add_epi32(s01, s11);
                ss2 = _mm256_add_epi32(ss2, s21);
                ss2 = _mm256_add_epi32(ss2, s31);
                ss2 = _mm256_add_epi32(ss2, s41);
                ss2 = _mm256_add_epi32(ss2, s51);
                ss2 = _mm256_add_epi32(ss2, s61);
                ss2 = _mm256_add_epi32(ss2, s71);
                ss2 = _mm256_add_epi32(ss2, s81);

                ss1 = _mm256_add_epi32(ss1, m_add_offset);
                ss1 = _mm256_srai_epi32(ss1, 6);

                ss2 = _mm256_add_epi32(ss2, m_add_offset);
                ss2 = _mm256_srai_epi32(ss2, 6);

                s = _mm256_packus_epi32(ss1, ss2);  /* 饱和到 [0,65535] (含下限 clip) */
                s = _mm256_min_epu16(s, max_val);    /* 上限 clip 到 max_pixel */
                if (j != x_pos_end15) {
                    _mm256_storeu_si256((__m256i*)(dst + j), s);
                } else {
                    _mm256_maskstore_epi32((int*)(dst + j), mask, s);
                    break;
                }
            }

            src += stride;
            dst += stride;
        }
    } else {
        /* lcu_width 为 16 的倍数: 全部 storeu, 无尾部掩码 */
        for (i = start_pos; i < end_pos; i++) {
            int y_up     = AVS2_CLIP3(start_pos, end_pos - 1, i - 1);
            int y_bottom = AVS2_CLIP3(start_pos, end_pos - 1, i + 1);
            img_pad1 = src + (y_bottom - i) * stride;
            img_pad2 = src + (y_up     - i) * stride;

            y_up     = AVS2_CLIP3(start_pos, end_pos - 1, i - 2);
            y_bottom = AVS2_CLIP3(start_pos, end_pos - 1, i + 2);
            img_pad3 = src + (y_bottom - i) * stride;
            img_pad4 = src + (y_up     - i) * stride;

            y_up     = AVS2_CLIP3(start_pos, end_pos - 1, i - 3);
            y_bottom = AVS2_CLIP3(start_pos, end_pos - 1, i + 3);
            img_pad5 = src + (y_bottom - i) * stride;
            img_pad6 = src + (y_up     - i) * stride;

            for (j = 0; j < x_pos_end; j += 16) {
                /* coeff[0]: ±3 行同列 */
                t00 = _mm256_loadu_si256((const __m256i*)&img_pad6[j]);
                t01 = _mm256_loadu_si256((const __m256i*)&img_pad5[j]);
                e00 = _mm256_unpacklo_epi16(t00, t01);
                e01 = _mm256_unpackhi_epi16(t00, t01);
                s00 = _mm256_madd_epi16(e00, c0);
                s01 = _mm256_madd_epi16(e01, c0);

                /* coeff[1]: ±2 行同列 */
                t10 = _mm256_loadu_si256((const __m256i*)&img_pad4[j]);
                t11 = _mm256_loadu_si256((const __m256i*)&img_pad3[j]);
                e10 = _mm256_unpacklo_epi16(t10, t11);
                e11 = _mm256_unpackhi_epi16(t10, t11);
                s10 = _mm256_madd_epi16(e10, c1);
                s11 = _mm256_madd_epi16(e11, c1);

                /* coeff[2]: +1 行右1 + -1 行左1 (交叉) */
                t20 = _mm256_loadu_si256((const __m256i*)&img_pad2[j - 1]);
                t21 = _mm256_loadu_si256((const __m256i*)&img_pad1[j + 1]);
                e20 = _mm256_unpacklo_epi16(t20, t21);
                e21 = _mm256_unpackhi_epi16(t20, t21);
                s20 = _mm256_madd_epi16(e20, c2);
                s21 = _mm256_madd_epi16(e21, c2);

                /* coeff[3]: ±1 行同列 */
                t30 = _mm256_loadu_si256((const __m256i*)&img_pad2[j]);
                t31 = _mm256_loadu_si256((const __m256i*)&img_pad1[j]);
                e30 = _mm256_unpacklo_epi16(t30, t31);
                e31 = _mm256_unpackhi_epi16(t30, t31);
                s30 = _mm256_madd_epi16(e30, c3);
                s31 = _mm256_madd_epi16(e31, c3);

                /* coeff[4]: +1 行左1 + -1 行右1 (交叉) */
                t40 = _mm256_loadu_si256((const __m256i*)&img_pad2[j + 1]);
                t41 = _mm256_loadu_si256((const __m256i*)&img_pad1[j - 1]);
                e40 = _mm256_unpacklo_epi16(t40, t41);
                e41 = _mm256_unpackhi_epi16(t40, t41);
                s40 = _mm256_madd_epi16(e40, c4);
                s41 = _mm256_madd_epi16(e41, c4);

                /* coeff[5]: 0 行 ±3 列 */
                t40 = _mm256_loadu_si256((const __m256i*)&src[j - 3]);
                t41 = _mm256_loadu_si256((const __m256i*)&src[j + 3]);
                e40 = _mm256_unpacklo_epi16(t40, t41);
                e41 = _mm256_unpackhi_epi16(t40, t41);
                s50 = _mm256_madd_epi16(e40, c5);
                s51 = _mm256_madd_epi16(e41, c5);

                /* coeff[6]: 0 行 ±2 列 */
                t40 = _mm256_loadu_si256((const __m256i*)&src[j - 2]);
                t41 = _mm256_loadu_si256((const __m256i*)&src[j + 2]);
                e40 = _mm256_unpacklo_epi16(t40, t41);
                e41 = _mm256_unpackhi_epi16(t40, t41);
                s60 = _mm256_madd_epi16(e40, c6);
                s61 = _mm256_madd_epi16(e41, c6);

                /* coeff[7]: 0 行 ±1 列 */
                t40 = _mm256_loadu_si256((const __m256i*)&src[j - 1]);
                t41 = _mm256_loadu_si256((const __m256i*)&src[j + 1]);
                e40 = _mm256_unpacklo_epi16(t40, t41);
                e41 = _mm256_unpackhi_epi16(t40, t41);
                s70 = _mm256_madd_epi16(e40, c7);
                s71 = _mm256_madd_epi16(e41, c7);

                /* coeff[8]: 0 行 0 列 (中心) */
                t40 = _mm256_loadu_si256((const __m256i*)&src[j]);
                e40 = _mm256_unpacklo_epi16(t40, zero);
                e41 = _mm256_unpackhi_epi16(t40, zero);
                s80 = _mm256_madd_epi16(e40, c8);
                s81 = _mm256_madd_epi16(e41, c8);

                /* 累加前 8 个像素 */
                ss1 = _mm256_add_epi32(s00, s10);
                ss1 = _mm256_add_epi32(ss1, s20);
                ss1 = _mm256_add_epi32(ss1, s30);
                ss1 = _mm256_add_epi32(ss1, s40);
                ss1 = _mm256_add_epi32(ss1, s50);
                ss1 = _mm256_add_epi32(ss1, s60);
                ss1 = _mm256_add_epi32(ss1, s70);
                ss1 = _mm256_add_epi32(ss1, s80);

                /* 累加后 8 个像素 */
                ss2 = _mm256_add_epi32(s01, s11);
                ss2 = _mm256_add_epi32(ss2, s21);
                ss2 = _mm256_add_epi32(ss2, s31);
                ss2 = _mm256_add_epi32(ss2, s41);
                ss2 = _mm256_add_epi32(ss2, s51);
                ss2 = _mm256_add_epi32(ss2, s61);
                ss2 = _mm256_add_epi32(ss2, s71);
                ss2 = _mm256_add_epi32(ss2, s81);

                ss1 = _mm256_add_epi32(ss1, m_add_offset);
                ss1 = _mm256_srai_epi32(ss1, 6);

                ss2 = _mm256_add_epi32(ss2, m_add_offset);
                ss2 = _mm256_srai_epi32(ss2, 6);

                s = _mm256_packus_epi32(ss1, ss2);
                s = _mm256_min_epu16(s, max_val);

                _mm256_storeu_si256((__m256i*)(dst + j), s);
            }

            src += stride;
            dst += stride;
        }
    }
}

/* ---------------------------------------------------------------------------
 * ALF SIMD 初始化 (x86)
 * SSE4.1 暂未实现 (占位); AVX2 注册 block1 主体滤波。
 * block2 (角点边界修正) 仅 4 像素, SIMD 无收益, 保留 C 实现 (avs2_alf_init 设置)。
 * ------------------------------------------------------------------------- */
void avs2_alf_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }

void avs2_alf_init_avx2(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.alf_block[0] = alf_filter_block1_avx2;
}

#else

/* 非 x86 平台: 暂用 C 回退 */
void avs2_alf_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_alf_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }

#endif
