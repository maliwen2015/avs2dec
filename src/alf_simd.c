/*
 * alf_simd.c - ALF SIMD 实现 (x86 SSE4.1, 10-bit/8-bit)
 *
 * 从 davs2 intrinsic_alf_256.c 的 alf_flt_one_block_sse256_10bit 移植为 SSE4.1。
 * 9 抽头 7x7 对称滤波, 内层每次处理 8 个 16-bit 像素 (128-bit)。
 *
 * block1: 主体 7x7 滤波 -> alf_filter_block1_sse4 (SSE4.1)
 * block2: 4 角点边界修正, 仅 4 像素, SIMD 无收益 -> 保留 C 实现 (不在此覆盖)
 *
 * 依赖帧边缘 padding (AVS2_PAD_LUMA=64, AVS2_PAD_CHROMA=32) 处理越界读取。
 */

#include "internal.h"

#ifndef AVS2_CLIP3
#define AVS2_CLIP3(lo, hi, x) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <tmmintrin.h>
#include <smmintrin.h>

/* ---------------------------------------------------------------------------
 * block1: 主体 7x7 滤波 SSE4.1 (10-bit / 8-bit)
 * 对应 alf_filter_block1_c / davs2 alf_flt_one_block_sse256_10bit。
 *
 * 系数加权布局 (与 C 参考一致):
 *   coeff[0]: ±3 行同列            coeff[5]: 0 行 ±3 列
 *   coeff[1]: ±2 行同列            coeff[6]: 0 行 ±2 列
 *   coeff[2]: +1 行右1 + -1 行左1  coeff[7]: 0 行 ±1 列
 *   coeff[3]: ±1 行同列            coeff[8]: 0 行 0 列 (中心, 由重建得到)
 *   coeff[4]: +1 行左1 + -1 行右1
 *
 * 每次迭代将一对像素 (img_pad6[j]+img_pad5[j] 等) 交错打包后用 _mm_madd_epi16
 * 完成 系数*(p1+p2) 的 int32 乘加, 9 项累加后 (+32)>>6, 饱和到 [0, max_pixel]。
 * 128-bit 每次处理 8 像素: ss1 累加前 4 像素, ss2 累加后 4 像素。
 * ------------------------------------------------------------------------- */
static void alf_filter_block1_sse4(uint8_t *_dst, const uint8_t *_src, int stride,
                                   int lcu_pix_x, int lcu_pix_y,
                                   int lcu_width, int lcu_height,
                                   int *alf_coeff, int b_top_avail, int b_down_avail,
                                   int bit_depth)
{
    if (bit_depth <= 8) {
        /* 8-bit SSE4.1: 加载 8 字节→cvtepu8 扩展为 uint16, 复用 16-bit madd 逻辑,
         * packus 压缩回 8 字节存储. 尾部不足 8 像素用标量处理. */
        uint8_t *dst8 = (uint8_t *)_dst;
        const uint8_t *src8 = (const uint8_t *)_src;
        const uint8_t *img_pad1_8, *img_pad2_8, *img_pad3_8, *img_pad4_8, *img_pad5_8, *img_pad6_8;
        const int max_pel = (1 << bit_depth) - 1;
        int i, j;
        int start_pos = b_top_avail ? (lcu_pix_y - 4) : lcu_pix_y;
        int end_pos   = b_down_avail ? (lcu_pix_y + lcu_height - 4) : (lcu_pix_y + lcu_height);
        int x_pos_end = lcu_width;
        int x_pos_end8 = x_pos_end & ~7;

        __m128i t00, t01, t10, t11, t20, t21, t30, t31, t40, t41;
        __m128i e00, e01, e10, e11, e20, e21, e30, e31, e40, e41;
        __m128i c0, c1, c2, c3, c4, c5, c6, c7, c8;
        __m128i s00, s01, s10, s11, s20, s21, s30, s31, s40, s41;
        __m128i s50, s51, s60, s61, s70, s71, s80, s81;
        __m128i ss1, ss2, s, m_add_offset;
        __m128i zero = _mm_setzero_si128();
        __m128i max_val = _mm_set1_epi16((short)max_pel);

        src8 += (start_pos * stride) + lcu_pix_x;
        dst8 += (start_pos * stride) + lcu_pix_x;

        c0 = _mm_set1_epi16((short)alf_coeff[0]);
        c1 = _mm_set1_epi16((short)alf_coeff[1]);
        c2 = _mm_set1_epi16((short)alf_coeff[2]);
        c3 = _mm_set1_epi16((short)alf_coeff[3]);
        c4 = _mm_set1_epi16((short)alf_coeff[4]);
        c5 = _mm_set1_epi16((short)alf_coeff[5]);
        c6 = _mm_set1_epi16((short)alf_coeff[6]);
        c7 = _mm_set1_epi16((short)alf_coeff[7]);
        c8 = _mm_set1_epi16((short)alf_coeff[8]);
        m_add_offset = _mm_set1_epi32(32);

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

            /* SIMD 主循环: 每次 8 像素 */
            for (j = 0; j < x_pos_end8; j += 8) {
                /* coeff[0]: ±3 行同列 */
                t00 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&img_pad6_8[j]));
                t01 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&img_pad5_8[j]));
                e00 = _mm_unpacklo_epi16(t00, t01);
                e01 = _mm_unpackhi_epi16(t00, t01);
                s00 = _mm_madd_epi16(e00, c0);
                s01 = _mm_madd_epi16(e01, c0);

                /* coeff[1]: ±2 行同列 */
                t10 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&img_pad4_8[j]));
                t11 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&img_pad3_8[j]));
                e10 = _mm_unpacklo_epi16(t10, t11);
                e11 = _mm_unpackhi_epi16(t10, t11);
                s10 = _mm_madd_epi16(e10, c1);
                s11 = _mm_madd_epi16(e11, c1);

                /* coeff[2]: +1 行右1 + -1 行左1 (交叉) */
                t20 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&img_pad2_8[j - 1]));
                t21 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&img_pad1_8[j + 1]));
                e20 = _mm_unpacklo_epi16(t20, t21);
                e21 = _mm_unpackhi_epi16(t20, t21);
                s20 = _mm_madd_epi16(e20, c2);
                s21 = _mm_madd_epi16(e21, c2);

                /* coeff[3]: ±1 行同列 */
                t30 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&img_pad2_8[j]));
                t31 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&img_pad1_8[j]));
                e30 = _mm_unpacklo_epi16(t30, t31);
                e31 = _mm_unpackhi_epi16(t30, t31);
                s30 = _mm_madd_epi16(e30, c3);
                s31 = _mm_madd_epi16(e31, c3);

                /* coeff[4]: +1 行左1 + -1 行右1 (交叉) */
                t40 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&img_pad2_8[j + 1]));
                t41 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&img_pad1_8[j - 1]));
                e40 = _mm_unpacklo_epi16(t40, t41);
                e41 = _mm_unpackhi_epi16(t40, t41);
                s40 = _mm_madd_epi16(e40, c4);
                s41 = _mm_madd_epi16(e41, c4);

                /* coeff[5]: 0 行 ±3 列 */
                t40 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&src8[j - 3]));
                t41 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&src8[j + 3]));
                e40 = _mm_unpacklo_epi16(t40, t41);
                e41 = _mm_unpackhi_epi16(t40, t41);
                s50 = _mm_madd_epi16(e40, c5);
                s51 = _mm_madd_epi16(e41, c5);

                /* coeff[6]: 0 行 ±2 列 */
                t40 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&src8[j - 2]));
                t41 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&src8[j + 2]));
                e40 = _mm_unpacklo_epi16(t40, t41);
                e41 = _mm_unpackhi_epi16(t40, t41);
                s60 = _mm_madd_epi16(e40, c6);
                s61 = _mm_madd_epi16(e41, c6);

                /* coeff[7]: 0 行 ±1 列 */
                t40 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&src8[j - 1]));
                t41 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&src8[j + 1]));
                e40 = _mm_unpacklo_epi16(t40, t41);
                e41 = _mm_unpackhi_epi16(t40, t41);
                s70 = _mm_madd_epi16(e40, c7);
                s71 = _mm_madd_epi16(e41, c7);

                /* coeff[8]: 0 行 0 列 (中心, 与 0 交错使 madd 退化为单乘) */
                t40 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)&src8[j]));
                e40 = _mm_unpacklo_epi16(t40, zero);
                e41 = _mm_unpackhi_epi16(t40, zero);
                s80 = _mm_madd_epi16(e40, c8);
                s81 = _mm_madd_epi16(e41, c8);

                /* 累加前 4 个像素 */
                ss1 = _mm_add_epi32(s00, s10);
                ss1 = _mm_add_epi32(ss1, s20);
                ss1 = _mm_add_epi32(ss1, s30);
                ss1 = _mm_add_epi32(ss1, s40);
                ss1 = _mm_add_epi32(ss1, s50);
                ss1 = _mm_add_epi32(ss1, s60);
                ss1 = _mm_add_epi32(ss1, s70);
                ss1 = _mm_add_epi32(ss1, s80);

                /* 累加后 4 个像素 */
                ss2 = _mm_add_epi32(s01, s11);
                ss2 = _mm_add_epi32(ss2, s21);
                ss2 = _mm_add_epi32(ss2, s31);
                ss2 = _mm_add_epi32(ss2, s41);
                ss2 = _mm_add_epi32(ss2, s51);
                ss2 = _mm_add_epi32(ss2, s61);
                ss2 = _mm_add_epi32(ss2, s71);
                ss2 = _mm_add_epi32(ss2, s81);

                ss1 = _mm_add_epi32(ss1, m_add_offset);
                ss1 = _mm_srai_epi32(ss1, 6);
                ss2 = _mm_add_epi32(ss2, m_add_offset);
                ss2 = _mm_srai_epi32(ss2, 6);

                s = _mm_packus_epi32(ss1, ss2);   /* 8 uint16 */
                s = _mm_min_epu16(s, max_val);     /* clip to max_pel */
                s = _mm_packus_epi16(s, zero);     /* 8 uint8 in low 64 bits */
                _mm_storel_epi64((__m128i*)&dst8[j], s);
            }

            /* 标量尾部: 不足 8 像素逐个处理 */
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

    /* 10-bit SSE4.1 */
    const uint16_t *img_pad1, *img_pad2, *img_pad3, *img_pad4, *img_pad5, *img_pad6;
    uint16_t *dst;
    const uint16_t *src;
    dst = (uint16_t *)_dst;
    src = (const uint16_t *)_src;

    __m128i t00, t01, t10, t11, t20, t21, t30, t31, t40, t41;
    __m128i e00, e01, e10, e11, e20, e21, e30, e31, e40, e41;
    __m128i c0, c1, c2, c3, c4, c5, c6, c7, c8;
    __m128i s00, s01, s10, s11, s20, s21, s30, s31, s40, s41;
    __m128i s50, s51, s60, s61, s70, s71, s80, s81;
    __m128i ss1, ss2, s, m_add_offset;
    __m128i zero = _mm_setzero_si128();
    __m128i max_val = _mm_set1_epi16((short)((1 << bit_depth) - 1));

    int i, j;
    int start_pos = b_top_avail ? (lcu_pix_y - 4) : lcu_pix_y;
    int end_pos   = b_down_avail ? (lcu_pix_y + lcu_height - 4) : (lcu_pix_y + lcu_height);
    int x_pos_end = lcu_width;
    int x_pos_end8 = x_pos_end & ~7;

    src += (start_pos * stride) + lcu_pix_x;
    dst += (start_pos * stride) + lcu_pix_x;

    c0 = _mm_set1_epi16((short)alf_coeff[0]);
    c1 = _mm_set1_epi16((short)alf_coeff[1]);
    c2 = _mm_set1_epi16((short)alf_coeff[2]);
    c3 = _mm_set1_epi16((short)alf_coeff[3]);
    c4 = _mm_set1_epi16((short)alf_coeff[4]);
    c5 = _mm_set1_epi16((short)alf_coeff[5]);
    c6 = _mm_set1_epi16((short)alf_coeff[6]);
    c7 = _mm_set1_epi16((short)alf_coeff[7]);
    c8 = _mm_set1_epi16((short)alf_coeff[8]);
    m_add_offset = _mm_set1_epi32(32);

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

        /* SIMD 主循环: 每次 8 像素 */
        for (j = 0; j < x_pos_end8; j += 8) {
            /* coeff[0]: ±3 行同列 */
            t00 = _mm_loadu_si128((const __m128i*)&img_pad6[j]);
            t01 = _mm_loadu_si128((const __m128i*)&img_pad5[j]);
            e00 = _mm_unpacklo_epi16(t00, t01);
            e01 = _mm_unpackhi_epi16(t00, t01);
            s00 = _mm_madd_epi16(e00, c0);
            s01 = _mm_madd_epi16(e01, c0);

            /* coeff[1]: ±2 行同列 */
            t10 = _mm_loadu_si128((const __m128i*)&img_pad4[j]);
            t11 = _mm_loadu_si128((const __m128i*)&img_pad3[j]);
            e10 = _mm_unpacklo_epi16(t10, t11);
            e11 = _mm_unpackhi_epi16(t10, t11);
            s10 = _mm_madd_epi16(e10, c1);
            s11 = _mm_madd_epi16(e11, c1);

            /* coeff[2]: +1 行右1 + -1 行左1 (交叉) */
            t20 = _mm_loadu_si128((const __m128i*)&img_pad2[j - 1]);
            t21 = _mm_loadu_si128((const __m128i*)&img_pad1[j + 1]);
            e20 = _mm_unpacklo_epi16(t20, t21);
            e21 = _mm_unpackhi_epi16(t20, t21);
            s20 = _mm_madd_epi16(e20, c2);
            s21 = _mm_madd_epi16(e21, c2);

            /* coeff[3]: ±1 行同列 */
            t30 = _mm_loadu_si128((const __m128i*)&img_pad2[j]);
            t31 = _mm_loadu_si128((const __m128i*)&img_pad1[j]);
            e30 = _mm_unpacklo_epi16(t30, t31);
            e31 = _mm_unpackhi_epi16(t30, t31);
            s30 = _mm_madd_epi16(e30, c3);
            s31 = _mm_madd_epi16(e31, c3);

            /* coeff[4]: +1 行左1 + -1 行右1 (交叉) */
            t40 = _mm_loadu_si128((const __m128i*)&img_pad2[j + 1]);
            t41 = _mm_loadu_si128((const __m128i*)&img_pad1[j - 1]);
            e40 = _mm_unpacklo_epi16(t40, t41);
            e41 = _mm_unpackhi_epi16(t40, t41);
            s40 = _mm_madd_epi16(e40, c4);
            s41 = _mm_madd_epi16(e41, c4);

            /* coeff[5]: 0 行 ±3 列 */
            t40 = _mm_loadu_si128((const __m128i*)&src[j - 3]);
            t41 = _mm_loadu_si128((const __m128i*)&src[j + 3]);
            e40 = _mm_unpacklo_epi16(t40, t41);
            e41 = _mm_unpackhi_epi16(t40, t41);
            s50 = _mm_madd_epi16(e40, c5);
            s51 = _mm_madd_epi16(e41, c5);

            /* coeff[6]: 0 行 ±2 列 */
            t40 = _mm_loadu_si128((const __m128i*)&src[j - 2]);
            t41 = _mm_loadu_si128((const __m128i*)&src[j + 2]);
            e40 = _mm_unpacklo_epi16(t40, t41);
            e41 = _mm_unpackhi_epi16(t40, t41);
            s60 = _mm_madd_epi16(e40, c6);
            s61 = _mm_madd_epi16(e41, c6);

            /* coeff[7]: 0 行 ±1 列 */
            t40 = _mm_loadu_si128((const __m128i*)&src[j - 1]);
            t41 = _mm_loadu_si128((const __m128i*)&src[j + 1]);
            e40 = _mm_unpacklo_epi16(t40, t41);
            e41 = _mm_unpackhi_epi16(t40, t41);
            s70 = _mm_madd_epi16(e40, c7);
            s71 = _mm_madd_epi16(e41, c7);

            /* coeff[8]: 0 行 0 列 (中心, 与 0 交错使 madd 退化为单乘) */
            t40 = _mm_loadu_si128((const __m128i*)&src[j]);
            e40 = _mm_unpacklo_epi16(t40, zero);
            e41 = _mm_unpackhi_epi16(t40, zero);
            s80 = _mm_madd_epi16(e40, c8);
            s81 = _mm_madd_epi16(e41, c8);

            /* 累加前 4 个像素 */
            ss1 = _mm_add_epi32(s00, s10);
            ss1 = _mm_add_epi32(ss1, s20);
            ss1 = _mm_add_epi32(ss1, s30);
            ss1 = _mm_add_epi32(ss1, s40);
            ss1 = _mm_add_epi32(ss1, s50);
            ss1 = _mm_add_epi32(ss1, s60);
            ss1 = _mm_add_epi32(ss1, s70);
            ss1 = _mm_add_epi32(ss1, s80);

            /* 累加后 4 个像素 */
            ss2 = _mm_add_epi32(s01, s11);
            ss2 = _mm_add_epi32(ss2, s21);
            ss2 = _mm_add_epi32(ss2, s31);
            ss2 = _mm_add_epi32(ss2, s41);
            ss2 = _mm_add_epi32(ss2, s51);
            ss2 = _mm_add_epi32(ss2, s61);
            ss2 = _mm_add_epi32(ss2, s71);
            ss2 = _mm_add_epi32(ss2, s81);

            ss1 = _mm_add_epi32(ss1, m_add_offset);
            ss1 = _mm_srai_epi32(ss1, 6);
            ss2 = _mm_add_epi32(ss2, m_add_offset);
            ss2 = _mm_srai_epi32(ss2, 6);

            s = _mm_packus_epi32(ss1, ss2);   /* 8 uint16 */
            s = _mm_min_epu16(s, max_val);     /* clip to max_pel */
            _mm_storeu_si128((__m128i*)(dst + j), s);
        }

        /* 标量尾部: 不足 8 像素逐个处理 */
        for (; j < x_pos_end; j++) {
            int sum = 0;
            sum += alf_coeff[0] * (img_pad6[j] + img_pad5[j]);
            sum += alf_coeff[1] * (img_pad4[j] + img_pad3[j]);
            sum += alf_coeff[2] * (img_pad2[j - 1] + img_pad1[j + 1]);
            sum += alf_coeff[3] * (img_pad2[j] + img_pad1[j]);
            sum += alf_coeff[4] * (img_pad2[j + 1] + img_pad1[j - 1]);
            sum += alf_coeff[5] * (src[j - 3] + src[j + 3]);
            sum += alf_coeff[6] * (src[j - 2] + src[j + 2]);
            sum += alf_coeff[7] * (src[j - 1] + src[j + 1]);
            sum += alf_coeff[8] * src[j];
            dst[j] = (uint16_t)AVS2_CLIP3(0, (1 << bit_depth) - 1, (sum + 32) >> 6);
        }

        src += stride;
        dst += stride;
    }
}

/* ---------------------------------------------------------------------------
 * ALF SIMD 初始化 (x86)
 * SSE4.1 注册 block1 主体滤波; AVX2 路径已清空 (降级为 SSE4.1)。
 * block2 (角点边界修正) 仅 4 像素, SIMD 无收益, 保留 C 实现 (avs2_alf_init 设置)。
 * ------------------------------------------------------------------------- */
void avs2_alf_init_sse41(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.alf_block[0] = alf_filter_block1_sse4;
}

void avs2_alf_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }

#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)

#include <arm_neon.h>

/* ---- C 回退函数声明 (在 alf.c 中定义, 供 8-bit 路径回退) ---- */
extern void alf_filter_block1_c(uint8_t *dst, const uint8_t *src, int stride,
                                int lcu_pix_x, int lcu_pix_y,
                                int lcu_width, int lcu_height,
                                int *alf_coeff, int b_top_avail, int b_down_avail,
                                int bit_depth);

/* ===========================================================================
 * ALF block1 主体 7x7 滤波 NEON (由 SSE4.1 逐行移植).
 *
 * 策略 (与 mc/lf NEON 一致): 10-bit 路径原生 NEON, 8-bit 回退 C 参考.
 *
 * 位精确性: 每对像素交错打包 (v_unpacklo/hi_epi16 = vzip_s16) 后
 *   v_madd_epi16 完成 系数*(p1+p2) 的 int32 乘加, 与 SSE _mm_madd_epi16 一致.
 * =========================================================================== */

typedef int16x8_t v128_t;

static inline int32x4_t v128_to_s32(v128_t a)  { return vreinterpretq_s32_s16(a); }
static inline v128_t    v128_from_s32(int32x4_t a) { return vreinterpretq_s16_s32(a); }

/* _mm_madd_epi16 */
static inline v128_t v_madd_epi16(v128_t a, v128_t b)
{
    int32x4_t lo = vmull_s16(vget_low_s16(a),  vget_low_s16(b));
    int32x4_t hi = vmull_s16(vget_high_s16(a), vget_high_s16(b));
    int32x2_t sl = vpadd_s32(vget_low_s32(lo),  vget_high_s32(lo));
    int32x2_t sh = vpadd_s32(vget_low_s32(hi),  vget_high_s32(hi));
    return v128_from_s32(vcombine_s32(sl, sh));
}

static inline v128_t v_unpacklo_epi16(v128_t a, v128_t b)
{
    int16x4x2_t z = vzip_s16(vget_low_s16(a), vget_low_s16(b));
    return vcombine_s16(z.val[0], z.val[1]);
}
static inline v128_t v_unpackhi_epi16(v128_t a, v128_t b)
{
    int16x4x2_t z = vzip_s16(vget_high_s16(a), vget_high_s16(b));
    return vcombine_s16(z.val[0], z.val[1]);
}

/* _mm_packus_epi32 (无符号饱和) */
static inline v128_t v_packus_epi32(v128_t a, v128_t b)
{
    const int32x4_t zero = vdupq_n_s32(0);
    uint16x4_t lo = vqmovn_u32((uint32x4_t)vmaxq_s32(v128_to_s32(a), zero));
    uint16x4_t hi = vqmovn_u32((uint32x4_t)vmaxq_s32(v128_to_s32(b), zero));
    return vreinterpretq_s16_u16(vcombine_u16(lo, hi));
}

static inline v128_t v_add_epi32(v128_t a, v128_t b) { return v128_from_s32(vaddq_s32(v128_to_s32(a), v128_to_s32(b))); }
static inline v128_t v_srai_epi32(v128_t a, int n)   { return v128_from_s32(vshrq_n_s32(v128_to_s32(a), n)); }
static inline v128_t v_min_epu16(v128_t a, v128_t b) { return vreinterpretq_s16_u16(vminq_u16(vreinterpretq_u16_s16(a), vreinterpretq_u16_s16(b))); }
static inline v128_t v_set1_epi16(int16_t x) { return vdupq_n_s16(x); }
static inline v128_t v_set1_epi32(int32_t x) { return v128_from_s32(vdupq_n_s32(x)); }
static inline v128_t v_load(const void *p)   { return vld1q_s16((const int16_t*)p); }
static inline void   v_store(void *p, v128_t v) { vst1q_s16((int16_t*)p, v); }

static void alf_filter_block1_neon(uint8_t *_dst, const uint8_t *_src, int stride,
                                   int lcu_pix_x, int lcu_pix_y,
                                   int lcu_width, int lcu_height,
                                   int *alf_coeff, int b_top_avail, int b_down_avail,
                                   int bit_depth)
{
    if (bit_depth <= 8) {
        alf_filter_block1_c(_dst, _src, stride, lcu_pix_x, lcu_pix_y,
                            lcu_width, lcu_height, alf_coeff,
                            b_top_avail, b_down_avail, bit_depth);
        return;
    }

    /* 10-bit NEON */
    const uint16_t *img_pad1, *img_pad2, *img_pad3, *img_pad4, *img_pad5, *img_pad6;
    uint16_t *dst;
    const uint16_t *src;
    dst = (uint16_t *)_dst;
    src = (const uint16_t *)_src;

    v128_t c0, c1, c2, c3, c4, c5, c6, c7, c8;
    v128_t m_add_offset;
    v128_t zero = v_set1_epi16(0);
    v128_t max_val;

    int i, j;
    int start_pos = b_top_avail ? (lcu_pix_y - 4) : lcu_pix_y;
    int end_pos   = b_down_avail ? (lcu_pix_y + lcu_height - 4) : (lcu_pix_y + lcu_height);
    int x_pos_end = lcu_width;
    int x_pos_end8 = x_pos_end & ~7;

    src += (start_pos * stride) + lcu_pix_x;
    dst += (start_pos * stride) + lcu_pix_x;

    c0 = v_set1_epi16((int16_t)alf_coeff[0]);
    c1 = v_set1_epi16((int16_t)alf_coeff[1]);
    c2 = v_set1_epi16((int16_t)alf_coeff[2]);
    c3 = v_set1_epi16((int16_t)alf_coeff[3]);
    c4 = v_set1_epi16((int16_t)alf_coeff[4]);
    c5 = v_set1_epi16((int16_t)alf_coeff[5]);
    c6 = v_set1_epi16((int16_t)alf_coeff[6]);
    c7 = v_set1_epi16((int16_t)alf_coeff[7]);
    c8 = v_set1_epi16((int16_t)alf_coeff[8]);
    m_add_offset = v_set1_epi32(32);
    max_val = v_set1_epi16((int16_t)((1 << bit_depth) - 1));

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

        for (j = 0; j < x_pos_end8; j += 8) {
            v128_t t00, t01, t10, t11, t20, t21, t30, t31;
            v128_t e00, e01, e10, e11, e20, e21, e30, e31;
            v128_t s00, s01, s10, s11, s20, s21, s30, s31;
            v128_t s40, s41, s50, s51, s60, s61, s70, s71, s80, s81;
            v128_t ss1, ss2, s;

            /* coeff[0]: ±3 行同列 */
            t00 = v_load(&img_pad6[j]);
            t01 = v_load(&img_pad5[j]);
            e00 = v_unpacklo_epi16(t00, t01);
            e01 = v_unpackhi_epi16(t00, t01);
            s00 = v_madd_epi16(e00, c0);
            s01 = v_madd_epi16(e01, c0);

            /* coeff[1]: ±2 行同列 */
            t10 = v_load(&img_pad4[j]);
            t11 = v_load(&img_pad3[j]);
            e10 = v_unpacklo_epi16(t10, t11);
            e11 = v_unpackhi_epi16(t10, t11);
            s10 = v_madd_epi16(e10, c1);
            s11 = v_madd_epi16(e11, c1);

            /* coeff[2]: +1 行右1 + -1 行左1 (交叉) */
            t20 = v_load(&img_pad2[j - 1]);
            t21 = v_load(&img_pad1[j + 1]);
            e20 = v_unpacklo_epi16(t20, t21);
            e21 = v_unpackhi_epi16(t20, t21);
            s20 = v_madd_epi16(e20, c2);
            s21 = v_madd_epi16(e21, c2);

            /* coeff[3]: ±1 行同列 */
            t30 = v_load(&img_pad2[j]);
            t31 = v_load(&img_pad1[j]);
            e30 = v_unpacklo_epi16(t30, t31);
            e31 = v_unpackhi_epi16(t30, t31);
            s30 = v_madd_epi16(e30, c3);
            s31 = v_madd_epi16(e31, c3);

            /* coeff[4]: +1 行左1 + -1 行右1 (交叉) */
            t20 = v_load(&img_pad2[j + 1]);
            t21 = v_load(&img_pad1[j - 1]);
            e20 = v_unpacklo_epi16(t20, t21);
            e21 = v_unpackhi_epi16(t20, t21);
            s40 = v_madd_epi16(e20, c4);
            s41 = v_madd_epi16(e21, c4);

            /* coeff[5]: 0 行 ±3 列 */
            t20 = v_load(&src[j - 3]);
            t21 = v_load(&src[j + 3]);
            e20 = v_unpacklo_epi16(t20, t21);
            e21 = v_unpackhi_epi16(t20, t21);
            s50 = v_madd_epi16(e20, c5);
            s51 = v_madd_epi16(e21, c5);

            /* coeff[6]: 0 行 ±2 列 */
            t20 = v_load(&src[j - 2]);
            t21 = v_load(&src[j + 2]);
            e20 = v_unpacklo_epi16(t20, t21);
            e21 = v_unpackhi_epi16(t20, t21);
            s60 = v_madd_epi16(e20, c6);
            s61 = v_madd_epi16(e21, c6);

            /* coeff[7]: 0 行 ±1 列 */
            t20 = v_load(&src[j - 1]);
            t21 = v_load(&src[j + 1]);
            e20 = v_unpacklo_epi16(t20, t21);
            e21 = v_unpackhi_epi16(t20, t21);
            s70 = v_madd_epi16(e20, c7);
            s71 = v_madd_epi16(e21, c7);

            /* coeff[8]: 0 行 0 列 (中心, 与 0 交错使 madd 退化为单乘) */
            t20 = v_load(&src[j]);
            e20 = v_unpacklo_epi16(t20, zero);
            e21 = v_unpackhi_epi16(t20, zero);
            s80 = v_madd_epi16(e20, c8);
            s81 = v_madd_epi16(e21, c8);

            /* 累加前 4 个像素 */
            ss1 = v_add_epi32(s00, s10);
            ss1 = v_add_epi32(ss1, s20);
            ss1 = v_add_epi32(ss1, s30);
            ss1 = v_add_epi32(ss1, s40);
            ss1 = v_add_epi32(ss1, s50);
            ss1 = v_add_epi32(ss1, s60);
            ss1 = v_add_epi32(ss1, s70);
            ss1 = v_add_epi32(ss1, s80);

            /* 累加后 4 个像素 */
            ss2 = v_add_epi32(s01, s11);
            ss2 = v_add_epi32(ss2, s21);
            ss2 = v_add_epi32(ss2, s31);
            ss2 = v_add_epi32(ss2, s41);
            ss2 = v_add_epi32(ss2, s51);
            ss2 = v_add_epi32(ss2, s61);
            ss2 = v_add_epi32(ss2, s71);
            ss2 = v_add_epi32(ss2, s81);

            ss1 = v_add_epi32(ss1, m_add_offset);
            ss1 = v_srai_epi32(ss1, 6);
            ss2 = v_add_epi32(ss2, m_add_offset);
            ss2 = v_srai_epi32(ss2, 6);

            s = v_packus_epi32(ss1, ss2);   /* 8 uint16 */
            s = v_min_epu16(s, max_val);     /* clip to max_pel */
            v_store(dst + j, s);
        }

        /* 标量尾部: 不足 8 像素逐个处理 */
        for (; j < x_pos_end; j++) {
            int sum = 0;
            sum += alf_coeff[0] * (img_pad6[j] + img_pad5[j]);
            sum += alf_coeff[1] * (img_pad4[j] + img_pad3[j]);
            sum += alf_coeff[2] * (img_pad2[j - 1] + img_pad1[j + 1]);
            sum += alf_coeff[3] * (img_pad2[j] + img_pad1[j]);
            sum += alf_coeff[4] * (img_pad2[j + 1] + img_pad1[j - 1]);
            sum += alf_coeff[5] * (src[j - 3] + src[j + 3]);
            sum += alf_coeff[6] * (src[j - 2] + src[j + 2]);
            sum += alf_coeff[7] * (src[j - 1] + src[j + 1]);
            sum += alf_coeff[8] * src[j];
            dst[j] = (uint16_t)AVS2_CLIP3(0, (1 << bit_depth) - 1, (sum + 32) >> 6);
        }

        src += stride;
        dst += stride;
    }
}

/* ---------------------------------------------------------------------------
 * ALF SIMD 初始化 (NEON)
 * NEON 注册 block1 主体滤波; block2 (角点边界修正) 保留 C 实现.
 * ------------------------------------------------------------------------- */
void avs2_alf_init_neon(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.alf_block[0] = alf_filter_block1_neon;
}

/* SSE/AVX 接口空实现以供链接 (非 x86 平台不调用) */
void avs2_alf_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_alf_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }

#else

/* 非 x86 平台: 暂用 C 回退; NEON 路径待移植, 保留空实现以供链接 */
void avs2_alf_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_alf_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_alf_init_neon(const avs2_cpu_flags *flags) { (void)flags; }

#endif
