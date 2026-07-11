/*
 * sao_simd.c - SAO SIMD 实现 (x86 AVX2)
 *
 * 当前实现:
 *   - AVX2: 10-bit EO_0(水平), EO_90(垂直), BO(带偏移)
 *   - EO_135, EO_45 保持 C 回退 (符号缓存行依赖复杂, 暂不实现 SIMD)
 *
 * 算法:
 *   EO (Edge Offset): 每像素根据邻居符号差计算 edge_type (0..4), 加偏移
 *   BO (Band Offset): 按像素值右移分带 (32 带), 加对应偏移
 *
 * SIMD 要点:
 *   - 16 像素/寄存器 (__m256i = 16 x int16)
 *   - 符号计算: cmpgt + sub 组合得到 +1/0/-1
 *     sign(diff) = (zero>diff) - (diff>zero)  即 lt - gt
 *   - EO 偏移查表: 5 类比较选择 (cmpeq + and + or)
 *   - BO 偏移查表: cvtepi16_epi32 扩展 + i32gather_epi32 收集 32 带偏移
 *   - 32->16 压缩: packs_epi32 (有符号饱和) + permute4x64_epi64 通道重排
 *   - 裁剪: max_epi16(>=0) + min_epi16(<=max_pel), 有符号比较
 *
 * 边界处理:
 *   - 帧数据有 padding (AVS2_PAD_LUMA>=64), 越界访问 padding 区域安全
 *   - 左右上下邻域可用性由 avail 数组控制循环起止位置
 *   - 不足 16 像素的尾部用标量处理
 */

#include "internal.h"
#include "aec_internal.h"

#ifndef AVS2_CLIP3
#define AVS2_CLIP3(lo, hi, x) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(x) (void)(x)
#endif

/* SAO 邻域方向索引 (与 sao.c 一致) */
enum {
    sao_nb_t  = 0,   /* top         */
    sao_nb_d  = 1,   /* down        */
    sao_nb_l  = 2,   /* left        */
    sao_nb_r  = 3,   /* right       */
    sao_nb_tl = 4,   /* top-left    */
    sao_nb_tr = 5,   /* top-right   */
    sao_nb_dl = 6,   /* down-left   */
    sao_nb_dr = 7    /* down-right  */
};

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <immintrin.h>

/* ===========================================================================
 * 辅助函数
 * =========================================================================== */

/* 计算符号向量: diff > 0 -> +1, diff < 0 -> -1, diff == 0 -> 0
 * 原理: cmpgt_epi16 返回 -1(0xFFFF) 为真, 0 为假
 *   gt = (diff > 0) ? -1 : 0
 *   lt = (0 > diff) ? -1 : 0
 *   sign = lt - gt  =>  diff>0: 0-(-1)=+1, diff<0: -1-0=-1, diff==0: 0-0=0 */
static inline __m256i sao_sign_epi16(__m256i diff)
{
    __m256i zero = _mm256_setzero_si256();
    __m256i gt = _mm256_cmpgt_epi16(diff, zero);
    __m256i lt = _mm256_cmpgt_epi16(zero, diff);
    return _mm256_sub_epi16(lt, gt);
}

/* ===========================================================================
 * EO_0: 水平方向 (左右邻居)
 *
 * 对每行 [sx, ex) 范围的像素:
 *   left_sign  = sign(src[x] - src[x-1])
 *   right_sign = sign(src[x] - src[x+1])
 *   edge_type  = left_sign + right_sign + 2   (0..4)
 *   dst[x]     = clip(src[x] + offset[edge_type])
 * =========================================================================== */
static void sao_eo_0_avx2(uint8_t *_dst, int dst_stride, const uint8_t *_src,
                          int src_stride, int w, int h, int bit_depth,
                          const int *avail, const int *offset)
{
    int is_8bit = (bit_depth <= 8);
    uint8_t *dst8 = (uint8_t *)_dst;
    const uint8_t *src8 = (const uint8_t *)_src;
    uint16_t *dst = (uint16_t *)_dst;
    const uint16_t *src = (const uint16_t *)_src;
    const int max_pel = (1 << bit_depth) - 1;
    int sx = avail[sao_nb_l] ? 0 : 1;
    int ex = avail[sao_nb_r] ? w : (w - 1);
    int y;

    __m256i v_zero = _mm256_setzero_si256();
    __m256i v_max  = _mm256_set1_epi16((short)max_pel);
    __m256i v_two  = _mm256_set1_epi16(2);

    /* EO 偏移广播 (5 类, edge_type 0..4) */
    __m256i v_off0 = _mm256_set1_epi16((short)offset[0]);
    __m256i v_off1 = _mm256_set1_epi16((short)offset[1]);
    __m256i v_off2 = _mm256_set1_epi16((short)offset[2]);
    __m256i v_off3 = _mm256_set1_epi16((short)offset[3]);
    __m256i v_off4 = _mm256_set1_epi16((short)offset[4]);
    __m256i v_et1  = _mm256_set1_epi16(1);
    __m256i v_et2  = _mm256_set1_epi16(2);
    __m256i v_et3  = _mm256_set1_epi16(3);
    __m256i v_et4  = _mm256_set1_epi16(4);

    for (y = 0; y < h; y++) {
        int x = sx;

        /* SIMD 主循环: 每次 16 像素 */
        for (; x + 16 <= ex; x += 16) {
            __m256i v_left, v_center, v_right;
            if (is_8bit) {
                /* 8-bit: 加载 16 字节 → 扩展为 16 个 uint16 */
                v_left   = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(src8 + x - 1)));
                v_center = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(src8 + x)));
                v_right  = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(src8 + x + 1)));
            } else {
                v_left   = _mm256_loadu_si256((const __m256i *)(src + x - 1));
                v_center = _mm256_loadu_si256((const __m256i *)(src + x));
                v_right  = _mm256_loadu_si256((const __m256i *)(src + x + 1));
            }

            __m256i left_sign  = sao_sign_epi16(_mm256_sub_epi16(v_center, v_left));
            __m256i right_sign = sao_sign_epi16(_mm256_sub_epi16(v_center, v_right));

            __m256i edge_type = _mm256_add_epi16(
                _mm256_add_epi16(left_sign, right_sign), v_two);

            /* 按 edge_type 查找偏移量 (0..4) */
            __m256i v_off = _mm256_and_si256(
                _mm256_cmpeq_epi16(edge_type, v_zero), v_off0);
            v_off = _mm256_or_si256(v_off, _mm256_and_si256(
                _mm256_cmpeq_epi16(edge_type, v_et1), v_off1));
            v_off = _mm256_or_si256(v_off, _mm256_and_si256(
                _mm256_cmpeq_epi16(edge_type, v_et2), v_off2));
            v_off = _mm256_or_si256(v_off, _mm256_and_si256(
                _mm256_cmpeq_epi16(edge_type, v_et3), v_off3));
            v_off = _mm256_or_si256(v_off, _mm256_and_si256(
                _mm256_cmpeq_epi16(edge_type, v_et4), v_off4));

            __m256i v_result = _mm256_add_epi16(v_center, v_off);
            v_result = _mm256_max_epi16(v_result, v_zero);
            v_result = _mm256_min_epi16(v_result, v_max);

            if (is_8bit) {
                /* 8-bit: packus 压缩 16 个 uint16 → 16 字节, 存 128 位 */
                __m256i packed = _mm256_permute4x64_epi64(
                    _mm256_packus_epi16(v_result, v_zero), 0xD8);
                _mm_storeu_si128((__m128i *)(dst8 + x),
                                 _mm256_castsi256_si128(packed));
            } else {
                _mm256_storeu_si256((__m256i *)(dst + x), v_result);
            }
        }

        /* 标量尾部 */
        for (; x < ex; x++) {
            int c, pd_l, pd_r, ls, rs, et, val;
            if (is_8bit) {
                c = src8[x]; pd_l = c - src8[x - 1]; pd_r = c - src8[x + 1];
            } else {
                c = src[x];  pd_l = c - src[x - 1];  pd_r = c - src[x + 1];
            }
            ls = (pd_l > 0) ? 1 : ((pd_l < 0) ? -1 : 0);
            rs = (pd_r > 0) ? 1 : ((pd_r < 0) ? -1 : 0);
            et = ls + rs + 2;
            val = AVS2_CLIP3(0, max_pel, c + offset[et]);
            if (is_8bit) dst8[x] = (uint8_t)val;
            else         dst[x]  = (uint16_t)val;
        }

        if (is_8bit) { src8 += src_stride; dst8 += dst_stride; }
        else         { src  += src_stride; dst  += dst_stride; }
    }
}

/* ===========================================================================
 * EO_90: 垂直方向 (上下邻居)
 *
 * 对 [sy, ey) 行的每个像素:
 *   top_sign    = sign(src[y] - src[y-1])
 *   bottom_sign = sign(src[y] - src[y+1])
 *   edge_type   = top_sign + bottom_sign + 2   (0..4)
 *   dst[y]      = clip(src[y] + offset[edge_type])
 *
 * 注: C 代码按列遍历 (x 外层, y 内层) 利用 left_sign 传递优化,
 *     SIMD 改为按行遍历 (y 外层, x 内层), 每像素独立计算 top/bottom sign,
 *     结果等价 (left_sign 传递 = sign(src[y]-src[y-1]) 的展开形式).
 * =========================================================================== */
static void sao_eo_90_avx2(uint8_t *_dst, int dst_stride, const uint8_t *_src,
                           int src_stride, int w, int h, int bit_depth,
                           const int *avail, const int *offset)
{
    int is_8bit = (bit_depth <= 8);
    uint8_t *dst8 = (uint8_t *)_dst;
    const uint8_t *src8 = (const uint8_t *)_src;
    uint16_t *dst = (uint16_t *)_dst;
    const uint16_t *src = (const uint16_t *)_src;
    const int max_pel = (1 << bit_depth) - 1;
    int sy = avail[sao_nb_t] ? 0 : 1;
    int ey = avail[sao_nb_d] ? h : (h - 1);
    int y;

    __m256i v_zero = _mm256_setzero_si256();
    __m256i v_max  = _mm256_set1_epi16((short)max_pel);
    __m256i v_two  = _mm256_set1_epi16(2);

    __m256i v_off0 = _mm256_set1_epi16((short)offset[0]);
    __m256i v_off1 = _mm256_set1_epi16((short)offset[1]);
    __m256i v_off2 = _mm256_set1_epi16((short)offset[2]);
    __m256i v_off3 = _mm256_set1_epi16((short)offset[3]);
    __m256i v_off4 = _mm256_set1_epi16((short)offset[4]);
    __m256i v_et1  = _mm256_set1_epi16(1);
    __m256i v_et2  = _mm256_set1_epi16(2);
    __m256i v_et3  = _mm256_set1_epi16(3);
    __m256i v_et4  = _mm256_set1_epi16(4);

    for (y = sy; y < ey; y++) {
        const uint8_t  *row_top8, *row_center8, *row_bottom8;
        uint8_t        *row_dst8;
        const uint16_t *row_top, *row_center, *row_bottom;
        uint16_t       *row_dst;
        int x = 0;

        if (is_8bit) {
            row_top8    = src8 + (y - 1) * src_stride;
            row_center8 = src8 + y * src_stride;
            row_bottom8 = src8 + (y + 1) * src_stride;
            row_dst8    = dst8 + y * dst_stride;
        } else {
            row_top    = src + (y - 1) * src_stride;
            row_center = src + y * src_stride;
            row_bottom = src + (y + 1) * src_stride;
            row_dst    = dst + y * dst_stride;
        }

        /* SIMD 主循环: 每次 16 像素 */
        for (; x + 16 <= w; x += 16) {
            __m256i v_top, v_center, v_bottom;
            if (is_8bit) {
                v_top    = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(row_top8 + x)));
                v_center = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(row_center8 + x)));
                v_bottom = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(row_bottom8 + x)));
            } else {
                v_top    = _mm256_loadu_si256((const __m256i *)(row_top + x));
                v_center = _mm256_loadu_si256((const __m256i *)(row_center + x));
                v_bottom = _mm256_loadu_si256((const __m256i *)(row_bottom + x));
            }

            __m256i top_sign    = sao_sign_epi16(_mm256_sub_epi16(v_center, v_top));
            __m256i bottom_sign = sao_sign_epi16(_mm256_sub_epi16(v_center, v_bottom));

            __m256i edge_type = _mm256_add_epi16(
                _mm256_add_epi16(top_sign, bottom_sign), v_two);

            __m256i v_off = _mm256_and_si256(
                _mm256_cmpeq_epi16(edge_type, v_zero), v_off0);
            v_off = _mm256_or_si256(v_off, _mm256_and_si256(
                _mm256_cmpeq_epi16(edge_type, v_et1), v_off1));
            v_off = _mm256_or_si256(v_off, _mm256_and_si256(
                _mm256_cmpeq_epi16(edge_type, v_et2), v_off2));
            v_off = _mm256_or_si256(v_off, _mm256_and_si256(
                _mm256_cmpeq_epi16(edge_type, v_et3), v_off3));
            v_off = _mm256_or_si256(v_off, _mm256_and_si256(
                _mm256_cmpeq_epi16(edge_type, v_et4), v_off4));

            __m256i v_result = _mm256_add_epi16(v_center, v_off);
            v_result = _mm256_max_epi16(v_result, v_zero);
            v_result = _mm256_min_epi16(v_result, v_max);

            if (is_8bit) {
                __m256i packed = _mm256_permute4x64_epi64(
                    _mm256_packus_epi16(v_result, v_zero), 0xD8);
                _mm_storeu_si128((__m128i *)(row_dst8 + x),
                                 _mm256_castsi256_si128(packed));
            } else {
                _mm256_storeu_si256((__m256i *)(row_dst + x), v_result);
            }
        }

        /* 标量尾部 */
        for (; x < w; x++) {
            int c, pd_t, pd_b, ts, bs, et, val;
            if (is_8bit) {
                c = row_center8[x]; pd_t = c - row_top8[x]; pd_b = c - row_bottom8[x];
            } else {
                c = row_center[x];  pd_t = c - row_top[x];  pd_b = c - row_bottom[x];
            }
            ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
            bs = (pd_b > 0) ? 1 : ((pd_b < 0) ? -1 : 0);
            et = ts + bs + 2;
            val = AVS2_CLIP3(0, max_pel, c + offset[et]);
            if (is_8bit) row_dst8[x] = (uint8_t)val;
            else         row_dst[x]  = (uint16_t)val;
        }
    }
}

/* ===========================================================================
 * BO: 带偏移 (32 带)
 *
 * 对每个像素:
 *   edge_type = src[x] >> band_shift   (band_shift = bit_depth - 5)
 *   dst[x]    = clip(src[x] + offset[edge_type])
 *
 * SIMD 流程:
 *   1. 加载 16 个 uint16
 *   2. 拆分为两个 128-bit, 各扩展为 32-bit (cvtepi16_epi32)
 *   3. 算术右移 band_shift 得到带索引 (值非负, 等价逻辑右移)
 *   4. i32gather_epi32 收集 32 带偏移
 *   5. packs_epi32 压缩回 16-bit (有符号饱和, SAO 偏移在 int16 范围内)
 *   6. permute4x64_epi64(0xD8) 修正 packs 的 128-bit 通道交错
 *   7. 加偏移并裁剪
 * =========================================================================== */
static void sao_bo_avx2(uint8_t *_dst, int dst_stride, const uint8_t *_src,
                        int src_stride, int w, int h, int bit_depth,
                        const int *offset)
{
    int is_8bit = (bit_depth <= 8);
    uint8_t *dst8 = (uint8_t *)_dst;
    const uint8_t *src8 = (const uint8_t *)_src;
    uint16_t *dst = (uint16_t *)_dst;
    const uint16_t *src = (const uint16_t *)_src;
    const int max_pel = (1 << bit_depth) - 1;
    const int band_shift = bit_depth - NUM_SAO_BO_CLASSES_IN_BIT;
    int y;

    __m256i v_zero = _mm256_setzero_si256();
    __m256i v_max  = _mm256_set1_epi16((short)max_pel);

    for (y = 0; y < h; y++) {
        int x = 0;

        /* SIMD 主循环: 每次 16 像素 */
        for (; x + 16 <= w; x += 16) {
            __m256i v_src;
            if (is_8bit) {
                /* 8-bit: 加载 16 字节 → 扩展为 16 个 uint16 */
                v_src = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(src8 + x)));
            } else {
                v_src = _mm256_loadu_si256((const __m256i *)(src + x));
            }

            /* 拆分为两个 128-bit (各 8 像素), 扩展为 32-bit 以便 gather */
            __m128i v_lo16 = _mm256_castsi256_si128(v_src);
            __m128i v_hi16 = _mm256_extracti128_si256(v_src, 1);

            __m256i v_idx_lo = _mm256_cvtepi16_epi32(v_lo16);
            __m256i v_idx_hi = _mm256_cvtepi16_epi32(v_hi16);

            /* 计算带索引 (像素值非负, 算术右移 = 逻辑右移) */
            v_idx_lo = _mm256_srai_epi32(v_idx_lo, band_shift);
            v_idx_hi = _mm256_srai_epi32(v_idx_hi, band_shift);

            /* 收集偏移量 (32 带, scale=sizeof(int)=4) */
            __m256i v_off_lo = _mm256_i32gather_epi32(offset, v_idx_lo, sizeof(int));
            __m256i v_off_hi = _mm256_i32gather_epi32(offset, v_idx_hi, sizeof(int));

            /* 压缩回 16-bit (有符号饱和) + 通道重排 */
            __m256i v_off16 = _mm256_packs_epi32(v_off_lo, v_off_hi);
            v_off16 = _mm256_permute4x64_epi64(v_off16, 0xD8);

            /* 加偏移并裁剪 */
            __m256i v_result = _mm256_add_epi16(v_src, v_off16);
            v_result = _mm256_max_epi16(v_result, v_zero);
            v_result = _mm256_min_epi16(v_result, v_max);

            if (is_8bit) {
                /* 8-bit: packus 压缩 16 个 uint16 → 16 字节, 存 128 位 */
                __m256i packed = _mm256_permute4x64_epi64(
                    _mm256_packus_epi16(v_result, v_zero), 0xD8);
                _mm_storeu_si128((__m128i *)(dst8 + x),
                                 _mm256_castsi256_si128(packed));
            } else {
                _mm256_storeu_si256((__m256i *)(dst + x), v_result);
            }
        }

        /* 标量尾部 */
        for (; x < w; x++) {
            int c, edge_type, val;
            if (is_8bit) c = src8[x]; else c = src[x];
            edge_type = c >> band_shift;
            val = AVS2_CLIP3(0, max_pel, c + offset[edge_type]);
            if (is_8bit) dst8[x] = (uint8_t)val;
            else         dst[x]  = (uint16_t)val;
        }

        if (is_8bit) { src8 += src_stride; dst8 += dst_stride; }
        else         { src  += src_stride; dst  += dst_stride; }
    }
}

/* ===========================================================================
 * 注册函数
 * =========================================================================== */

/* SSE4.1: 暂用 C 回退 */
void avs2_sao_init_sse41(const avs2_cpu_flags *flags)
{
    UNUSED_PARAMETER(flags);
}

/* AVX2: 注册 EO_0, EO_90, BO (EO_135/EO_45 保持 C 回退) */
void avs2_sao_init_avx2(const avs2_cpu_flags *flags)
{
    UNUSED_PARAMETER(flags);
    avs2_dsp_table.sao_eo[0] = sao_eo_0_avx2;
    avs2_dsp_table.sao_eo[1] = sao_eo_90_avx2;
    /* sao_eo[2] (EO_135) 和 sao_eo[3] (EO_45) 保持 C 实现 */
    avs2_dsp_table.sao_bo   = sao_bo_avx2;
}

#else /* 非 x86 平台 */

void avs2_sao_init_sse41(const avs2_cpu_flags *flags)
{
    UNUSED_PARAMETER(flags);
}

void avs2_sao_init_avx2(const avs2_cpu_flags *flags)
{
    UNUSED_PARAMETER(flags);
}

#endif
