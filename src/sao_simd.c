/*
 * sao_simd.c - SAO SIMD 实现 (x86 SSE4.1)
 *
 * 当前实现:
 *   - SSE4.1: 10-bit/8-bit EO_0(水平), EO_90(垂直), BO(带偏移)
 *   - EO_135, EO_45 保持 C 回退 (符号缓存行依赖复杂, 暂不实现 SIMD)
 *
 * 算法:
 *   EO (Edge Offset): 每像素根据邻居符号差计算 edge_type (0..4), 加偏移
 *   BO (Band Offset): 按像素值右移分带 (32 带), 加对应偏移
 *
 * SIMD 要点:
 *   - 8 像素/寄存器 (__m128i = 8 x int16)
 *   - 符号计算: cmpgt + sub 组合得到 +1/0/-1
 *     sign(diff) = (zero>diff) - (diff>zero)  即 lt - gt
 *   - EO 偏移查表: 5 类比较选择 (cmpeq + and + or)
 *   - BO 偏移查表: 标量提取索引 + 数组查表 + setr_epi16 组装
 *     (SSE4.1 无 gather 指令, 用标量 gather 替代 AVX2 _mm256_i32gather_epi32)
 *   - 裁剪: max_epi16(>=0) + min_epi16(<=max_pel), 有符号比较
 *
 * 边界处理:
 *   - 帧数据有 padding (AVS2_PAD_LUMA>=64), 越界访问 padding 区域安全
 *   - 左右上下邻域可用性由 avail 数组控制循环起止位置
 *   - 不足 8 像素的尾部用标量处理
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

#include <tmmintrin.h>
#include <smmintrin.h>

/* ===========================================================================
 * 辅助函数
 * =========================================================================== */

/* 计算符号向量: diff > 0 -> +1, diff < 0 -> -1, diff == 0 -> 0
 * 原理: cmpgt_epi16 返回 -1(0xFFFF) 为真, 0 为假
 *   gt = (diff > 0) ? -1 : 0
 *   lt = (0 > diff) ? -1 : 0
 *   sign = lt - gt  =>  diff>0: 0-(-1)=+1, diff<0: -1-0=-1, diff==0: 0-0=0 */
static inline __m128i sao_sign_epi16(__m128i diff)
{
    __m128i zero = _mm_setzero_si128();
    __m128i gt = _mm_cmpgt_epi16(diff, zero);
    __m128i lt = _mm_cmpgt_epi16(zero, diff);
    return _mm_sub_epi16(lt, gt);
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
static void sao_eo_0_sse4(uint8_t *_dst, int dst_stride, const uint8_t *_src,
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

    __m128i v_zero = _mm_setzero_si128();
    __m128i v_max  = _mm_set1_epi16((short)max_pel);
    __m128i v_two  = _mm_set1_epi16(2);

    /* EO 偏移广播 (5 类, edge_type 0..4) */
    __m128i v_off0 = _mm_set1_epi16((short)offset[0]);
    __m128i v_off1 = _mm_set1_epi16((short)offset[1]);
    __m128i v_off2 = _mm_set1_epi16((short)offset[2]);
    __m128i v_off3 = _mm_set1_epi16((short)offset[3]);
    __m128i v_off4 = _mm_set1_epi16((short)offset[4]);
    __m128i v_et1  = _mm_set1_epi16(1);
    __m128i v_et2  = _mm_set1_epi16(2);
    __m128i v_et3  = _mm_set1_epi16(3);
    __m128i v_et4  = _mm_set1_epi16(4);

    for (y = 0; y < h; y++) {
        int x = sx;

        /* SIMD 主循环: 每次 8 像素 */
        for (; x + 8 <= ex; x += 8) {
            __m128i v_left, v_center, v_right;
            if (is_8bit) {
                /* 8-bit: 加载 8 字节 → 扩展为 8 个 uint16 */
                v_left   = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(src8 + x - 1)));
                v_center = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(src8 + x)));
                v_right  = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(src8 + x + 1)));
            } else {
                v_left   = _mm_loadu_si128((const __m128i *)(src + x - 1));
                v_center = _mm_loadu_si128((const __m128i *)(src + x));
                v_right  = _mm_loadu_si128((const __m128i *)(src + x + 1));
            }

            __m128i left_sign  = sao_sign_epi16(_mm_sub_epi16(v_center, v_left));
            __m128i right_sign = sao_sign_epi16(_mm_sub_epi16(v_center, v_right));

            __m128i edge_type = _mm_add_epi16(
                _mm_add_epi16(left_sign, right_sign), v_two);

            /* 按 edge_type 查找偏移量 (0..4) */
            __m128i v_off = _mm_and_si128(
                _mm_cmpeq_epi16(edge_type, v_zero), v_off0);
            v_off = _mm_or_si128(v_off, _mm_and_si128(
                _mm_cmpeq_epi16(edge_type, v_et1), v_off1));
            v_off = _mm_or_si128(v_off, _mm_and_si128(
                _mm_cmpeq_epi16(edge_type, v_et2), v_off2));
            v_off = _mm_or_si128(v_off, _mm_and_si128(
                _mm_cmpeq_epi16(edge_type, v_et3), v_off3));
            v_off = _mm_or_si128(v_off, _mm_and_si128(
                _mm_cmpeq_epi16(edge_type, v_et4), v_off4));

            __m128i v_result = _mm_add_epi16(v_center, v_off);
            v_result = _mm_max_epi16(v_result, v_zero);
            v_result = _mm_min_epi16(v_result, v_max);

            if (is_8bit) {
                /* 8-bit: packus 压缩 8 个 uint16 → 8 字节, 存 64 位 */
                __m128i packed = _mm_packus_epi16(v_result, v_zero);
                _mm_storel_epi64((__m128i *)(dst8 + x), packed);
            } else {
                _mm_storeu_si128((__m128i *)(dst + x), v_result);
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
 * =========================================================================== */
static void sao_eo_90_sse4(uint8_t *_dst, int dst_stride, const uint8_t *_src,
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

    __m128i v_zero = _mm_setzero_si128();
    __m128i v_max  = _mm_set1_epi16((short)max_pel);
    __m128i v_two  = _mm_set1_epi16(2);

    __m128i v_off0 = _mm_set1_epi16((short)offset[0]);
    __m128i v_off1 = _mm_set1_epi16((short)offset[1]);
    __m128i v_off2 = _mm_set1_epi16((short)offset[2]);
    __m128i v_off3 = _mm_set1_epi16((short)offset[3]);
    __m128i v_off4 = _mm_set1_epi16((short)offset[4]);
    __m128i v_et1  = _mm_set1_epi16(1);
    __m128i v_et2  = _mm_set1_epi16(2);
    __m128i v_et3  = _mm_set1_epi16(3);
    __m128i v_et4  = _mm_set1_epi16(4);

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

        /* SIMD 主循环: 每次 8 像素 */
        for (; x + 8 <= w; x += 8) {
            __m128i v_top, v_center, v_bottom;
            if (is_8bit) {
                v_top    = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(row_top8 + x)));
                v_center = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(row_center8 + x)));
                v_bottom = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(row_bottom8 + x)));
            } else {
                v_top    = _mm_loadu_si128((const __m128i *)(row_top + x));
                v_center = _mm_loadu_si128((const __m128i *)(row_center + x));
                v_bottom = _mm_loadu_si128((const __m128i *)(row_bottom + x));
            }

            __m128i top_sign    = sao_sign_epi16(_mm_sub_epi16(v_center, v_top));
            __m128i bottom_sign = sao_sign_epi16(_mm_sub_epi16(v_center, v_bottom));

            __m128i edge_type = _mm_add_epi16(
                _mm_add_epi16(top_sign, bottom_sign), v_two);

            __m128i v_off = _mm_and_si128(
                _mm_cmpeq_epi16(edge_type, v_zero), v_off0);
            v_off = _mm_or_si128(v_off, _mm_and_si128(
                _mm_cmpeq_epi16(edge_type, v_et1), v_off1));
            v_off = _mm_or_si128(v_off, _mm_and_si128(
                _mm_cmpeq_epi16(edge_type, v_et2), v_off2));
            v_off = _mm_or_si128(v_off, _mm_and_si128(
                _mm_cmpeq_epi16(edge_type, v_et3), v_off3));
            v_off = _mm_or_si128(v_off, _mm_and_si128(
                _mm_cmpeq_epi16(edge_type, v_et4), v_off4));

            __m128i v_result = _mm_add_epi16(v_center, v_off);
            v_result = _mm_max_epi16(v_result, v_zero);
            v_result = _mm_min_epi16(v_result, v_max);

            if (is_8bit) {
                __m128i packed = _mm_packus_epi16(v_result, v_zero);
                _mm_storel_epi64((__m128i *)(row_dst8 + x), packed);
            } else {
                _mm_storeu_si128((__m128i *)(row_dst + x), v_result);
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
 * EO_135: 135 度对角方向 (左上-右下邻居)
 *
 * 算法: edge_type = sign(c-tl) + sign(c-dr) + 2, dst = clip(c + offset[et])
 *   tl = src[y-stride-1] (左上), dr = src[y+stride+1] (右下)
 *
 * 边界处理: 首尾行有角落像素 (TL/DR) 可用性问题, 用标量处理.
 *   内部 LCU (avail 全 1): 所有行用 SIMD.
 *   边界 LCU: 首尾行用标量, 中间行用 SIMD.
 * =========================================================================== */
static void sao_eo_135_sse4(uint8_t *_dst, int dst_stride, const uint8_t *_src,
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

    __m128i v_zero = _mm_setzero_si128();
    __m128i v_max  = _mm_set1_epi16((short)max_pel);
    __m128i v_two  = _mm_set1_epi16(2);
    __m128i v_off0 = _mm_set1_epi16((short)offset[0]);
    __m128i v_off1 = _mm_set1_epi16((short)offset[1]);
    __m128i v_off2 = _mm_set1_epi16((short)offset[2]);
    __m128i v_off3 = _mm_set1_epi16((short)offset[3]);
    __m128i v_off4 = _mm_set1_epi16((short)offset[4]);
    __m128i v_et1  = _mm_set1_epi16(1);
    __m128i v_et2  = _mm_set1_epi16(2);
    __m128i v_et3  = _mm_set1_epi16(3);
    __m128i v_et4  = _mm_set1_epi16(4);

    /* 首行: 标量处理 (角落 TL/DR 可用性不同) */
    if (avail[sao_nb_t]) {
        int sx0 = avail[sao_nb_tl] ? 0 : 1;
        int ex0 = ex;
        int x;
        y = 0;
        if (is_8bit) {
            const uint8_t *rc = src8 + y * src_stride;
            uint8_t *rd = dst8 + y * dst_stride;
            for (x = sx0; x < ex0; x++) {
                int c = rc[x], pd_t = c - rc[-src_stride + x - 1];
                int pd_d = c - rc[src_stride + x + 1];
                int ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                int ds = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                int et = ts + ds + 2;
                rd[x] = (uint8_t)AVS2_CLIP3(0, max_pel, c + offset[et]);
            }
        } else {
            const uint16_t *rc = src + y * src_stride;
            uint16_t *rd = dst + y * dst_stride;
            for (x = sx0; x < ex0; x++) {
                int c = rc[x], pd_t = c - rc[-src_stride + x - 1];
                int pd_d = c - rc[src_stride + x + 1];
                int ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                int ds = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                int et = ts + ds + 2;
                rd[x] = (uint16_t)AVS2_CLIP3(0, max_pel, c + offset[et]);
            }
        }
    }

    /* 中间行: SIMD 处理 */
    {
        int sy_mid = avail[sao_nb_t] ? 1 : 1;
        int ey_mid = avail[sao_nb_d] ? (h - 1) : (h - 1);
        for (y = sy_mid; y < ey_mid; y++) {
            int x = sx;
            if (is_8bit) {
                const uint8_t *rc = src8 + y * src_stride;
                const uint8_t *rtl = rc - src_stride - 1;
                const uint8_t *rdr = rc + src_stride + 1;
                uint8_t *rd = dst8 + y * dst_stride;
                for (; x + 8 <= ex; x += 8) {
                    __m128i vc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(rc + x)));
                    __m128i vtl = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(rtl + x)));
                    __m128i vdr = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(rdr + x)));
                    __m128i ts = sao_sign_epi16(_mm_sub_epi16(vc, vtl));
                    __m128i ds = sao_sign_epi16(_mm_sub_epi16(vc, vdr));
                    __m128i et = _mm_add_epi16(_mm_add_epi16(ts, ds), v_two);
                    __m128i vo = _mm_and_si128(_mm_cmpeq_epi16(et, v_zero), v_off0);
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et1), v_off1));
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et2), v_off2));
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et3), v_off3));
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et4), v_off4));
                    __m128i vr = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(vc, vo), v_zero), v_max);
                    _mm_storel_epi64((__m128i*)(rd + x), _mm_packus_epi16(vr, v_zero));
                }
                for (; x < ex; x++) {
                    int c = rc[x], pd_t = c - rtl[x], pd_d = c - rdr[x];
                    int ts2 = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                    int ds2 = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                    int et2 = ts2 + ds2 + 2;
                    rd[x] = (uint8_t)AVS2_CLIP3(0, max_pel, c + offset[et2]);
                }
            } else {
                const uint16_t *rc = src + y * src_stride;
                const uint16_t *rtl = rc - src_stride - 1;
                const uint16_t *rdr = rc + src_stride + 1;
                uint16_t *rd = dst + y * dst_stride;
                for (; x + 8 <= ex; x += 8) {
                    __m128i vc = _mm_loadu_si128((const __m128i*)(rc + x));
                    __m128i vtl = _mm_loadu_si128((const __m128i*)(rtl + x));
                    __m128i vdr = _mm_loadu_si128((const __m128i*)(rdr + x));
                    __m128i ts = sao_sign_epi16(_mm_sub_epi16(vc, vtl));
                    __m128i ds = sao_sign_epi16(_mm_sub_epi16(vc, vdr));
                    __m128i et = _mm_add_epi16(_mm_add_epi16(ts, ds), v_two);
                    __m128i vo = _mm_and_si128(_mm_cmpeq_epi16(et, v_zero), v_off0);
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et1), v_off1));
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et2), v_off2));
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et3), v_off3));
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et4), v_off4));
                    __m128i vr = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(vc, vo), v_zero), v_max);
                    _mm_storeu_si128((__m128i*)(rd + x), vr);
                }
                for (; x < ex; x++) {
                    int c = rc[x], pd_t = c - rtl[x], pd_d = c - rdr[x];
                    int ts2 = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                    int ds2 = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                    int et2 = ts2 + ds2 + 2;
                    rd[x] = (uint16_t)AVS2_CLIP3(0, max_pel, c + offset[et2]);
                }
            }
        }
    }

    /* 末行: 标量处理 */
    if (avail[sao_nb_d]) {
        int sxn = avail[sao_nb_l] ? 0 : 1;
        int exn = avail[sao_nb_dr] ? w : (w - 1);
        int x;
        y = h - 1;
        if (is_8bit) {
            const uint8_t *rc = src8 + y * src_stride;
            uint8_t *rd = dst8 + y * dst_stride;
            for (x = sxn; x < exn; x++) {
                int c = rc[x], pd_t = c - rc[-src_stride + x - 1];
                int pd_d = c - rc[src_stride + x + 1];
                int ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                int ds = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                int et = ts + ds + 2;
                rd[x] = (uint8_t)AVS2_CLIP3(0, max_pel, c + offset[et]);
            }
        } else {
            const uint16_t *rc = src + y * src_stride;
            uint16_t *rd = dst + y * dst_stride;
            for (x = sxn; x < exn; x++) {
                int c = rc[x], pd_t = c - rc[-src_stride + x - 1];
                int pd_d = c - rc[src_stride + x + 1];
                int ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                int ds = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                int et = ts + ds + 2;
                rd[x] = (uint16_t)AVS2_CLIP3(0, max_pel, c + offset[et]);
            }
        }
    }
}

/* ===========================================================================
 * EO_45: 45 度对角方向 (右上-左下邻居)
 *
 * 算法: edge_type = sign(c-tr) + sign(c-dl) + 2, dst = clip(c + offset[et])
 *   tr = src[y-stride+1] (右上), dl = src[y+stride-1] (左下)
 *
 * 边界处理: 首尾行有角落像素 (TR/DL) 可用性问题, 用标量处理.
 * =========================================================================== */
static void sao_eo_45_sse4(uint8_t *_dst, int dst_stride, const uint8_t *_src,
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

    __m128i v_zero = _mm_setzero_si128();
    __m128i v_max  = _mm_set1_epi16((short)max_pel);
    __m128i v_two  = _mm_set1_epi16(2);
    __m128i v_off0 = _mm_set1_epi16((short)offset[0]);
    __m128i v_off1 = _mm_set1_epi16((short)offset[1]);
    __m128i v_off2 = _mm_set1_epi16((short)offset[2]);
    __m128i v_off3 = _mm_set1_epi16((short)offset[3]);
    __m128i v_off4 = _mm_set1_epi16((short)offset[4]);
    __m128i v_et1  = _mm_set1_epi16(1);
    __m128i v_et2  = _mm_set1_epi16(2);
    __m128i v_et3  = _mm_set1_epi16(3);
    __m128i v_et4  = _mm_set1_epi16(4);

    /* 首行: 标量处理 */
    if (avail[sao_nb_t]) {
        int sx0 = sx;
        int ex0 = avail[sao_nb_tr] ? w : (w - 1);
        int x;
        y = 0;
        if (is_8bit) {
            const uint8_t *rc = src8 + y * src_stride;
            uint8_t *rd = dst8 + y * dst_stride;
            for (x = sx0; x < ex0; x++) {
                int c = rc[x], pd_t = c - rc[-src_stride + x + 1];
                int pd_d = c - rc[src_stride + x - 1];
                int ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                int ds = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                int et = ts + ds + 2;
                rd[x] = (uint8_t)AVS2_CLIP3(0, max_pel, c + offset[et]);
            }
        } else {
            const uint16_t *rc = src + y * src_stride;
            uint16_t *rd = dst + y * dst_stride;
            for (x = sx0; x < ex0; x++) {
                int c = rc[x], pd_t = c - rc[-src_stride + x + 1];
                int pd_d = c - rc[src_stride + x - 1];
                int ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                int ds = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                int et = ts + ds + 2;
                rd[x] = (uint16_t)AVS2_CLIP3(0, max_pel, c + offset[et]);
            }
        }
    }

    /* 中间行: SIMD 处理 */
    {
        int sy_mid = avail[sao_nb_t] ? 1 : 1;
        int ey_mid = avail[sao_nb_d] ? (h - 1) : (h - 1);
        for (y = sy_mid; y < ey_mid; y++) {
            int x = sx;
            if (is_8bit) {
                const uint8_t *rc = src8 + y * src_stride;
                const uint8_t *rtr = rc - src_stride + 1;
                const uint8_t *rdl = rc + src_stride - 1;
                uint8_t *rd = dst8 + y * dst_stride;
                for (; x + 8 <= ex; x += 8) {
                    __m128i vc = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(rc + x)));
                    __m128i vtr = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(rtr + x)));
                    __m128i vdl = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(rdl + x)));
                    __m128i ts = sao_sign_epi16(_mm_sub_epi16(vc, vtr));
                    __m128i ds = sao_sign_epi16(_mm_sub_epi16(vc, vdl));
                    __m128i et = _mm_add_epi16(_mm_add_epi16(ts, ds), v_two);
                    __m128i vo = _mm_and_si128(_mm_cmpeq_epi16(et, v_zero), v_off0);
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et1), v_off1));
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et2), v_off2));
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et3), v_off3));
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et4), v_off4));
                    __m128i vr = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(vc, vo), v_zero), v_max);
                    _mm_storel_epi64((__m128i*)(rd + x), _mm_packus_epi16(vr, v_zero));
                }
                for (; x < ex; x++) {
                    int c = rc[x], pd_t = c - rtr[x], pd_d = c - rdl[x];
                    int ts2 = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                    int ds2 = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                    int et2 = ts2 + ds2 + 2;
                    rd[x] = (uint8_t)AVS2_CLIP3(0, max_pel, c + offset[et2]);
                }
            } else {
                const uint16_t *rc = src + y * src_stride;
                const uint16_t *rtr = rc - src_stride + 1;
                const uint16_t *rdl = rc + src_stride - 1;
                uint16_t *rd = dst + y * dst_stride;
                for (; x + 8 <= ex; x += 8) {
                    __m128i vc = _mm_loadu_si128((const __m128i*)(rc + x));
                    __m128i vtr = _mm_loadu_si128((const __m128i*)(rtr + x));
                    __m128i vdl = _mm_loadu_si128((const __m128i*)(rdl + x));
                    __m128i ts = sao_sign_epi16(_mm_sub_epi16(vc, vtr));
                    __m128i ds = sao_sign_epi16(_mm_sub_epi16(vc, vdl));
                    __m128i et = _mm_add_epi16(_mm_add_epi16(ts, ds), v_two);
                    __m128i vo = _mm_and_si128(_mm_cmpeq_epi16(et, v_zero), v_off0);
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et1), v_off1));
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et2), v_off2));
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et3), v_off3));
                    vo = _mm_or_si128(vo, _mm_and_si128(_mm_cmpeq_epi16(et, v_et4), v_off4));
                    __m128i vr = _mm_min_epi16(_mm_max_epi16(_mm_add_epi16(vc, vo), v_zero), v_max);
                    _mm_storeu_si128((__m128i*)(rd + x), vr);
                }
                for (; x < ex; x++) {
                    int c = rc[x], pd_t = c - rtr[x], pd_d = c - rdl[x];
                    int ts2 = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                    int ds2 = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                    int et2 = ts2 + ds2 + 2;
                    rd[x] = (uint16_t)AVS2_CLIP3(0, max_pel, c + offset[et2]);
                }
            }
        }
    }

    /* 末行: 标量处理 */
    if (avail[sao_nb_d]) {
        int sxn = avail[sao_nb_dl] ? 0 : 1;
        int exn = ex;
        int x;
        y = h - 1;
        if (is_8bit) {
            const uint8_t *rc = src8 + y * src_stride;
            uint8_t *rd = dst8 + y * dst_stride;
            for (x = sxn; x < exn; x++) {
                int c = rc[x], pd_t = c - rc[-src_stride + x + 1];
                int pd_d = c - rc[src_stride + x - 1];
                int ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                int ds = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                int et = ts + ds + 2;
                rd[x] = (uint8_t)AVS2_CLIP3(0, max_pel, c + offset[et]);
            }
        } else {
            const uint16_t *rc = src + y * src_stride;
            uint16_t *rd = dst + y * dst_stride;
            for (x = sxn; x < exn; x++) {
                int c = rc[x], pd_t = c - rc[-src_stride + x + 1];
                int pd_d = c - rc[src_stride + x - 1];
                int ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                int ds = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                int et = ts + ds + 2;
                rd[x] = (uint16_t)AVS2_CLIP3(0, max_pel, c + offset[et]);
            }
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
 * SSE4.1 SIMD 流程 (8 像素/次):
 *   1. 加载 8 个 uint16
 *   2. 算术右移 band_shift 得到带索引 (0..31, 值非负)
 *   3. 标量 gather: 逐个提取索引查表 (SSE4.1 无 gather 指令)
 *   4. 用 _mm_setr_epi16 组装偏移向量
 *   5. 加偏移并裁剪
 * =========================================================================== */
static void sao_bo_sse4(uint8_t *_dst, int dst_stride, const uint8_t *_src,
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

    __m128i v_zero = _mm_setzero_si128();
    __m128i v_max  = _mm_set1_epi16((short)max_pel);

    for (y = 0; y < h; y++) {
        int x = 0;

        /* SIMD 主循环: 每次 8 像素 */
        for (; x + 8 <= w; x += 8) {
            __m128i v_src;
            if (is_8bit) {
                /* 8-bit: 加载 8 字节 → 扩展为 8 个 uint16 */
                v_src = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(src8 + x)));
            } else {
                v_src = _mm_loadu_si128((const __m128i *)(src + x));
            }

            /* 计算带索引 (像素值非负, 算术右移 = 逻辑右移) */
            __m128i v_idx = _mm_srai_epi16(v_src, band_shift);

            /* 标量 gather: 逐个提取索引查表, 组装为偏移向量 */
            __m128i v_off = _mm_setr_epi16(
                (short)offset[_mm_extract_epi16(v_idx, 0)],
                (short)offset[_mm_extract_epi16(v_idx, 1)],
                (short)offset[_mm_extract_epi16(v_idx, 2)],
                (short)offset[_mm_extract_epi16(v_idx, 3)],
                (short)offset[_mm_extract_epi16(v_idx, 4)],
                (short)offset[_mm_extract_epi16(v_idx, 5)],
                (short)offset[_mm_extract_epi16(v_idx, 6)],
                (short)offset[_mm_extract_epi16(v_idx, 7)]
            );

            /* 加偏移并裁剪 */
            __m128i v_result = _mm_add_epi16(v_src, v_off);
            v_result = _mm_max_epi16(v_result, v_zero);
            v_result = _mm_min_epi16(v_result, v_max);

            if (is_8bit) {
                /* 8-bit: packus 压缩 8 个 uint16 → 8 字节, 存 64 位 */
                __m128i packed = _mm_packus_epi16(v_result, v_zero);
                _mm_storel_epi64((__m128i *)(dst8 + x), packed);
            } else {
                _mm_storeu_si128((__m128i *)(dst + x), v_result);
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

/* SSE4.1: 注册全部 SAO 函数 (EO_0, EO_90, EO_135, EO_45, BO) */
void avs2_sao_init_sse41(const avs2_cpu_flags *flags)
{
    UNUSED_PARAMETER(flags);
    avs2_dsp_table.sao_eo[0] = sao_eo_0_sse4;
    avs2_dsp_table.sao_eo[1] = sao_eo_90_sse4;
    avs2_dsp_table.sao_eo[2] = sao_eo_135_sse4;
    avs2_dsp_table.sao_eo[3] = sao_eo_45_sse4;
    avs2_dsp_table.sao_bo   = sao_bo_sse4;
}

/* AVX2: 已清空 (降级为 SSE4.1) */
void avs2_sao_init_avx2(const avs2_cpu_flags *flags)
{
    UNUSED_PARAMETER(flags);
}

#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)

#include <arm_neon.h>

/* ---- C 回退函数声明 (在 sao.c 中定义, 供 8-bit 路径回退) ---- */
extern void sao_eo_0_c(uint8_t *dst, int dst_stride, const uint8_t *src, int src_stride,
                       int w, int h, int bit_depth, const int *avail, const int *offset);
extern void sao_eo_90_c(uint8_t *dst, int dst_stride, const uint8_t *src, int src_stride,
                        int w, int h, int bit_depth, const int *avail, const int *offset);
extern void sao_eo_135_c(uint8_t *dst, int dst_stride, const uint8_t *src, int src_stride,
                         int w, int h, int bit_depth, const int *avail, const int *offset);
extern void sao_eo_45_c(uint8_t *dst, int dst_stride, const uint8_t *src, int src_stride,
                        int w, int h, int bit_depth, const int *avail, const int *offset);
extern void sao_bo_c(uint8_t *dst, int dst_stride, const uint8_t *src, int src_stride,
                     int w, int h, int bit_depth, const int *offset);

/* ===========================================================================
 * NEON SAO 实现 (由 SSE4.1 版本逐行移植).
 *
 * 策略 (与 lf/mc NEON 一致): 10-bit 路径原生 NEON, 8-bit 回退 C 参考.
 *
 * 位精确性:
 *   sign(diff) = (0>diff) - (diff>0)   <- vcgtq_s16 返回 0/-1, 与 SSE cmpgt 一致
 *   edge_type 偏移选择: 5 路 vceqq_s16 + vandq_s16 + vorrq_s16, 与 SSE 一致
 * =========================================================================== */

/* 符号向量: diff>0 -> +1, diff<0 -> -1, diff==0 -> 0
 * 等价于 SSE sao_sign_epi16: sign = lt - gt */
static inline int16x8_t sao_sign_neon(int16x8_t diff)
{
    const int16x8_t zero = vdupq_n_s16(0);
    /* vcgtq_s16 返回 uint16x8_t 掩码 (0/0xFFFF), 重解释为 int16 再相减 */
    return vsubq_s16(vreinterpretq_s16_u16(vcgtq_s16(zero, diff)),
                     vreinterpretq_s16_u16(vcgtq_s16(diff, zero)));
}

/* 按 edge_type (0..4) 选择偏移 (5 路比较-选择, 与 SSE 一致) */
static inline int16x8_t sao_off_select_neon(int16x8_t et,
                                            int16x8_t c1, int16x8_t c2, int16x8_t c3, int16x8_t c4,
                                            int16x8_t v0, int16x8_t v1, int16x8_t v2,
                                            int16x8_t v3, int16x8_t v4, int16x8_t zero)
{
    int16x8_t vo = vandq_s16(vreinterpretq_s16_u16(vceqq_s16(et, zero)), v0);
    vo = vorrq_s16(vo, vandq_s16(vreinterpretq_s16_u16(vceqq_s16(et, c1)), v1));
    vo = vorrq_s16(vo, vandq_s16(vreinterpretq_s16_u16(vceqq_s16(et, c2)), v2));
    vo = vorrq_s16(vo, vandq_s16(vreinterpretq_s16_u16(vceqq_s16(et, c3)), v3));
    vo = vorrq_s16(vo, vandq_s16(vreinterpretq_s16_u16(vceqq_s16(et, c4)), v4));
    return vo;
}

/* ===========================================================================
 * EO_0: 水平方向 (左右邻居), 10-bit NEON
 * =========================================================================== */
static void sao_eo_0_neon(uint8_t *_dst, int dst_stride, const uint8_t *_src,
                          int src_stride, int w, int h, int bit_depth,
                          const int *avail, const int *offset)
{
    if (bit_depth <= 8) {
        sao_eo_0_c(_dst, dst_stride, _src, src_stride, w, h, bit_depth, avail, offset);
        return;
    }
    uint16_t *dst = (uint16_t *)_dst;
    const uint16_t *src = (const uint16_t *)_src;
    const int max_pel = (1 << bit_depth) - 1;
    int sx = avail[sao_nb_l] ? 0 : 1;
    int ex = avail[sao_nb_r] ? w : (w - 1);
    int y;

    int16x8_t v_zero = vdupq_n_s16(0);
    int16x8_t v_max  = vdupq_n_s16((int16_t)max_pel);
    int16x8_t v_two  = vdupq_n_s16(2);
    int16x8_t v_et1  = vdupq_n_s16(1);
    int16x8_t v_et2  = vdupq_n_s16(2);
    int16x8_t v_et3  = vdupq_n_s16(3);
    int16x8_t v_et4  = vdupq_n_s16(4);
    int16x8_t v_off0 = vdupq_n_s16((int16_t)offset[0]);
    int16x8_t v_off1 = vdupq_n_s16((int16_t)offset[1]);
    int16x8_t v_off2 = vdupq_n_s16((int16_t)offset[2]);
    int16x8_t v_off3 = vdupq_n_s16((int16_t)offset[3]);
    int16x8_t v_off4 = vdupq_n_s16((int16_t)offset[4]);

    for (y = 0; y < h; y++) {
        int x = sx;

        for (; x + 8 <= ex; x += 8) {
            int16x8_t v_left   = vreinterpretq_s16_u16(vld1q_u16(src + x - 1));
            int16x8_t v_center = vreinterpretq_s16_u16(vld1q_u16(src + x));
            int16x8_t v_right  = vreinterpretq_s16_u16(vld1q_u16(src + x + 1));

            int16x8_t left_sign  = sao_sign_neon(vsubq_s16(v_center, v_left));
            int16x8_t right_sign = sao_sign_neon(vsubq_s16(v_center, v_right));

            int16x8_t edge_type = vaddq_s16(vaddq_s16(left_sign, right_sign), v_two);
            int16x8_t v_off = sao_off_select_neon(edge_type, v_et1, v_et2, v_et3, v_et4,
                                                  v_off0, v_off1, v_off2, v_off3, v_off4, v_zero);
            int16x8_t v_result = vminq_s16(vmaxq_s16(vaddq_s16(v_center, v_off), v_zero), v_max);
            vst1q_u16(dst + x, vreinterpretq_u16_s16(v_result));
        }

        for (; x < ex; x++) {
            int c, pd_l, pd_r, ls, rs, et, val;
            c = src[x]; pd_l = c - src[x - 1]; pd_r = c - src[x + 1];
            ls = (pd_l > 0) ? 1 : ((pd_l < 0) ? -1 : 0);
            rs = (pd_r > 0) ? 1 : ((pd_r < 0) ? -1 : 0);
            et = ls + rs + 2;
            val = AVS2_CLIP3(0, max_pel, c + offset[et]);
            dst[x] = (uint16_t)val;
        }

        src += src_stride;
        dst += dst_stride;
    }
}

/* ===========================================================================
 * EO_90: 垂直方向 (上下邻居), 10-bit NEON
 * =========================================================================== */
static void sao_eo_90_neon(uint8_t *_dst, int dst_stride, const uint8_t *_src,
                           int src_stride, int w, int h, int bit_depth,
                           const int *avail, const int *offset)
{
    if (bit_depth <= 8) {
        sao_eo_90_c(_dst, dst_stride, _src, src_stride, w, h, bit_depth, avail, offset);
        return;
    }
    uint16_t *dst = (uint16_t *)_dst;
    const uint16_t *src = (const uint16_t *)_src;
    const int max_pel = (1 << bit_depth) - 1;
    int sy = avail[sao_nb_t] ? 0 : 1;
    int ey = avail[sao_nb_d] ? h : (h - 1);
    int y;

    int16x8_t v_zero = vdupq_n_s16(0);
    int16x8_t v_max  = vdupq_n_s16((int16_t)max_pel);
    int16x8_t v_two  = vdupq_n_s16(2);
    int16x8_t v_et1  = vdupq_n_s16(1);
    int16x8_t v_et2  = vdupq_n_s16(2);
    int16x8_t v_et3  = vdupq_n_s16(3);
    int16x8_t v_et4  = vdupq_n_s16(4);
    int16x8_t v_off0 = vdupq_n_s16((int16_t)offset[0]);
    int16x8_t v_off1 = vdupq_n_s16((int16_t)offset[1]);
    int16x8_t v_off2 = vdupq_n_s16((int16_t)offset[2]);
    int16x8_t v_off3 = vdupq_n_s16((int16_t)offset[3]);
    int16x8_t v_off4 = vdupq_n_s16((int16_t)offset[4]);

    for (y = sy; y < ey; y++) {
        const uint16_t *row_top    = src + (y - 1) * src_stride;
        const uint16_t *row_center = src + y * src_stride;
        const uint16_t *row_bottom = src + (y + 1) * src_stride;
        uint16_t *row_dst = dst + y * dst_stride;
        int x = 0;

        for (; x + 8 <= w; x += 8) {
            int16x8_t v_top    = vreinterpretq_s16_u16(vld1q_u16(row_top + x));
            int16x8_t v_center = vreinterpretq_s16_u16(vld1q_u16(row_center + x));
            int16x8_t v_bottom = vreinterpretq_s16_u16(vld1q_u16(row_bottom + x));

            int16x8_t top_sign    = sao_sign_neon(vsubq_s16(v_center, v_top));
            int16x8_t bottom_sign = sao_sign_neon(vsubq_s16(v_center, v_bottom));

            int16x8_t edge_type = vaddq_s16(vaddq_s16(top_sign, bottom_sign), v_two);
            int16x8_t v_off = sao_off_select_neon(edge_type, v_et1, v_et2, v_et3, v_et4,
                                                  v_off0, v_off1, v_off2, v_off3, v_off4, v_zero);
            int16x8_t v_result = vminq_s16(vmaxq_s16(vaddq_s16(v_center, v_off), v_zero), v_max);
            vst1q_u16(row_dst + x, vreinterpretq_u16_s16(v_result));
        }

        for (; x < w; x++) {
            int c, pd_t, pd_b, ts, bs, et, val;
            c = row_center[x]; pd_t = c - row_top[x]; pd_b = c - row_bottom[x];
            ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
            bs = (pd_b > 0) ? 1 : ((pd_b < 0) ? -1 : 0);
            et = ts + bs + 2;
            val = AVS2_CLIP3(0, max_pel, c + offset[et]);
            row_dst[x] = (uint16_t)val;
        }
    }
}

/* ===========================================================================
 * EO_135: 135 度对角方向 (左上-右下邻居), 10-bit NEON
 * 首/末行标量 (角落可用性), 中间行 SIMD.
 * =========================================================================== */
static void sao_eo_135_neon(uint8_t *_dst, int dst_stride, const uint8_t *_src,
                            int src_stride, int w, int h, int bit_depth,
                            const int *avail, const int *offset)
{
    if (bit_depth <= 8) {
        sao_eo_135_c(_dst, dst_stride, _src, src_stride, w, h, bit_depth, avail, offset);
        return;
    }
    uint16_t *dst = (uint16_t *)_dst;
    const uint16_t *src = (const uint16_t *)_src;
    const int max_pel = (1 << bit_depth) - 1;
    int sx = avail[sao_nb_l] ? 0 : 1;
    int ex = avail[sao_nb_r] ? w : (w - 1);
    int y;

    int16x8_t v_zero = vdupq_n_s16(0);
    int16x8_t v_max  = vdupq_n_s16((int16_t)max_pel);
    int16x8_t v_two  = vdupq_n_s16(2);
    int16x8_t v_et1  = vdupq_n_s16(1);
    int16x8_t v_et2  = vdupq_n_s16(2);
    int16x8_t v_et3  = vdupq_n_s16(3);
    int16x8_t v_et4  = vdupq_n_s16(4);
    int16x8_t v_off0 = vdupq_n_s16((int16_t)offset[0]);
    int16x8_t v_off1 = vdupq_n_s16((int16_t)offset[1]);
    int16x8_t v_off2 = vdupq_n_s16((int16_t)offset[2]);
    int16x8_t v_off3 = vdupq_n_s16((int16_t)offset[3]);
    int16x8_t v_off4 = vdupq_n_s16((int16_t)offset[4]);

    /* 首行: 标量处理 (角落 TL/DR 可用性不同) */
    if (avail[sao_nb_t]) {
        int sx0 = avail[sao_nb_tl] ? 0 : 1;
        int ex0 = ex;
        int x;
        y = 0;
        {
            const uint16_t *rc = src + y * src_stride;
            uint16_t *rd = dst + y * dst_stride;
            for (x = sx0; x < ex0; x++) {
                int c = rc[x], pd_t = c - rc[-src_stride + x - 1];
                int pd_d = c - rc[src_stride + x + 1];
                int ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                int ds = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                int et = ts + ds + 2;
                rd[x] = (uint16_t)AVS2_CLIP3(0, max_pel, c + offset[et]);
            }
        }
    }

    /* 中间行: SIMD 处理 */
    for (y = 1; y < h - 1; y++) {
        const uint16_t *rc = src + y * src_stride;
        const uint16_t *rtl = rc - src_stride - 1;
        const uint16_t *rdr = rc + src_stride + 1;
        uint16_t *rd = dst + y * dst_stride;
        int x = sx;
        for (; x + 8 <= ex; x += 8) {
            int16x8_t vc  = vreinterpretq_s16_u16(vld1q_u16(rc + x));
            int16x8_t vtl = vreinterpretq_s16_u16(vld1q_u16(rtl + x));
            int16x8_t vdr = vreinterpretq_s16_u16(vld1q_u16(rdr + x));
            int16x8_t ts = sao_sign_neon(vsubq_s16(vc, vtl));
            int16x8_t ds = sao_sign_neon(vsubq_s16(vc, vdr));
            int16x8_t et = vaddq_s16(vaddq_s16(ts, ds), v_two);
            int16x8_t vo = sao_off_select_neon(et, v_et1, v_et2, v_et3, v_et4,
                                               v_off0, v_off1, v_off2, v_off3, v_off4, v_zero);
            int16x8_t vr = vminq_s16(vmaxq_s16(vaddq_s16(vc, vo), v_zero), v_max);
            vst1q_u16(rd + x, vreinterpretq_u16_s16(vr));
        }
        for (; x < ex; x++) {
            int c = rc[x], pd_t = c - rtl[x], pd_d = c - rdr[x];
            int ts2 = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
            int ds2 = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
            int et2 = ts2 + ds2 + 2;
            rd[x] = (uint16_t)AVS2_CLIP3(0, max_pel, c + offset[et2]);
        }
    }

    /* 末行: 标量处理 */
    if (avail[sao_nb_d]) {
        int sxn = avail[sao_nb_l] ? 0 : 1;
        int exn = avail[sao_nb_dr] ? w : (w - 1);
        int x;
        y = h - 1;
        {
            const uint16_t *rc = src + y * src_stride;
            uint16_t *rd = dst + y * dst_stride;
            for (x = sxn; x < exn; x++) {
                int c = rc[x], pd_t = c - rc[-src_stride + x - 1];
                int pd_d = c - rc[src_stride + x + 1];
                int ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                int ds = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                int et = ts + ds + 2;
                rd[x] = (uint16_t)AVS2_CLIP3(0, max_pel, c + offset[et]);
            }
        }
    }
}

/* ===========================================================================
 * EO_45: 45 度对角方向 (右上-左下邻居), 10-bit NEON
 * =========================================================================== */
static void sao_eo_45_neon(uint8_t *_dst, int dst_stride, const uint8_t *_src,
                           int src_stride, int w, int h, int bit_depth,
                           const int *avail, const int *offset)
{
    if (bit_depth <= 8) {
        sao_eo_45_c(_dst, dst_stride, _src, src_stride, w, h, bit_depth, avail, offset);
        return;
    }
    uint16_t *dst = (uint16_t *)_dst;
    const uint16_t *src = (const uint16_t *)_src;
    const int max_pel = (1 << bit_depth) - 1;
    int sx = avail[sao_nb_l] ? 0 : 1;
    int ex = avail[sao_nb_r] ? w : (w - 1);
    int y;

    int16x8_t v_zero = vdupq_n_s16(0);
    int16x8_t v_max  = vdupq_n_s16((int16_t)max_pel);
    int16x8_t v_two  = vdupq_n_s16(2);
    int16x8_t v_et1  = vdupq_n_s16(1);
    int16x8_t v_et2  = vdupq_n_s16(2);
    int16x8_t v_et3  = vdupq_n_s16(3);
    int16x8_t v_et4  = vdupq_n_s16(4);
    int16x8_t v_off0 = vdupq_n_s16((int16_t)offset[0]);
    int16x8_t v_off1 = vdupq_n_s16((int16_t)offset[1]);
    int16x8_t v_off2 = vdupq_n_s16((int16_t)offset[2]);
    int16x8_t v_off3 = vdupq_n_s16((int16_t)offset[3]);
    int16x8_t v_off4 = vdupq_n_s16((int16_t)offset[4]);

    /* 首行: 标量处理 */
    if (avail[sao_nb_t]) {
        int sx0 = sx;
        int ex0 = avail[sao_nb_tr] ? w : (w - 1);
        int x;
        y = 0;
        {
            const uint16_t *rc = src + y * src_stride;
            uint16_t *rd = dst + y * dst_stride;
            for (x = sx0; x < ex0; x++) {
                int c = rc[x], pd_t = c - rc[-src_stride + x + 1];
                int pd_d = c - rc[src_stride + x - 1];
                int ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                int ds = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                int et = ts + ds + 2;
                rd[x] = (uint16_t)AVS2_CLIP3(0, max_pel, c + offset[et]);
            }
        }
    }

    /* 中间行: SIMD 处理 */
    for (y = 1; y < h - 1; y++) {
        const uint16_t *rc = src + y * src_stride;
        const uint16_t *rtr = rc - src_stride + 1;
        const uint16_t *rdl = rc + src_stride - 1;
        uint16_t *rd = dst + y * dst_stride;
        int x = sx;
        for (; x + 8 <= ex; x += 8) {
            int16x8_t vc  = vreinterpretq_s16_u16(vld1q_u16(rc + x));
            int16x8_t vtr = vreinterpretq_s16_u16(vld1q_u16(rtr + x));
            int16x8_t vdl = vreinterpretq_s16_u16(vld1q_u16(rdl + x));
            int16x8_t ts = sao_sign_neon(vsubq_s16(vc, vtr));
            int16x8_t ds = sao_sign_neon(vsubq_s16(vc, vdl));
            int16x8_t et = vaddq_s16(vaddq_s16(ts, ds), v_two);
            int16x8_t vo = sao_off_select_neon(et, v_et1, v_et2, v_et3, v_et4,
                                               v_off0, v_off1, v_off2, v_off3, v_off4, v_zero);
            int16x8_t vr = vminq_s16(vmaxq_s16(vaddq_s16(vc, vo), v_zero), v_max);
            vst1q_u16(rd + x, vreinterpretq_u16_s16(vr));
        }
        for (; x < ex; x++) {
            int c = rc[x], pd_t = c - rtr[x], pd_d = c - rdl[x];
            int ts2 = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
            int ds2 = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
            int et2 = ts2 + ds2 + 2;
            rd[x] = (uint16_t)AVS2_CLIP3(0, max_pel, c + offset[et2]);
        }
    }

    /* 末行: 标量处理 */
    if (avail[sao_nb_d]) {
        int sxn = avail[sao_nb_dl] ? 0 : 1;
        int exn = ex;
        int x;
        y = h - 1;
        {
            const uint16_t *rc = src + y * src_stride;
            uint16_t *rd = dst + y * dst_stride;
            for (x = sxn; x < exn; x++) {
                int c = rc[x], pd_t = c - rc[-src_stride + x + 1];
                int pd_d = c - rc[src_stride + x - 1];
                int ts = (pd_t > 0) ? 1 : ((pd_t < 0) ? -1 : 0);
                int ds = (pd_d > 0) ? 1 : ((pd_d < 0) ? -1 : 0);
                int et = ts + ds + 2;
                rd[x] = (uint16_t)AVS2_CLIP3(0, max_pel, c + offset[et]);
            }
        }
    }
}

/* ===========================================================================
 * BO: 带偏移 (32 带), 10-bit NEON
 * 与 SSE4.1 相同: 标量 gather (NEON 无 gather 指令)
 * =========================================================================== */
static void sao_bo_neon(uint8_t *_dst, int dst_stride, const uint8_t *_src,
                        int src_stride, int w, int h, int bit_depth,
                        const int *offset)
{
    if (bit_depth <= 8) {
        sao_bo_c(_dst, dst_stride, _src, src_stride, w, h, bit_depth, offset);
        return;
    }
    uint16_t *dst = (uint16_t *)_dst;
    const uint16_t *src = (const uint16_t *)_src;
    const int max_pel = (1 << bit_depth) - 1;
    const int band_shift = bit_depth - NUM_SAO_BO_CLASSES_IN_BIT;
    int y;

    int16x8_t v_zero = vdupq_n_s16(0);
    int16x8_t v_max  = vdupq_n_s16((int16_t)max_pel);

    for (y = 0; y < h; y++) {
        int x = 0;

        for (; x + 8 <= w; x += 8) {
            int16x8_t v_src = vreinterpretq_s16_u16(vld1q_u16(src + x));

            /* 计算带索引 (像素值非负, 算术右移 = 逻辑右移) */
            int16x8_t v_idx = vshrq_n_s16(v_src, band_shift);

            /* 标量 gather: 逐个提取索引查表, 组装为偏移向量 */
            int16x8_t v_off = vdupq_n_s16(0);
            v_off = vsetq_lane_s16((int16_t)offset[vgetq_lane_s16(v_idx, 0)], v_off, 0);
            v_off = vsetq_lane_s16((int16_t)offset[vgetq_lane_s16(v_idx, 1)], v_off, 1);
            v_off = vsetq_lane_s16((int16_t)offset[vgetq_lane_s16(v_idx, 2)], v_off, 2);
            v_off = vsetq_lane_s16((int16_t)offset[vgetq_lane_s16(v_idx, 3)], v_off, 3);
            v_off = vsetq_lane_s16((int16_t)offset[vgetq_lane_s16(v_idx, 4)], v_off, 4);
            v_off = vsetq_lane_s16((int16_t)offset[vgetq_lane_s16(v_idx, 5)], v_off, 5);
            v_off = vsetq_lane_s16((int16_t)offset[vgetq_lane_s16(v_idx, 6)], v_off, 6);
            v_off = vsetq_lane_s16((int16_t)offset[vgetq_lane_s16(v_idx, 7)], v_off, 7);

            /* 加偏移并裁剪 */
            int16x8_t v_result = vminq_s16(vmaxq_s16(vaddq_s16(v_src, v_off), v_zero), v_max);
            vst1q_u16(dst + x, vreinterpretq_u16_s16(v_result));
        }

        for (; x < w; x++) {
            int c, edge_type, val;
            c = src[x];
            edge_type = c >> band_shift;
            val = AVS2_CLIP3(0, max_pel, c + offset[edge_type]);
            dst[x] = (uint16_t)val;
        }

        src += src_stride;
        dst += dst_stride;
    }
}

/* ===========================================================================
 * 注册函数
 * =========================================================================== */

/* NEON: 注册全部 SAO 函数 (EO_0, EO_90, EO_135, EO_45, BO; 10-bit NEON, 8-bit C) */
void avs2_sao_init_neon(const avs2_cpu_flags *flags)
{
    UNUSED_PARAMETER(flags);
    avs2_dsp_table.sao_eo[0] = sao_eo_0_neon;
    avs2_dsp_table.sao_eo[1] = sao_eo_90_neon;
    avs2_dsp_table.sao_eo[2] = sao_eo_135_neon;
    avs2_dsp_table.sao_eo[3] = sao_eo_45_neon;
    avs2_dsp_table.sao_bo    = sao_bo_neon;
}

/* SSE/AVX 接口空实现以供链接 (非 x86 平台不调用) */
void avs2_sao_init_sse41(const avs2_cpu_flags *flags)
{
    UNUSED_PARAMETER(flags);
}

void avs2_sao_init_avx2(const avs2_cpu_flags *flags)
{
    UNUSED_PARAMETER(flags);
}

#else /* 其它平台: 回退到 C */

/* NEON 路径回退到 C, 保留空实现以供链接 */
void avs2_sao_init_sse41(const avs2_cpu_flags *flags)
{
    UNUSED_PARAMETER(flags);
}

void avs2_sao_init_avx2(const avs2_cpu_flags *flags)
{
    UNUSED_PARAMETER(flags);
}

void avs2_sao_init_neon(const avs2_cpu_flags *flags)
{
    UNUSED_PARAMETER(flags);
}

#endif
