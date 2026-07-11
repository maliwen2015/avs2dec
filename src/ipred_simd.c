/*
 * ipred_simd.c - 帧内预测 SIMD 实现 (x86)
 *
 * 当前实现:
 *   - AVX2: 垂直 / 水平 / DC / Plane 预测 (256-bit 路径, 参考 libudavs2 xPredIntra*Adi_sse256_10bit)
 *     8-bit 与 10-bit 共用同一路径 (内部均按 pel_t/uint16_t 处理;
 *     8-bit 由 avs2_get_intra_pred 分配 uint16_t 临时缓冲, 再打包回 uint8_t)
 *   - SSE4.1: 角度预测 - 水平方向 (模式 3..11) 4 抽头滤波
 *   - 小块 (bsx < 8 的 Plane) 回退到 C 实现 (ipred.c)
 *
 * 函数签名 (avs2_intra_pred_fn):
 *   void (*)(uint8_t *src, uint8_t *dst, int i_dst,
 *            int mode, int bsx, int bsy, int bit_depth)
 *   - src: EP 参考样本数组 (uint16_t 内部表示), 左侧用负索引, 上方用正索引
 *   - dst: 预测输出, 步长 i_dst (10-bit 为 uint16 元素步长, 8-bit 为字节步长)
 *   - bsx/bsy: 块宽高 (4..64)
 *   - mode: 预测模式 (DC 时高 8 位=b_top, 低 8 位=b_left)
 *
 * 对齐说明:
 *   - EP 参考样本数组 (ipred.c 中 ep_buf) 已声明 32 字节对齐,
 *     EP+1 (上方参考起始) 为 32 字节对齐, ver 路径可用对齐加载
 *   - dst 为帧数据, 位于 4 像素边界:
 *       10-bit 即 8 字节对齐, 8-bit 即 4 字节对齐, 均不保证 32 字节对齐
 *   - stride 为 32 字节倍数 (帧步长 64/128B, pred_buf 步长 w*2>=32B),
 *     每行 dst 对齐相同, 头部偏移只需计算一次
 *   - bsx >= 16: 头部未对齐用 128/64-bit storeu, 主体用 256-bit 对齐 store,
 *     尾部用 128/64-bit storeu
 *   - bsx == 8/4: 数据不保证 128-bit 对齐, 保持 loadu/storeu
 *   - 参考 libudavs2, 按块宽度选择 256-bit / 128-bit / 64-bit 路径
 */

#include "internal.h"
#include "tables.h"

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(x) (void)(x)
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <immintrin.h>

/* 10-bit 像素类型 (与 ipred.c 一致) */
typedef uint16_t pel_t;

/* C 回退函数声明 (在 ipred.c 中定义, 供 8-bit 或不支持尺寸回退) */
extern void intra_pred_ver_c(uint8_t *src, uint8_t *dst, int i_dst, int mode,
                             int bsx, int bsy, int bit_depth);
extern void intra_pred_hor_c(uint8_t *src, uint8_t *dst, int i_dst, int mode,
                             int bsx, int bsy, int bit_depth);
extern void intra_pred_dc_c(uint8_t *src, uint8_t *dst, int i_dst, int mode,
                            int bsx, int bsy, int bit_depth);
extern void intra_pred_plane_c(uint8_t *src, uint8_t *dst, int i_dst, int mode,
                               int bsx, int bsy, int bit_depth);
/* C 回退: 角度预测垂直/对角方向及双线性 (供 SSE4.1 8-bit 回退) */
extern void intra_pred_ang_y_c(uint8_t *src, uint8_t *dst, int i_dst, int dir_mode,
                                int bsx, int bsy, int bit_depth);
extern void intra_pred_ang_xy_c(uint8_t *src, uint8_t *dst, int i_dst, int dir_mode,
                                 int bsx, int bsy, int bit_depth);
extern void intra_pred_bilinear_c(uint8_t *src, uint8_t *dst, int i_dst, int mode,
                                   int bsx, int bsy, int bit_depth);

/* ===========================================================================
 * 垂直预测: 复制上方参考行 (src + 1) 到 dst
 * =========================================================================== */
static void intra_pred_ver_avx2(uint8_t *src_u8, uint8_t *dst_u8, int i_dst, int mode,
                                int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t *p_src = src + 1;
    int y;

    UNUSED_PARAMETER(mode);

    if (bsx >= 16) {
        /* stride 为 32 字节倍数, 每行 dst 对齐相同 */
        int head = (int)((-(uintptr_t)dst) & 31) >> 1;
        for (y = 0; y < bsy; y++) {
            int x = 0;
            /* 头部未对齐部分 (0, 4, 8 或 12 元素) */
            for (; x + 8 <= head; x += 8) {
                __m128i v = _mm_loadu_si128((const __m128i*)(p_src + x));
                _mm_storeu_si128((__m128i*)(dst + x), v);
            }
            for (; x + 4 <= head; x += 4) {
                __m128i v = _mm_loadl_epi64((const __m128i*)(p_src + x));
                _mm_storel_epi64((__m128i*)(dst + x), v);
            }
            /* 对齐主体: 256-bit 对齐存储 (src 偏移后可能未对齐, 用 loadu) */
            for (; x + 16 <= bsx; x += 16) {
                __m256i v = _mm256_loadu_si256((const __m256i*)(p_src + x));
                _mm256_store_si256((__m256i*)(dst + x), v);
            }
            /* 尾部未对齐部分 */
            for (; x + 8 <= bsx; x += 8) {
                __m128i v = _mm_loadu_si128((const __m128i*)(p_src + x));
                _mm_storeu_si128((__m128i*)(dst + x), v);
            }
            for (; x + 4 <= bsx; x += 4) {
                __m128i v = _mm_loadl_epi64((const __m128i*)(p_src + x));
                _mm_storel_epi64((__m128i*)(dst + x), v);
            }
            dst += i_dst;
        }
    } else if (bsx == 8) {
        __m128i v = _mm_loadu_si128((const __m128i*)p_src);
        for (y = 0; y < bsy; y++) {
            _mm_storeu_si128((__m128i*)dst, v);
            dst += i_dst;
        }
    } else {
        /* bsx == 4 */
        __m128i v = _mm_loadl_epi64((const __m128i*)p_src);
        for (y = 0; y < bsy; y++) {
            _mm_storel_epi64((__m128i*)dst, v);
            dst += i_dst;
        }
    }
}

/* ===========================================================================
 * 水平预测: 复制左侧参考列 (src - 1, 按 src[-1-y] 取值) 广播到每行
 * =========================================================================== */
static void intra_pred_hor_avx2(uint8_t *src_u8, uint8_t *dst_u8, int i_dst, int mode,
                                int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t *p_src = src - 1;
    int y;

    UNUSED_PARAMETER(mode);

    if (bsx >= 16) {
        int head = (int)((-(uintptr_t)dst) & 31) >> 1;
        for (y = 0; y < bsy; y++) {
            __m256i v = _mm256_set1_epi16((short)p_src[-y]);
            int x = 0;
            /* 头部未对齐部分 */
            for (; x + 8 <= head; x += 8) {
                __m128i v128 = _mm256_castsi256_si128(v);
                _mm_storeu_si128((__m128i*)(dst + x), v128);
            }
            for (; x + 4 <= head; x += 4) {
                __m128i v128 = _mm256_castsi256_si128(v);
                _mm_storel_epi64((__m128i*)(dst + x), v128);
            }
            /* 对齐主体: 256-bit 对齐存储 */
            for (; x + 16 <= bsx; x += 16) {
                _mm256_store_si256((__m256i*)(dst + x), v);
            }
            /* 尾部未对齐部分 */
            for (; x + 8 <= bsx; x += 8) {
                __m128i v128 = _mm256_castsi256_si128(v);
                _mm_storeu_si128((__m128i*)(dst + x), v128);
            }
            for (; x + 4 <= bsx; x += 4) {
                __m128i v128 = _mm256_castsi256_si128(v);
                _mm_storel_epi64((__m128i*)(dst + x), v128);
            }
            dst += i_dst;
        }
    } else if (bsx == 8) {
        for (y = 0; y < bsy; y++) {
            __m128i v = _mm_set1_epi16((short)p_src[-y]);
            _mm_storeu_si128((__m128i*)dst, v);
            dst += i_dst;
        }
    } else {
        /* bsx == 4 */
        for (y = 0; y < bsy; y++) {
            __m128i v = _mm_set1_epi16((short)p_src[-y]);
            _mm_storel_epi64((__m128i*)dst, v);
            dst += i_dst;
        }
    }
}

/* ===========================================================================
 * DC 预测: 取左侧和上方参考样本均值, 广播填充
 * mode 高 8 位 = b_top, 低 8 位 = b_left (与 intra_pred_dc_c 一致)
 * =========================================================================== */
static void intra_pred_dc_avx2(uint8_t *src_u8, uint8_t *dst_u8, int i_dst, int mode,
                               int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int b_top  = mode >> 8;
    int b_left = mode & 0xFF;
    pel_t *p_src = src - 1;
    int dc_value = 0;
    int max_val = (1 << bit_depth) - 1;
    int x, y;

    /* 计算 DC 值 (与 C 版完全一致) */
    if (b_left) {
        for (y = 0; y < bsy; y++) {
            dc_value += p_src[-y];
        }
        p_src = src + 1;
        if (b_top) {
            for (x = 0; x < bsx; x++) {
                dc_value += p_src[x];
            }
            dc_value += ((bsx + bsy) >> 1);
            dc_value = (dc_value * (512 / (bsx + bsy))) >> 9;
        } else {
            dc_value += bsy / 2;
            dc_value /= bsy;
        }
    } else {
        p_src = src + 1;
        if (b_top) {
            for (x = 0; x < bsx; x++) {
                dc_value += p_src[x];
            }
            dc_value += bsx / 2;
            dc_value /= bsx;
        } else {
            dc_value = 1 << (bit_depth - 1);
        }
    }

    /* 裁剪并广播填充 */
    if (dc_value < 0) dc_value = 0;
    if (dc_value > max_val) dc_value = max_val;

    if (bsx >= 16) {
        __m256i v = _mm256_set1_epi16((short)dc_value);
        int head = (int)((-(uintptr_t)dst) & 31) >> 1;
        for (y = 0; y < bsy; y++) {
            int x = 0;
            /* 头部未对齐部分 */
            for (; x + 8 <= head; x += 8) {
                __m128i v128 = _mm256_castsi256_si128(v);
                _mm_storeu_si128((__m128i*)(dst + x), v128);
            }
            for (; x + 4 <= head; x += 4) {
                __m128i v128 = _mm256_castsi256_si128(v);
                _mm_storel_epi64((__m128i*)(dst + x), v128);
            }
            /* 对齐主体: 256-bit 对齐存储 */
            for (; x + 16 <= bsx; x += 16) {
                _mm256_store_si256((__m256i*)(dst + x), v);
            }
            /* 尾部未对齐部分 */
            for (; x + 8 <= bsx; x += 8) {
                __m128i v128 = _mm256_castsi256_si128(v);
                _mm_storeu_si128((__m128i*)(dst + x), v128);
            }
            for (; x + 4 <= bsx; x += 4) {
                __m128i v128 = _mm256_castsi256_si128(v);
                _mm_storel_epi64((__m128i*)(dst + x), v128);
            }
            dst += i_dst;
        }
    } else if (bsx == 8) {
        __m128i v = _mm_set1_epi16((short)dc_value);
        for (y = 0; y < bsy; y++) {
            _mm_storeu_si128((__m128i*)dst, v);
            dst += i_dst;
        }
    } else {
        /* bsx == 4 */
        __m128i v = _mm_set1_epi16((short)dc_value);
        for (y = 0; y < bsy; y++) {
            _mm_storel_epi64((__m128i*)dst, v);
            dst += i_dst;
        }
    }
}

/* ===========================================================================
 * Plane 预测 AVX2: 内层循环用 8x int32 累加 + 限幅 + 打包
 * =========================================================================== */
static inline int intra_log2_sz(int sz)
{
    return (sz == 4) ? 2 : (sz == 8) ? 3 : (sz == 16) ? 4 : (sz == 32) ? 5 : 6;
}

static inline int intra_clip3(int low, int high, int v)
{
    return v < low ? low : (v > high ? high : v);
}

static void intra_pred_plane_avx2(uint8_t *src_u8, uint8_t *dst_u8, int i_dst, int mode,
                                  int bsx, int bsy, int bit_depth)
{
    static const int ib_mult[5]  = { 13, 17,  5, 11, 23 };
    static const int ib_shift[5] = {  7, 10, 11, 15, 19 };
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int log2_bsx = intra_log2_sz(bsx);
    int log2_bsy = intra_log2_sz(bsy);
    int im_h = ib_mult[log2_bsx - 2];
    int im_v = ib_mult[log2_bsy - 2];
    int is_h = ib_shift[log2_bsx - 2];
    int is_v = ib_shift[log2_bsy - 2];
    int iW2 = bsx >> 1;
    int iH2 = bsy >> 1;
    int iH = 0, iV = 0;
    int iA, iB, iC;
    int x, y;
    int iTmp, iTmp2;
    int max_val = (1 << bit_depth) - 1;
    pel_t *p_src;

    UNUSED_PARAMETER(mode);

    /* 计算水平梯度 iH */
    p_src = src + 1 + (iW2 - 1);
    for (x = 1; x < iW2 + 1; x++) {
        iH += x * (p_src[x] - p_src[-x]);
    }
    /* 计算垂直梯度 iV */
    p_src = src - 1 - (iH2 - 1);
    for (y = 1; y < iH2 + 1; y++) {
        iV += y * (p_src[-y] - p_src[y]);
    }
    /* 计算平面参数 */
    p_src = src;
    iA = (p_src[-1 - (bsy - 1)] + p_src[1 + bsx - 1]) << 4;
    iB = ((iH << 5) * im_h + (1 << (is_h - 1))) >> is_h;
    iC = ((iV << 5) * im_v + (1 << (is_v - 1))) >> is_v;

    iTmp = iA - (iH2 - 1) * iC - (iW2 - 1) * iB + 16;

    if (bsx >= 8) {
        __m256i v_max = _mm256_set1_epi32(max_val);
        __m256i v_zero = _mm256_setzero_si256();
        /* bsx >= 16: 128-bit 对齐存储; bsx == 8: 不保证对齐, 保持 storeu */
        int use_aligned = (bsx >= 16);
        int head = use_aligned ? (int)((-(uintptr_t)dst) & 15) >> 1 : 0;
        for (y = 0; y < bsy; y++) {
            iTmp2 = iTmp;
            x = 0;
            /* 头部未对齐: 标量处理 */
            for (; x < head; x++) {
                dst[x] = (pel_t)intra_clip3(0, max_val, iTmp2 >> 5);
                iTmp2 += iB;
            }
            /* 初始化 8 通道: iTmp2, iTmp2+iB, ..., iTmp2+7*iB */
            __m256i v_val = _mm256_setr_epi32(
                iTmp2,        iTmp2 + iB,  iTmp2 + 2*iB, iTmp2 + 3*iB,
                iTmp2 + 4*iB, iTmp2 + 5*iB, iTmp2 + 6*iB, iTmp2 + 7*iB);
            __m256i v_step8 = _mm256_set1_epi32(iB * 8);

            /* 主体: 8 像素一组 */
            for (; x + 7 < bsx; x += 8) {
                __m256i v_s = _mm256_srai_epi32(v_val, 5);
                v_s = _mm256_max_epi32(v_s, v_zero);
                v_s = _mm256_min_epi32(v_s, v_max);
                /* packus: 8x int32 -> 8x uint16, permute 消除 lane 交错 */
                __m256i v_pk = _mm256_packus_epi32(v_s, v_s);
                v_pk = _mm256_permute4x64_epi64(v_pk, 0xD8);
                if (use_aligned) {
                    _mm_store_si128((__m128i *)(dst + x), _mm256_castsi256_si128(v_pk));
                } else {
                    _mm_storeu_si128((__m128i *)(dst + x), _mm256_castsi256_si128(v_pk));
                }
                v_val = _mm256_add_epi32(v_val, v_step8);
            }
            /* 尾部标量 */
            iTmp2 = iTmp + x * iB;
            for (; x < bsx; x++) {
                dst[x] = (pel_t)intra_clip3(0, max_val, iTmp2 >> 5);
                iTmp2 += iB;
            }
            dst += i_dst;
            iTmp += iC;
        }
    } else {
        /* 小块: 回退到 C */
        intra_pred_plane_c(src_u8, dst_u8, i_dst, mode, bsx, bsy, bit_depth);
    }
}

/* ===========================================================================
 * 角度预测 - 水平方向 SSE4.1 (模式 3..11)
 * 通用 4 抽头滤波: dst[i] = (src[iX]*c1 + src[iX+1]*c2 + src[iX+2]*c3 + src[iX+3]*c4 + 64) >> 7
 * 每行内系数恒定, 源偏移递增 1, 可向量化处理 8 像素
 * =========================================================================== */

/* get_context_pixel 辅助: 计算 iX 和 offset (c4)
 * 原始公式: iTempDn = iTempD * imult >> ishift; offset = ((iTempD * imult * 32) >> ishift) - iTempDn * 32
 * 优化: 提取公共子表达式 tmp = iTempD * imult, 避免重复乘法 */
static int get_context_pixel(int mode, int i_temp_d, int *offset)
{
    extern const int8_t tab_auc_dir_dxdy[2][NUM_INTRA_MODE][2];
    int imult = tab_auc_dir_dxdy[0][mode][0];
    int ishift = tab_auc_dir_dxdy[0][mode][1];
    int tmp = i_temp_d * imult;
    int i_temp_dn = tmp >> ishift;
    *offset = ((tmp * 32) >> ishift) - (i_temp_dn * 32);
    return i_temp_dn;
}

/* get_context_pixel_y: 垂直方向 (dy/dx), 使用 tab_auc_dir_dxdy[1] */
static int get_context_pixel_y(int mode, int i_temp_d, int *offset)
{
    extern const int8_t tab_auc_dir_dxdy[2][NUM_INTRA_MODE][2];
    int imult = tab_auc_dir_dxdy[1][mode][0];
    int ishift = tab_auc_dir_dxdy[1][mode][1];
    int tmp = i_temp_d * imult;
    int i_temp_dn = tmp >> ishift;
    *offset = ((tmp * 32) >> ishift) - (i_temp_dn * 32);
    return i_temp_dn;
}

static void intra_pred_ang_x_sse41(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                   int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    /* 预计算所有行的 iX 和 c4, 避免循环内重复查表 */
    int idx_tab[64];
    int c4_tab[64];
    int j;

    /* 一次性预计算所有行的 iX 和 c4 */
    for (j = 0; j < bsy; j++) {
        idx_tab[j] = get_context_pixel(dir_mode, j + 1, &c4_tab[j]);
    }

    for (j = 0; j < bsy; j++) {
        int c4 = c4_tab[j];
        int iX = idx_tab[j];
        int c1 = 32 - c4;
        int c2 = 64 - c4;
        int c3 = 32 + c4;

        {
            __m128i v_c01 = _mm_set_epi16(c2, c1, c2, c1, c2, c1, c2, c1);
            __m128i v_c23 = _mm_set_epi16(c4, c3, c4, c3, c4, c3, c4, c3);
            __m128i v_off = _mm_set1_epi32(64);
            int i = 0;

            for (; i + 8 <= bsx; i += 8, iX += 8) {
                __m128i a, b, s0, s1, s2, s3;
                __m128i t0, t1, t2, t3, r0, r1, r2, r3;
                __m128i res_lo, res_hi;

                a = _mm_loadu_si128((const __m128i *)(src + iX));
                b = _mm_loadu_si128((const __m128i *)(src + iX + 8));
                s0 = a;
                s1 = _mm_alignr_epi8(b, a, 2);
                s2 = _mm_alignr_epi8(b, a, 4);
                s3 = _mm_alignr_epi8(b, a, 6);

                t0 = _mm_unpacklo_epi16(s0, s1);
                t1 = _mm_unpackhi_epi16(s0, s1);
                t2 = _mm_unpacklo_epi16(s2, s3);
                t3 = _mm_unpackhi_epi16(s2, s3);

                r0 = _mm_madd_epi16(t0, v_c01);
                r1 = _mm_madd_epi16(t1, v_c01);
                r2 = _mm_madd_epi16(t2, v_c23);
                r3 = _mm_madd_epi16(t3, v_c23);

                res_lo = _mm_add_epi32(_mm_add_epi32(r0, r2), v_off);
                res_hi = _mm_add_epi32(_mm_add_epi32(r1, r3), v_off);
                res_lo = _mm_srli_epi32(res_lo, 7);
                res_hi = _mm_srli_epi32(res_hi, 7);

                {
                    __m128i packed = _mm_packus_epi32(res_lo, res_hi);
                    _mm_storeu_si128((__m128i *)(dst + i), packed);
                }
            }

            for (; i < bsx; i++, iX++) {
                dst[i] = (pel_t)((src[iX] * c1 + src[iX + 1] * c2 +
                                  src[iX + 2] * c3 + src[iX + 3] * c4 + 64) >> 7);
            }
        }

        dst += i_dst;
    }
}

/* ===========================================================================
 * 角度预测 - 垂直方向 SSE4.1 (模式 25..32)
 * 4 抽头滤波: dst[i] = (src[idx]*(32-off) + src[idx-1]*(64-off) +
 *                       src[idx-2]*(32+off) + src[idx-3]*off + 64) >> 7
 * 每列的源位置 idx 和系数 offset 随列变化 (gather 模式), 标量收集后向量化计算
 * =========================================================================== */
static void intra_pred_ang_y_sse41(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                   int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int xsteps[64];
    int offsets[64];
    int i, j;

    if (bit_depth <= 8) {
        intra_pred_ang_y_c(src_u8, dst_u8, i_dst, dir_mode, bsx, bsy, bit_depth);
        return;
    }

    /* 预计算每列的 xsteps 和 offsets (dy/dx 查表, uiXYflag=1) */
    for (i = 0; i < bsx; i++) {
        xsteps[i] = get_context_pixel_y(dir_mode, i + 1, &offsets[i]);
    }

    for (j = 0; j < bsy; j++) {
        int i2 = 0;
        for (; i2 + 8 <= bsx; i2 += 8) {
            /* 标量收集 8 列的非连续源数据 (gather 模式) */
            pel_t s0[8], s1[8], s2[8], s3[8];
            int16_t c0[8], c1[8], c2[8], c3[8];
            __m128i v_s0, v_s1, v_s2, v_s3;
            __m128i v_c0, v_c1, v_c2, v_c3;
            __m128i t0, t1, t2, t3, r0, r1, r2, r3;
            __m128i res_lo, res_hi, res;
            __m128i v_off = _mm_set1_epi32(64);
            int k;

            for (k = 0; k < 8; k++) {
                int col = i2 + k;
                int iY = j + xsteps[col];
                int idx = -iY;
                int off = offsets[col];
                s0[k] = src[idx];
                s1[k] = src[idx - 1];
                s2[k] = src[idx - 2];
                s3[k] = src[idx - 3];
                c0[k] = (int16_t)(32 - off);
                c1[k] = (int16_t)(64 - off);
                c2[k] = (int16_t)(32 + off);
                c3[k] = (int16_t)off;
            }

            v_s0 = _mm_loadu_si128((const __m128i *)s0);
            v_s1 = _mm_loadu_si128((const __m128i *)s1);
            v_s2 = _mm_loadu_si128((const __m128i *)s2);
            v_s3 = _mm_loadu_si128((const __m128i *)s3);
            v_c0 = _mm_loadu_si128((const __m128i *)c0);
            v_c1 = _mm_loadu_si128((const __m128i *)c1);
            v_c2 = _mm_loadu_si128((const __m128i *)c2);
            v_c3 = _mm_loadu_si128((const __m128i *)c3);

            /* 4 抽头滤波: s0*c0 + s1*c1 + s2*c2 + s3*c3 + 64, >> 7 */
            t0 = _mm_unpacklo_epi16(v_s0, v_s1);
            t1 = _mm_unpacklo_epi16(v_c0, v_c1);
            t2 = _mm_unpackhi_epi16(v_s0, v_s1);
            t3 = _mm_unpackhi_epi16(v_c0, v_c1);
            r0 = _mm_madd_epi16(t0, t1);
            r1 = _mm_madd_epi16(t2, t3);

            t0 = _mm_unpacklo_epi16(v_s2, v_s3);
            t1 = _mm_unpacklo_epi16(v_c2, v_c3);
            t2 = _mm_unpackhi_epi16(v_s2, v_s3);
            t3 = _mm_unpackhi_epi16(v_c2, v_c3);
            r2 = _mm_madd_epi16(t0, t1);
            r3 = _mm_madd_epi16(t2, t3);

            res_lo = _mm_add_epi32(_mm_add_epi32(r0, r2), v_off);
            res_hi = _mm_add_epi32(_mm_add_epi32(r1, r3), v_off);
            res_lo = _mm_srli_epi32(res_lo, 7);
            res_hi = _mm_srli_epi32(res_hi, 7);
            res = _mm_packus_epi32(res_lo, res_hi);
            _mm_storeu_si128((__m128i *)(dst + i2), res);
        }

        /* 标量尾部 */
        for (; i2 < bsx; i2++) {
            int iY = j + xsteps[i2];
            int idx = -iY;
            int off = offsets[i2];
            dst[i2] = (pel_t)((src[idx] * (32 - off) + src[idx - 1] * (64 - off) +
                               src[idx - 2] * (32 + off) + src[idx - 3] * off + 64) >> 7);
        }
        dst += i_dst;
    }
}

/* ===========================================================================
 * 角度预测 - 对角方向 (模式 13..23)
 * XY 方向存在分支 (iYy <= -1), SIMD 向量化收益有限, 回退到 C 实现
 * =========================================================================== */
static void intra_pred_ang_xy_sse41(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                    int dir_mode, int bsx, int bsy, int bit_depth)
{
    intra_pred_ang_xy_c(src_u8, dst_u8, i_dst, dir_mode, bsx, bsy, bit_depth);
}

/* ===========================================================================
 * Bilinear 预测 SSE4.1 (模式 2)
 * 内层循环每像素: result = ((predx<<iy + p_top<<ix + wxy + offset) >> ixy) clip
 * predx = p_left[y] + (x+1)*p_l[y], wxy = x*wy[y], p_top[x] 累加 p_t[x]
 * 4 列一组用 int32 SIMD 处理 (中间值超过 int16 范围)
 * =========================================================================== */
static void intra_pred_bilinear_sse41(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                      int mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int32_t p_top[MAX_CU_SIZE];
    int32_t p_left[MAX_CU_SIZE];
    int32_t p_t[MAX_CU_SIZE];
    int32_t p_l[MAX_CU_SIZE];
    int32_t wy_tab[MAX_CU_SIZE];
    int ishift_x  = intra_log2_sz(bsx);
    int ishift_y  = intra_log2_sz(bsy);
    int ishift    = (ishift_x < ishift_y) ? ishift_x : ishift_y;
    int ishift_xy = ishift_x + ishift_y + 1;
    int offset    = 1 << (ishift_x + ishift_y);
    int a, b, c, w, t;
    int x, y;
    int max_value = (1 << bit_depth) - 1;

    UNUSED_PARAMETER(mode);

    if (bit_depth <= 8) {
        intra_pred_bilinear_c(src_u8, dst_u8, i_dst, mode, bsx, bsy, bit_depth);
        return;
    }

    /* 初始化 p_top 和 p_left */
    for (x = 0; x < bsx; x++) {
        p_top[x] = src[1 + x];
    }
    for (y = 0; y < bsy; y++) {
        p_left[y] = src[-1 - y];
    }

    a = p_top[bsx - 1];
    b = p_left[bsy - 1];
    c = (bsx == bsy) ? (a + b + 1) >> 1
                     : (((a << ishift_x) + (b << ishift_y)) * 13 + (1 << (ishift + 5))) >> (ishift + 6);
    w = (c << 1) - a - b;

    for (x = 0; x < bsx; x++) {
        p_t[x] = (int32_t)(b - p_top[x]);
        p_top[x] <<= ishift_y;
    }
    t = 0;
    for (y = 0; y < bsy; y++) {
        p_l[y] = (int32_t)(a - p_left[y]);
        p_left[y] <<= ishift_x;
        wy_tab[y] = (int32_t)t;
        t += w;
    }

    /* 主循环: 每行用 SSE4.1 处理 4 列 (int32) */
    {
        __m128i v_zero = _mm_setzero_si128();
        __m128i v_max = _mm_set1_epi32(max_value);
        __m128i v_offset = _mm_set1_epi32(offset);
        __m128i v_base = _mm_setr_epi32(0, 1, 2, 3);

        for (y = 0; y < bsy; y++) {
            int predx_base = p_left[y];
            int pl = p_l[y];
            int wy_y = wy_tab[y];
            int x2 = 0;

            for (; x2 + 4 <= bsx; x2 += 4) {
                __m128i v_top = _mm_loadu_si128((const __m128i *)(p_top + x2));
                __m128i v_pt  = _mm_loadu_si128((const __m128i *)(p_t + x2));
                __m128i v_col = _mm_add_epi32(v_base, _mm_set1_epi32(x2));
                __m128i v_xp1 = _mm_add_epi32(v_col, _mm_set1_epi32(1));
                __m128i v_predx, v_wxy, v_res, v_pk;
                __m128i v_top_new;

                /* predx = p_left[y] + (col+1) * p_l[y] */
                v_predx = _mm_add_epi32(_mm_set1_epi32(predx_base),
                                        _mm_mullo_epi32(v_xp1, _mm_set1_epi32(pl)));
                /* wxy = col * wy[y] */
                v_wxy = _mm_mullo_epi32(v_col, _mm_set1_epi32(wy_y));

                /* result = ((predx<<iy) + (p_top<<ix) + wxy + offset) >> ixy */
                v_res = _mm_add_epi32(_mm_slli_epi32(v_predx, ishift_y),
                                      _mm_slli_epi32(v_top, ishift_x));
                v_res = _mm_add_epi32(v_res, v_wxy);
                v_res = _mm_add_epi32(v_res, v_offset);
                v_res = _mm_srai_epi32(v_res, ishift_xy);
                v_res = _mm_max_epi32(v_res, v_zero);
                v_res = _mm_min_epi32(v_res, v_max);

                /* 打包 4x int32 -> 4x uint16 并存储 */
                v_pk = _mm_packus_epi32(v_res, v_res);
                _mm_storel_epi64((__m128i *)(dst + x2), v_pk);

                /* 更新 p_top: p_top[x] += p_t[x] */
                v_top_new = _mm_add_epi32(v_top, v_pt);
                _mm_storeu_si128((__m128i *)(p_top + x2), v_top_new);
            }

            /* 标量尾部: 从 SIMD 停止处继续累加 */
            {
                int predx = predx_base + x2 * pl;
                int wxy = wy_y * (x2 - 1);
                for (; x2 < bsx; x2++) {
                    predx += pl;
                    wxy += wy_y;
                    p_top[x2] += p_t[x2];
                    dst[x2] = (pel_t)intra_clip3(0, max_value,
                            (((predx << ishift_y) + (p_top[x2] << ishift_x) + wxy + offset) >> ishift_xy));
                }
            }
            dst += i_dst;
        }
    }
}

/* ===========================================================================
 * 注册函数
 * =========================================================================== */

/* SSE4.1: 角度预测 - 水平方向 (模式 3..11) 4 抽头滤波, 8/10-bit 共用 */
void avs2_ipred_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }

/* AVX2: 注册垂直/水平/DC/Plane 预测 (8/10-bit 共用 256-bit 路径) */
void avs2_ipred_init_avx2(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.ipred_mode[DC_PRED]    = intra_pred_dc_avx2;
    avs2_dsp_table.ipred_mode[VERT_PRED]  = intra_pred_ver_avx2;
    avs2_dsp_table.ipred_mode[HOR_PRED]   = intra_pred_hor_avx2;
    avs2_dsp_table.ipred_mode[PLANE_PRED] = intra_pred_plane_avx2;

    /* 角度预测 - 水平方向 (模式 3..11): SSE4.1 4 抽头滤波 */
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_3]  = intra_pred_ang_x_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_4]  = intra_pred_ang_x_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_5]  = intra_pred_ang_x_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_6]  = intra_pred_ang_x_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_7]  = intra_pred_ang_x_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_8]  = intra_pred_ang_x_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_9]  = intra_pred_ang_x_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_10] = intra_pred_ang_x_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_11] = intra_pred_ang_x_sse41;

    /* 角度预测 - 对角方向 (模式 15/17/19/21): 回退到 C (分支复杂, SIMD 收益有限) */
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_15] = intra_pred_ang_xy_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_17] = intra_pred_ang_xy_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_19] = intra_pred_ang_xy_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_21] = intra_pred_ang_xy_sse41;

    /* 角度预测 - 垂直方向 (模式 25..32): SSE4.1 gather 4 抽头滤波 */
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_25] = intra_pred_ang_y_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_26] = intra_pred_ang_y_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_27] = intra_pred_ang_y_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_28] = intra_pred_ang_y_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_29] = intra_pred_ang_y_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_30] = intra_pred_ang_y_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_31] = intra_pred_ang_y_sse41;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_32] = intra_pred_ang_y_sse41;

    /* Bilinear 预测 (模式 2): SSE4.1 int32 向量化 */
    avs2_dsp_table.ipred_mode[BI_PRED] = intra_pred_bilinear_sse41;
}

/* AVX512 预留 */
void avs2_ipred_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }

#else  /* 非 x86 平台 */

void avs2_ipred_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_ipred_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_ipred_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }

#endif
