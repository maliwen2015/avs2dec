/*
 * lf_simd.c - 去块滤波 SIMD 实现 (x86)
 *
 * 当前实现 (SSE4.1, 从 AVX2 降级):
 *   - SSE4.1: 10-bit 亮度去块滤波 (垂直边 + 水平边), 一次处理 8 像素
 *   - SSE4.1: 10-bit 色度去块滤波 (垂直边 + 水平边), U/V 交错处理 4 像素
 *
 * 算法参考: libudavs2 x86_256/intrinsic_deblock_256.c (8-bit AVX2)
 *           libudavs2 x86/intrinsic_deblock.c (10-bit SSE4.1 色度)
 * 移植改动: 10-bit 像素为 uint16_t, 无需 8↔16 位扩展/打包,
 *           加载用 loadu_si128 (128-bit = 8 像素), 存储用 storeu_si128.
 *
 * 核心思路 (128-bit 分别处理 L/R 两侧):
 *   原 AVX2 将 L/R 打包到 256-bit 同时计算, SSE4 降级为分别计算 L 侧和 R 侧.
 *   FS (滤波强度) 对 L/R 两侧相同, 只需计算一次.
 *   用 blendv 按 FS 值选择不同强度 (0..4) 的滤波结果.
 *
 * 对齐说明:
 *   帧数据在 8x8 网格边界处是 16 字节对齐的 (10-bit: 8 像素 × 2 字节 = 16 字节;
 *   帧步长 stride 为 128 字节对齐, 基地址经 padding 后 64 字节对齐).
 *   但垂直边滤波需从 src-3 处加载 (3 个 uint16_t = 6 字节, 相对 16 字节
 *   边界偏移 -6), 不满足 16 字节对齐, 故必须用 loadu/storeu.
 *   水平边滤波各访问点虽恰好 16 字节对齐 (同行同列, stride 为 128 字节
 *   的倍数), 但为与垂直边保持一致也用 loadu/storeu.
 *   ver 边需 8x8 转置, 加载/存储 8 像素/行 (位置 -3..+4, 多余 2 像素原样写回)
 */

#include "internal.h"

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(x) (void)(x)
#endif

/* ---- C 回退函数声明 (在 loopfilter.c 中定义) ---- */
extern void deblock_luma_ver(void *src, int stride, int alpha, int beta,
                             uint8_t *flt_flag, int bit_depth);
extern void deblock_luma_hor(void *src, int stride, int alpha, int beta,
                             uint8_t *flt_flag, int bit_depth);
extern void deblock_chroma_ver(void *src_u, void *src_v, int stride,
                               int alpha, int beta, uint8_t *flt_flag, int bit_depth);
extern void deblock_chroma_hor(void *src_u, void *src_v, int stride,
                               int alpha, int beta, uint8_t *flt_flag, int bit_depth);

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <tmmintrin.h>
#include <smmintrin.h>

/* ===========================================================================
 * 8x8 uint16 转置 (SSE 128-bit, 3 级 interleave)
 * 输入: r0..r7 各 8 个 uint16 (一行)
 * 输出: c0..c7 各 8 个 uint16 (一列)
 * 转置是其自身的逆运算 (转置两次恢复原矩阵)
 * =========================================================================== */
static inline void transpose8x8_epu16(
    __m128i r0, __m128i r1, __m128i r2, __m128i r3,
    __m128i r4, __m128i r5, __m128i r6, __m128i r7,
    __m128i *c0, __m128i *c1, __m128i *c2, __m128i *c3,
    __m128i *c4, __m128i *c5, __m128i *c6, __m128i *c7)
{
    __m128i t0 = _mm_unpacklo_epi16(r0, r1);
    __m128i t1 = _mm_unpacklo_epi16(r2, r3);
    __m128i t2 = _mm_unpacklo_epi16(r4, r5);
    __m128i t3 = _mm_unpacklo_epi16(r6, r7);
    __m128i t4 = _mm_unpackhi_epi16(r0, r1);
    __m128i t5 = _mm_unpackhi_epi16(r2, r3);
    __m128i t6 = _mm_unpackhi_epi16(r4, r5);
    __m128i t7 = _mm_unpackhi_epi16(r6, r7);

    __m128i u0 = _mm_unpacklo_epi32(t0, t1);
    __m128i u1 = _mm_unpacklo_epi32(t2, t3);
    __m128i u2 = _mm_unpackhi_epi32(t0, t1);
    __m128i u3 = _mm_unpackhi_epi32(t2, t3);
    __m128i u4 = _mm_unpacklo_epi32(t4, t5);
    __m128i u5 = _mm_unpacklo_epi32(t6, t7);
    __m128i u6 = _mm_unpackhi_epi32(t4, t5);
    __m128i u7 = _mm_unpackhi_epi32(t6, t7);

    *c0 = _mm_unpacklo_epi64(u0, u1);
    *c1 = _mm_unpackhi_epi64(u0, u1);
    *c2 = _mm_unpacklo_epi64(u2, u3);
    *c3 = _mm_unpackhi_epi64(u2, u3);
    *c4 = _mm_unpacklo_epi64(u4, u5);
    *c5 = _mm_unpackhi_epi64(u4, u5);
    *c6 = _mm_unpacklo_epi64(u6, u7);
    *c7 = _mm_unpackhi_epi64(u6, u7);
}

/* ===========================================================================
 * 核心滤波: 处理 8 个亮度像素的去块滤波
 *
 * 输入: TL2,TL1,TL0,TR0,TR1,TR2 — 各 128-bit (8 个 uint16)
 *       alpha, beta — 阈值 (已按位深移位)
 *       flag0, flag1 — 前4/后4 像素的滤波标志
 * 输出: oTL0,oTL1,oTL2,oTR0,oTR1,oTR2 — 滤波后的值 (未滤波的像素保持原值)
 *
 * 滤波强度 FS 判定 (与 C 代码 deblock_edge 完全一致):
 *   flat_l = (|L1-L0|<beta ? 2:0) + (|L2-L0|<beta ? 1:0)
 *   flat_r = (|R0-R1|<beta ? 2:0) + (|R0-R2|<beta ? 1:0)
 *   sum = flat_l + flat_r
 *   sum==6: fs = 3 + (L1==L0 && R1==R0)
 *   sum==5: fs = 2 + (L1==L0 && R1==R0)
 *   sum==4: fs = 1 + (flat_l==2)
 *   sum==3: fs = (|L1-R1|<beta)
 *   else:   fs = 0
 * =========================================================================== */
static inline void deblock_core_luma_sse4(
    __m128i TL2, __m128i TL1, __m128i TL0,
    __m128i TR0, __m128i TR1, __m128i TR2,
    int alpha, int beta, int flag0, int flag1,
    __m128i *oTL0, __m128i *oTL1, __m128i *oTL2,
    __m128i *oTR0, __m128i *oTR1, __m128i *oTR2)
{
    /* 常量 */
    __m128i c_0   = _mm_set1_epi16(0);
    __m128i c_1   = _mm_set1_epi16(1);
    __m128i c_2   = _mm_set1_epi16(2);
    __m128i c_3   = _mm_set1_epi16(3);
    __m128i c_4   = _mm_set1_epi16(4);
    __m128i c_8   = _mm_set1_epi16(8);
    __m128i c_16  = _mm_set1_epi16(16);
    __m128i ALPHA = _mm_set1_epi16((short)alpha);
    __m128i BETA  = _mm_set1_epi16((short)beta);

    __m128i T0, T1, T2, M0, M1;
    __m128i FLT, FLT_L, FLT_R, FS, FS3, FS4, FS56;

    /* --- 主掩码 M0 = flt_flag && (|L0-R0|>1) && (|L0-R0|<alpha) --- */
    T0 = _mm_abs_epi16(_mm_subs_epi16(TL0, TR0));
    T1 = _mm_cmpgt_epi16(T0, c_1);
    T2 = _mm_cmpgt_epi16(ALPHA, T0);

    int mflag0 = flag0 ? -1 : 0;
    int mflag1 = flag1 ? -1 : 0;
    M0 = _mm_set_epi32(mflag1, mflag1, mflag0, mflag0);
    M0 = _mm_and_si128(M0, _mm_and_si128(T1, T2));

    /* --- 平坦度: flat_l 和 flat_r 分别计算 --- */
    /* flat_l = (|L1-L0|<beta ? 2:0) + (|L2-L0|<beta ? 1:0) */
    T0 = _mm_abs_epi16(_mm_subs_epi16(TL1, TL0));
    FLT_L = _mm_and_si128(_mm_cmpgt_epi16(BETA, T0), c_2);
    T0 = _mm_abs_epi16(_mm_subs_epi16(TL2, TL0));
    FLT_L = _mm_add_epi16(FLT_L, _mm_and_si128(_mm_cmpgt_epi16(BETA, T0), c_1));

    /* flat_r = (|R0-R1|<beta ? 2:0) + (|R0-R2|<beta ? 1:0) */
    T0 = _mm_abs_epi16(_mm_subs_epi16(TR0, TR1));
    FLT_R = _mm_and_si128(_mm_cmpgt_epi16(BETA, T0), c_2);
    T0 = _mm_abs_epi16(_mm_subs_epi16(TR0, TR2));
    FLT_R = _mm_add_epi16(FLT_R, _mm_and_si128(_mm_cmpgt_epi16(BETA, T0), c_1));

    FLT = _mm_add_epi16(FLT_L, FLT_R);

    /* --- M1 = (L1==L0) && (R1==R0) --- */
    M1 = _mm_and_si128(_mm_cmpeq_epi16(TL1, TL0), _mm_cmpeq_epi16(TR1, TR0));

    /* --- 确定 FS --- */
    T0 = _mm_subs_epi16(FLT, c_2);  /* FLT-2 (饱和) */
    T1 = _mm_subs_epi16(FLT, c_3);  /* FLT-3 (饱和) */
    T2 = _mm_abs_epi16(_mm_subs_epi16(TL1, TR1));

    FS56 = _mm_blendv_epi8(T1, T0, M1);  /* M1 ? FLT-2 : FLT-3 */
    FS4  = _mm_blendv_epi8(c_1, c_2, _mm_cmpeq_epi16(FLT_L, c_2));  /* flat_l==2 ? 2 : 1 */
    FS3  = _mm_blendv_epi8(c_0, c_1, _mm_cmpgt_epi16(BETA, T2));    /* |L1-R1|<beta ? 1 : 0 */

    FS = _mm_blendv_epi8(c_0, FS56, _mm_cmpgt_epi16(FLT, c_4));    /* FLT>4 -> FS56 */
    FS = _mm_blendv_epi8(FS, FS4, _mm_cmpeq_epi16(FLT, c_4));      /* FLT==4 -> FS4 */
    FS = _mm_blendv_epi8(FS, FS3, _mm_cmpeq_epi16(FLT, c_3));      /* FLT==3 -> FS3 */
    FS = _mm_and_si128(FS, M0);

    /* --- 滤波: L 侧和 R 侧分别计算 --- */
    __m128i L0w = TL0, L1w = TL1, L2w = TL2;
    __m128i R0w = TR0, R1w = TR1, R2w = TR2;

    /* sumLR = L0+R0+2, 后续每个 fs 级别左移 1 位 (2x, 4x) */
    __m128i sumLR = _mm_add_epi16(_mm_add_epi16(TL0, TR0), c_2);

    /* fs==1: L0'=(3*L0+R0+2)>>2, R0'=(3*R0+L0+2)>>2 */
    T0 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(TL0, 1), sumLR), 2);
    L0w = _mm_blendv_epi8(L0w, T0, _mm_cmpeq_epi16(FS, c_1));
    T0 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(TR0, 1), sumLR), 2);
    R0w = _mm_blendv_epi8(R0w, T0, _mm_cmpeq_epi16(FS, c_1));

    /* fs==2: sumLR *= 2 -> 2*(L0+R0+2) */
    sumLR = _mm_slli_epi16(sumLR, 1);
    /* L0'=(10*L0+3*L1+3*R0+8)>>4 = (8*L0 + (3*L1+R0) + 2*(L0+R0+2) + 4)>>4 */
    T0 = _mm_add_epi16(_mm_slli_epi16(TL1, 1), _mm_add_epi16(TL1, TR0));  /* 3*L1+R0 */
    T0 = _mm_add_epi16(_mm_slli_epi16(TL0, 3), _mm_add_epi16(T0, sumLR));
    T0 = _mm_srli_epi16(_mm_add_epi16(T0, c_4), 4);
    L0w = _mm_blendv_epi8(L0w, T0, _mm_cmpeq_epi16(FS, c_2));
    /* R0'=(10*R0+3*R1+3*L0+8)>>4 */
    T0 = _mm_add_epi16(_mm_slli_epi16(TR1, 1), _mm_add_epi16(TR1, TL0));  /* 3*R1+L0 */
    T0 = _mm_add_epi16(_mm_slli_epi16(TR0, 3), _mm_add_epi16(T0, sumLR));
    T0 = _mm_srli_epi16(_mm_add_epi16(T0, c_4), 4);
    R0w = _mm_blendv_epi8(R0w, T0, _mm_cmpeq_epi16(FS, c_2));

    /* fs==3: sumLR *= 2 -> 4*(L0+R0+2) */
    sumLR = _mm_slli_epi16(sumLR, 1);
    /* L0'=(6*L0+4*L1+L2+4*R0+R1+8)>>4 = (2*L0 + (4*L1+L2+R1) + 4*(L0+R0+2))>>4 */
    T0 = _mm_add_epi16(_mm_slli_epi16(TL1, 2), _mm_add_epi16(TL2, TR1));  /* 4*L1+L2+R1 */
    T0 = _mm_add_epi16(_mm_slli_epi16(TL0, 1), _mm_add_epi16(T0, sumLR));
    T0 = _mm_srli_epi16(T0, 4);
    L0w = _mm_blendv_epi8(L0w, T0, _mm_cmpeq_epi16(FS, c_3));
    /* R0'=(6*R0+4*R1+R2+4*L0+L1+8)>>4 */
    T0 = _mm_add_epi16(_mm_slli_epi16(TR1, 2), _mm_add_epi16(TR2, TL1));  /* 4*R1+R2+L1 */
    T0 = _mm_add_epi16(_mm_slli_epi16(TR0, 1), _mm_add_epi16(T0, sumLR));
    T0 = _mm_srli_epi16(T0, 4);
    R0w = _mm_blendv_epi8(R0w, T0, _mm_cmpeq_epi16(FS, c_3));
    /* L1'=(3*L2+8*L1+4*L0+R0+8)>>4 */
    T0 = _mm_add_epi16(_mm_add_epi16(TL2, TR0), _mm_slli_epi16(TL2, 1));  /* 3*L2+R0 */
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TL1, 3));  /* + 8*L1 */
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TL0, 2));  /* + 4*L0 */
    T0 = _mm_srli_epi16(_mm_add_epi16(T0, c_8), 4);
    L1w = _mm_blendv_epi8(L1w, T0, _mm_cmpeq_epi16(FS, c_3));
    /* R1'=(3*R2+8*R1+4*R0+L0+8)>>4 */
    T0 = _mm_add_epi16(_mm_add_epi16(TR2, TL0), _mm_slli_epi16(TR2, 1));  /* 3*R2+L0 */
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TR1, 3));  /* + 8*R1 */
    T0 = _mm_add_epi16(T0, _mm_slli_epi16(TR0, 2));  /* + 4*R0 */
    T0 = _mm_srli_epi16(_mm_add_epi16(T0, c_8), 4);
    R1w = _mm_blendv_epi8(R1w, T0, _mm_cmpeq_epi16(FS, c_3));

    /* fs==4: 仅当存在 fs==4 的像素时才计算 */
    {
        __m128i FS4_mask = _mm_cmpeq_epi16(FS, c_4);
        if (_mm_extract_epi64(FS4_mask, 0) || _mm_extract_epi64(FS4_mask, 1)) {
            /* L0'=(9*L0+9*L2+8*R0+6*R2+16)>>5 */
            T0 = _mm_slli_epi16(_mm_add_epi16(_mm_add_epi16(TL0, TL2), TR0), 3);  /* 8*(L0+L2+R0) */
            T0 = _mm_add_epi16(_mm_add_epi16(T0, c_16), _mm_add_epi16(TL0, TL2));  /* + 16 + (L0+L2) */
            T2 = _mm_add_epi16(_mm_slli_epi16(TR2, 1), _mm_slli_epi16(TR2, 2));    /* 6*R2 */
            T0 = _mm_srli_epi16(_mm_add_epi16(T0, T2), 5);
            L0w = _mm_blendv_epi8(L0w, T0, FS4_mask);
            /* R0'=(9*R0+9*R2+8*L0+6*L2+16)>>5 */
            T0 = _mm_slli_epi16(_mm_add_epi16(_mm_add_epi16(TR0, TR2), TL0), 3);  /* 8*(R0+R2+L0) */
            T0 = _mm_add_epi16(_mm_add_epi16(T0, c_16), _mm_add_epi16(TR0, TR2));  /* + 16 + (R0+R2) */
            T2 = _mm_add_epi16(_mm_slli_epi16(TL2, 1), _mm_slli_epi16(TL2, 2));    /* 6*L2 */
            T0 = _mm_srli_epi16(_mm_add_epi16(T0, T2), 5);
            R0w = _mm_blendv_epi8(R0w, T0, FS4_mask);

            /* L1'=(7*L0+6*L2+3*R0+8)>>4 */
            T0 = _mm_slli_epi16(_mm_add_epi16(TL2, TR0), 1);  /* 2*(L2+R0) */
            T0 = _mm_add_epi16(T0, _mm_sub_epi16(_mm_slli_epi16(TL0, 3), TL0));  /* + 7*L0 */
            T2 = _mm_add_epi16(_mm_slli_epi16(TL2, 2), _mm_add_epi16(TR0, c_8));  /* 4*L2 + R0 + 8 */
            T0 = _mm_srli_epi16(_mm_add_epi16(T0, T2), 4);
            L1w = _mm_blendv_epi8(L1w, T0, FS4_mask);
            /* R1'=(7*R0+6*R2+3*L0+8)>>4 */
            T0 = _mm_slli_epi16(_mm_add_epi16(TR2, TL0), 1);  /* 2*(R2+L0) */
            T0 = _mm_add_epi16(T0, _mm_sub_epi16(_mm_slli_epi16(TR0, 3), TR0));  /* + 7*R0 */
            T2 = _mm_add_epi16(_mm_slli_epi16(TR2, 2), _mm_add_epi16(TL0, c_8));  /* 4*R2 + L0 + 8 */
            T0 = _mm_srli_epi16(_mm_add_epi16(T0, T2), 4);
            R1w = _mm_blendv_epi8(R1w, T0, FS4_mask);

            /* L2'=(4*L0+3*L2+R0+4)>>3 */
            T0 = _mm_add_epi16(_mm_slli_epi16(TL2, 1), TL2);  /* 3*L2 */
            T2 = _mm_add_epi16(_mm_slli_epi16(TL0, 2), TR0);   /* 4*L0 + R0 */
            T0 = _mm_srli_epi16(_mm_add_epi16(T0, _mm_add_epi16(T2, c_4)), 3);
            L2w = _mm_blendv_epi8(L2w, T0, FS4_mask);
            /* R2'=(4*R0+3*R2+L0+4)>>3 */
            T0 = _mm_add_epi16(_mm_slli_epi16(TR2, 1), TR2);  /* 3*R2 */
            T2 = _mm_add_epi16(_mm_slli_epi16(TR0, 2), TL0);   /* 4*R0 + L0 */
            T0 = _mm_srli_epi16(_mm_add_epi16(T0, _mm_add_epi16(T2, c_4)), 3);
            R2w = _mm_blendv_epi8(R2w, T0, FS4_mask);
        }
    }

    *oTL0 = L0w; *oTL1 = L1w; *oTL2 = L2w;
    *oTR0 = R0w; *oTR1 = R1w; *oTR2 = R2w;
}

/* ===========================================================================
 * 亮度垂直边去块滤波 (EDGE_VER)
 * src 指向 R0 (边界右侧首像素), stride 为行步长 (uint16_t 元素)
 * 跨边方向: 水平 (inc1=1), 沿边方向: 垂直 (ptr_inc=stride)
 * 加载 8 行 × 8 像素, 转置后滤波, 转置回后存储
 * =========================================================================== */
static void deblock_luma_ver_sse4(void *src_v, int stride, int alpha,
                                  int beta, uint8_t *flt_flag, int bit_depth)
{
    if (bit_depth <= 8) {
        /* 8-bit: 加载 8 字节→扩展为 uint16, 复用 16-bit 核心滤波, packus 压缩回 8 字节 */
        uint8_t *src = (uint8_t *)src_v;
        int s1 = stride, s2 = stride * 2, s3 = stride * 3;
        int s4 = stride * 4, s5 = stride * 5, s6 = stride * 6, s7 = stride * 7;

        __m128i r0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src - 3)));
        __m128i r1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src - 3 + s1)));
        __m128i r2 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src - 3 + s2)));
        __m128i r3 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src - 3 + s3)));
        __m128i r4 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src - 3 + s4)));
        __m128i r5 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src - 3 + s5)));
        __m128i r6 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src - 3 + s6)));
        __m128i r7 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src - 3 + s7)));

        __m128i c0, c1, c2, c3, c4, c5, c6, c7;
        transpose8x8_epu16(r0, r1, r2, r3, r4, r5, r6, r7,
                           &c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7);

        /* Early-exit: c2=L0, c3=R0, 检查 |L0-R0| 是否需要滤波 */
        {
            __m128i diff = _mm_abs_epi16(_mm_subs_epi16(c2, c3));
            __m128i c_1 = _mm_set1_epi16(1);
            __m128i av = _mm_set1_epi16((short)alpha);
            __m128i need = _mm_and_si128(_mm_cmpgt_epi16(diff, c_1),
                                          _mm_cmpgt_epi16(av, diff));
            int mf0 = flt_flag[0] ? -1 : 0;
            int mf1 = flt_flag[1] ? -1 : 0;
            __m128i mask = _mm_set_epi32(mf1, mf1, mf0, mf0);
            need = _mm_and_si128(need, mask);
            if (_mm_movemask_epi8(need) == 0)
                return;
        }

        __m128i oL0, oL1, oL2, oR0, oR1, oR2;
        deblock_core_luma_sse4(c0, c1, c2, c3, c4, c5,
                               alpha, beta, flt_flag[0], flt_flag[1],
                               &oL0, &oL1, &oL2, &oR0, &oR1, &oR2);

        c0 = oL2; c1 = oL1; c2 = oL0; c3 = oR0; c4 = oR1; c5 = oR2;
        transpose8x8_epu16(c0, c1, c2, c3, c4, c5, c6, c7,
                           &r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);

        __m128i z = _mm_setzero_si128();
        _mm_storel_epi64((__m128i*)(src - 3),       _mm_packus_epi16(r0, z));
        _mm_storel_epi64((__m128i*)(src - 3 + s1),  _mm_packus_epi16(r1, z));
        _mm_storel_epi64((__m128i*)(src - 3 + s2),  _mm_packus_epi16(r2, z));
        _mm_storel_epi64((__m128i*)(src - 3 + s3),  _mm_packus_epi16(r3, z));
        _mm_storel_epi64((__m128i*)(src - 3 + s4),  _mm_packus_epi16(r4, z));
        _mm_storel_epi64((__m128i*)(src - 3 + s5),  _mm_packus_epi16(r5, z));
        _mm_storel_epi64((__m128i*)(src - 3 + s6),  _mm_packus_epi16(r6, z));
        _mm_storel_epi64((__m128i*)(src - 3 + s7),  _mm_packus_epi16(r7, z));
        return;
    }

    uint16_t *src = (uint16_t *)src_v;
    int stride2 = stride * 2;
    int stride3 = stride * 3;
    int stride4 = stride * 4;
    int stride5 = stride * 5;
    int stride6 = stride * 6;
    int stride7 = stride * 7;

    /* 加载 8 行 × 8 像素 (从 src-3 开始, 覆盖 L2..R4) */
    __m128i r0 = _mm_loadu_si128((const __m128i*)(src - 3));
    __m128i r1 = _mm_loadu_si128((const __m128i*)(src - 3 + stride));
    __m128i r2 = _mm_loadu_si128((const __m128i*)(src - 3 + stride2));
    __m128i r3 = _mm_loadu_si128((const __m128i*)(src - 3 + stride3));
    __m128i r4 = _mm_loadu_si128((const __m128i*)(src - 3 + stride4));
    __m128i r5 = _mm_loadu_si128((const __m128i*)(src - 3 + stride5));
    __m128i r6 = _mm_loadu_si128((const __m128i*)(src - 3 + stride6));
    __m128i r7 = _mm_loadu_si128((const __m128i*)(src - 3 + stride7));

    /* 转置: 行→列 */
    __m128i c0, c1, c2, c3, c4, c5, c6, c7;
    transpose8x8_epu16(r0, r1, r2, r3, r4, r5, r6, r7,
                       &c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7);

    /* c0=L2, c1=L1, c2=L0, c3=R0, c4=R1, c5=R2, c6=R3(不变), c7=R4(不变) */
    /* Early-exit: 检查是否有像素需要滤波, 跳过滤波+转置回+存储 */
    {
        __m128i diff = _mm_abs_epi16(_mm_subs_epi16(c2, c3));
        __m128i c_1 = _mm_set1_epi16(1);
        __m128i av = _mm_set1_epi16((short)alpha);
        __m128i need = _mm_and_si128(_mm_cmpgt_epi16(diff, c_1),
                                      _mm_cmpgt_epi16(av, diff));
        int mf0 = flt_flag[0] ? -1 : 0;
        int mf1 = flt_flag[1] ? -1 : 0;
        __m128i mask = _mm_set_epi32(mf1, mf1, mf0, mf0);
        need = _mm_and_si128(need, mask);
        if (_mm_movemask_epi8(need) == 0)
            return;  /* 全部像素无需滤波, 数据未修改, 跳过写回 */
    }

    __m128i oL0, oL1, oL2, oR0, oR1, oR2;
    deblock_core_luma_sse4(c0, c1, c2, c3, c4, c5,
                           alpha, beta, flt_flag[0], flt_flag[1],
                           &oL0, &oL1, &oL2, &oR0, &oR1, &oR2);

    /* 替换修改后的列向量 */
    c0 = oL2; c1 = oL1; c2 = oL0; c3 = oR0; c4 = oR1; c5 = oR2;

    /* 转置回: 列→行 */
    __m128i o0, o1, o2, o3, o4, o5, o6, o7;
    transpose8x8_epu16(c0, c1, c2, c3, c4, c5, c6, c7,
                       &o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7);

    /* 存储 8 行 × 8 像素 (位置 -3..+4, R3/R4 原样写回) */
    _mm_storeu_si128((__m128i*)(src - 3),           o0);
    _mm_storeu_si128((__m128i*)(src - 3 + stride),  o1);
    _mm_storeu_si128((__m128i*)(src - 3 + stride2), o2);
    _mm_storeu_si128((__m128i*)(src - 3 + stride3), o3);
    _mm_storeu_si128((__m128i*)(src - 3 + stride4), o4);
    _mm_storeu_si128((__m128i*)(src - 3 + stride5), o5);
    _mm_storeu_si128((__m128i*)(src - 3 + stride6), o6);
    _mm_storeu_si128((__m128i*)(src - 3 + stride7), o7);
}

/* ===========================================================================
 * 亮度水平边去块滤波 (EDGE_HOR)
 * src 指向 R0 (边界下方首像素), stride 为行步长
 * 跨边方向: 垂直 (inc1=stride), 沿边方向: 水平 (ptr_inc=1)
 * 每行 8 像素连续, 无需转置
 * =========================================================================== */
static void deblock_luma_hor_sse4(void *src_v, int stride, int alpha,
                                  int beta, uint8_t *flt_flag, int bit_depth)
{
    if (bit_depth <= 8) {
        /* 8-bit: 加载 8 字节→扩展为 uint16, 复用 16-bit 核心滤波, packus 压缩回 8 字节 */
        uint8_t *src = (uint8_t *)src_v;
        int s2 = stride * 2, s3 = stride * 3;

        /* Early-exit: 先加载 L0/R0 检查是否需要滤波 */
        {
            __m128i eL0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src - stride)));
            __m128i eR0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src)));
            __m128i diff = _mm_abs_epi16(_mm_subs_epi16(eL0, eR0));
            __m128i c_1 = _mm_set1_epi16(1);
            __m128i av = _mm_set1_epi16((short)alpha);
            __m128i need = _mm_and_si128(_mm_cmpgt_epi16(diff, c_1),
                                          _mm_cmpgt_epi16(av, diff));
            int mf0 = flt_flag[0] ? -1 : 0;
            int mf1 = flt_flag[1] ? -1 : 0;
            __m128i mask = _mm_set_epi32(mf1, mf1, mf0, mf0);
            need = _mm_and_si128(need, mask);
            if (_mm_movemask_epi8(need) == 0)
                return;
        }

        __m128i L2 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src - s3)));
        __m128i L1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src - s2)));
        __m128i L0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src - stride)));
        __m128i R0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src)));
        __m128i R1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src + stride)));
        __m128i R2 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(src + s2)));

        __m128i oL0, oL1, oL2, oR0, oR1, oR2;
        deblock_core_luma_sse4(L2, L1, L0, R0, R1, R2,
                               alpha, beta, flt_flag[0], flt_flag[1],
                               &oL0, &oL1, &oL2, &oR0, &oR1, &oR2);

        __m128i z = _mm_setzero_si128();
        _mm_storel_epi64((__m128i*)(src - s3),    _mm_packus_epi16(oL2, z));
        _mm_storel_epi64((__m128i*)(src - s2),    _mm_packus_epi16(oL1, z));
        _mm_storel_epi64((__m128i*)(src - stride), _mm_packus_epi16(oL0, z));
        _mm_storel_epi64((__m128i*)(src),          _mm_packus_epi16(oR0, z));
        _mm_storel_epi64((__m128i*)(src + stride), _mm_packus_epi16(oR1, z));
        _mm_storel_epi64((__m128i*)(src + s2),     _mm_packus_epi16(oR2, z));
        return;
    }

    uint16_t *src = (uint16_t *)src_v;
    int stride2 = stride * 2;
    int stride3 = stride * 3;

    /* Early-exit: 先加载 L0/R0 检查是否有像素需要滤波
     * 条件: flt_flag && |L0-R0|>1 && |L0-R0|<alpha
     * 全部不需要时跳过其余加载/滤波/存储 (大量平滑边可省开销) */
    {
        __m128i eL0 = _mm_loadu_si128((const __m128i*)(src - stride));
        __m128i eR0 = _mm_loadu_si128((const __m128i*)(src));
        __m128i diff = _mm_abs_epi16(_mm_subs_epi16(eL0, eR0));
        __m128i c_1 = _mm_set1_epi16(1);
        __m128i av = _mm_set1_epi16((short)alpha);
        __m128i need = _mm_and_si128(_mm_cmpgt_epi16(diff, c_1),
                                      _mm_cmpgt_epi16(av, diff));
        int mf0 = flt_flag[0] ? -1 : 0;
        int mf1 = flt_flag[1] ? -1 : 0;
        __m128i mask = _mm_set_epi32(mf1, mf1, mf0, mf0);
        need = _mm_and_si128(need, mask);
        if (_mm_movemask_epi8(need) == 0)
            return;
    }

    /* 直接加载 6 行 × 8 像素 (各连续) */
    __m128i L2 = _mm_loadu_si128((const __m128i*)(src - stride3));
    __m128i L1 = _mm_loadu_si128((const __m128i*)(src - stride2));
    __m128i L0 = _mm_loadu_si128((const __m128i*)(src - stride));
    __m128i R0 = _mm_loadu_si128((const __m128i*)(src));
    __m128i R1 = _mm_loadu_si128((const __m128i*)(src + stride));
    __m128i R2 = _mm_loadu_si128((const __m128i*)(src + stride2));

    __m128i oL0, oL1, oL2, oR0, oR1, oR2;
    deblock_core_luma_sse4(L2, L1, L0, R0, R1, R2,
                           alpha, beta, flt_flag[0], flt_flag[1],
                           &oL0, &oL1, &oL2, &oR0, &oR1, &oR2);

    /* 直接存储 6 行 */
    _mm_storeu_si128((__m128i*)(src - stride3), oL2);
    _mm_storeu_si128((__m128i*)(src - stride2), oL1);
    _mm_storeu_si128((__m128i*)(src - stride),  oL0);
    _mm_storeu_si128((__m128i*)(src),           oR0);
    _mm_storeu_si128((__m128i*)(src + stride),  oR1);
    _mm_storeu_si128((__m128i*)(src + stride2), oR2);
}

/* ===========================================================================
 * 色度垂直边去块滤波 (EDGE_VER) - SSE4.1 128-bit
 * 同时处理 U/V (低 64 位=U, 高 64 位=V), 加载 4 行 × 8 像素
 * 色度滤波强度比亮度减 1: 无 fs=4, FLT==3 时 fs=0
 * 参考: deblock_edge_ver_c_sse128_10bit
 * =========================================================================== */
static void deblock_chroma_ver_sse4(void *src_u_v, void *src_v_v,
                                    int stride, int alpha, int beta,
                                    uint8_t *flt_flag, int bit_depth)
{
    int is_8bit = (bit_depth <= 8);
    int flag0 = flt_flag[0] ? -1 : 0;
    int flag1 = flt_flag[1] ? -1 : 0;

    __m128i uvl0, uvl1, uvl2, uvr0, uvr1, uvr2;
    __m128i tl0, tl1, tl2, tl3;
    __m128i tr0, tr1, tr2, tr3;
    __m128i t0, t1, t2, t3, t4, t5, t6, t7;
    __m128i v0, v1, v2, v3;
    __m128i m0, m1, m2, m3, m4, m5, m6, m7;
    __m128i flt_l, flt_r, flt, fs;
    __m128i fs4, fs56;

    __m128i ALPHA = _mm_set1_epi16((short)alpha);
    __m128i BETA  = _mm_set1_epi16((short)beta);
    __m128i c_0 = _mm_set1_epi16(0);
    __m128i c_1 = _mm_set1_epi16(1);
    __m128i c_2 = _mm_set1_epi16(2);
    __m128i c_3 = _mm_set1_epi16(3);
    __m128i c_4 = _mm_set1_epi16(4);
    __m128i c_8 = _mm_set1_epi16(8);

    /* 加载 U/V 各 4 行 × 8 像素 (位置 -4..3, 覆盖 L3..R3) */
    if (is_8bit) {
        uint8_t *p = (uint8_t *)src_u_v - 4;
        t0 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(p)));
        t1 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(p + stride)));
        t2 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(p + stride * 2)));
        t3 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(p + stride * 3)));
        p = (uint8_t *)src_v_v - 4;
        t4 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(p)));
        t5 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(p + stride)));
        t6 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(p + stride * 2)));
        t7 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(p + stride * 3)));
    } else {
        uint16_t *p = (uint16_t *)src_u_v - 4;
        t0 = _mm_loadu_si128((__m128i*)(p));
        t1 = _mm_loadu_si128((__m128i*)(p + stride));
        t2 = _mm_loadu_si128((__m128i*)(p + stride * 2));
        t3 = _mm_loadu_si128((__m128i*)(p + stride * 3));
        p = (uint16_t *)src_v_v - 4;
        t4 = _mm_loadu_si128((__m128i*)(p));
        t5 = _mm_loadu_si128((__m128i*)(p + stride));
        t6 = _mm_loadu_si128((__m128i*)(p + stride * 2));
        t7 = _mm_loadu_si128((__m128i*)(p + stride * 3));
    }

    /* 转置 4x8 U 和 4x8 V, 交错 U/V (低 64=U, 高 64=V) */
    m0 = _mm_unpacklo_epi16(t0, t1);
    m1 = _mm_unpackhi_epi16(t0, t1);
    m2 = _mm_unpacklo_epi16(t2, t3);
    m3 = _mm_unpackhi_epi16(t2, t3);
    m4 = _mm_unpacklo_epi16(t4, t5);
    m5 = _mm_unpackhi_epi16(t4, t5);
    m6 = _mm_unpacklo_epi16(t6, t7);
    m7 = _mm_unpackhi_epi16(t6, t7);

    t0 = _mm_unpacklo_epi32(m0, m2);
    t1 = _mm_unpackhi_epi32(m0, m2);
    t2 = _mm_unpacklo_epi32(m1, m3);
    t3 = _mm_unpackhi_epi32(m1, m3);
    t4 = _mm_unpacklo_epi32(m4, m6);
    t5 = _mm_unpackhi_epi32(m4, m6);
    t6 = _mm_unpacklo_epi32(m5, m7);
    t7 = _mm_unpackhi_epi32(m5, m7);

    tl3 = _mm_unpacklo_epi64(t0, t4);
    tl2 = _mm_unpackhi_epi64(t0, t4);
    tr0 = _mm_unpacklo_epi64(t2, t6);
    tr1 = _mm_unpackhi_epi64(t2, t6);
    tl1 = _mm_unpacklo_epi64(t1, t5);
    tl0 = _mm_unpackhi_epi64(t1, t5);
    tr2 = _mm_unpacklo_epi64(t3, t7);
    tr3 = _mm_unpackhi_epi64(t3, t7);

#define _mm_subabs_epu16(a, b) _mm_abs_epi16(_mm_subs_epi16(a, b))

    /* --- 主掩码 m0 = flt_flag && (|L0-R0|>1) && (|L0-R0|<alpha) --- */
    t0 = _mm_subabs_epu16(tl0, tr0);
    t1 = _mm_cmpgt_epi16(t0, c_1);
    t2 = _mm_cmpgt_epi16(ALPHA, t0);
    m0 = _mm_set_epi32(flag1, flag0, flag1, flag0);
    m0 = _mm_and_si128(m0, _mm_and_si128(t1, t2));

    /* --- 平坦度 flt = flat_l + flat_r --- */
    t0 = _mm_subabs_epu16(tl1, tl0);
    t1 = _mm_subabs_epu16(tr1, tr0);
    flt_l = _mm_and_si128(_mm_cmpgt_epi16(BETA, t0), c_2);
    flt_r = _mm_and_si128(_mm_cmpgt_epi16(BETA, t1), c_2);

    t0 = _mm_subabs_epu16(tl2, tl0);
    t1 = _mm_subabs_epu16(tr2, tr0);
    m1 = _mm_cmpgt_epi16(BETA, t0);
    m2 = _mm_cmpgt_epi16(BETA, t1);
    flt_l = _mm_add_epi16(_mm_and_si128(m1, c_1), flt_l);
    flt_r = _mm_add_epi16(_mm_and_si128(m2, c_1), flt_r);
    flt = _mm_add_epi16(flt_l, flt_r);

    /* --- FS 计算 (色度: 无 FS3, FS4=0/1, 对应亮度 fs 减 1) --- */
    m1 = _mm_and_si128(_mm_cmpeq_epi16(tr0, tr1), _mm_cmpeq_epi16(tl0, tl1));
    t0 = _mm_sub_epi16(flt, c_3);   /* FLT-3 */
    t1 = _mm_sub_epi16(flt, c_4);   /* FLT-4 */
    t2 = _mm_subabs_epu16(tl1, tr1);

    fs56 = _mm_blendv_epi8(t1, t0, m1);               /* M1 ? FLT-3 : FLT-4 */
    fs4  = _mm_blendv_epi8(c_0, c_1, _mm_cmpeq_epi16(flt_l, c_2));  /* flat_l==2 ? 1 : 0 */

    fs = _mm_blendv_epi8(c_0, fs56, _mm_cmpgt_epi16(flt, c_4));  /* FLT>4 */
    fs = _mm_blendv_epi8(fs, fs4, _mm_cmpeq_epi16(flt, c_4));    /* FLT==4 */
    /* FLT==3: 无滤波 (色度 fs=0, 对应亮度 fs=1 减 1) */

    fs = _mm_and_si128(fs, m0);

    /* 全部像素无需滤波, 数据未修改, 跳过滤波计算和写回.
     * 注意: 不能用 _mm_movemask_epi8(fs)==0, 因为 fs 值 1/2/3 的 MSB 为 0,
     * 会误判为全零. 使用 _mm_testz_si128 正确检测全零. */
    if (_mm_testz_si128(fs, fs)) return;

#undef _mm_subabs_epu16

    /* 保存原始值用于滤波计算 */
    uvl0 = tl0;  uvl1 = tl1;  uvl2 = tl2;
    uvr0 = tr0;  uvr1 = tr1;  uvr2 = tr2;

    /* fs==1: L0'=(3*L0+R0+2)>>2, R0'=(3*R0+L0+2)>>2 */
    t2 = _mm_add_epi16(_mm_add_epi16(uvl0, uvr0), c_2);
    v0 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(uvl0, 1), t2), 2);
    v1 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(uvr0, 1), t2), 2);
    tl0 = _mm_blendv_epi8(tl0, v0, _mm_cmpeq_epi16(fs, c_1));
    tr0 = _mm_blendv_epi8(tr0, v1, _mm_cmpeq_epi16(fs, c_1));

    /* fs==2: L0'=(8*L0+3*L1+3*R0+L1+R0+4)>>4, R0'=(8*R0+3*R1+3*L0+R1+L0+4)>>4 */
    t2 = _mm_slli_epi16(t2, 1);  /* 2*(L0+R0+2) */
    t3 = _mm_slli_epi16(t3, 1);

    t0 = _mm_add_epi16(_mm_slli_epi16(uvl1, 1), _mm_add_epi16(uvl1, uvr0));
    t0 = _mm_add_epi16(_mm_slli_epi16(uvl0, 3), _mm_add_epi16(t0, t2));
    v0 = _mm_srli_epi16(_mm_add_epi16(t0, c_4), 4);

    t0 = _mm_add_epi16(_mm_slli_epi16(uvr1, 1), _mm_add_epi16(uvr1, uvl0));
    t0 = _mm_add_epi16(_mm_slli_epi16(uvr0, 3), _mm_add_epi16(t0, t2));
    v1 = _mm_srli_epi16(_mm_add_epi16(t0, c_4), 4);

    tl0 = _mm_blendv_epi8(tl0, v0, _mm_cmpeq_epi16(fs, c_2));
    tr0 = _mm_blendv_epi8(tr0, v1, _mm_cmpeq_epi16(fs, c_2));

    /* fs==3: L0'=(2*L0+4*L1+4*L0+L2+4*R0+R1+8)>>4, etc. */
    t2 = _mm_slli_epi16(t2, 1);  /* 4*(L0+R0+2) */
    t3 = _mm_slli_epi16(t3, 1);

    t0 = _mm_add_epi16(_mm_slli_epi16(uvl1, 2), _mm_add_epi16(uvl2, uvr1));
    t0 = _mm_add_epi16(_mm_slli_epi16(uvl0, 1), _mm_add_epi16(t0, t2));
    v0 = _mm_srli_epi16(t0, 4);

    t0 = _mm_add_epi16(_mm_slli_epi16(uvr1, 2), _mm_add_epi16(uvr2, uvl1));
    t0 = _mm_add_epi16(_mm_slli_epi16(uvr0, 1), _mm_add_epi16(t0, t2));
    v1 = _mm_srli_epi16(t0, 4);

    tl0 = _mm_blendv_epi8(tl0, v0, _mm_cmpeq_epi16(fs, c_3));
    tr0 = _mm_blendv_epi8(tr0, v1, _mm_cmpeq_epi16(fs, c_3));

    /* fs==3: L1'=(3*L2+8*L1+4*L0+R0+8)>>4 */
    t0 = _mm_add_epi16(_mm_add_epi16(uvl2, uvr0), _mm_slli_epi16(uvl2, 1));
    t0 = _mm_add_epi16(t0, _mm_slli_epi16(uvl1, 3));
    t0 = _mm_add_epi16(t0, _mm_slli_epi16(uvl0, 2));
    v2 = _mm_srli_epi16(_mm_add_epi16(t0, c_8), 4);

    t0 = _mm_add_epi16(_mm_add_epi16(uvr2, uvl0), _mm_slli_epi16(uvr2, 1));
    t0 = _mm_add_epi16(t0, _mm_slli_epi16(uvr1, 3));
    t0 = _mm_add_epi16(t0, _mm_slli_epi16(uvr0, 2));
    v3 = _mm_srli_epi16(_mm_add_epi16(t0, c_8), 4);

    tl1 = _mm_blendv_epi8(tl1, v2, _mm_cmpeq_epi16(fs, c_3));
    tr1 = _mm_blendv_epi8(tr1, v3, _mm_cmpeq_epi16(fs, c_3));

    /* 转置回行序, 分离 U/V 并存储 */
    m0 = _mm_unpacklo_epi16(tl3, tl2);
    m1 = _mm_unpackhi_epi16(tl3, tl2);
    m2 = _mm_unpacklo_epi16(tl1, tl0);
    m3 = _mm_unpackhi_epi16(tl1, tl0);
    m4 = _mm_unpacklo_epi16(tr0, tr1);
    m5 = _mm_unpackhi_epi16(tr0, tr1);
    m6 = _mm_unpacklo_epi16(tr2, tr3);
    m7 = _mm_unpackhi_epi16(tr2, tr3);

    t0 = _mm_unpacklo_epi32(m0, m2);
    t1 = _mm_unpackhi_epi32(m0, m2);
    t2 = _mm_unpacklo_epi32(m1, m3);
    t3 = _mm_unpackhi_epi32(m1, m3);
    t4 = _mm_unpacklo_epi32(m4, m6);
    t5 = _mm_unpackhi_epi32(m4, m6);
    t6 = _mm_unpacklo_epi32(m5, m7);
    t7 = _mm_unpackhi_epi32(m5, m7);

    m0 = _mm_unpacklo_epi64(t0, t4);
    m1 = _mm_unpackhi_epi64(t0, t4);
    m4 = _mm_unpacklo_epi64(t2, t6);
    m5 = _mm_unpackhi_epi64(t2, t6);
    m2 = _mm_unpacklo_epi64(t1, t5);
    m3 = _mm_unpackhi_epi64(t1, t5);
    m6 = _mm_unpacklo_epi64(t3, t7);
    m7 = _mm_unpackhi_epi64(t3, t7);

    if (is_8bit) {
        /* 8-bit: packus 将 uint16 压回 uint8, 存储 8 字节/行 */
        uint8_t *pu = (uint8_t *)src_u_v - 4;
        uint8_t *pv = (uint8_t *)src_v_v - 4;
        __m128i z = _mm_setzero_si128();
        _mm_storel_epi64((__m128i*)(pu),              _mm_packus_epi16(m0, z));
        _mm_storel_epi64((__m128i*)(pu + stride),     _mm_packus_epi16(m1, z));
        _mm_storel_epi64((__m128i*)(pu + stride * 2), _mm_packus_epi16(m2, z));
        _mm_storel_epi64((__m128i*)(pu + stride * 3), _mm_packus_epi16(m3, z));
        _mm_storel_epi64((__m128i*)(pv),              _mm_packus_epi16(m4, z));
        _mm_storel_epi64((__m128i*)(pv + stride),     _mm_packus_epi16(m5, z));
        _mm_storel_epi64((__m128i*)(pv + stride * 2), _mm_packus_epi16(m6, z));
        _mm_storel_epi64((__m128i*)(pv + stride * 3), _mm_packus_epi16(m7, z));
    } else {
        /* 10-bit: 直接存储 16 字节 (8 个 uint16)/行 */
        uint16_t *pu = (uint16_t *)src_u_v - 4;
        uint16_t *pv = (uint16_t *)src_v_v - 4;
        _mm_storeu_si128((__m128i*)(pu),              m0);
        _mm_storeu_si128((__m128i*)(pu + stride),     m1);
        _mm_storeu_si128((__m128i*)(pu + stride * 2), m2);
        _mm_storeu_si128((__m128i*)(pu + stride * 3), m3);
        _mm_storeu_si128((__m128i*)(pv),              m4);
        _mm_storeu_si128((__m128i*)(pv + stride),     m5);
        _mm_storeu_si128((__m128i*)(pv + stride * 2), m6);
        _mm_storeu_si128((__m128i*)(pv + stride * 3), m7);
    }
}

/* ===========================================================================
 * 色度水平边去块滤波 (EDGE_HOR) - SSE4.1 128-bit
 * 同时处理 U/V (低 64 位=U, 高 64 位=V), 每行 4 像素
 * 色度滤波强度比亮度减 1: 无 fs=4, FLT==3 时 fs=0
 * 参考: deblock_edge_hor_c_sse128_10bit
 * =========================================================================== */
static void deblock_chroma_hor_sse4(void *src_u_v, void *src_v_v,
                                    int stride, int alpha, int beta,
                                    uint8_t *flt_flag, int bit_depth)
{
    int is_8bit = (bit_depth <= 8);
    uint8_t  *src_u8 = (uint8_t  *)src_u_v;
    uint8_t  *src_v8 = (uint8_t  *)src_v_v;
    uint16_t *src_u  = (uint16_t *)src_u_v;
    uint16_t *src_v  = (uint16_t *)src_v_v;
    int inc  = stride;
    int inc2 = inc << 1;
    int inc3 = inc + inc2;
    int flag0 = flt_flag[0] ? -1 : 0;
    int flag1 = flt_flag[1] ? -1 : 0;

    __m128i ul0, ul1, ur0, ur1;
    __m128i tl0, tl1, tl2;
    __m128i tr0, tr1, tr2;
    __m128i t0, t1, t2;
    __m128i v0, v1, v2, v3;
    __m128i m0, m1, m2;
    __m128i flt_l, flt_r, flt, fs;
    __m128i fs4, fs56;

    __m128i ALPHA = _mm_set1_epi16((short)alpha);
    __m128i BETA  = _mm_set1_epi16((short)beta);
    __m128i c_0 = _mm_set1_epi16(0);
    __m128i c_1 = _mm_set1_epi16(1);
    __m128i c_2 = _mm_set1_epi16(2);
    __m128i c_3 = _mm_set1_epi16(3);
    __m128i c_4 = _mm_set1_epi16(4);
    __m128i c_8 = _mm_set1_epi16(8);

    /* 加载 U/V 各 4 像素, 交错 (低 64=U, 高 64=V) */
    if (is_8bit) {
        /* 8-bit: 加载 4 字节→扩展为 4 个 uint16 (低 64 位) */
        __m128i u, v;
        u = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(src_u8 - inc)));
        v = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(src_v8 - inc)));
        tl0 = _mm_unpacklo_epi64(u, v);
        u = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(src_u8 - inc2)));
        v = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(src_v8 - inc2)));
        tl1 = _mm_unpacklo_epi64(u, v);
        u = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(src_u8 - inc3)));
        v = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(src_v8 - inc3)));
        tl2 = _mm_unpacklo_epi64(u, v);
        u = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(src_u8)));
        v = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(src_v8)));
        tr0 = _mm_unpacklo_epi64(u, v);
        u = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(src_u8 + inc)));
        v = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(src_v8 + inc)));
        tr1 = _mm_unpacklo_epi64(u, v);
        u = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(src_u8 + inc2)));
        v = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*(const int*)(src_v8 + inc2)));
        tr2 = _mm_unpacklo_epi64(u, v);
    } else {
        /* 10-bit: 加载 8 字节 (4 个 uint16) */
        tl0 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i*)(src_u - inc)),  _mm_loadl_epi64((__m128i*)(src_v - inc)));
        tl1 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i*)(src_u - inc2)), _mm_loadl_epi64((__m128i*)(src_v - inc2)));
        tl2 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i*)(src_u - inc3)), _mm_loadl_epi64((__m128i*)(src_v - inc3)));
        tr0 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i*)(src_u)),        _mm_loadl_epi64((__m128i*)(src_v)));
        tr1 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i*)(src_u + inc)),  _mm_loadl_epi64((__m128i*)(src_v + inc)));
        tr2 = _mm_unpacklo_epi64(_mm_loadl_epi64((__m128i*)(src_u + inc2)), _mm_loadl_epi64((__m128i*)(src_v + inc2)));
    }

#define _mm_subabs_epu16(a, b) _mm_abs_epi16(_mm_subs_epi16(a, b))

    /* --- 主掩码 m0 = flt_flag && (|L0-R0|>1) && (|L0-R0|<alpha) --- */
    t0 = _mm_subabs_epu16(tl0, tr0);
    t1 = _mm_cmpgt_epi16(t0, c_1);
    t2 = _mm_cmpgt_epi16(ALPHA, t0);
    m0 = _mm_set_epi32(flag1, flag0, flag1, flag0);
    m0 = _mm_and_si128(m0, _mm_and_si128(t1, t2));

    /* --- 平坦度 flt = flat_l + flat_r --- */
    t0 = _mm_subabs_epu16(tl1, tl0);
    t1 = _mm_subabs_epu16(tr1, tr0);
    flt_l = _mm_and_si128(_mm_cmpgt_epi16(BETA, t0), c_2);
    flt_r = _mm_and_si128(_mm_cmpgt_epi16(BETA, t1), c_2);

    t0 = _mm_subabs_epu16(tl2, tl0);
    t1 = _mm_subabs_epu16(tr2, tr0);
    m1 = _mm_cmpgt_epi16(BETA, t0);
    m2 = _mm_cmpgt_epi16(BETA, t1);
    flt_l = _mm_add_epi16(_mm_and_si128(m1, c_1), flt_l);
    flt_r = _mm_add_epi16(_mm_and_si128(m2, c_1), flt_r);
    flt = _mm_add_epi16(flt_l, flt_r);

    /* --- FS 计算 (色度: 无 FS3, FS4=0/1, 对应亮度 fs 减 1) --- */
    m1 = _mm_and_si128(_mm_cmpeq_epi16(tr0, tr1), _mm_cmpeq_epi16(tl0, tl1));
    t0 = _mm_subs_epi16(flt, c_3);   /* FLT-3 (饱和) */
    t1 = _mm_subs_epi16(flt, c_4);   /* FLT-4 (饱和) */

    fs56 = _mm_blendv_epi8(t1, t0, m1);               /* M1 ? FLT-3 : FLT-4 */
    fs4  = _mm_blendv_epi8(c_0, c_1, _mm_cmpeq_epi16(flt_l, c_2));  /* flat_l==2 ? 1 : 0 */

    fs = _mm_blendv_epi8(c_0, fs56, _mm_cmpgt_epi16(flt, c_4));  /* FLT>4 */
    fs = _mm_blendv_epi8(fs, fs4, _mm_cmpeq_epi16(flt, c_4));    /* FLT==4 */
    /* FLT==3: 无滤波 (色度 fs=0, 对应亮度 fs=1 减 1) */

    fs = _mm_and_si128(fs, m0);

    /* 全部像素无需滤波, 数据未修改, 跳过滤波计算和写回 */
    if (_mm_testz_si128(fs, fs)) return;

#undef _mm_subabs_epu16

    ur0 = tr0;  ur1 = tr1;
    ul0 = tl0;  ul1 = tl1;

    /* fs==1: L0'=(3*L0+R0+2)>>2, R0'=(3*R0+L0+2)>>2 */
    t2 = _mm_add_epi16(_mm_add_epi16(tl0, tr0), c_2);
    v0 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(tl0, 1), t2), 2);
    v1 = _mm_srli_epi16(_mm_add_epi16(_mm_slli_epi16(tr0, 1), t2), 2);
    ul0 = _mm_blendv_epi8(tl0, v0, _mm_cmpeq_epi16(fs, c_1));
    ur0 = _mm_blendv_epi8(tr0, v1, _mm_cmpeq_epi16(fs, c_1));

    /* fs==2: L0'=(8*L0+3*L1+3*R0+L1+R0+4)>>4, R0'=(8*R0+3*R1+3*L0+R1+L0+4)>>4 */
    t2 = _mm_slli_epi16(t2, 1);  /* 2*(L0+R0+2) */

    t0 = _mm_add_epi16(_mm_slli_epi16(tl1, 1), _mm_add_epi16(tl1, tr0));
    t0 = _mm_add_epi16(_mm_slli_epi16(tl0, 3), _mm_add_epi16(t0, t2));
    v0 = _mm_srli_epi16(_mm_add_epi16(t0, c_4), 4);

    t0 = _mm_add_epi16(_mm_slli_epi16(tr1, 1), _mm_add_epi16(tr1, tl0));
    t0 = _mm_add_epi16(_mm_slli_epi16(tr0, 3), _mm_add_epi16(t0, t2));
    v1 = _mm_srli_epi16(_mm_add_epi16(t0, c_4), 4);

    ul0 = _mm_blendv_epi8(ul0, v0, _mm_cmpeq_epi16(fs, c_2));
    ur0 = _mm_blendv_epi8(ur0, v1, _mm_cmpeq_epi16(fs, c_2));

    /* fs==3: L0'=(2*L0+4*L1+4*L0+L2+4*R0+R1+8)>>4, etc. */
    t2 = _mm_slli_epi16(t2, 1);  /* 4*(L0+R0+2) */

    t0 = _mm_add_epi16(_mm_slli_epi16(tl1, 2), _mm_add_epi16(tl2, tr1));
    t0 = _mm_add_epi16(_mm_slli_epi16(tl0, 1), _mm_add_epi16(t0, t2));
    v0 = _mm_srli_epi16(t0, 4);

    t0 = _mm_add_epi16(_mm_slli_epi16(tr1, 2), _mm_add_epi16(tr2, tl1));
    t0 = _mm_add_epi16(_mm_slli_epi16(tr0, 1), _mm_add_epi16(t0, t2));
    v1 = _mm_srli_epi16(t0, 4);

    ul0 = _mm_blendv_epi8(ul0, v0, _mm_cmpeq_epi16(fs, c_3));
    ur0 = _mm_blendv_epi8(ur0, v1, _mm_cmpeq_epi16(fs, c_3));

    /* fs==3: L1'=(3*L2+8*L1+4*L0+R0+8)>>4 */
    t0 = _mm_add_epi16(_mm_add_epi16(tl2, tr0), _mm_slli_epi16(tl2, 1));
    t0 = _mm_add_epi16(t0, _mm_slli_epi16(tl1, 3));
    t0 = _mm_add_epi16(t0, _mm_slli_epi16(tl0, 2));
    v2 = _mm_srli_epi16(_mm_add_epi16(t0, c_8), 4);

    t0 = _mm_add_epi16(_mm_add_epi16(tr2, tl0), _mm_slli_epi16(tr2, 1));
    t0 = _mm_add_epi16(t0, _mm_slli_epi16(tr1, 3));
    t0 = _mm_add_epi16(t0, _mm_slli_epi16(tr0, 2));
    v3 = _mm_srli_epi16(_mm_add_epi16(t0, c_8), 4);

    ul1 = _mm_blendv_epi8(ul1, v2, _mm_cmpeq_epi16(fs, c_3));
    ur1 = _mm_blendv_epi8(ur1, v3, _mm_cmpeq_epi16(fs, c_3));

    /* 存储: 低 64 位=U, 高 64 位=V */
    if (is_8bit) {
        /* 8-bit: 低/高 64 位各 4 uint16 → packus 压为 4 uint8 → 存 4 字节 */
        __m128i z = _mm_setzero_si128();
        *(uint32_t*)(src_u8 - inc)  = _mm_cvtsi128_si32(_mm_packus_epi16(ul0, z));
        *(uint32_t*)(src_u8)        = _mm_cvtsi128_si32(_mm_packus_epi16(ur0, z));
        *(uint32_t*)(src_u8 - inc2) = _mm_cvtsi128_si32(_mm_packus_epi16(ul1, z));
        *(uint32_t*)(src_u8 + inc)  = _mm_cvtsi128_si32(_mm_packus_epi16(ur1, z));
        *(uint32_t*)(src_v8 - inc)  = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_unpackhi_epi64(ul0, ul0), z));
        *(uint32_t*)(src_v8)        = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_unpackhi_epi64(ur0, ur0), z));
        *(uint32_t*)(src_v8 - inc2) = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_unpackhi_epi64(ul1, ul1), z));
        *(uint32_t*)(src_v8 + inc)  = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_unpackhi_epi64(ur1, ur1), z));
    } else {
        /* 10-bit: 直接存 8 字节 (4 个 uint16) */
        *(uint64_t*)(src_u - inc)  = _mm_extract_epi64(ul0, 0);
        *(uint64_t*)(src_u)        = _mm_extract_epi64(ur0, 0);
        *(uint64_t*)(src_u - inc2) = _mm_extract_epi64(ul1, 0);
        *(uint64_t*)(src_u + inc)  = _mm_extract_epi64(ur1, 0);
        *(uint64_t*)(src_v - inc)  = _mm_extract_epi64(ul0, 1);
        *(uint64_t*)(src_v)        = _mm_extract_epi64(ur0, 1);
        *(uint64_t*)(src_v - inc2) = _mm_extract_epi64(ul1, 1);
        *(uint64_t*)(src_v + inc)  = _mm_extract_epi64(ur1, 1);
    }
}

/* ===========================================================================
 * 验证模式: 对比 SIMD 与 C 参考实现的输出差异
 * =========================================================================== */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static int g_lf_verify = -1;  /* -1=未初始化, 0=关闭, 1=开启 */

static int lf_verify_enabled(void)
{
    if (g_lf_verify < 0) {
        const char *env = getenv("AVS2_LF_VERIFY");
        g_lf_verify = (env && atoi(env) > 0) ? 1 : 0;
    }
    return g_lf_verify;
}

/* 亮度垂直边验证包装 */
static void deblock_luma_ver_verify(void *src, int stride, int alpha, int beta,
                                    uint8_t *flt_flag, int bit_depth)
{
    if (!lf_verify_enabled()) {
        deblock_luma_ver_sse4(src, stride, alpha, beta, flt_flag, bit_depth);
        return;
    }

    /* 保存原始数据: 8 行 × 8 像素 (从 src-3 开始) */
    int bps = (bit_depth > 8) ? 2 : 1;
    int row_bytes = 8 * bps;
    uint8_t backup[8 * 64];  /* 足够大 */
    uint8_t simd_result[8 * 64];
    uint8_t c_result[8 * 64];

    for (int i = 0; i < 8; i++) {
        memcpy(backup + i * row_bytes,
               (uint8_t*)src - 3 * bps + i * stride * bps, row_bytes);
    }

    /* 运行 SIMD */
    deblock_luma_ver_sse4(src, stride, alpha, beta, flt_flag, bit_depth);

    /* 保存 SIMD 结果 */
    for (int i = 0; i < 8; i++) {
        memcpy(simd_result + i * row_bytes,
               (uint8_t*)src - 3 * bps + i * stride * bps, row_bytes);
    }

    /* 恢复原始数据 */
    for (int i = 0; i < 8; i++) {
        memcpy((uint8_t*)src - 3 * bps + i * stride * bps,
               backup + i * row_bytes, row_bytes);
    }

    /* 运行 C 参考 */
    deblock_luma_ver(src, stride, alpha, beta, flt_flag, bit_depth);

    /* 保存 C 结果 */
    for (int i = 0; i < 8; i++) {
        memcpy(c_result + i * row_bytes,
               (uint8_t*)src - 3 * bps + i * stride * bps, row_bytes);
    }

    /* 比较 (仅 C 修改的 6 像素: 位置 -3..+2) */
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 6; j++) {
            int off = (i * 8 + j) * bps;
            if (memcmp(simd_result + off, c_result + off, bps) != 0) {
                int pos = j - 3;  /* 相对边界的位置 */
                if (bps == 2) {
                    uint16_t sv = *(uint16_t*)(simd_result + off);
                    uint16_t cv = *(uint16_t*)(c_result + off);
                    fprintf(stderr, "LF_VERIFY LUMA_VER diff: row=%d pos=%d "
                            "simd=%d c=%d alpha=%d beta=%d flag=[%d,%d]\n",
                            i, pos, sv, cv, alpha, beta,
                            flt_flag[0], flt_flag[1]);
                } else {
                    fprintf(stderr, "LF_VERIFY LUMA_VER diff: row=%d pos=%d "
                            "simd=%d c=%d alpha=%d beta=%d flag=[%d,%d]\n",
                            i, pos, simd_result[off], c_result[off],
                            alpha, beta, flt_flag[0], flt_flag[1]);
                }
                /* 恢复为 SIMD 结果 */
                goto restore_simd;
            }
        }
    }
restore_simd:
    /* 恢复为 SIMD 结果 */
    for (int i = 0; i < 8; i++) {
        memcpy((uint8_t*)src - 3 * bps + i * stride * bps,
               simd_result + i * row_bytes, row_bytes);
    }
}

/* 亮度水平边验证包装 */
static void deblock_luma_hor_verify(void *src, int stride, int alpha, int beta,
                                    uint8_t *flt_flag, int bit_depth)
{
    if (!lf_verify_enabled()) {
        deblock_luma_hor_sse4(src, stride, alpha, beta, flt_flag, bit_depth);
        return;
    }

    int bps = (bit_depth > 8) ? 2 : 1;
    int row_bytes = 8 * bps;
    uint8_t backup[6 * 64];
    uint8_t simd_result[6 * 64];
    uint8_t c_result[6 * 64];
    int offsets[6] = { -3, -2, -1, 0, 1, 2 };  /* L2..R2 相对 src 的行偏移 */

    for (int i = 0; i < 6; i++) {
        memcpy(backup + i * row_bytes,
               (uint8_t*)src + offsets[i] * stride * bps, row_bytes);
    }

    deblock_luma_hor_sse4(src, stride, alpha, beta, flt_flag, bit_depth);

    for (int i = 0; i < 6; i++) {
        memcpy(simd_result + i * row_bytes,
               (uint8_t*)src + offsets[i] * stride * bps, row_bytes);
    }

    for (int i = 0; i < 6; i++) {
        memcpy((uint8_t*)src + offsets[i] * stride * bps,
               backup + i * row_bytes, row_bytes);
    }

    deblock_luma_hor(src, stride, alpha, beta, flt_flag, bit_depth);

    for (int i = 0; i < 6; i++) {
        memcpy(c_result + i * row_bytes,
               (uint8_t*)src + offsets[i] * stride * bps, row_bytes);
    }

    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 8; j++) {
            int off = (i * 8 + j) * bps;
            if (memcmp(simd_result + off, c_result + off, bps) != 0) {
                if (bps == 2) {
                    uint16_t sv = *(uint16_t*)(simd_result + off);
                    uint16_t cv = *(uint16_t*)(c_result + off);
                    fprintf(stderr, "LF_VERIFY LUMA_HOR diff: row_offset=%d pix=%d "
                            "simd=%d c=%d alpha=%d beta=%d flag=[%d,%d]\n",
                            offsets[i], j, sv, cv, alpha, beta,
                            flt_flag[0], flt_flag[1]);
                } else {
                    fprintf(stderr, "LF_VERIFY LUMA_HOR diff: row_offset=%d pix=%d "
                            "simd=%d c=%d alpha=%d beta=%d flag=[%d,%d]\n",
                            offsets[i], j, simd_result[off], c_result[off],
                            alpha, beta, flt_flag[0], flt_flag[1]);
                }
                goto restore_simd_hor;
            }
        }
    }
restore_simd_hor:
    for (int i = 0; i < 6; i++) {
        memcpy((uint8_t*)src + offsets[i] * stride * bps,
               simd_result + i * row_bytes, row_bytes);
    }
}

/* 色度垂直边验证包装 */
static void deblock_chroma_ver_verify(void *src_u, void *src_v, int stride,
                                      int alpha, int beta, uint8_t *flt_flag,
                                      int bit_depth)
{
    if (!lf_verify_enabled()) {
        deblock_chroma_ver_sse4(src_u, src_v, stride, alpha, beta, flt_flag, bit_depth);
        return;
    }

    int bps = (bit_depth > 8) ? 2 : 1;
    int row_bytes = 8 * bps;
    uint8_t backup_u[4 * 64], backup_v[4 * 64];
    uint8_t simd_u[4 * 64], simd_v[4 * 64];
    uint8_t c_u[4 * 64], c_v[4 * 64];

    for (int i = 0; i < 4; i++) {
        memcpy(backup_u + i * row_bytes,
               (uint8_t*)src_u - 4 * bps + i * stride * bps, row_bytes);
        memcpy(backup_v + i * row_bytes,
               (uint8_t*)src_v - 4 * bps + i * stride * bps, row_bytes);
    }

    deblock_chroma_ver_sse4(src_u, src_v, stride, alpha, beta, flt_flag, bit_depth);

    for (int i = 0; i < 4; i++) {
        memcpy(simd_u + i * row_bytes,
               (uint8_t*)src_u - 4 * bps + i * stride * bps, row_bytes);
        memcpy(simd_v + i * row_bytes,
               (uint8_t*)src_v - 4 * bps + i * stride * bps, row_bytes);
    }

    for (int i = 0; i < 4; i++) {
        memcpy((uint8_t*)src_u - 4 * bps + i * stride * bps,
               backup_u + i * row_bytes, row_bytes);
        memcpy((uint8_t*)src_v - 4 * bps + i * stride * bps,
               backup_v + i * row_bytes, row_bytes);
    }

    deblock_chroma_ver(src_u, src_v, stride, alpha, beta, flt_flag, bit_depth);

    for (int i = 0; i < 4; i++) {
        memcpy(c_u + i * row_bytes,
               (uint8_t*)src_u - 4 * bps + i * stride * bps, row_bytes);
        memcpy(c_v + i * row_bytes,
               (uint8_t*)src_v - 4 * bps + i * stride * bps, row_bytes);
    }

    /* 比较 U 和 V (C 修改的 6 像素: 位置 -3..+2) */
    static int s_cv_diff_count = 0;
    for (int plane = 0; plane < 2; plane++) {
        uint8_t *sr = (plane == 0) ? simd_u : simd_v;
        uint8_t *cr = (plane == 0) ? c_u : c_v;
        uint8_t *orig = (plane == 0) ? backup_u : backup_v;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 6; j++) {
                int off = (i * 8 + j) * bps;
                if (memcmp(sr + off, cr + off, bps) != 0) {
                    int pos = j - 4;  /* 相对边界的位置 */
                    if (bps == 2 && s_cv_diff_count < 5) {
                        s_cv_diff_count++;
                        uint16_t sv = *(uint16_t*)(sr + off);
                        uint16_t cv = *(uint16_t*)(cr + off);
                        /* 打印输入像素: L2,L1,L0,R0,R1,R2 = 位置 -3..+2 */
                        uint16_t L2 = *(uint16_t*)(orig + (i * 8 + 1) * bps);
                        uint16_t L1 = *(uint16_t*)(orig + (i * 8 + 2) * bps);
                        uint16_t L0 = *(uint16_t*)(orig + (i * 8 + 3) * bps);
                        uint16_t R0 = *(uint16_t*)(orig + (i * 8 + 4) * bps);
                        uint16_t R1 = *(uint16_t*)(orig + (i * 8 + 5) * bps);
                        uint16_t R2 = *(uint16_t*)(orig + (i * 8 + 6) * bps);
                        /* 手动计算 FS */
                        int fl = 0, fr = 0, fs_manual;
                        fl = (abs(L1 - L0) < beta) ? 2 : 0;
                        fl += (abs(L2 - L0) < beta) ? 1 : 0;
                        fr = (abs(R0 - R1) < beta) ? 2 : 0;
                        fr += (abs(R0 - R2) < beta) ? 1 : 0;
                        int flt = fl + fr;
                        int m1 = (R1 == R0) && (L0 == L1);
                        switch (flt) {
                            case 6: fs_manual = 3 + m1; break;
                            case 5: fs_manual = 2 + m1; break;
                            case 4: fs_manual = 1 + (fl == 2); break;
                            case 3: fs_manual = (abs(L1 - R1) < beta); break;
                            default: fs_manual = 0; break;
                        }
                        fs_manual -= (fs_manual > 0);
                        fprintf(stderr, "LF_VERIFY CHROMA_VER diff: plane=%c row=%d pos=%d "
                                "simd=%d c=%d alpha=%d beta=%d flag=[%d,%d] "
                                "L2=%d L1=%d L0=%d R0=%d R1=%d R2=%d "
                                "flt=%d fl=%d fr=%d m1=%d fs=%d\n",
                                plane ? 'V' : 'U', i, pos, sv, cv, alpha, beta,
                                flt_flag[0], flt_flag[1],
                                L2, L1, L0, R0, R1, R2,
                                flt, fl, fr, m1, fs_manual);
                    }
                    goto restore_simd_cv;
                }
            }
        }
    }
restore_simd_cv:
    for (int i = 0; i < 4; i++) {
        memcpy((uint8_t*)src_u - 4 * bps + i * stride * bps,
               simd_u + i * row_bytes, row_bytes);
        memcpy((uint8_t*)src_v - 4 * bps + i * stride * bps,
               simd_v + i * row_bytes, row_bytes);
    }
}

/* 色度水平边验证包装 */
static void deblock_chroma_hor_verify(void *src_u, void *src_v, int stride,
                                      int alpha, int beta, uint8_t *flt_flag,
                                      int bit_depth)
{
    if (!lf_verify_enabled()) {
        deblock_chroma_hor_sse4(src_u, src_v, stride, alpha, beta, flt_flag, bit_depth);
        return;
    }

    int bps = (bit_depth > 8) ? 2 : 1;
    int npx = 4;  /* 色度每行 4 像素 */
    int row_bytes = npx * bps;
    uint8_t backup_u[6 * 32], backup_v[6 * 32];
    uint8_t simd_u[6 * 32], simd_v[6 * 32];
    uint8_t c_u[6 * 32], c_v[6 * 32];
    int offsets[6] = { -3, -2, -1, 0, 1, 2 };

    for (int i = 0; i < 6; i++) {
        memcpy(backup_u + i * row_bytes,
               (uint8_t*)src_u + offsets[i] * stride * bps, row_bytes);
        memcpy(backup_v + i * row_bytes,
               (uint8_t*)src_v + offsets[i] * stride * bps, row_bytes);
    }

    deblock_chroma_hor_sse4(src_u, src_v, stride, alpha, beta, flt_flag, bit_depth);

    for (int i = 0; i < 6; i++) {
        memcpy(simd_u + i * row_bytes,
               (uint8_t*)src_u + offsets[i] * stride * bps, row_bytes);
        memcpy(simd_v + i * row_bytes,
               (uint8_t*)src_v + offsets[i] * stride * bps, row_bytes);
    }

    for (int i = 0; i < 6; i++) {
        memcpy((uint8_t*)src_u + offsets[i] * stride * bps,
               backup_u + i * row_bytes, row_bytes);
        memcpy((uint8_t*)src_v + offsets[i] * stride * bps,
               backup_v + i * row_bytes, row_bytes);
    }

    deblock_chroma_hor(src_u, src_v, stride, alpha, beta, flt_flag, bit_depth);

    for (int i = 0; i < 6; i++) {
        memcpy(c_u + i * row_bytes,
               (uint8_t*)src_u + offsets[i] * stride * bps, row_bytes);
        memcpy(c_v + i * row_bytes,
               (uint8_t*)src_v + offsets[i] * stride * bps, row_bytes);
    }

    for (int plane = 0; plane < 2; plane++) {
        uint8_t *sr = (plane == 0) ? simd_u : simd_v;
        uint8_t *cr = (plane == 0) ? c_u : c_v;
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < npx; j++) {
                int off = (i * npx + j) * bps;
                if (memcmp(sr + off, cr + off, bps) != 0) {
                    if (bps == 2) {
                        uint16_t sv = *(uint16_t*)(sr + off);
                        uint16_t cv = *(uint16_t*)(cr + off);
                        fprintf(stderr, "LF_VERIFY CHROMA_HOR diff: plane=%c row_off=%d pix=%d "
                                "simd=%d c=%d alpha=%d beta=%d flag=[%d,%d]\n",
                                plane ? 'V' : 'U', offsets[i], j, sv, cv, alpha, beta,
                                flt_flag[0], flt_flag[1]);
                    } else {
                        fprintf(stderr, "LF_VERIFY CHROMA_HOR diff: plane=%c row_off=%d pix=%d "
                                "simd=%d c=%d alpha=%d beta=%d flag=[%d,%d]\n",
                                plane ? 'V' : 'U', offsets[i], j, sr[off], cr[off],
                                alpha, beta, flt_flag[0], flt_flag[1]);
                    }
                    goto restore_simd_ch;
                }
            }
        }
    }
restore_simd_ch:
    for (int i = 0; i < 6; i++) {
        memcpy((uint8_t*)src_u + offsets[i] * stride * bps,
               simd_u + i * row_bytes, row_bytes);
        memcpy((uint8_t*)src_v + offsets[i] * stride * bps,
               simd_v + i * row_bytes, row_bytes);
    }
}

/* ===========================================================================
 * 注册函数
 * =========================================================================== */

/* SSE4.1: 注册亮度+色度去块滤波 */
void avs2_lf_init_sse41(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.deblock_luma[EDGE_VER]   = deblock_luma_ver_verify;
    avs2_dsp_table.deblock_luma[EDGE_HOR]   = deblock_luma_hor_verify;
    avs2_dsp_table.deblock_chroma[EDGE_VER] = deblock_chroma_ver_verify;
    avs2_dsp_table.deblock_chroma[EDGE_HOR] = deblock_chroma_hor_verify;
}

/* AVX2: 已降级为 SSE4, 无额外注册 */
void avs2_lf_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }

/* AVX512 预留 */
void avs2_lf_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }

#else /* 非 x86 平台 */

void avs2_lf_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_lf_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_lf_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }

#endif
