/*
 * mc_simd.c - 运动补偿 SIMD 实现 (x86 SSE4.1)
 *
 * 当前实现:
 *   - SSE4.1: 10-bit/8-bit 亮度 8 抽头 / 色度 4 抽头插值
 *   - SSE4.1: 8-bit 块拷贝 (block_copy) / 双向预测平均 (bi_avg) / 块填充 (fill_block)
 *
 * 对齐要求:
 *   - avs2_mem_alloc/allocz 统一返回 32 字节对齐内存
 *   - 参考帧/目标帧数据由 avs2_frame_alloc 分配, 保证 32 字节对齐
 *   - 中间缓冲区用 AVS2_ALIGN16 声明, 步长 (MC_TMP_STRIDE=64) 保证行起始 16 字节对齐,
 *     因此对中间缓冲区的访问使用 _mm_load_si128 / _mm_store_si128 (aligned)
 *   - src (参考帧, 运动矢量偏移): MV 整数部分可为任意值, src 处于任意像素偏移,
 *     无法保证对齐, 因此使用 _mm_loadu_si128 (unaligned)
 *   - dst (目标帧, CU 位置): 10-bit 下 CU 位置为 8 像素倍数 = 16 字节对齐,
 *     128-bit 存储使用 _mm_store_si128 (aligned);
 *     8-bit 下 dst 仅 8 字节对齐, 128-bit 存储使用 _mm_storeu_si128 (unaligned)
 *   - 系数表为 int8_t, 使用 _mm_cvtepi8_epi16 扩展为 16-bit
 *
 * 算法说明:
 *   - 亮度: 8 抽头滤波, 1/4 像素精度, 系数之和 64, shift=6
 *   - 色度: 4 抽头滤波, 1/8 像素精度, 系数之和 64, shift=6
 *   - 双向插值 (ext): 先水平后垂直
 *     shift1 = bit_depth - 8 (10-bit 时为 2)
 *     shift2 = 20 - bit_depth (10-bit 时为 10)
 *
 * 就地安全: src != dst (运动补偿不会就地操作)
 */

#include "internal.h"
#include "tables.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <tmmintrin.h>
#include <smmintrin.h>

/* ---- 对齐宏 ---- */
#if defined(_MSC_VER)
#define AVS2_ALIGN32(x) __declspec(align(32)) x
#define AVS2_ALIGN16(x) __declspec(align(16)) x
#else
#define AVS2_ALIGN32(x) x __attribute__((aligned(32)))
#define AVS2_ALIGN16(x) x __attribute__((aligned(16)))
#endif

/* ---- C 回退函数声明 (在 mc.c 中定义) ---- */
extern void mc_luma_c(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                      ptrdiff_t dstride, int w, int h, int mx, int my,
                      int bit_depth);
extern void mc_chroma_c(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                        ptrdiff_t dstride, int w, int h, int mx, int my,
                        int bit_depth);
extern void mc_luma_avg_c(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                          ptrdiff_t dstride, int w, int h, int mx, int my,
                          int bit_depth);
extern void mc_chroma_avg_c(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                            ptrdiff_t dstride, int w, int h, int mx, int my,
                            int bit_depth);

/* 10-bit 像素类型 */
typedef uint16_t pel_t;

/* 中间缓冲区步长 (覆盖最大块 64x64) */
#define MC_TMP_STRIDE 64

/* ===========================================================================
 * 第一部分: 块拷贝
 * =========================================================================== */

/* 整像素块拷贝 (10-bit, SSE4.1)
 * width 取值: 4,8,16,32,64 (亮度); 2,4,8,16,32 (色度)
 * 使用 _mm_loadu_si128 / _mm_store_si128 (128-bit = 8 个 uint16) */
static void block_copy_10bit_sse4(pel_t *dst, int i_dst,
                                   const pel_t *src, int i_src,
                                   int width, int height)
{
    int y, x;
    for (y = 0; y < height; y++) {
        if (y + 8 < height)
            _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
        for (x = 0; x < width; x += 8) {
            __m128i v = _mm_loadu_si128((const __m128i*)(src + x));
            _mm_storeu_si128((__m128i*)(dst + x), v);
        }
        /* 尾部标量 */
        for (; x < width; x++) {
            dst[x] = src[x];
        }
        src += i_src;
        dst += i_dst;
    }
}

/* 整像素块拷贝 (8-bit, SSE4.1) */
static void block_copy_8bit_sse41(uint8_t *dst, int i_dst,
                                    const uint8_t *src, int i_src,
                                    int width, int height)
{
    int y, x;

    if ((width & 15) == 0) {
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 16) {
                __m128i v = _mm_loadu_si128((const __m128i*)(src + x));
                _mm_storeu_si128((__m128i*)(dst + x), v);
            }
            src += i_src;
            dst += i_dst;
        }
    } else if ((width & 7) == 0) {
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 8) {
                __m128i v = _mm_loadl_epi64((const __m128i*)(src + x));
                _mm_storel_epi64((__m128i*)(dst + x), v);
            }
            src += i_src;
            dst += i_dst;
        }
    } else if ((width & 3) == 0) {
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 4) {
                *(uint32_t*)(dst + x) = *(const uint32_t*)(src + x);
            }
            src += i_src;
            dst += i_dst;
        }
    } else {
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x++) {
                dst[x] = src[x];
            }
            src += i_src;
            dst += i_dst;
        }
    }
}

/* ===========================================================================
 * 第二部分: 亮度 8 抽头水平插值 (10-bit SSE4.1)
 *
 * 算法: 对每个输出像素, 取 src[x-3..x+4] 共 8 个像素,
 *       与 8 个 int8_t 系数相乘累加, 加 32 后右移 6 位, 裁剪到 [0, max_val]
 *
 * SSE4.1 实现: 每次处理 8 个输出像素
 *   - 加载 8 个连续的 128-bit 窗口 (8 像素/窗口, p++ 滑动 1 像素)
 *   - 每个窗口与 mCoef [c0..c7] 做 madd_epi16 得 4 个 int32
 *   - 3 级 hadd 合并为 8 个 int32 结果
 * =========================================================================== */
static void ip_filter_luma_hor_10bit_sse4(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height, const int8_t *coeff, int max_val)
{
    int j, i;
    __m128i max_val1 = _mm_set1_epi16((short)max_val);
    __m128i offset = _mm_set1_epi32(32);
    __m128i mCoef = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)coeff));

    src -= 3;

    for (j = 0; j < height; j++) {
        const pel_t *p = src;
        if (j + 8 < height)
            _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
        for (i = 0; i + 7 < width; i += 8) {
            __m128i T0, T1, T2, T3, T4, T5, T6, T7;
            __m128i M0, M1, M2, M3, M4, M5, M6, M7;
            __m128i A0, A1, A2, A3, B0, B1;

            T0 = _mm_loadu_si128((__m128i*)p++);
            T1 = _mm_loadu_si128((__m128i*)p++);
            T2 = _mm_loadu_si128((__m128i*)p++);
            T3 = _mm_loadu_si128((__m128i*)p++);
            T4 = _mm_loadu_si128((__m128i*)p++);
            T5 = _mm_loadu_si128((__m128i*)p++);
            T6 = _mm_loadu_si128((__m128i*)p++);
            T7 = _mm_loadu_si128((__m128i*)p++);

            M0 = _mm_madd_epi16(T0, mCoef);
            M1 = _mm_madd_epi16(T1, mCoef);
            M2 = _mm_madd_epi16(T2, mCoef);
            M3 = _mm_madd_epi16(T3, mCoef);
            M4 = _mm_madd_epi16(T4, mCoef);
            M5 = _mm_madd_epi16(T5, mCoef);
            M6 = _mm_madd_epi16(T6, mCoef);
            M7 = _mm_madd_epi16(T7, mCoef);

            A0 = _mm_hadd_epi32(M0, M1);
            A1 = _mm_hadd_epi32(M2, M3);
            A2 = _mm_hadd_epi32(M4, M5);
            A3 = _mm_hadd_epi32(M6, M7);

            B0 = _mm_hadd_epi32(A0, A1);
            B1 = _mm_hadd_epi32(A2, A3);

            B0 = _mm_add_epi32(B0, offset);
            B1 = _mm_add_epi32(B1, offset);
            B0 = _mm_srai_epi32(B0, 6);
            B1 = _mm_srai_epi32(B1, 6);
            B0 = _mm_packus_epi32(B0, B1);
            B0 = _mm_min_epu16(B0, max_val1);
            _mm_storeu_si128((__m128i*)(dst + i), B0);
        }
        for (; i < width; i++) {
            int v = src[i]     * coeff[0] + src[i + 1] * coeff[1]
                  + src[i + 2] * coeff[2] + src[i + 3] * coeff[3]
                  + src[i + 4] * coeff[4] + src[i + 5] * coeff[5]
                  + src[i + 6] * coeff[6] + src[i + 7] * coeff[7];
            v = (v + 32) >> 6;
            dst[i] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
        }
        dst += i_dst;
        src += i_src;
    }
}

/* ===========================================================================
 * 第三部分: 亮度 8 抽头垂直插值 (10-bit SSE4.1)
 *
 * 算法: 对每个输出像素, 取 src[y-3..y+4] 共 8 行同一列像素,
 *       与 8 个 int8_t 系数相乘累加
 *
 * SSE4.1 实现: 每次处理 8 个像素 (一行)
 *   - 加载 8 行各 8 像素, unpacklo/hi 交错相邻行
 *   - madd 与 4 组系数对相乘, 4 组部分和相加
 * =========================================================================== */
static void ip_filter_luma_ver_10bit_sse4(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height, const int8_t *coeff, int max_val)
{
    int i, j;
    __m128i max_val1 = _mm_set1_epi16((short)max_val);
    __m128i mAddOffset = _mm_set1_epi32(32);

    __m128i c0 = _mm_set1_epi16(*(const short*)(coeff + 0));
    __m128i c1 = _mm_set1_epi16(*(const short*)(coeff + 2));
    __m128i c2 = _mm_set1_epi16(*(const short*)(coeff + 4));
    __m128i c3 = _mm_set1_epi16(*(const short*)(coeff + 6));
    __m128i coeff00 = _mm_cvtepi8_epi16(c0);
    __m128i coeff01 = _mm_cvtepi8_epi16(c1);
    __m128i coeff02 = _mm_cvtepi8_epi16(c2);
    __m128i coeff03 = _mm_cvtepi8_epi16(c3);

    src -= 3 * i_src;

    for (j = 0; j < height; j++) {
        const pel_t *p = src;
        for (i = 0; i + 7 < width; i += 8) {
            __m128i T0, T1, T2, T3, T4, T5, T6, T7;
            __m128i M0_lo, M0_hi, M1_lo, M1_hi, M2_lo, M2_hi, M3_lo, M3_hi;
            __m128i N0_lo, N0_hi, N1_lo, N1_hi, N2_lo, N2_hi, N3_lo, N3_hi;
            __m128i sum_lo, sum_hi;

            _mm_prefetch((const char*)(p + 16 * i_src), _MM_HINT_T0);
            _mm_prefetch((const char*)(p + 18 * i_src), _MM_HINT_T0);

            T0 = _mm_loadu_si128((__m128i*)(p));
            T1 = _mm_loadu_si128((__m128i*)(p + i_src));
            T2 = _mm_loadu_si128((__m128i*)(p + 2 * i_src));
            T3 = _mm_loadu_si128((__m128i*)(p + 3 * i_src));
            T4 = _mm_loadu_si128((__m128i*)(p + 4 * i_src));
            T5 = _mm_loadu_si128((__m128i*)(p + 5 * i_src));
            T6 = _mm_loadu_si128((__m128i*)(p + 6 * i_src));
            T7 = _mm_loadu_si128((__m128i*)(p + 7 * i_src));

            M0_lo = _mm_unpacklo_epi16(T0, T1);
            M0_hi = _mm_unpackhi_epi16(T0, T1);
            M1_lo = _mm_unpacklo_epi16(T2, T3);
            M1_hi = _mm_unpackhi_epi16(T2, T3);
            M2_lo = _mm_unpacklo_epi16(T4, T5);
            M2_hi = _mm_unpackhi_epi16(T4, T5);
            M3_lo = _mm_unpacklo_epi16(T6, T7);
            M3_hi = _mm_unpackhi_epi16(T6, T7);

            N0_lo = _mm_madd_epi16(M0_lo, coeff00);
            N0_hi = _mm_madd_epi16(M0_hi, coeff00);
            N1_lo = _mm_madd_epi16(M1_lo, coeff01);
            N1_hi = _mm_madd_epi16(M1_hi, coeff01);
            N2_lo = _mm_madd_epi16(M2_lo, coeff02);
            N2_hi = _mm_madd_epi16(M2_hi, coeff02);
            N3_lo = _mm_madd_epi16(M3_lo, coeff03);
            N3_hi = _mm_madd_epi16(M3_hi, coeff03);

            sum_lo = _mm_add_epi32(N0_lo, N1_lo);
            sum_lo = _mm_add_epi32(sum_lo, N2_lo);
            sum_lo = _mm_add_epi32(sum_lo, N3_lo);

            sum_hi = _mm_add_epi32(N0_hi, N1_hi);
            sum_hi = _mm_add_epi32(sum_hi, N2_hi);
            sum_hi = _mm_add_epi32(sum_hi, N3_hi);

            sum_lo = _mm_add_epi32(sum_lo, mAddOffset);
            sum_hi = _mm_add_epi32(sum_hi, mAddOffset);
            sum_lo = _mm_srai_epi32(sum_lo, 6);
            sum_hi = _mm_srai_epi32(sum_hi, 6);

            sum_lo = _mm_packus_epi32(sum_lo, sum_hi);
            sum_lo = _mm_min_epu16(sum_lo, max_val1);
            _mm_storeu_si128((__m128i*)(dst + i), sum_lo);

            p += 8;
        }
        for (; i < width; i++) {
            int v = src[i]              * coeff[0]
                  + src[i + i_src]      * coeff[1]
                  + src[i + 2 * i_src]  * coeff[2]
                  + src[i + 3 * i_src]  * coeff[3]
                  + src[i + 4 * i_src]  * coeff[4]
                  + src[i + 5 * i_src]  * coeff[5]
                  + src[i + 6 * i_src]  * coeff[6]
                  + src[i + 7 * i_src]  * coeff[7];
            v = (v + 32) >> 6;
            dst[i] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ===========================================================================
 * 第四部分: 亮度 8 抽头双向插值 (10-bit SSE4.1)
 *
 * 算法: 先水平滤波到中间缓冲 (shift1), 再垂直滤波到目标 (shift2)
 *       shift1 = bit_depth - 8 = 2 (10-bit)
 *       shift2 = 20 - bit_depth = 10 (10-bit)
 * =========================================================================== */
static void ip_filter_luma_ext_10bit_sse4(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y, int max_val)
{
    AVS2_ALIGN16(short tmp_res[(64 + 7) * 64]);
    short *tmp = tmp_res;
    const int i_tmp = MC_TMP_STRIDE;
    int add1, shift1, add2, shift2;
    int i, j;
    __m128i offset;
    __m128i max_val1 = _mm_set1_epi16((short)max_val);
    __m128i mCoef = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)coef_x));
    __m128i c0, c1, c2, c3, coeff00, coeff01, coeff02, coeff03;

    shift1 = 2;
    shift2 = 10;
    add1 = (1 << shift1) >> 1;
    add2 = 1 << (shift2 - 1);

    /* ---- 第一级: 水平滤波 ---- */
    src += -3 * i_src - 3;
    offset = _mm_set1_epi32(add1);

    for (j = -3; j < height + 4; j++) {
        const pel_t *p = src;
        if (j + 8 < height + 4)
            _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
        for (i = 0; i + 7 < width; i += 8) {
            __m128i T0, T1, T2, T3, T4, T5, T6, T7;
            __m128i M0, M1, M2, M3, M4, M5, M6, M7;
            __m128i A0, A1, A2, A3, B0, B1;

            T0 = _mm_loadu_si128((__m128i*)p++);
            T1 = _mm_loadu_si128((__m128i*)p++);
            T2 = _mm_loadu_si128((__m128i*)p++);
            T3 = _mm_loadu_si128((__m128i*)p++);
            T4 = _mm_loadu_si128((__m128i*)p++);
            T5 = _mm_loadu_si128((__m128i*)p++);
            T6 = _mm_loadu_si128((__m128i*)p++);
            T7 = _mm_loadu_si128((__m128i*)p++);

            M0 = _mm_madd_epi16(T0, mCoef);
            M1 = _mm_madd_epi16(T1, mCoef);
            M2 = _mm_madd_epi16(T2, mCoef);
            M3 = _mm_madd_epi16(T3, mCoef);
            M4 = _mm_madd_epi16(T4, mCoef);
            M5 = _mm_madd_epi16(T5, mCoef);
            M6 = _mm_madd_epi16(T6, mCoef);
            M7 = _mm_madd_epi16(T7, mCoef);

            A0 = _mm_hadd_epi32(M0, M1);
            A1 = _mm_hadd_epi32(M2, M3);
            A2 = _mm_hadd_epi32(M4, M5);
            A3 = _mm_hadd_epi32(M6, M7);

            B0 = _mm_hadd_epi32(A0, A1);
            B1 = _mm_hadd_epi32(A2, A3);

            B0 = _mm_add_epi32(B0, offset);
            B1 = _mm_add_epi32(B1, offset);
            B0 = _mm_srai_epi32(B0, shift1);
            B1 = _mm_srai_epi32(B1, shift1);
            B0 = _mm_packs_epi32(B0, B1);
            _mm_store_si128((__m128i*)(tmp + i), B0);
        }
        for (; i < width; i++) {
            int v = src[i]     * coef_x[0] + src[i + 1] * coef_x[1]
                  + src[i + 2] * coef_x[2] + src[i + 3] * coef_x[3]
                  + src[i + 4] * coef_x[4] + src[i + 5] * coef_x[5]
                  + src[i + 6] * coef_x[6] + src[i + 7] * coef_x[7];
            v = (v + add1) >> shift1;
            tmp[i] = (short)v;
        }
        tmp += i_tmp;
        src += i_src;
    }

    /* ---- 第二级: 垂直滤波 ---- */
    offset = _mm_set1_epi32(add2);
    tmp = tmp_res;

    c0 = _mm_set1_epi16(*(const short*)(coef_y + 0));
    c1 = _mm_set1_epi16(*(const short*)(coef_y + 2));
    c2 = _mm_set1_epi16(*(const short*)(coef_y + 4));
    c3 = _mm_set1_epi16(*(const short*)(coef_y + 6));
    coeff00 = _mm_cvtepi8_epi16(c0);
    coeff01 = _mm_cvtepi8_epi16(c1);
    coeff02 = _mm_cvtepi8_epi16(c2);
    coeff03 = _mm_cvtepi8_epi16(c3);

    for (j = 0; j < height; j++) {
        const short *p = tmp;
        for (i = 0; i + 7 < width; i += 8) {
            __m128i T0, T1, T2, T3, T4, T5, T6, T7;
            __m128i M0_lo, M0_hi, M1_lo, M1_hi, M2_lo, M2_hi, M3_lo, M3_hi;
            __m128i N0_lo, N0_hi, N1_lo, N1_hi, N2_lo, N2_hi, N3_lo, N3_hi;
            __m128i sum_lo, sum_hi;

            T0 = _mm_load_si128((__m128i*)(p));
            T1 = _mm_load_si128((__m128i*)(p + i_tmp));
            T2 = _mm_load_si128((__m128i*)(p + 2 * i_tmp));
            T3 = _mm_load_si128((__m128i*)(p + 3 * i_tmp));
            T4 = _mm_load_si128((__m128i*)(p + 4 * i_tmp));
            T5 = _mm_load_si128((__m128i*)(p + 5 * i_tmp));
            T6 = _mm_load_si128((__m128i*)(p + 6 * i_tmp));
            T7 = _mm_load_si128((__m128i*)(p + 7 * i_tmp));

            M0_lo = _mm_unpacklo_epi16(T0, T1);
            M0_hi = _mm_unpackhi_epi16(T0, T1);
            M1_lo = _mm_unpacklo_epi16(T2, T3);
            M1_hi = _mm_unpackhi_epi16(T2, T3);
            M2_lo = _mm_unpacklo_epi16(T4, T5);
            M2_hi = _mm_unpackhi_epi16(T4, T5);
            M3_lo = _mm_unpacklo_epi16(T6, T7);
            M3_hi = _mm_unpackhi_epi16(T6, T7);

            N0_lo = _mm_madd_epi16(M0_lo, coeff00);
            N0_hi = _mm_madd_epi16(M0_hi, coeff00);
            N1_lo = _mm_madd_epi16(M1_lo, coeff01);
            N1_hi = _mm_madd_epi16(M1_hi, coeff01);
            N2_lo = _mm_madd_epi16(M2_lo, coeff02);
            N2_hi = _mm_madd_epi16(M2_hi, coeff02);
            N3_lo = _mm_madd_epi16(M3_lo, coeff03);
            N3_hi = _mm_madd_epi16(M3_hi, coeff03);

            sum_lo = _mm_add_epi32(N0_lo, N1_lo);
            sum_lo = _mm_add_epi32(sum_lo, N2_lo);
            sum_lo = _mm_add_epi32(sum_lo, N3_lo);

            sum_hi = _mm_add_epi32(N0_hi, N1_hi);
            sum_hi = _mm_add_epi32(sum_hi, N2_hi);
            sum_hi = _mm_add_epi32(sum_hi, N3_hi);

            sum_lo = _mm_add_epi32(sum_lo, offset);
            sum_hi = _mm_add_epi32(sum_hi, offset);
            sum_lo = _mm_srai_epi32(sum_lo, shift2);
            sum_hi = _mm_srai_epi32(sum_hi, shift2);

            sum_lo = _mm_packus_epi32(sum_lo, sum_hi);
            sum_lo = _mm_min_epu16(sum_lo, max_val1);
            _mm_storeu_si128((__m128i*)(dst + i), sum_lo);

            p += 8;
        }
        for (; i < width; i++) {
            int v = tmp[i]              * coef_y[0]
                  + tmp[i + i_tmp]      * coef_y[1]
                  + tmp[i + 2 * i_tmp]  * coef_y[2]
                  + tmp[i + 3 * i_tmp]  * coef_y[3]
                  + tmp[i + 4 * i_tmp]  * coef_y[4]
                  + tmp[i + 5 * i_tmp]  * coef_y[5]
                  + tmp[i + 6 * i_tmp]  * coef_y[6]
                  + tmp[i + 7 * i_tmp]  * coef_y[7];
            v = (v + add2) >> shift2;
            dst[i] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
        }
        dst += i_dst;
        tmp += i_tmp;
    }
}

/* ---- 融合 MC+avg: 双向预测第二路, dst[i] = (dst[i] + mc(src)[i] + 1) >> 1 ---- */
static void ip_filter_luma_ext_10bit_avg_sse4(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y, int max_val)
{
    AVS2_ALIGN16(short tmp_res[(64 + 7) * 64]);
    short *tmp = tmp_res;
    const int i_tmp = MC_TMP_STRIDE;
    int add1, shift1, add2, shift2;
    int i, j;
    __m128i offset;
    __m128i max_val1 = _mm_set1_epi16((short)max_val);
    __m128i one = _mm_set1_epi16(1);
    __m128i mCoef = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)coef_x));
    __m128i c0, c1, c2, c3, coeff00, coeff01, coeff02, coeff03;

    shift1 = 2;
    shift2 = 10;
    add1 = (1 << shift1) >> 1;
    add2 = 1 << (shift2 - 1);

    /* ---- 第一级: 水平滤波 (与普通版相同) ---- */
    src += -3 * i_src - 3;
    offset = _mm_set1_epi32(add1);

    for (j = -3; j < height + 4; j++) {
        const pel_t *p = src;
        if (j + 8 < height + 4)
            _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
        for (i = 0; i + 7 < width; i += 8) {
            __m128i T0, T1, T2, T3, T4, T5, T6, T7;
            __m128i M0, M1, M2, M3, M4, M5, M6, M7;
            __m128i A0, A1, A2, A3, B0, B1;

            T0 = _mm_loadu_si128((__m128i*)p++);
            T1 = _mm_loadu_si128((__m128i*)p++);
            T2 = _mm_loadu_si128((__m128i*)p++);
            T3 = _mm_loadu_si128((__m128i*)p++);
            T4 = _mm_loadu_si128((__m128i*)p++);
            T5 = _mm_loadu_si128((__m128i*)p++);
            T6 = _mm_loadu_si128((__m128i*)p++);
            T7 = _mm_loadu_si128((__m128i*)p++);

            M0 = _mm_madd_epi16(T0, mCoef);
            M1 = _mm_madd_epi16(T1, mCoef);
            M2 = _mm_madd_epi16(T2, mCoef);
            M3 = _mm_madd_epi16(T3, mCoef);
            M4 = _mm_madd_epi16(T4, mCoef);
            M5 = _mm_madd_epi16(T5, mCoef);
            M6 = _mm_madd_epi16(T6, mCoef);
            M7 = _mm_madd_epi16(T7, mCoef);

            A0 = _mm_hadd_epi32(M0, M1);
            A1 = _mm_hadd_epi32(M2, M3);
            A2 = _mm_hadd_epi32(M4, M5);
            A3 = _mm_hadd_epi32(M6, M7);

            B0 = _mm_hadd_epi32(A0, A1);
            B1 = _mm_hadd_epi32(A2, A3);

            B0 = _mm_add_epi32(B0, offset);
            B1 = _mm_add_epi32(B1, offset);
            B0 = _mm_srai_epi32(B0, shift1);
            B1 = _mm_srai_epi32(B1, shift1);
            B0 = _mm_packs_epi32(B0, B1);
            _mm_store_si128((__m128i*)(tmp + i), B0);
        }
        for (; i < width; i++) {
            int v = src[i]     * coef_x[0] + src[i + 1] * coef_x[1]
                  + src[i + 2] * coef_x[2] + src[i + 3] * coef_x[3]
                  + src[i + 4] * coef_x[4] + src[i + 5] * coef_x[5]
                  + src[i + 6] * coef_x[6] + src[i + 7] * coef_x[7];
            v = (v + add1) >> shift1;
            tmp[i] = (short)v;
        }
        tmp += i_tmp;
        src += i_src;
    }

    /* ---- 第二级: 垂直滤波 + 融合平均 ---- */
    offset = _mm_set1_epi32(add2);
    tmp = tmp_res;

    c0 = _mm_set1_epi16(*(const short*)(coef_y + 0));
    c1 = _mm_set1_epi16(*(const short*)(coef_y + 2));
    c2 = _mm_set1_epi16(*(const short*)(coef_y + 4));
    c3 = _mm_set1_epi16(*(const short*)(coef_y + 6));
    coeff00 = _mm_cvtepi8_epi16(c0);
    coeff01 = _mm_cvtepi8_epi16(c1);
    coeff02 = _mm_cvtepi8_epi16(c2);
    coeff03 = _mm_cvtepi8_epi16(c3);

    for (j = 0; j < height; j++) {
        const short *p = tmp;
        for (i = 0; i + 7 < width; i += 8) {
            __m128i T0, T1, T2, T3, T4, T5, T6, T7;
            __m128i M0_lo, M0_hi, M1_lo, M1_hi, M2_lo, M2_hi, M3_lo, M3_hi;
            __m128i N0_lo, N0_hi, N1_lo, N1_hi, N2_lo, N2_hi, N3_lo, N3_hi;
            __m128i sum_lo, sum_hi, vdst, vsum;

            T0 = _mm_load_si128((__m128i*)(p));
            T1 = _mm_load_si128((__m128i*)(p + i_tmp));
            T2 = _mm_load_si128((__m128i*)(p + 2 * i_tmp));
            T3 = _mm_load_si128((__m128i*)(p + 3 * i_tmp));
            T4 = _mm_load_si128((__m128i*)(p + 4 * i_tmp));
            T5 = _mm_load_si128((__m128i*)(p + 5 * i_tmp));
            T6 = _mm_load_si128((__m128i*)(p + 6 * i_tmp));
            T7 = _mm_load_si128((__m128i*)(p + 7 * i_tmp));

            M0_lo = _mm_unpacklo_epi16(T0, T1);
            M0_hi = _mm_unpackhi_epi16(T0, T1);
            M1_lo = _mm_unpacklo_epi16(T2, T3);
            M1_hi = _mm_unpackhi_epi16(T2, T3);
            M2_lo = _mm_unpacklo_epi16(T4, T5);
            M2_hi = _mm_unpackhi_epi16(T4, T5);
            M3_lo = _mm_unpacklo_epi16(T6, T7);
            M3_hi = _mm_unpackhi_epi16(T6, T7);

            N0_lo = _mm_madd_epi16(M0_lo, coeff00);
            N0_hi = _mm_madd_epi16(M0_hi, coeff00);
            N1_lo = _mm_madd_epi16(M1_lo, coeff01);
            N1_hi = _mm_madd_epi16(M1_hi, coeff01);
            N2_lo = _mm_madd_epi16(M2_lo, coeff02);
            N2_hi = _mm_madd_epi16(M2_hi, coeff02);
            N3_lo = _mm_madd_epi16(M3_lo, coeff03);
            N3_hi = _mm_madd_epi16(M3_hi, coeff03);

            sum_lo = _mm_add_epi32(N0_lo, N1_lo);
            sum_lo = _mm_add_epi32(sum_lo, N2_lo);
            sum_lo = _mm_add_epi32(sum_lo, N3_lo);

            sum_hi = _mm_add_epi32(N0_hi, N1_hi);
            sum_hi = _mm_add_epi32(sum_hi, N2_hi);
            sum_hi = _mm_add_epi32(sum_hi, N3_hi);

            sum_lo = _mm_add_epi32(sum_lo, offset);
            sum_hi = _mm_add_epi32(sum_hi, offset);
            sum_lo = _mm_srai_epi32(sum_lo, shift2);
            sum_hi = _mm_srai_epi32(sum_hi, shift2);

            sum_lo = _mm_packus_epi32(sum_lo, sum_hi);
            sum_lo = _mm_min_epu16(sum_lo, max_val1);

            /* 融合平均: dst[i] = (dst[i] + result[i] + 1) >> 1 */
            vdst = _mm_loadu_si128((const __m128i*)(dst + i));
            vsum = _mm_add_epi16(vdst, sum_lo);
            vsum = _mm_add_epi16(vsum, one);
            vsum = _mm_srli_epi16(vsum, 1);
            _mm_storeu_si128((__m128i*)(dst + i), vsum);

            p += 8;
        }
        for (; i < width; i++) {
            int v = tmp[i]              * coef_y[0]
                  + tmp[i + i_tmp]      * coef_y[1]
                  + tmp[i + 2 * i_tmp]  * coef_y[2]
                  + tmp[i + 3 * i_tmp]  * coef_y[3]
                  + tmp[i + 4 * i_tmp]  * coef_y[4]
                  + tmp[i + 5 * i_tmp]  * coef_y[5]
                  + tmp[i + 6 * i_tmp]  * coef_y[6]
                  + tmp[i + 7 * i_tmp]  * coef_y[7];
            v = (v + add2) >> shift2;
            v = v < 0 ? 0 : (v > max_val ? max_val : v);
            dst[i] = (pel_t)((dst[i] + v + 1) >> 1);
        }
        dst += i_dst;
        tmp += i_tmp;
    }
}

/* ===========================================================================
 * 第五部分: 色度 4 抽头水平插值 (10-bit SSE4.1)
 *
 * 算法: 4 抽头, shift=6, offset=32
 * SSE4.1: 每次处理 8 个输出像素
 *   - 2 组 shuffle 产生 2 对 4-tap 窗口, madd + hadd 得 4 结果
 *   - 两轮覆盖 8 输出
 * =========================================================================== */
static void ip_filter_chroma_hor_10bit_sse4(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height, const int8_t *coeff, int max_val)
{
    int row, col;
    const int offset_val = 32;
    const int shift = 6;

    __m128i mCoef = _mm_cvtepi8_epi16(_mm_set1_epi32(*(const int32_t*)coeff));
    __m128i mSwitch1 = _mm_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9);
    __m128i mSwitch2 = _mm_setr_epi8(
        4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13);
    __m128i mAddOffset = _mm_set1_epi32(offset_val);
    __m128i max_val1 = _mm_set1_epi16((short)max_val);

    src -= 1;

    for (row = 0; row < height; row++) {
        const pel_t *p = src;
        if (row + 8 < height)
            _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
        for (col = 0; col + 7 < width; col += 8) {
            __m128i s, s1, s2, s3, s4;
            __m128i m1, m2, m3, m4, sum_lo, sum_hi;

            /* 前 4 输出: 从 p[col] 加载 8 像素 */
            s = _mm_loadu_si128((__m128i*)(p + col));
            s1 = _mm_shuffle_epi8(s, mSwitch1);
            s2 = _mm_shuffle_epi8(s, mSwitch2);
            m1 = _mm_madd_epi16(s1, mCoef);
            m2 = _mm_madd_epi16(s2, mCoef);
            sum_lo = _mm_hadd_epi32(m1, m2);

            /* 后 4 输出: 从 p[col+4] 加载 8 像素 */
            s3 = _mm_loadu_si128((__m128i*)(p + col + 4));
            s3 = _mm_shuffle_epi8(s3, mSwitch1);
            s4 = _mm_loadu_si128((__m128i*)(p + col + 4));
            s4 = _mm_shuffle_epi8(s4, mSwitch2);
            m3 = _mm_madd_epi16(s3, mCoef);
            m4 = _mm_madd_epi16(s4, mCoef);
            sum_hi = _mm_hadd_epi32(m3, m4);

            sum_lo = _mm_add_epi32(sum_lo, mAddOffset);
            sum_hi = _mm_add_epi32(sum_hi, mAddOffset);
            sum_lo = _mm_srai_epi32(sum_lo, shift);
            sum_hi = _mm_srai_epi32(sum_hi, shift);
            sum_lo = _mm_packus_epi32(sum_lo, sum_hi);
            sum_lo = _mm_min_epu16(sum_lo, max_val1);
            _mm_storeu_si128((__m128i*)(dst + col), sum_lo);
        }
        for (; col < width; col++) {
            int v = src[col]     * coeff[0] + src[col + 1] * coeff[1]
                  + src[col + 2] * coeff[2] + src[col + 3] * coeff[3];
            v = (v + offset_val) >> shift;
            dst[col] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ===========================================================================
 * 第六部分: 色度 4 抽头垂直插值 (10-bit SSE4.1)
 * =========================================================================== */
static void ip_filter_chroma_ver_10bit_sse4(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height, const int8_t *coeff, int max_val)
{
    int row, col;
    const int offset_val = 32;
    const int shift = 6;
    __m128i mAddOffset = _mm_set1_epi32(offset_val);
    __m128i max_val1 = _mm_set1_epi16((short)max_val);
    const int i_src2 = i_src * 2;
    const int i_src3 = i_src * 3;

    __m128i c0 = _mm_set1_epi16(*(const short*)(coeff + 0));
    __m128i c1 = _mm_set1_epi16(*(const short*)(coeff + 2));
    __m128i coeff0 = _mm_cvtepi8_epi16(c0);
    __m128i coeff1 = _mm_cvtepi8_epi16(c1);

    src -= i_src;

    for (row = 0; row < height; row++) {
        const pel_t *p = src;
        for (col = 0; col + 7 < width; col += 8) {
            __m128i S0, S1, S2, S3;
            __m128i T0_lo, T0_hi, T1_lo, T1_hi;
            __m128i N0_lo, N0_hi, N1_lo, N1_hi;
            __m128i sum_lo, sum_hi;

            _mm_prefetch((const char*)(p + col + 12 * i_src), _MM_HINT_T0);

            S0 = _mm_loadu_si128((__m128i*)(p + col));
            S1 = _mm_loadu_si128((__m128i*)(p + col + i_src));
            S2 = _mm_loadu_si128((__m128i*)(p + col + i_src2));
            S3 = _mm_loadu_si128((__m128i*)(p + col + i_src3));

            T0_lo = _mm_unpacklo_epi16(S0, S1);
            T0_hi = _mm_unpackhi_epi16(S0, S1);
            T1_lo = _mm_unpacklo_epi16(S2, S3);
            T1_hi = _mm_unpackhi_epi16(S2, S3);

            N0_lo = _mm_madd_epi16(T0_lo, coeff0);
            N0_hi = _mm_madd_epi16(T0_hi, coeff0);
            N1_lo = _mm_madd_epi16(T1_lo, coeff1);
            N1_hi = _mm_madd_epi16(T1_hi, coeff1);

            sum_lo = _mm_add_epi32(N0_lo, N1_lo);
            sum_hi = _mm_add_epi32(N0_hi, N1_hi);

            sum_lo = _mm_add_epi32(sum_lo, mAddOffset);
            sum_hi = _mm_add_epi32(sum_hi, mAddOffset);
            sum_lo = _mm_srai_epi32(sum_lo, shift);
            sum_hi = _mm_srai_epi32(sum_hi, shift);
            sum_lo = _mm_packus_epi32(sum_lo, sum_hi);
            sum_lo = _mm_min_epu16(sum_lo, max_val1);
            _mm_storeu_si128((__m128i*)(dst + col), sum_lo);
        }
        for (; col < width; col++) {
            int v = src[col]             * coeff[0]
                  + src[col + i_src]     * coeff[1]
                  + src[col + i_src2]    * coeff[2]
                  + src[col + i_src3]    * coeff[3];
            v = (v + offset_val) >> shift;
            dst[col] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ===========================================================================
 * 第七部分: 色度 4 抽头双向插值 (10-bit SSE4.1)
 * =========================================================================== */
static void ip_filter_chroma_ext_10bit_sse4(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y, int max_val)
{
    AVS2_ALIGN16(short tmp_res[(32 + 3) * 32]);
    short *tmp = tmp_res;
    const int i_tmp = 32;
    int shift1, shift2, add1, add2;
    int row, col;
    __m128i max_val1 = _mm_set1_epi16((short)max_val);
    __m128i mAddOffset;
    __m128i mCoef = _mm_cvtepi8_epi16(_mm_set1_epi32(*(const int32_t*)coef_x));
    __m128i mSwitch1 = _mm_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9);
    __m128i mSwitch2 = _mm_setr_epi8(
        4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13);
    __m128i cy0, cy1, coeffy0, coeffy1;

    shift1 = 2;
    shift2 = 10;
    add1 = (1 << shift1) >> 1;
    add2 = 1 << (shift2 - 1);

    /* ---- 第一级: 水平滤波 ---- */
    mAddOffset = _mm_set1_epi32(add1);
    src = src - i_src - 1;

    for (row = -1; row < height + 2; row++) {
        const pel_t *p = src;
        if (row + 8 < height + 2)
            _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
        for (col = 0; col + 7 < width; col += 8) {
            __m128i s, s1, s2, s3, s4;
            __m128i m1, m2, m3, m4, sum_lo, sum_hi;

            s = _mm_loadu_si128((__m128i*)(p + col));
            s1 = _mm_shuffle_epi8(s, mSwitch1);
            s2 = _mm_shuffle_epi8(s, mSwitch2);
            m1 = _mm_madd_epi16(s1, mCoef);
            m2 = _mm_madd_epi16(s2, mCoef);
            sum_lo = _mm_hadd_epi32(m1, m2);

            s3 = _mm_loadu_si128((__m128i*)(p + col + 4));
            s3 = _mm_shuffle_epi8(s3, mSwitch1);
            s4 = _mm_loadu_si128((__m128i*)(p + col + 4));
            s4 = _mm_shuffle_epi8(s4, mSwitch2);
            m3 = _mm_madd_epi16(s3, mCoef);
            m4 = _mm_madd_epi16(s4, mCoef);
            sum_hi = _mm_hadd_epi32(m3, m4);

            sum_lo = _mm_add_epi32(sum_lo, mAddOffset);
            sum_hi = _mm_add_epi32(sum_hi, mAddOffset);
            sum_lo = _mm_srai_epi32(sum_lo, shift1);
            sum_hi = _mm_srai_epi32(sum_hi, shift1);
            sum_lo = _mm_packs_epi32(sum_lo, sum_hi);
            _mm_store_si128((__m128i*)(tmp + col), sum_lo);
        }
        for (; col < width; col++) {
            int v = src[col]     * coef_x[0] + src[col + 1] * coef_x[1]
                  + src[col + 2] * coef_x[2] + src[col + 3] * coef_x[3];
            v = (v + add1) >> shift1;
            tmp[col] = (short)v;
        }
        src += i_src;
        tmp += i_tmp;
    }

    /* ---- 第二级: 垂直滤波 ---- */
    tmp = tmp_res;
    mAddOffset = _mm_set1_epi32(add2);
    cy0 = _mm_set1_epi16(*(const short*)(coef_y + 0));
    cy1 = _mm_set1_epi16(*(const short*)(coef_y + 2));
    coeffy0 = _mm_cvtepi8_epi16(cy0);
    coeffy1 = _mm_cvtepi8_epi16(cy1);

    for (row = 0; row < height; row++) {
        for (col = 0; col + 7 < width; col += 8) {
            __m128i S0, S1, S2, S3;
            __m128i T0_lo, T0_hi, T1_lo, T1_hi;
            __m128i N0_lo, N0_hi, N1_lo, N1_hi;
            __m128i sum_lo, sum_hi;

            S0 = _mm_loadu_si128((__m128i*)(tmp + col));
            S1 = _mm_loadu_si128((__m128i*)(tmp + col + i_tmp));
            S2 = _mm_loadu_si128((__m128i*)(tmp + col + 2 * i_tmp));
            S3 = _mm_loadu_si128((__m128i*)(tmp + col + 3 * i_tmp));

            T0_lo = _mm_unpacklo_epi16(S0, S1);
            T0_hi = _mm_unpackhi_epi16(S0, S1);
            T1_lo = _mm_unpacklo_epi16(S2, S3);
            T1_hi = _mm_unpackhi_epi16(S2, S3);

            N0_lo = _mm_madd_epi16(T0_lo, coeffy0);
            N0_hi = _mm_madd_epi16(T0_hi, coeffy0);
            N1_lo = _mm_madd_epi16(T1_lo, coeffy1);
            N1_hi = _mm_madd_epi16(T1_hi, coeffy1);

            sum_lo = _mm_add_epi32(N0_lo, N1_lo);
            sum_hi = _mm_add_epi32(N0_hi, N1_hi);

            sum_lo = _mm_add_epi32(sum_lo, mAddOffset);
            sum_hi = _mm_add_epi32(sum_hi, mAddOffset);
            sum_lo = _mm_srai_epi32(sum_lo, shift2);
            sum_hi = _mm_srai_epi32(sum_hi, shift2);
            sum_lo = _mm_packus_epi32(sum_lo, sum_hi);
            sum_lo = _mm_min_epu16(sum_lo, max_val1);
            _mm_storeu_si128((__m128i*)(dst + col), sum_lo);
        }
        for (; col < width; col++) {
            int v = tmp[col]            * coef_y[0]
                  + tmp[col + i_tmp]    * coef_y[1]
                  + tmp[col + 2 * i_tmp] * coef_y[2]
                  + tmp[col + 3 * i_tmp] * coef_y[3];
            v = (v + add2) >> shift2;
            dst[col] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
        }
        dst += i_dst;
        tmp += i_tmp;
    }
}

/* ---- 融合 MC+avg: 色度双向预测第二路 ---- */
static void ip_filter_chroma_ext_10bit_avg_sse4(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y, int max_val)
{
    AVS2_ALIGN16(short tmp_res[(32 + 3) * 32]);
    short *tmp = tmp_res;
    const int i_tmp = 32;
    int shift1, shift2, add1, add2;
    int row, col;
    __m128i max_val1 = _mm_set1_epi16((short)max_val);
    __m128i one16 = _mm_set1_epi16(1);
    __m128i mAddOffset;
    __m128i mCoef = _mm_cvtepi8_epi16(_mm_set1_epi32(*(const int32_t*)coef_x));
    __m128i mSwitch1 = _mm_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9);
    __m128i mSwitch2 = _mm_setr_epi8(
        4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13);
    __m128i cy0, cy1, coeffy0, coeffy1;

    shift1 = 2;
    shift2 = 10;
    add1 = (1 << shift1) >> 1;
    add2 = 1 << (shift2 - 1);

    /* ---- 第一级: 水平滤波 (与普通版相同) ---- */
    mAddOffset = _mm_set1_epi32(add1);
    src = src - i_src - 1;

    for (row = -1; row < height + 2; row++) {
        const pel_t *p = src;
        if (row + 8 < height + 2)
            _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
        for (col = 0; col + 7 < width; col += 8) {
            __m128i s, s1, s2, s3, s4;
            __m128i m1, m2, m3, m4, sum_lo, sum_hi;

            s = _mm_loadu_si128((__m128i*)(p + col));
            s1 = _mm_shuffle_epi8(s, mSwitch1);
            s2 = _mm_shuffle_epi8(s, mSwitch2);
            m1 = _mm_madd_epi16(s1, mCoef);
            m2 = _mm_madd_epi16(s2, mCoef);
            sum_lo = _mm_hadd_epi32(m1, m2);

            s3 = _mm_loadu_si128((__m128i*)(p + col + 4));
            s3 = _mm_shuffle_epi8(s3, mSwitch1);
            s4 = _mm_loadu_si128((__m128i*)(p + col + 4));
            s4 = _mm_shuffle_epi8(s4, mSwitch2);
            m3 = _mm_madd_epi16(s3, mCoef);
            m4 = _mm_madd_epi16(s4, mCoef);
            sum_hi = _mm_hadd_epi32(m3, m4);

            sum_lo = _mm_add_epi32(sum_lo, mAddOffset);
            sum_hi = _mm_add_epi32(sum_hi, mAddOffset);
            sum_lo = _mm_srai_epi32(sum_lo, shift1);
            sum_hi = _mm_srai_epi32(sum_hi, shift1);
            sum_lo = _mm_packs_epi32(sum_lo, sum_hi);
            _mm_store_si128((__m128i*)(tmp + col), sum_lo);
        }
        for (; col < width; col++) {
            int v = src[col]     * coef_x[0] + src[col + 1] * coef_x[1]
                  + src[col + 2] * coef_x[2] + src[col + 3] * coef_x[3];
            v = (v + add1) >> shift1;
            tmp[col] = (short)v;
        }
        src += i_src;
        tmp += i_tmp;
    }

    /* ---- 第二级: 垂直滤波 + 融合平均 ---- */
    tmp = tmp_res;
    mAddOffset = _mm_set1_epi32(add2);
    cy0 = _mm_set1_epi16(*(const short*)(coef_y + 0));
    cy1 = _mm_set1_epi16(*(const short*)(coef_y + 2));
    coeffy0 = _mm_cvtepi8_epi16(cy0);
    coeffy1 = _mm_cvtepi8_epi16(cy1);

    for (row = 0; row < height; row++) {
        for (col = 0; col + 7 < width; col += 8) {
            __m128i S0, S1, S2, S3;
            __m128i T0_lo, T0_hi, T1_lo, T1_hi;
            __m128i N0_lo, N0_hi, N1_lo, N1_hi;
            __m128i sum_lo, sum_hi, vdst, vsum;

            S0 = _mm_loadu_si128((__m128i*)(tmp + col));
            S1 = _mm_loadu_si128((__m128i*)(tmp + col + i_tmp));
            S2 = _mm_loadu_si128((__m128i*)(tmp + col + 2 * i_tmp));
            S3 = _mm_loadu_si128((__m128i*)(tmp + col + 3 * i_tmp));

            T0_lo = _mm_unpacklo_epi16(S0, S1);
            T0_hi = _mm_unpackhi_epi16(S0, S1);
            T1_lo = _mm_unpacklo_epi16(S2, S3);
            T1_hi = _mm_unpackhi_epi16(S2, S3);

            N0_lo = _mm_madd_epi16(T0_lo, coeffy0);
            N0_hi = _mm_madd_epi16(T0_hi, coeffy0);
            N1_lo = _mm_madd_epi16(T1_lo, coeffy1);
            N1_hi = _mm_madd_epi16(T1_hi, coeffy1);

            sum_lo = _mm_add_epi32(N0_lo, N1_lo);
            sum_hi = _mm_add_epi32(N0_hi, N1_hi);

            sum_lo = _mm_add_epi32(sum_lo, mAddOffset);
            sum_hi = _mm_add_epi32(sum_hi, mAddOffset);
            sum_lo = _mm_srai_epi32(sum_lo, shift2);
            sum_hi = _mm_srai_epi32(sum_hi, shift2);
            sum_lo = _mm_packus_epi32(sum_lo, sum_hi);
            sum_lo = _mm_min_epu16(sum_lo, max_val1);

            /* 融合平均: dst[col] = (dst[col] + result[col] + 1) >> 1 */
            vdst = _mm_loadu_si128((const __m128i*)(dst + col));
            vsum = _mm_add_epi16(vdst, sum_lo);
            vsum = _mm_add_epi16(vsum, one16);
            vsum = _mm_srli_epi16(vsum, 1);
            _mm_storeu_si128((__m128i*)(dst + col), vsum);
        }
        for (; col < width; col++) {
            int v = tmp[col]            * coef_y[0]
                  + tmp[col + i_tmp]    * coef_y[1]
                  + tmp[col + 2 * i_tmp] * coef_y[2]
                  + tmp[col + 3 * i_tmp] * coef_y[3];
            v = (v + add2) >> shift2;
            v = v < 0 ? 0 : (v > max_val ? max_val : v);
            dst[col] = (pel_t)((dst[col] + v + 1) >> 1);
        }
        dst += i_dst;
        tmp += i_tmp;
    }
}

/* ===========================================================================
 * 第八部分: 8-bit 子像素插值 (SSE4.1)
 * =========================================================================== */

static void ip_filter_luma_hor_8bit_sse41(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height, const int8_t *coeff)
{
    int j, i;
    __m128i m_switch1 = _mm_setr_epi8(0,1,2,3,4,5,6,7, 1,2,3,4,5,6,7,8);
    __m128i m_switch2 = _mm_setr_epi8(2,3,4,5,6,7,8,9, 3,4,5,6,7,8,9,10);
    __m128i m_switch3 = _mm_setr_epi8(4,5,6,7,8,9,10,11, 5,6,7,8,9,10,11,12);
    __m128i m_switch4 = _mm_setr_epi8(6,7,8,9,10,11,12,13, 7,8,9,10,11,12,13,14);
    __m128i m_coef = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)coeff),
                                         _mm_loadl_epi64((const __m128i*)coeff));
    __m128i offset = _mm_set1_epi16(32);

    src -= 3;

    for (j = 0; j < height; j++) {
        const uint8_t *p = src;
        for (i = 0; i + 7 < width; i += 8) {
            __m128i s = _mm_loadu_si128((const __m128i*)p);
            __m128i t1, t2, t3, t4, sum;

            t1 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch1), m_coef);
            t2 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch2), m_coef);
            t3 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch3), m_coef);
            t4 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch4), m_coef);

            sum = _mm_hadd_epi16(t1, t2);
            sum = _mm_hadd_epi16(sum, _mm_hadd_epi16(t3, t4));

            sum = _mm_add_epi16(sum, offset);
            sum = _mm_srai_epi16(sum, 6);
            sum = _mm_packus_epi16(sum, sum);

            _mm_storel_epi64((__m128i*)(dst + i), sum);

            p += 8;
        }
        for (; i < width; i++) {
            int v = src[i]     * coeff[0] + src[i + 1] * coeff[1]
                  + src[i + 2] * coeff[2] + src[i + 3] * coeff[3]
                  + src[i + 4] * coeff[4] + src[i + 5] * coeff[5]
                  + src[i + 6] * coeff[6] + src[i + 7] * coeff[7];
            v = (v + 32) >> 6;
            dst[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        dst += i_dst;
        src += i_src;
    }
}

static void ip_filter_luma_ver_8bit_sse41(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height, const int8_t *coeff)
{
    int i, j;
    __m128i m_coef01 = _mm_set1_epi16(*(const short*)(coeff + 0));
    __m128i m_coef23 = _mm_set1_epi16(*(const short*)(coeff + 2));
    __m128i m_coef45 = _mm_set1_epi16(*(const short*)(coeff + 4));
    __m128i m_coef67 = _mm_set1_epi16(*(const short*)(coeff + 6));
    __m128i offset = _mm_set1_epi16(32);

    src -= 3 * i_src;

    for (j = 0; j < height; j++) {
        const uint8_t *p = src;
        for (i = 0; i + 7 < width; i += 8) {
            __m128i s0, s1, s2, s3, s4, s5, s6, s7;
            __m128i m0, m1, m2, m3, sum;

            s0 = _mm_loadl_epi64((const __m128i*)(p + i));
            s1 = _mm_loadl_epi64((const __m128i*)(p + i + i_src));
            s2 = _mm_loadl_epi64((const __m128i*)(p + i + 2 * i_src));
            s3 = _mm_loadl_epi64((const __m128i*)(p + i + 3 * i_src));
            s4 = _mm_loadl_epi64((const __m128i*)(p + i + 4 * i_src));
            s5 = _mm_loadl_epi64((const __m128i*)(p + i + 5 * i_src));
            s6 = _mm_loadl_epi64((const __m128i*)(p + i + 6 * i_src));
            s7 = _mm_loadl_epi64((const __m128i*)(p + i + 7 * i_src));

            m0 = _mm_unpacklo_epi8(s0, s1);
            m1 = _mm_unpacklo_epi8(s2, s3);
            m2 = _mm_unpacklo_epi8(s4, s5);
            m3 = _mm_unpacklo_epi8(s6, s7);

            sum = _mm_maddubs_epi16(m0, m_coef01);
            sum = _mm_add_epi16(sum, _mm_maddubs_epi16(m1, m_coef23));
            sum = _mm_add_epi16(sum, _mm_maddubs_epi16(m2, m_coef45));
            sum = _mm_add_epi16(sum, _mm_maddubs_epi16(m3, m_coef67));

            sum = _mm_add_epi16(sum, offset);
            sum = _mm_srai_epi16(sum, 6);
            sum = _mm_packus_epi16(sum, sum);

            _mm_storel_epi64((__m128i*)(dst + i), sum);
        }
        for (; i < width; i++) {
            int v = src[i]              * coeff[0]
                  + src[i + i_src]      * coeff[1]
                  + src[i + 2 * i_src]  * coeff[2]
                  + src[i + 3 * i_src]  * coeff[3]
                  + src[i + 4 * i_src]  * coeff[4]
                  + src[i + 5 * i_src]  * coeff[5]
                  + src[i + 6 * i_src]  * coeff[6]
                  + src[i + 7 * i_src]  * coeff[7];
            v = (v + 32) >> 6;
            dst[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        src += i_src;
        dst += i_dst;
    }
}

static void ip_filter_luma_ext_8bit_sse41(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y)
{
    AVS2_ALIGN16(short tmp_res[(64 + 7) * 64]);
    short *tmp = tmp_res;
    const int i_tmp = MC_TMP_STRIDE;
    const int shift2 = 12;
    const int add2 = 1 << (shift2 - 1);
    int i, j;
    __m128i m_switch1 = _mm_setr_epi8(0,1,2,3,4,5,6,7, 1,2,3,4,5,6,7,8);
    __m128i m_switch2 = _mm_setr_epi8(2,3,4,5,6,7,8,9, 3,4,5,6,7,8,9,10);
    __m128i m_switch3 = _mm_setr_epi8(4,5,6,7,8,9,10,11, 5,6,7,8,9,10,11,12);
    __m128i m_switch4 = _mm_setr_epi8(6,7,8,9,10,11,12,13, 7,8,9,10,11,12,13,14);
    __m128i m_coef = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)coef_x),
                                         _mm_loadl_epi64((const __m128i*)coef_x));
    __m128i cy0 = _mm_cvtepi8_epi16(_mm_set1_epi16(*(const short*)(coef_y + 0)));
    __m128i cy1 = _mm_cvtepi8_epi16(_mm_set1_epi16(*(const short*)(coef_y + 2)));
    __m128i cy2 = _mm_cvtepi8_epi16(_mm_set1_epi16(*(const short*)(coef_y + 4)));
    __m128i cy3 = _mm_cvtepi8_epi16(_mm_set1_epi16(*(const short*)(coef_y + 6)));
    __m128i offset2 = _mm_set1_epi32(add2);

    src += -3 * i_src - 3;

    for (j = -3; j < height + 4; j++) {
        const uint8_t *p = src;
        for (i = 0; i + 7 < width; i += 8) {
            __m128i s = _mm_loadu_si128((const __m128i*)p);
            __m128i t1, t2, t3, t4, sum;

            t1 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch1), m_coef);
            t2 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch2), m_coef);
            t3 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch3), m_coef);
            t4 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch4), m_coef);

            sum = _mm_hadd_epi16(t1, t2);
            sum = _mm_hadd_epi16(sum, _mm_hadd_epi16(t3, t4));

            _mm_store_si128((__m128i*)(tmp + i), sum);

            p += 8;
        }
        for (; i < width; i++) {
            int v = src[i]     * coef_x[0] + src[i + 1] * coef_x[1]
                  + src[i + 2] * coef_x[2] + src[i + 3] * coef_x[3]
                  + src[i + 4] * coef_x[4] + src[i + 5] * coef_x[5]
                  + src[i + 6] * coef_x[6] + src[i + 7] * coef_x[7];
            tmp[i] = (short)v;
        }
        src += i_src;
        tmp += i_tmp;
    }

    tmp = tmp_res;

    for (j = 0; j < height; j++) {
        const short *p = tmp;
        for (i = 0; i + 7 < width; i += 8) {
            __m128i t0, t1, t2, t3, t4, t5, t6, t7;
            __m128i m0_lo, m0_hi, m1_lo, m1_hi, m2_lo, m2_hi, m3_lo, m3_hi;
            __m128i n0_lo, n0_hi, n1_lo, n1_hi, n2_lo, n2_hi, n3_lo, n3_hi;
            __m128i sum_lo, sum_hi, result;

            t0 = _mm_load_si128((__m128i*)(p));
            t1 = _mm_load_si128((__m128i*)(p + i_tmp));
            t2 = _mm_load_si128((__m128i*)(p + 2 * i_tmp));
            t3 = _mm_load_si128((__m128i*)(p + 3 * i_tmp));
            t4 = _mm_load_si128((__m128i*)(p + 4 * i_tmp));
            t5 = _mm_load_si128((__m128i*)(p + 5 * i_tmp));
            t6 = _mm_load_si128((__m128i*)(p + 6 * i_tmp));
            t7 = _mm_load_si128((__m128i*)(p + 7 * i_tmp));

            m0_lo = _mm_unpacklo_epi16(t0, t1);
            m0_hi = _mm_unpackhi_epi16(t0, t1);
            m1_lo = _mm_unpacklo_epi16(t2, t3);
            m1_hi = _mm_unpackhi_epi16(t2, t3);
            m2_lo = _mm_unpacklo_epi16(t4, t5);
            m2_hi = _mm_unpackhi_epi16(t4, t5);
            m3_lo = _mm_unpacklo_epi16(t6, t7);
            m3_hi = _mm_unpackhi_epi16(t6, t7);

            n0_lo = _mm_madd_epi16(m0_lo, cy0);
            n0_hi = _mm_madd_epi16(m0_hi, cy0);
            n1_lo = _mm_madd_epi16(m1_lo, cy1);
            n1_hi = _mm_madd_epi16(m1_hi, cy1);
            n2_lo = _mm_madd_epi16(m2_lo, cy2);
            n2_hi = _mm_madd_epi16(m2_hi, cy2);
            n3_lo = _mm_madd_epi16(m3_lo, cy3);
            n3_hi = _mm_madd_epi16(m3_hi, cy3);

            sum_lo = _mm_add_epi32(n0_lo, n1_lo);
            sum_lo = _mm_add_epi32(sum_lo, n2_lo);
            sum_lo = _mm_add_epi32(sum_lo, n3_lo);

            sum_hi = _mm_add_epi32(n0_hi, n1_hi);
            sum_hi = _mm_add_epi32(sum_hi, n2_hi);
            sum_hi = _mm_add_epi32(sum_hi, n3_hi);

            sum_lo = _mm_add_epi32(sum_lo, offset2);
            sum_hi = _mm_add_epi32(sum_hi, offset2);
            sum_lo = _mm_srai_epi32(sum_lo, shift2);
            sum_hi = _mm_srai_epi32(sum_hi, shift2);

            result = _mm_packus_epi32(sum_lo, sum_hi);
            result = _mm_packus_epi16(result, result);

            _mm_storel_epi64((__m128i*)(dst + i), result);

            p += 8;
        }
        for (; i < width; i++) {
            int v = tmp[i]              * coef_y[0]
                  + tmp[i + i_tmp]      * coef_y[1]
                  + tmp[i + 2 * i_tmp]  * coef_y[2]
                  + tmp[i + 3 * i_tmp]  * coef_y[3]
                  + tmp[i + 4 * i_tmp]  * coef_y[4]
                  + tmp[i + 5 * i_tmp]  * coef_y[5]
                  + tmp[i + 6 * i_tmp]  * coef_y[6]
                  + tmp[i + 7 * i_tmp]  * coef_y[7];
            v = (v + add2) >> shift2;
            dst[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        dst += i_dst;
        tmp += i_tmp;
    }
}

static void ip_filter_chroma_hor_4bit_sse41(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height, const int8_t *coeff)
{
    int row, col;
    __m128i m_switch1 = _mm_setr_epi8(0,1, 1,2, 2,3, 3,4, 4,5, 5,6, 6,7, 7,8);
    __m128i m_switch2 = _mm_setr_epi8(2,3, 3,4, 4,5, 5,6, 6,7, 7,8, 8,9, 9,10);
    __m128i m_coef1 = _mm_set1_epi16(*(const short*)(coeff + 0));
    __m128i m_coef2 = _mm_set1_epi16(*(const short*)(coeff + 2));
    __m128i offset = _mm_set1_epi16(32);

    src -= 1;

    for (row = 0; row < height; row++) {
        const uint8_t *p = src;
        for (col = 0; col + 7 < width; col += 8) {
            __m128i s = _mm_loadu_si128((const __m128i*)p);
            __m128i t1, t2, sum;

            t1 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch1), m_coef1);
            t2 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch2), m_coef2);
            sum = _mm_add_epi16(t1, t2);

            sum = _mm_add_epi16(sum, offset);
            sum = _mm_srai_epi16(sum, 6);
            sum = _mm_packus_epi16(sum, sum);

            _mm_storel_epi64((__m128i*)(dst + col), sum);

            p += 8;
        }
        for (; col < width; col++) {
            int v = src[col]     * coeff[0] + src[col + 1] * coeff[1]
                  + src[col + 2] * coeff[2] + src[col + 3] * coeff[3];
            v = (v + 32) >> 6;
            dst[col] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        src += i_src;
        dst += i_dst;
    }
}

static void ip_filter_chroma_ver_4bit_sse41(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height, const int8_t *coeff)
{
    int row, col;
    __m128i m_coef01 = _mm_set1_epi16(*(const short*)(coeff + 0));
    __m128i m_coef23 = _mm_set1_epi16(*(const short*)(coeff + 2));
    __m128i offset = _mm_set1_epi16(32);
    const int i_src2 = i_src * 2;
    const int i_src3 = i_src * 3;

    src -= i_src;

    for (row = 0; row < height; row++) {
        const uint8_t *p = src;
        for (col = 0; col + 7 < width; col += 8) {
            __m128i s0, s1, s2, s3;
            __m128i m01, m23, sum;

            s0 = _mm_loadl_epi64((const __m128i*)(p + col));
            s1 = _mm_loadl_epi64((const __m128i*)(p + col + i_src));
            s2 = _mm_loadl_epi64((const __m128i*)(p + col + i_src2));
            s3 = _mm_loadl_epi64((const __m128i*)(p + col + i_src3));

            m01 = _mm_unpacklo_epi8(s0, s1);
            m23 = _mm_unpacklo_epi8(s2, s3);

            sum = _mm_maddubs_epi16(m01, m_coef01);
            sum = _mm_add_epi16(sum, _mm_maddubs_epi16(m23, m_coef23));

            sum = _mm_add_epi16(sum, offset);
            sum = _mm_srai_epi16(sum, 6);
            sum = _mm_packus_epi16(sum, sum);

            _mm_storel_epi64((__m128i*)(dst + col), sum);
        }
        for (; col < width; col++) {
            int v = src[col]             * coeff[0]
                  + src[col + i_src]     * coeff[1]
                  + src[col + i_src2]    * coeff[2]
                  + src[col + i_src3]    * coeff[3];
            v = (v + 32) >> 6;
            dst[col] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        src += i_src;
        dst += i_dst;
    }
}

static void ip_filter_chroma_ext_4bit_sse41(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y)
{
    AVS2_ALIGN16(short tmp_res[(32 + 3) * 32]);
    short *tmp = tmp_res;
    const int i_tmp = 32;
    const int shift2 = 12;
    const int add2 = 1 << (shift2 - 1);
    int row, col;
    __m128i m_switch1 = _mm_setr_epi8(0,1, 1,2, 2,3, 3,4, 4,5, 5,6, 6,7, 7,8);
    __m128i m_switch2 = _mm_setr_epi8(2,3, 3,4, 4,5, 5,6, 6,7, 7,8, 8,9, 9,10);
    __m128i m_coef1 = _mm_set1_epi16(*(const short*)(coef_x + 0));
    __m128i m_coef2 = _mm_set1_epi16(*(const short*)(coef_x + 2));
    __m128i cy0 = _mm_cvtepi8_epi16(_mm_set1_epi16(*(const short*)(coef_y + 0)));
    __m128i cy1 = _mm_cvtepi8_epi16(_mm_set1_epi16(*(const short*)(coef_y + 2)));
    __m128i offset2 = _mm_set1_epi32(add2);

    src = src - i_src - 1;

    for (row = -1; row < height + 2; row++) {
        const uint8_t *p = src;
        for (col = 0; col + 7 < width; col += 8) {
            __m128i s = _mm_loadu_si128((const __m128i*)p);
            __m128i t1 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch1), m_coef1);
            __m128i t2 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch2), m_coef2);
            __m128i sum = _mm_add_epi16(t1, t2);

            _mm_store_si128((__m128i*)(tmp + col), sum);

            p += 8;
        }
        for (; col < width; col++) {
            int v = src[col]     * coef_x[0] + src[col + 1] * coef_x[1]
                  + src[col + 2] * coef_x[2] + src[col + 3] * coef_x[3];
            tmp[col] = (short)v;
        }
        src += i_src;
        tmp += i_tmp;
    }

    tmp = tmp_res;

    for (row = 0; row < height; row++) {
        const short *p = tmp;
        for (col = 0; col + 7 < width; col += 8) {
            __m128i t0, t1, t2, t3;
            __m128i m0_lo, m0_hi, m1_lo, m1_hi;
            __m128i n0_lo, n0_hi, n1_lo, n1_hi;
            __m128i sum_lo, sum_hi, result;

            t0 = _mm_load_si128((__m128i*)(p));
            t1 = _mm_load_si128((__m128i*)(p + i_tmp));
            t2 = _mm_load_si128((__m128i*)(p + 2 * i_tmp));
            t3 = _mm_load_si128((__m128i*)(p + 3 * i_tmp));

            m0_lo = _mm_unpacklo_epi16(t0, t1);
            m0_hi = _mm_unpackhi_epi16(t0, t1);
            m1_lo = _mm_unpacklo_epi16(t2, t3);
            m1_hi = _mm_unpackhi_epi16(t2, t3);

            n0_lo = _mm_madd_epi16(m0_lo, cy0);
            n0_hi = _mm_madd_epi16(m0_hi, cy0);
            n1_lo = _mm_madd_epi16(m1_lo, cy1);
            n1_hi = _mm_madd_epi16(m1_hi, cy1);

            sum_lo = _mm_add_epi32(n0_lo, n1_lo);
            sum_hi = _mm_add_epi32(n0_hi, n1_hi);

            sum_lo = _mm_add_epi32(sum_lo, offset2);
            sum_hi = _mm_add_epi32(sum_hi, offset2);
            sum_lo = _mm_srai_epi32(sum_lo, shift2);
            sum_hi = _mm_srai_epi32(sum_hi, shift2);

            result = _mm_packus_epi32(sum_lo, sum_hi);
            result = _mm_packus_epi16(result, result);

            _mm_storel_epi64((__m128i*)(dst + col), result);

            p += 8;
        }
        for (; col < width; col++) {
            int v = tmp[col]            * coef_y[0]
                  + tmp[col + i_tmp]    * coef_y[1]
                  + tmp[col + 2 * i_tmp] * coef_y[2]
                  + tmp[col + 3 * i_tmp] * coef_y[3];
            v = (v + add2) >> shift2;
            dst[col] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        dst += i_dst;
        tmp += i_tmp;
    }
}

/* ===========================================================================
 * 第九部分: 入口函数
 * =========================================================================== */

static void mc_luma_sse4(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                         ptrdiff_t dstride, int w, int h, int mx, int my,
                         int bit_depth)
{
    if (bit_depth > 8) {
        int dx = mx & 3;
        int dy = my & 3;
        int max_val = (1 << bit_depth) - 1;
        pel_t *dst16 = (pel_t*)(void *)dst;
        const pel_t *src16 = (const pel_t*)(const void *)src;
        int i_dst = (int)(dstride >> 1);
        int i_src = (int)(sstride >> 1);

        if (dx == 0 && dy == 0) {
            block_copy_10bit_sse4(dst16, i_dst, src16, i_src, w, h);
        } else if (dx == 0) {
            ip_filter_luma_ver_10bit_sse4(dst16, i_dst, src16, i_src,
                                          w, h, avs2_intpl_filters[dy], max_val);
        } else if (dy == 0) {
            ip_filter_luma_hor_10bit_sse4(dst16, i_dst, src16, i_src,
                                          w, h, avs2_intpl_filters[dx], max_val);
        } else {
            ip_filter_luma_ext_10bit_sse4(dst16, i_dst, src16, i_src,
                                          w, h,
                                          avs2_intpl_filters[dx],
                                          avs2_intpl_filters[dy], max_val);
        }
    } else {
        int dx = mx & 3;
        int dy = my & 3;
        if (dx == 0 && dy == 0) {
            block_copy_8bit_sse41(dst, (int)dstride, src, (int)sstride, w, h);
        } else if (dx == 0) {
            ip_filter_luma_ver_8bit_sse41(dst, (int)dstride, src, (int)sstride,
                                          w, h, avs2_intpl_filters[dy]);
        } else if (dy == 0) {
            ip_filter_luma_hor_8bit_sse41(dst, (int)dstride, src, (int)sstride,
                                          w, h, avs2_intpl_filters[dx]);
        } else {
            ip_filter_luma_ext_8bit_sse41(dst, (int)dstride, src, (int)sstride,
                                          w, h,
                                          avs2_intpl_filters[dx],
                                          avs2_intpl_filters[dy]);
        }
    }
}

static void mc_chroma_sse4(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                           ptrdiff_t dstride, int w, int h, int mx, int my,
                           int bit_depth)
{
    if (bit_depth > 8) {
        int dx = mx & 7;
        int dy = my & 7;
        int max_val = (1 << bit_depth) - 1;
        pel_t *dst16 = (pel_t*)(void *)dst;
        const pel_t *src16 = (const pel_t*)(const void *)src;
        int i_dst = (int)(dstride >> 1);
        int i_src = (int)(sstride >> 1);

        if (dx == 0 && dy == 0) {
            block_copy_10bit_sse4(dst16, i_dst, src16, i_src, w, h);
        } else if (dx == 0) {
            ip_filter_chroma_ver_10bit_sse4(dst16, i_dst, src16, i_src,
                                            w, h, avs2_intpl_filters_c[dy], max_val);
        } else if (dy == 0) {
            ip_filter_chroma_hor_10bit_sse4(dst16, i_dst, src16, i_src,
                                            w, h, avs2_intpl_filters_c[dx], max_val);
        } else {
            ip_filter_chroma_ext_10bit_sse4(dst16, i_dst, src16, i_src,
                                            w, h,
                                            avs2_intpl_filters_c[dx],
                                            avs2_intpl_filters_c[dy], max_val);
        }
    } else {
        int dx = mx & 7;
        int dy = my & 7;
        if (dx == 0 && dy == 0) {
            block_copy_8bit_sse41(dst, (int)dstride, src, (int)sstride, w, h);
        } else if (dx == 0) {
            ip_filter_chroma_ver_4bit_sse41(dst, (int)dstride, src, (int)sstride,
                                            w, h, avs2_intpl_filters_c[dy]);
        } else if (dy == 0) {
            ip_filter_chroma_hor_4bit_sse41(dst, (int)dstride, src, (int)sstride,
                                            w, h, avs2_intpl_filters_c[dx]);
        } else {
            ip_filter_chroma_ext_4bit_sse41(dst, (int)dstride, src, (int)sstride,
                                            w, h,
                                            avs2_intpl_filters_c[dx],
                                            avs2_intpl_filters_c[dy]);
        }
    }
}

/* ===========================================================================
 * 第十部分: 双向预测平均 / 块填充
 * =========================================================================== */

static void bi_avg_8bit_sse41(uint8_t *dst, int i_dst,
                                const int16_t *pred2, int pred2_stride,
                                int width, int height)
{
    int y, x;

    for (y = 0; y < height; y++) {
        uint8_t *d = dst;
        const int16_t *p = pred2;
        x = 0;
        for (; x + 15 < width; x += 16) {
            __m128i vd = _mm_loadu_si128((const __m128i*)(d + x));
            __m128i vp_lo = _mm_loadu_si128((const __m128i*)(p + x));
            __m128i vp_hi = _mm_loadu_si128((const __m128i*)(p + x + 8));
            __m128i vp = _mm_packus_epi16(vp_lo, vp_hi);
            __m128i avg = _mm_avg_epu8(vd, vp);
            _mm_storeu_si128((__m128i*)(d + x), avg);
        }
        for (; x + 7 < width; x += 8) {
            __m128i vd = _mm_loadl_epi64((const __m128i*)(d + x));
            __m128i vp = _mm_loadu_si128((const __m128i*)(p + x));
            vp = _mm_packus_epi16(vp, _mm_setzero_si128());
            __m128i avg = _mm_avg_epu8(vd, vp);
            _mm_storel_epi64((__m128i*)(d + x), avg);
        }
        for (; x < width; x++) {
            int v = d[x] + (uint8_t)p[x];
            d[x] = (uint8_t)((v + 1) >> 1);
        }
        dst += i_dst;
        pred2 += pred2_stride;
    }
}

/* 8-bit 双向预测平均: pred2 为 uint8_t[] (步长 pred2_stride 字节).
 * 与 bi_avg_8bit_sse41 的区别: pred2 按 uint8_t 步长读取, 无需 packus_epi16 转换. */
static void bi_avg_8bit_u8_sse41(uint8_t *dst, int i_dst,
                                   const uint8_t *pred2, int pred2_stride,
                                   int width, int height)
{
    int y, x;
    for (y = 0; y < height; y++) {
        uint8_t *d = dst;
        const uint8_t *p = pred2;
        x = 0;
        for (; x + 15 < width; x += 16) {
            __m128i vd = _mm_loadu_si128((const __m128i*)(d + x));
            __m128i vp = _mm_loadu_si128((const __m128i*)(p + x));
            __m128i avg = _mm_avg_epu8(vd, vp);
            _mm_storeu_si128((__m128i*)(d + x), avg);
        }
        for (; x + 7 < width; x += 8) {
            __m128i vd = _mm_loadl_epi64((const __m128i*)(d + x));
            __m128i vp = _mm_loadl_epi64((const __m128i*)(p + x));
            __m128i avg = _mm_avg_epu8(vd, vp);
            _mm_storel_epi64((__m128i*)(d + x), avg);
        }
        for (; x < width; x++) {
            int v = d[x] + p[x];
            d[x] = (uint8_t)((v + 1) >> 1);
        }
        dst += i_dst;
        pred2 += pred2_stride;
    }
}

static void fill_block_8bit_sse41(uint8_t *dst, int i_dst,
                                    int width, int height, int fill_val)
{
    int y, x;
    __m128i vfill = _mm_set1_epi8((char)fill_val);

    for (y = 0; y < height; y++) {
        uint8_t *d = dst;
        x = 0;
        for (; x + 16 <= width; x += 16) {
            _mm_storeu_si128((__m128i*)(d + x), vfill);
        }
        for (; x < width; x++) {
            d[x] = (uint8_t)fill_val;
        }
        dst += i_dst;
    }
}

static void bi_avg_sse4(uint8_t *dst, ptrdiff_t dst_stride, const int16_t *pred2,
                        int pred2_stride, int w, int h, int bit_depth)
{
    if (bit_depth > 8) {
        uint16_t *p16 = (uint16_t *)(void *)dst;
        int stride16 = (int)(dst_stride >> 1);
        const __m128i one = _mm_set1_epi16(1);
        int y, x;

        for (y = 0; y < h; y++) {
            uint16_t *d = p16 + y * stride16;
            const int16_t *p = pred2 + y * pred2_stride;
            x = 0;
            for (; x + 7 < w; x += 8) {
                __m128i vd = _mm_loadu_si128((const __m128i *)(d + x));
                __m128i vp = _mm_loadu_si128((const __m128i *)(p + x));
                __m128i sum = _mm_add_epi16(vd, vp);
                sum = _mm_add_epi16(sum, one);
                sum = _mm_srli_epi16(sum, 1);
                _mm_storeu_si128((__m128i *)(d + x), sum);
            }
            for (; x < w; x++) {
                int v = d[x] + p[x];
                d[x] = (uint16_t)((v + 1) >> 1);
            }
        }
    } else {
        bi_avg_8bit_sse41(dst, (int)dst_stride, pred2, pred2_stride, w, h);
    }
}

static void bi_avg_2src_sse4(uint8_t *dst, ptrdiff_t dst_stride,
                             const int16_t *pred1, int pred1_stride,
                             const int16_t *pred2, int pred2_stride,
                             int w, int h, int bit_depth)
{
    int y, x;
    if (bit_depth > 8) {
        uint16_t *p16 = (uint16_t *)(void *)dst;
        int stride16 = (int)(dst_stride >> 1);
        const __m128i one = _mm_set1_epi16(1);

        for (y = 0; y < h; y++) {
            uint16_t *d = p16 + y * stride16;
            const int16_t *p1 = pred1 + y * pred1_stride;
            const int16_t *p2 = pred2 + y * pred2_stride;
            x = 0;
            for (; x + 7 < w; x += 8) {
                __m128i v1 = _mm_loadu_si128((const __m128i *)(p1 + x));
                __m128i v2 = _mm_loadu_si128((const __m128i *)(p2 + x));
                __m128i sum = _mm_add_epi16(v1, v2);
                sum = _mm_add_epi16(sum, one);
                sum = _mm_srli_epi16(sum, 1);
                _mm_storeu_si128((__m128i *)(d + x), sum);
            }
            for (; x < w; x++) {
                int v = p1[x] + p2[x];
                d[x] = (uint16_t)((v + 1) >> 1);
            }
        }
    } else {
        for (y = 0; y < h; y++) {
            uint8_t *d = dst + y * dst_stride;
            const int16_t *p1 = pred1 + y * pred1_stride;
            const int16_t *p2 = pred2 + y * pred2_stride;
            x = 0;
            for (; x + 15 < w; x += 16) {
                __m128i v1_lo = _mm_loadu_si128((const __m128i *)(p1 + x));
                __m128i v1_hi = _mm_loadu_si128((const __m128i *)(p1 + x + 8));
                __m128i v2_lo = _mm_loadu_si128((const __m128i *)(p2 + x));
                __m128i v2_hi = _mm_loadu_si128((const __m128i *)(p2 + x + 8));
                __m128i s_lo = _mm_add_epi16(v1_lo, v2_lo);
                __m128i s_hi = _mm_add_epi16(v1_hi, v2_hi);
                __m128i one16 = _mm_set1_epi16(1);
                s_lo = _mm_add_epi16(s_lo, one16);
                s_hi = _mm_add_epi16(s_hi, one16);
                s_lo = _mm_srli_epi16(s_lo, 1);
                s_hi = _mm_srli_epi16(s_hi, 1);
                __m128i packed = _mm_packus_epi16(s_lo, s_hi);
                _mm_storeu_si128((__m128i *)(d + x), packed);
            }
            for (; x < w; x++) {
                int v = p1[x] + p2[x];
                d[x] = (uint8_t)((v + 1) >> 1);
            }
        }
    }
}

/* 块填充 (SSE4.1) */
static void fill_block_sse4(uint8_t *dst, ptrdiff_t dst_stride, int w, int h,
                             int fill_val, int bit_depth)
{
    if (bit_depth > 8) {
        uint16_t *p16 = (uint16_t *)(void *)dst;
        int stride16 = (int)(dst_stride >> 1);
        __m128i vfill = _mm_set1_epi16((short)fill_val);
        int y, x;

        for (y = 0; y < h; y++) {
            uint16_t *d = p16 + y * stride16;
            x = 0;
            for (; x + 7 < w; x += 8) {
                _mm_storeu_si128((__m128i *)(d + x), vfill);
            }
            for (; x < w; x++) {
                d[x] = (uint16_t)fill_val;
            }
        }
    } else {
        fill_block_8bit_sse41(dst, (int)dst_stride, w, h, fill_val);
    }
}

/* ===========================================================================
 * 第十一部分: 整像素块平均 (10-bit) + MC+avg 入口
 * =========================================================================== */

/* 10-bit 整像素块平均: dst[i] = (dst[i] + src[i] + 1) >> 1 */
static void block_avg_10bit_sse4(pel_t *dst, int i_dst,
                                  const pel_t *src, int i_src,
                                  int width, int height)
{
    const __m128i one = _mm_set1_epi16(1);
    int y, x;

    for (y = 0; y < height; y++) {
        pel_t *d = dst;
        const pel_t *s = src;
        x = 0;
        for (; x + 7 < width; x += 8) {
            __m128i vd = _mm_loadu_si128((const __m128i *)(d + x));
            __m128i vs = _mm_loadu_si128((const __m128i *)(s + x));
            __m128i sum = _mm_add_epi16(vd, vs);
            sum = _mm_add_epi16(sum, one);
            sum = _mm_srli_epi16(sum, 1);
            _mm_storeu_si128((__m128i *)(d + x), sum);
        }
        for (; x < width; x++) {
            int v = d[x] + s[x];
            d[x] = (pel_t)((v + 1) >> 1);
        }
        dst += i_dst;
        src += i_src;
    }
}

/* MC + 双向平均: 亮度 (SSE4.1) */
static void mc_luma_avg_sse4(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                              ptrdiff_t dstride, int w, int h, int mx, int my,
                              int bit_depth)
{
    if (bit_depth > 8) {
        int dx = mx & 3;
        int dy = my & 3;
        int max_val = (1 << bit_depth) - 1;
        pel_t *dst16 = (pel_t *)(void *)dst;
        const pel_t *src16 = (const pel_t *)(const void *)src;
        int i_dst = (int)(dstride >> 1);
        int i_src = (int)(sstride >> 1);

        if (dx == 0 && dy == 0) {
            block_avg_10bit_sse4(dst16, i_dst, src16, i_src, w, h);
        } else if (dx == 0) {
            AVS2_ALIGN16(short pred2[64 * 64]);
            ip_filter_luma_ver_10bit_sse4((pel_t *)pred2, w, src16, i_src,
                                          w, h, avs2_intpl_filters[dy], max_val);
            block_avg_10bit_sse4(dst16, i_dst, (const pel_t *)pred2, w, w, h);
        } else if (dy == 0) {
            AVS2_ALIGN16(short pred2[64 * 64]);
            ip_filter_luma_hor_10bit_sse4((pel_t *)pred2, w, src16, i_src,
                                          w, h, avs2_intpl_filters[dx], max_val);
            block_avg_10bit_sse4(dst16, i_dst, (const pel_t *)pred2, w, w, h);
        } else {
            ip_filter_luma_ext_10bit_avg_sse4(dst16, i_dst, src16, i_src,
                                              w, h,
                                              avs2_intpl_filters[dx],
                                              avs2_intpl_filters[dy], max_val);
        }
    } else {
        /* 8-bit: pred2 为 uint8_t[], 步长 w 字节.
         * 若用 int16_t[] + 步长 w*2, mc_luma_sse4 按 uint8_t 写入每行前 w 字节,
         * 后 w 字节未初始化, bi_avg 读取 (int16_t 步长) 会读到随机值. */
        uint8_t pred2[64 * 64];
        mc_luma_sse4(src, sstride, pred2, w,
                     w, h, mx, my, bit_depth);
        bi_avg_8bit_u8_sse41(dst, (int)dstride, pred2, w, w, h);
    }
}

/* MC + 双向平均: 色度 (SSE4.1) */
static void mc_chroma_avg_sse4(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                                ptrdiff_t dstride, int w, int h, int mx, int my,
                                int bit_depth)
{
    if (bit_depth > 8) {
        int dx = mx & 7;
        int dy = my & 7;
        int max_val = (1 << bit_depth) - 1;
        pel_t *dst16 = (pel_t *)(void *)dst;
        const pel_t *src16 = (const pel_t *)(const void *)src;
        int i_dst = (int)(dstride >> 1);
        int i_src = (int)(sstride >> 1);

        if (dx == 0 && dy == 0) {
            block_avg_10bit_sse4(dst16, i_dst, src16, i_src, w, h);
        } else if (dx == 0) {
            AVS2_ALIGN16(short pred2[32 * 32]);
            ip_filter_chroma_ver_10bit_sse4((pel_t *)pred2, w, src16, i_src,
                                            w, h, avs2_intpl_filters_c[dy], max_val);
            block_avg_10bit_sse4(dst16, i_dst, (const pel_t *)pred2, w, w, h);
        } else if (dy == 0) {
            AVS2_ALIGN16(short pred2[32 * 32]);
            ip_filter_chroma_hor_10bit_sse4((pel_t *)pred2, w, src16, i_src,
                                            w, h, avs2_intpl_filters_c[dx], max_val);
            block_avg_10bit_sse4(dst16, i_dst, (const pel_t *)pred2, w, w, h);
        } else {
            ip_filter_chroma_ext_10bit_avg_sse4(dst16, i_dst, src16, i_src,
                                                w, h,
                                                avs2_intpl_filters_c[dx],
                                                avs2_intpl_filters_c[dy], max_val);
        }
    } else {
        uint8_t pred2[32 * 32];
        mc_chroma_sse4(src, sstride, pred2, w,
                       w, h, mx, my, bit_depth);
        bi_avg_8bit_u8_sse41(dst, (int)dstride, pred2, w, w, h);
    }
}

/* ===========================================================================
 * 第十二部分: DSP 初始化
 * =========================================================================== */

void avs2_mc_init_sse41(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.mc_luma       = mc_luma_sse4;
    avs2_dsp_table.mc_chroma     = mc_chroma_sse4;
    avs2_dsp_table.mc_luma_avg   = mc_luma_avg_sse4;
    avs2_dsp_table.mc_chroma_avg = mc_chroma_avg_sse4;
    avs2_dsp_table.bi_avg        = bi_avg_sse4;
    avs2_dsp_table.bi_avg_2src   = bi_avg_2src_sse4;
    avs2_dsp_table.fill_block    = fill_block_sse4;
}

void avs2_mc_init_avx2(const avs2_cpu_flags *flags)
{
    (void)flags;
}

void avs2_mc_init_avx512(const avs2_cpu_flags *flags)
{
    (void)flags;
}

#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON)

#include <arm_neon.h>

/* ---- 对齐宏 ---- */
#define AVS2_ALIGN16(x) x __attribute__((aligned(16)))

/* 10-bit 像素类型 (与 x86 段一致) */
typedef uint16_t pel_t;

/* 中间缓冲区步长 (覆盖最大块 64x64) */
#define MC_TMP_STRIDE 64

/* ---- C 回退函数声明 (在 mc.c 中定义, 供 8-bit 路径回退) ---- */
extern void mc_luma_c(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                      ptrdiff_t dstride, int w, int h, int mx, int my,
                      int bit_depth);
extern void mc_chroma_c(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                        ptrdiff_t dstride, int w, int h, int mx, int my,
                        int bit_depth);
extern void mc_luma_avg_c(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                          ptrdiff_t dstride, int w, int h, int mx, int my,
                          int bit_depth);
extern void mc_chroma_avg_c(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                            ptrdiff_t dstride, int w, int h, int mx, int my,
                            int bit_depth);
extern void bi_avg_c(uint8_t *dst, ptrdiff_t dst_stride, const int16_t *pred2,
                     int pred2_stride, int w, int h, int bit_depth);
extern void bi_avg_2src_c(uint8_t *dst, ptrdiff_t dst_stride,
                          const int16_t *pred1, int pred1_stride,
                          const int16_t *pred2, int pred2_stride,
                          int w, int h, int bit_depth);
extern void fill_block_c(uint8_t *dst, ptrdiff_t dst_stride, int w, int h,
                         int fill_val, int bit_depth);

/* ===========================================================================
 * NEON 运动补偿实现 (由 SSE4.1 版本逐行移植).
 *
 * 策略 (与 lf_simd.c NEON 一致): 10-bit 路径 (bit_depth>8) 使用原生 NEON,
 * 8-bit 路径回退到经过验证的 C 参考 (本机码流为 10-bit, 8-bit NEON 无法哈希
 * 校验, 不引入未验证代码).
 *
 * 位精确性: 所有 SSE 内联函数均有对应的 NEON 等价:
 *   _mm_madd_epi16    -> v_madd_epi16 (vmull_s16 + vpadd_s32)
 *   _mm_hadd_epi32    -> v_hadd_epi32 (vpaddq_s32)
 *   _mm_unpacklo/hi   -> v_unpacklo/hi_epi16 (vzip_s16)
 *   _mm_packs_epi32   -> v_packs_epi32 (vqmovn_s32, 有符号饱和)
 *   _mm_packus_epi32  -> v_packus_epi32 (vmax+ vqmovn_u32, 无符号饱和)
 *   _mm_shuffle_epi8  -> vextq_u16 组合 (色度 4 抽头滑窗)
 * =========================================================================== */

/* 通用 128-bit 寄存器类型 (类比 __m128i), 内部按 int16x8 存储 */
typedef int16x8_t v128_t;

static inline int32x4_t v128_to_s32(v128_t a)            { return vreinterpretq_s32_s16(a); }
static inline v128_t    v128_from_s32(int32x4_t a)       { return vreinterpretq_s16_s32(a); }

/* _mm_madd_epi16: 8×int16 相乘, 相邻对相加 → 4×int32, 返回 v128 */
static inline v128_t v_madd_epi16(v128_t a, v128_t b)
{
    int32x4_t lo = vmull_s16(vget_low_s16(a),  vget_low_s16(b));
    int32x4_t hi = vmull_s16(vget_high_s16(a), vget_high_s16(b));
    int32x2_t sl = vpadd_s32(vget_low_s32(lo),  vget_high_s32(lo));
    int32x2_t sh = vpadd_s32(vget_low_s32(hi),  vget_high_s32(hi));
    return v128_from_s32(vcombine_s32(sl, sh));
}

/* _mm_hadd_epi32: [a0+a1, a2+a3, b0+b1, b2+b3] */
static inline v128_t v_hadd_epi32(v128_t a, v128_t b)
{
    return v128_from_s32(vpaddq_s32(v128_to_s32(a), v128_to_s32(b)));
}

/* _mm_unpacklo_epi16 / hi */
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

/* _mm_packs_epi32: 2×(4 int32) 有符号饱和缩窄为 8 int16 */
static inline v128_t v_packs_epi32(v128_t a, v128_t b)
{
    return vcombine_s16(vqmovn_s32(v128_to_s32(a)), vqmovn_s32(v128_to_s32(b)));
}

/* _mm_packus_epi32: 2×(4 int32) 无符号饱和缩窄为 8 uint16
 * 负数饱和为 0, >65535 饱和为 65535 (先 max 到 0, 再无符号饱和窄化) */
static inline v128_t v_packus_epi32(v128_t a, v128_t b)
{
    const int32x4_t zero = vdupq_n_s32(0);
    uint16x4_t lo = vqmovn_u32((uint32x4_t)vmaxq_s32(v128_to_s32(a), zero));
    uint16x4_t hi = vqmovn_u32((uint32x4_t)vmaxq_s32(v128_to_s32(b), zero));
    return vreinterpretq_s16_u16(vcombine_u16(lo, hi));
}

/* int32 运算 */
static inline v128_t v_add_epi32(v128_t a, v128_t b) { return v128_from_s32(vaddq_s32(v128_to_s32(a), v128_to_s32(b))); }
static inline v128_t v_srai_epi32(v128_t a, int n)   { return v128_from_s32(vshrq_n_s32(v128_to_s32(a), n)); }

/* int16 运算 */
static inline v128_t v_add_epi16(v128_t a, v128_t b) { return vaddq_s16(a, b); }
static inline v128_t v_srli_epi16(v128_t a, int n)   { return vreinterpretq_s16_u16(vshrq_n_u16(vreinterpretq_u16_s16(a), (unsigned)n)); }
static inline v128_t v_min_epu16(v128_t a, v128_t b) { return vreinterpretq_s16_u16(vminq_u16(vreinterpretq_u16_s16(a), vreinterpretq_u16_s16(b))); }

/* 常量与访存 */
static inline v128_t v_set1_epi16(int16_t x) { return vdupq_n_s16(x); }
static inline v128_t v_set1_epi32(int32_t x) { return v128_from_s32(vdupq_n_s32(x)); }
static inline v128_t v_load(const void *p)   { return vld1q_s16((const int16_t*)p); }
static inline void   v_store(void *p, v128_t v) { vst1q_s16((int16_t*)p, v); }

/* ===========================================================================
 * 块拷贝 (10-bit NEON): 与 SSE4.1 相同的 8 像素/次 128-bit 循环
 * =========================================================================== */
static void block_copy_10bit_neon(pel_t *dst, int i_dst,
                                  const pel_t *src, int i_src,
                                  int width, int height)
{
    int y, x;
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x += 8) {
            uint16x8_t v = vld1q_u16((const uint16_t*)(src + x));
            vst1q_u16((uint16_t*)(dst + x), v);
        }
        for (; x < width; x++) {
            dst[x] = src[x];
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ===========================================================================
 * 亮度 8 抽头水平插值 (10-bit NEON)
 * =========================================================================== */
static void ip_filter_luma_hor_10bit_neon(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height, const int8_t *coeff, int max_val)
{
    int j, i;
    v128_t max_val1 = v_set1_epi16((int16_t)max_val);
    v128_t offset = v_set1_epi32(32);
    v128_t mCoef = vmovl_s8(vld1_s8(coeff));  /* 8 int8 系数 → 8 int16 */

    src -= 3;

    for (j = 0; j < height; j++) {
        const pel_t *p = src;
        for (i = 0; i + 7 < width; i += 8) {
            v128_t T0, T1, T2, T3, T4, T5, T6, T7;
            v128_t M0, M1, M2, M3, M4, M5, M6, M7;
            v128_t A0, A1, A2, A3, B0, B1;

            T0 = v_load(p++);
            T1 = v_load(p++);
            T2 = v_load(p++);
            T3 = v_load(p++);
            T4 = v_load(p++);
            T5 = v_load(p++);
            T6 = v_load(p++);
            T7 = v_load(p++);

            M0 = v_madd_epi16(T0, mCoef);
            M1 = v_madd_epi16(T1, mCoef);
            M2 = v_madd_epi16(T2, mCoef);
            M3 = v_madd_epi16(T3, mCoef);
            M4 = v_madd_epi16(T4, mCoef);
            M5 = v_madd_epi16(T5, mCoef);
            M6 = v_madd_epi16(T6, mCoef);
            M7 = v_madd_epi16(T7, mCoef);

            A0 = v_hadd_epi32(M0, M1);
            A1 = v_hadd_epi32(M2, M3);
            A2 = v_hadd_epi32(M4, M5);
            A3 = v_hadd_epi32(M6, M7);

            B0 = v_hadd_epi32(A0, A1);
            B1 = v_hadd_epi32(A2, A3);

            B0 = v_add_epi32(B0, offset);
            B1 = v_add_epi32(B1, offset);
            B0 = v_srai_epi32(B0, 6);
            B1 = v_srai_epi32(B1, 6);
            B0 = v_packus_epi32(B0, B1);
            B0 = v_min_epu16(B0, max_val1);
            v_store(dst + i, B0);
        }
        for (; i < width; i++) {
            int v = src[i]     * coeff[0] + src[i + 1] * coeff[1]
                  + src[i + 2] * coeff[2] + src[i + 3] * coeff[3]
                  + src[i + 4] * coeff[4] + src[i + 5] * coeff[5]
                  + src[i + 6] * coeff[6] + src[i + 7] * coeff[7];
            v = (v + 32) >> 6;
            dst[i] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
        }
        dst += i_dst;
        src += i_src;
    }
}

/* ===========================================================================
 * 亮度 8 抽头垂直插值 (10-bit NEON)
 * =========================================================================== */
static void ip_filter_luma_ver_10bit_neon(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height, const int8_t *coeff, int max_val)
{
    int i, j;
    v128_t max_val1 = v_set1_epi16((int16_t)max_val);
    v128_t mAddOffset = v_set1_epi32(32);

    /* 每对相邻系数 (c0,c1)/(c2,c3)/(c4,c5)/(c6,c7) 重复 4 次, 与 SSE 一致 */
    v128_t coeff00 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coeff + 0))));
    v128_t coeff01 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coeff + 2))));
    v128_t coeff02 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coeff + 4))));
    v128_t coeff03 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coeff + 6))));

    src -= 3 * i_src;

    for (j = 0; j < height; j++) {
        const pel_t *p = src;
        for (i = 0; i + 7 < width; i += 8) {
            v128_t T0, T1, T2, T3, T4, T5, T6, T7;
            v128_t M0_lo, M0_hi, M1_lo, M1_hi, M2_lo, M2_hi, M3_lo, M3_hi;
            v128_t N0_lo, N0_hi, N1_lo, N1_hi, N2_lo, N2_hi, N3_lo, N3_hi;
            v128_t sum_lo, sum_hi;

            T0 = v_load(p);
            T1 = v_load(p + i_src);
            T2 = v_load(p + 2 * i_src);
            T3 = v_load(p + 3 * i_src);
            T4 = v_load(p + 4 * i_src);
            T5 = v_load(p + 5 * i_src);
            T6 = v_load(p + 6 * i_src);
            T7 = v_load(p + 7 * i_src);

            M0_lo = v_unpacklo_epi16(T0, T1);
            M0_hi = v_unpackhi_epi16(T0, T1);
            M1_lo = v_unpacklo_epi16(T2, T3);
            M1_hi = v_unpackhi_epi16(T2, T3);
            M2_lo = v_unpacklo_epi16(T4, T5);
            M2_hi = v_unpackhi_epi16(T4, T5);
            M3_lo = v_unpacklo_epi16(T6, T7);
            M3_hi = v_unpackhi_epi16(T6, T7);

            N0_lo = v_madd_epi16(M0_lo, coeff00);
            N0_hi = v_madd_epi16(M0_hi, coeff00);
            N1_lo = v_madd_epi16(M1_lo, coeff01);
            N1_hi = v_madd_epi16(M1_hi, coeff01);
            N2_lo = v_madd_epi16(M2_lo, coeff02);
            N2_hi = v_madd_epi16(M2_hi, coeff02);
            N3_lo = v_madd_epi16(M3_lo, coeff03);
            N3_hi = v_madd_epi16(M3_hi, coeff03);

            sum_lo = v_add_epi32(v_add_epi32(N0_lo, N1_lo), v_add_epi32(N2_lo, N3_lo));
            sum_hi = v_add_epi32(v_add_epi32(N0_hi, N1_hi), v_add_epi32(N2_hi, N3_hi));

            sum_lo = v_add_epi32(sum_lo, mAddOffset);
            sum_hi = v_add_epi32(sum_hi, mAddOffset);
            sum_lo = v_srai_epi32(sum_lo, 6);
            sum_hi = v_srai_epi32(sum_hi, 6);

            sum_lo = v_packus_epi32(sum_lo, sum_hi);
            sum_lo = v_min_epu16(sum_lo, max_val1);
            v_store(dst + i, sum_lo);

            p += 8;
        }
        for (; i < width; i++) {
            int v = src[i]              * coeff[0]
                  + src[i + i_src]      * coeff[1]
                  + src[i + 2 * i_src]  * coeff[2]
                  + src[i + 3 * i_src]  * coeff[3]
                  + src[i + 4 * i_src]  * coeff[4]
                  + src[i + 5 * i_src]  * coeff[5]
                  + src[i + 6 * i_src]  * coeff[6]
                  + src[i + 7 * i_src]  * coeff[7];
            v = (v + 32) >> 6;
            dst[i] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ===========================================================================
 * 亮度 8 抽头双向插值 (10-bit NEON): 先水平 (shift1=2) 后垂直 (shift2=10)
 * =========================================================================== */
static void ip_filter_luma_ext_10bit_neon(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y, int max_val)
{
    AVS2_ALIGN16(short tmp_res[(64 + 7) * 64]);
    short *tmp = tmp_res;
    const int i_tmp = MC_TMP_STRIDE;
    int shift1, shift2, add1, add2;
    int i, j;
    v128_t offset;
    v128_t max_val1 = v_set1_epi16((int16_t)max_val);
    v128_t mCoef = vmovl_s8(vld1_s8(coef_x));

    shift1 = 2;
    shift2 = 10;
    add1 = (1 << shift1) >> 1;
    add2 = 1 << (shift2 - 1);

    /* ---- 第一级: 水平滤波 ---- */
    src += -3 * i_src - 3;
    offset = v_set1_epi32(add1);

    for (j = -3; j < height + 4; j++) {
        const pel_t *p = src;
        for (i = 0; i + 7 < width; i += 8) {
            v128_t T0, T1, T2, T3, T4, T5, T6, T7;
            v128_t M0, M1, M2, M3, M4, M5, M6, M7;
            v128_t A0, A1, A2, A3, B0, B1;

            T0 = v_load(p++);
            T1 = v_load(p++);
            T2 = v_load(p++);
            T3 = v_load(p++);
            T4 = v_load(p++);
            T5 = v_load(p++);
            T6 = v_load(p++);
            T7 = v_load(p++);

            M0 = v_madd_epi16(T0, mCoef);
            M1 = v_madd_epi16(T1, mCoef);
            M2 = v_madd_epi16(T2, mCoef);
            M3 = v_madd_epi16(T3, mCoef);
            M4 = v_madd_epi16(T4, mCoef);
            M5 = v_madd_epi16(T5, mCoef);
            M6 = v_madd_epi16(T6, mCoef);
            M7 = v_madd_epi16(T7, mCoef);

            A0 = v_hadd_epi32(M0, M1);
            A1 = v_hadd_epi32(M2, M3);
            A2 = v_hadd_epi32(M4, M5);
            A3 = v_hadd_epi32(M6, M7);

            B0 = v_hadd_epi32(A0, A1);
            B1 = v_hadd_epi32(A2, A3);

            B0 = v_add_epi32(B0, offset);
            B1 = v_add_epi32(B1, offset);
            B0 = v_srai_epi32(B0, shift1);
            B1 = v_srai_epi32(B1, shift1);
            B0 = v_packs_epi32(B0, B1);   /* 有符号饱和 → int16 中间缓冲 */
            v_store(tmp + i, B0);
        }
        for (; i < width; i++) {
            int v = src[i]     * coef_x[0] + src[i + 1] * coef_x[1]
                  + src[i + 2] * coef_x[2] + src[i + 3] * coef_x[3]
                  + src[i + 4] * coef_x[4] + src[i + 5] * coef_x[5]
                  + src[i + 6] * coef_x[6] + src[i + 7] * coef_x[7];
            v = (v + add1) >> shift1;
            tmp[i] = (short)v;
        }
        tmp += i_tmp;
        src += i_src;
    }

    /* ---- 第二级: 垂直滤波 ---- */
    offset = v_set1_epi32(add2);
    tmp = tmp_res;

    {
        v128_t coeff00 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 0))));
        v128_t coeff01 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 2))));
        v128_t coeff02 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 4))));
        v128_t coeff03 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 6))));

        for (j = 0; j < height; j++) {
            const short *p = tmp;
            for (i = 0; i + 7 < width; i += 8) {
                v128_t T0, T1, T2, T3, T4, T5, T6, T7;
                v128_t M0_lo, M0_hi, M1_lo, M1_hi, M2_lo, M2_hi, M3_lo, M3_hi;
                v128_t N0_lo, N0_hi, N1_lo, N1_hi, N2_lo, N2_hi, N3_lo, N3_hi;
                v128_t sum_lo, sum_hi;

                T0 = v_load(p);
                T1 = v_load(p + i_tmp);
                T2 = v_load(p + 2 * i_tmp);
                T3 = v_load(p + 3 * i_tmp);
                T4 = v_load(p + 4 * i_tmp);
                T5 = v_load(p + 5 * i_tmp);
                T6 = v_load(p + 6 * i_tmp);
                T7 = v_load(p + 7 * i_tmp);

                M0_lo = v_unpacklo_epi16(T0, T1);
                M0_hi = v_unpackhi_epi16(T0, T1);
                M1_lo = v_unpacklo_epi16(T2, T3);
                M1_hi = v_unpackhi_epi16(T2, T3);
                M2_lo = v_unpacklo_epi16(T4, T5);
                M2_hi = v_unpackhi_epi16(T4, T5);
                M3_lo = v_unpacklo_epi16(T6, T7);
                M3_hi = v_unpackhi_epi16(T6, T7);

                N0_lo = v_madd_epi16(M0_lo, coeff00);
                N0_hi = v_madd_epi16(M0_hi, coeff00);
                N1_lo = v_madd_epi16(M1_lo, coeff01);
                N1_hi = v_madd_epi16(M1_hi, coeff01);
                N2_lo = v_madd_epi16(M2_lo, coeff02);
                N2_hi = v_madd_epi16(M2_hi, coeff02);
                N3_lo = v_madd_epi16(M3_lo, coeff03);
                N3_hi = v_madd_epi16(M3_hi, coeff03);

                sum_lo = v_add_epi32(v_add_epi32(N0_lo, N1_lo), v_add_epi32(N2_lo, N3_lo));
                sum_hi = v_add_epi32(v_add_epi32(N0_hi, N1_hi), v_add_epi32(N2_hi, N3_hi));

                sum_lo = v_add_epi32(sum_lo, offset);
                sum_hi = v_add_epi32(sum_hi, offset);
                sum_lo = v_srai_epi32(sum_lo, shift2);
                sum_hi = v_srai_epi32(sum_hi, shift2);

                sum_lo = v_packus_epi32(sum_lo, sum_hi);
                sum_lo = v_min_epu16(sum_lo, max_val1);
                v_store(dst + i, sum_lo);

                p += 8;
            }
            for (; i < width; i++) {
                int v = tmp[i]              * coef_y[0]
                      + tmp[i + i_tmp]      * coef_y[1]
                      + tmp[i + 2 * i_tmp]  * coef_y[2]
                      + tmp[i + 3 * i_tmp]  * coef_y[3]
                      + tmp[i + 4 * i_tmp]  * coef_y[4]
                      + tmp[i + 5 * i_tmp]  * coef_y[5]
                      + tmp[i + 6 * i_tmp]  * coef_y[6]
                      + tmp[i + 7 * i_tmp]  * coef_y[7];
                v = (v + add2) >> shift2;
                dst[i] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
            }
            dst += i_dst;
            tmp += i_tmp;
        }
    }
}

/* ---- 融合 MC+avg: 第二级垂直滤波后与 dst 平均 ---- */
static void ip_filter_luma_ext_10bit_avg_neon(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y, int max_val)
{
    AVS2_ALIGN16(short tmp_res[(64 + 7) * 64]);
    short *tmp = tmp_res;
    const int i_tmp = MC_TMP_STRIDE;
    int shift1, shift2, add1, add2;
    int i, j;
    v128_t offset;
    v128_t max_val1 = v_set1_epi16((int16_t)max_val);
    v128_t one = v_set1_epi16(1);
    v128_t mCoef = vmovl_s8(vld1_s8(coef_x));

    shift1 = 2;
    shift2 = 10;
    add1 = (1 << shift1) >> 1;
    add2 = 1 << (shift2 - 1);

    /* ---- 第一级: 水平滤波 (与普通版相同) ---- */
    src += -3 * i_src - 3;
    offset = v_set1_epi32(add1);

    for (j = -3; j < height + 4; j++) {
        const pel_t *p = src;
        for (i = 0; i + 7 < width; i += 8) {
            v128_t T0, T1, T2, T3, T4, T5, T6, T7;
            v128_t M0, M1, M2, M3, M4, M5, M6, M7;
            v128_t A0, A1, A2, A3, B0, B1;

            T0 = v_load(p++);
            T1 = v_load(p++);
            T2 = v_load(p++);
            T3 = v_load(p++);
            T4 = v_load(p++);
            T5 = v_load(p++);
            T6 = v_load(p++);
            T7 = v_load(p++);

            M0 = v_madd_epi16(T0, mCoef);
            M1 = v_madd_epi16(T1, mCoef);
            M2 = v_madd_epi16(T2, mCoef);
            M3 = v_madd_epi16(T3, mCoef);
            M4 = v_madd_epi16(T4, mCoef);
            M5 = v_madd_epi16(T5, mCoef);
            M6 = v_madd_epi16(T6, mCoef);
            M7 = v_madd_epi16(T7, mCoef);

            A0 = v_hadd_epi32(M0, M1);
            A1 = v_hadd_epi32(M2, M3);
            A2 = v_hadd_epi32(M4, M5);
            A3 = v_hadd_epi32(M6, M7);

            B0 = v_hadd_epi32(A0, A1);
            B1 = v_hadd_epi32(A2, A3);

            B0 = v_add_epi32(B0, offset);
            B1 = v_add_epi32(B1, offset);
            B0 = v_srai_epi32(B0, shift1);
            B1 = v_srai_epi32(B1, shift1);
            B0 = v_packs_epi32(B0, B1);
            v_store(tmp + i, B0);
        }
        for (; i < width; i++) {
            int v = src[i]     * coef_x[0] + src[i + 1] * coef_x[1]
                  + src[i + 2] * coef_x[2] + src[i + 3] * coef_x[3]
                  + src[i + 4] * coef_x[4] + src[i + 5] * coef_x[5]
                  + src[i + 6] * coef_x[6] + src[i + 7] * coef_x[7];
            v = (v + add1) >> shift1;
            tmp[i] = (short)v;
        }
        tmp += i_tmp;
        src += i_src;
    }

    /* ---- 第二级: 垂直滤波 + 融合平均 ---- */
    offset = v_set1_epi32(add2);
    tmp = tmp_res;

    {
        v128_t coeff00 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 0))));
        v128_t coeff01 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 2))));
        v128_t coeff02 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 4))));
        v128_t coeff03 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 6))));

        for (j = 0; j < height; j++) {
            const short *p = tmp;
            for (i = 0; i + 7 < width; i += 8) {
                v128_t T0, T1, T2, T3, T4, T5, T6, T7;
                v128_t M0_lo, M0_hi, M1_lo, M1_hi, M2_lo, M2_hi, M3_lo, M3_hi;
                v128_t N0_lo, N0_hi, N1_lo, N1_hi, N2_lo, N2_hi, N3_lo, N3_hi;
                v128_t sum_lo, sum_hi, vdst, vsum;

                T0 = v_load(p);
                T1 = v_load(p + i_tmp);
                T2 = v_load(p + 2 * i_tmp);
                T3 = v_load(p + 3 * i_tmp);
                T4 = v_load(p + 4 * i_tmp);
                T5 = v_load(p + 5 * i_tmp);
                T6 = v_load(p + 6 * i_tmp);
                T7 = v_load(p + 7 * i_tmp);

                M0_lo = v_unpacklo_epi16(T0, T1);
                M0_hi = v_unpackhi_epi16(T0, T1);
                M1_lo = v_unpacklo_epi16(T2, T3);
                M1_hi = v_unpackhi_epi16(T2, T3);
                M2_lo = v_unpacklo_epi16(T4, T5);
                M2_hi = v_unpackhi_epi16(T4, T5);
                M3_lo = v_unpacklo_epi16(T6, T7);
                M3_hi = v_unpackhi_epi16(T6, T7);

                N0_lo = v_madd_epi16(M0_lo, coeff00);
                N0_hi = v_madd_epi16(M0_hi, coeff00);
                N1_lo = v_madd_epi16(M1_lo, coeff01);
                N1_hi = v_madd_epi16(M1_hi, coeff01);
                N2_lo = v_madd_epi16(M2_lo, coeff02);
                N2_hi = v_madd_epi16(M2_hi, coeff02);
                N3_lo = v_madd_epi16(M3_lo, coeff03);
                N3_hi = v_madd_epi16(M3_hi, coeff03);

                sum_lo = v_add_epi32(v_add_epi32(N0_lo, N1_lo), v_add_epi32(N2_lo, N3_lo));
                sum_hi = v_add_epi32(v_add_epi32(N0_hi, N1_hi), v_add_epi32(N2_hi, N3_hi));

                sum_lo = v_add_epi32(sum_lo, offset);
                sum_hi = v_add_epi32(sum_hi, offset);
                sum_lo = v_srai_epi32(sum_lo, shift2);
                sum_hi = v_srai_epi32(sum_hi, shift2);

                sum_lo = v_packus_epi32(sum_lo, sum_hi);
                sum_lo = v_min_epu16(sum_lo, max_val1);

                /* 融合平均: dst[i] = (dst[i] + result[i] + 1) >> 1 */
                vdst = v_load(dst + i);
                vsum = v_add_epi16(vdst, sum_lo);
                vsum = v_add_epi16(vsum, one);
                vsum = v_srli_epi16(vsum, 1);
                v_store(dst + i, vsum);

                p += 8;
            }
            for (; i < width; i++) {
                int v = tmp[i]              * coef_y[0]
                      + tmp[i + i_tmp]      * coef_y[1]
                      + tmp[i + 2 * i_tmp]  * coef_y[2]
                      + tmp[i + 3 * i_tmp]  * coef_y[3]
                      + tmp[i + 4 * i_tmp]  * coef_y[4]
                      + tmp[i + 5 * i_tmp]  * coef_y[5]
                      + tmp[i + 6 * i_tmp]  * coef_y[6]
                      + tmp[i + 7 * i_tmp]  * coef_y[7];
                v = (v + add2) >> shift2;
                v = v < 0 ? 0 : (v > max_val ? max_val : v);
                dst[i] = (pel_t)((dst[i] + v + 1) >> 1);
            }
            dst += i_dst;
            tmp += i_tmp;
        }
    }
}

/* ===========================================================================
 * 色度 4 抽头水平插值 (10-bit NEON)
 *
 * SSE 用 shuffle_epi8 构造滑窗, NEON 用 vextq_u16 构造等价窗口:
 *   s1 = [p0,p1,p2,p3, p1,p2,p3,p4], s2 = [p2,p3,p4,p5, p3,p4,p5,p6]
 * =========================================================================== */
static void ip_filter_chroma_hor_10bit_neon(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height, const int8_t *coeff, int max_val)
{
    int row, col;
    const int offset_val = 32;
    const int shift = 6;

    /* 4 个 int8 系数 → [c0,c1,c2,c3, c0,c1,c2,c3] */
    v128_t mCoef = vmovl_s8(vreinterpret_s8_s32(vdup_n_s32(*(const int32_t*)coeff)));
    v128_t mAddOffset = v_set1_epi32(offset_val);
    v128_t max_val1 = v_set1_epi16((int16_t)max_val);

    src -= 1;

    for (row = 0; row < height; row++) {
        const pel_t *p = src;
        for (col = 0; col + 7 < width; col += 8) {
            uint16x8_t s;
            uint16x4_t lo1, lo2;
            v128_t s1, s2, m1, m2, sum_lo;
            v128_t s3, s4, m3, m4, sum_hi;

            /* 前 4 输出: p[col..col+6] */
            s = vld1q_u16(p + col);                  /* [p0..p7] */
            lo1 = vget_low_u16(vextq_u16(s, s, 1));  /* [p1,p2,p3,p4] */
            lo2 = vget_low_u16(vextq_u16(s, s, 3));  /* [p3,p4,p5,p6] */
            s1 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(s), lo1)); /* [p0,p1,p2,p3, p1,p2,p3,p4] */
            s2 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(vextq_u16(s, s, 2)), lo2)); /* [p2,p3,p4,p5, p3,p4,p5,p6] */
            m1 = v_madd_epi16(s1, mCoef);
            m2 = v_madd_epi16(s2, mCoef);
            sum_lo = v_hadd_epi32(m1, m2);

            /* 后 4 输出: p[col+4..col+10] */
            s = vld1q_u16(p + col + 4);
            lo1 = vget_low_u16(vextq_u16(s, s, 1));
            lo2 = vget_low_u16(vextq_u16(s, s, 3));
            s3 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(s), lo1));
            s4 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(vextq_u16(s, s, 2)), lo2));
            m3 = v_madd_epi16(s3, mCoef);
            m4 = v_madd_epi16(s4, mCoef);
            sum_hi = v_hadd_epi32(m3, m4);

            sum_lo = v_add_epi32(sum_lo, mAddOffset);
            sum_hi = v_add_epi32(sum_hi, mAddOffset);
            sum_lo = v_srai_epi32(sum_lo, shift);
            sum_hi = v_srai_epi32(sum_hi, shift);
            sum_lo = v_packus_epi32(sum_lo, sum_hi);
            sum_lo = v_min_epu16(sum_lo, max_val1);
            v_store(dst + col, sum_lo);
        }
        for (; col < width; col++) {
            int v = src[col]     * coeff[0] + src[col + 1] * coeff[1]
                  + src[col + 2] * coeff[2] + src[col + 3] * coeff[3];
            v = (v + offset_val) >> shift;
            dst[col] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ===========================================================================
 * 色度 4 抽头垂直插值 (10-bit NEON)
 * =========================================================================== */
static void ip_filter_chroma_ver_10bit_neon(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height, const int8_t *coeff, int max_val)
{
    int row, col;
    const int offset_val = 32;
    const int shift = 6;
    v128_t mAddOffset = v_set1_epi32(offset_val);
    v128_t max_val1 = v_set1_epi16((int16_t)max_val);
    const int i_src2 = i_src * 2;
    const int i_src3 = i_src * 3;

    v128_t coeff0 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coeff + 0))));
    v128_t coeff1 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coeff + 2))));

    src -= i_src;

    for (row = 0; row < height; row++) {
        const pel_t *p = src;
        for (col = 0; col + 7 < width; col += 8) {
            v128_t S0, S1, S2, S3;
            v128_t T0_lo, T0_hi, T1_lo, T1_hi;
            v128_t N0_lo, N0_hi, N1_lo, N1_hi;
            v128_t sum_lo, sum_hi;

            S0 = v_load(p + col);
            S1 = v_load(p + col + i_src);
            S2 = v_load(p + col + i_src2);
            S3 = v_load(p + col + i_src3);

            T0_lo = v_unpacklo_epi16(S0, S1);
            T0_hi = v_unpackhi_epi16(S0, S1);
            T1_lo = v_unpacklo_epi16(S2, S3);
            T1_hi = v_unpackhi_epi16(S2, S3);

            N0_lo = v_madd_epi16(T0_lo, coeff0);
            N0_hi = v_madd_epi16(T0_hi, coeff0);
            N1_lo = v_madd_epi16(T1_lo, coeff1);
            N1_hi = v_madd_epi16(T1_hi, coeff1);

            sum_lo = v_add_epi32(N0_lo, N1_lo);
            sum_hi = v_add_epi32(N0_hi, N1_hi);

            sum_lo = v_add_epi32(sum_lo, mAddOffset);
            sum_hi = v_add_epi32(sum_hi, mAddOffset);
            sum_lo = v_srai_epi32(sum_lo, shift);
            sum_hi = v_srai_epi32(sum_hi, shift);
            sum_lo = v_packus_epi32(sum_lo, sum_hi);
            sum_lo = v_min_epu16(sum_lo, max_val1);
            v_store(dst + col, sum_lo);
        }
        for (; col < width; col++) {
            int v = src[col]             * coeff[0]
                  + src[col + i_src]     * coeff[1]
                  + src[col + i_src2]    * coeff[2]
                  + src[col + i_src3]    * coeff[3];
            v = (v + offset_val) >> shift;
            dst[col] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ===========================================================================
 * 色度 4 抽头双向插值 (10-bit NEON): 先水平 (shift1=2) 后垂直 (shift2=10)
 * =========================================================================== */
static void ip_filter_chroma_ext_10bit_neon(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y, int max_val)
{
    AVS2_ALIGN16(short tmp_res[(32 + 3) * 32]);
    short *tmp = tmp_res;
    const int i_tmp = 32;
    int shift1, shift2, add1, add2;
    int row, col;
    v128_t max_val1 = v_set1_epi16((int16_t)max_val);
    v128_t mAddOffset;
    v128_t mCoef = vmovl_s8(vreinterpret_s8_s32(vdup_n_s32(*(const int32_t*)coef_x)));

    shift1 = 2;
    shift2 = 10;
    add1 = (1 << shift1) >> 1;
    add2 = 1 << (shift2 - 1);

    /* ---- 第一级: 水平滤波 ---- */
    mAddOffset = v_set1_epi32(add1);
    src = src - i_src - 1;

    for (row = -1; row < height + 2; row++) {
        const pel_t *p = src;
        for (col = 0; col + 7 < width; col += 8) {
            uint16x8_t s;
            uint16x4_t lo1, lo2;
            v128_t s1, s2, m1, m2, sum_lo;
            v128_t s3, s4, m3, m4, sum_hi;

            s = vld1q_u16(p + col);
            lo1 = vget_low_u16(vextq_u16(s, s, 1));
            lo2 = vget_low_u16(vextq_u16(s, s, 3));
            s1 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(s), lo1));
            s2 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(vextq_u16(s, s, 2)), lo2));
            m1 = v_madd_epi16(s1, mCoef);
            m2 = v_madd_epi16(s2, mCoef);
            sum_lo = v_hadd_epi32(m1, m2);

            s = vld1q_u16(p + col + 4);
            lo1 = vget_low_u16(vextq_u16(s, s, 1));
            lo2 = vget_low_u16(vextq_u16(s, s, 3));
            s3 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(s), lo1));
            s4 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(vextq_u16(s, s, 2)), lo2));
            m3 = v_madd_epi16(s3, mCoef);
            m4 = v_madd_epi16(s4, mCoef);
            sum_hi = v_hadd_epi32(m3, m4);

            sum_lo = v_add_epi32(sum_lo, mAddOffset);
            sum_hi = v_add_epi32(sum_hi, mAddOffset);
            sum_lo = v_srai_epi32(sum_lo, shift1);
            sum_hi = v_srai_epi32(sum_hi, shift1);
            sum_lo = v_packs_epi32(sum_lo, sum_hi);
            v_store(tmp + col, sum_lo);
        }
        for (; col < width; col++) {
            int v = src[col]     * coef_x[0] + src[col + 1] * coef_x[1]
                  + src[col + 2] * coef_x[2] + src[col + 3] * coef_x[3];
            v = (v + add1) >> shift1;
            tmp[col] = (short)v;
        }
        src += i_src;
        tmp += i_tmp;
    }

    /* ---- 第二级: 垂直滤波 ---- */
    tmp = tmp_res;
    mAddOffset = v_set1_epi32(add2);

    {
        v128_t coeffy0 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 0))));
        v128_t coeffy1 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 2))));

        for (row = 0; row < height; row++) {
            for (col = 0; col + 7 < width; col += 8) {
                v128_t S0, S1, S2, S3;
                v128_t T0_lo, T0_hi, T1_lo, T1_hi;
                v128_t N0_lo, N0_hi, N1_lo, N1_hi;
                v128_t sum_lo, sum_hi;

                S0 = v_load(tmp + col);
                S1 = v_load(tmp + col + i_tmp);
                S2 = v_load(tmp + col + 2 * i_tmp);
                S3 = v_load(tmp + col + 3 * i_tmp);

                T0_lo = v_unpacklo_epi16(S0, S1);
                T0_hi = v_unpackhi_epi16(S0, S1);
                T1_lo = v_unpacklo_epi16(S2, S3);
                T1_hi = v_unpackhi_epi16(S2, S3);

                N0_lo = v_madd_epi16(T0_lo, coeffy0);
                N0_hi = v_madd_epi16(T0_hi, coeffy0);
                N1_lo = v_madd_epi16(T1_lo, coeffy1);
                N1_hi = v_madd_epi16(T1_hi, coeffy1);

                sum_lo = v_add_epi32(N0_lo, N1_lo);
                sum_hi = v_add_epi32(N0_hi, N1_hi);

                sum_lo = v_add_epi32(sum_lo, mAddOffset);
                sum_hi = v_add_epi32(sum_hi, mAddOffset);
                sum_lo = v_srai_epi32(sum_lo, shift2);
                sum_hi = v_srai_epi32(sum_hi, shift2);
                sum_lo = v_packus_epi32(sum_lo, sum_hi);
                sum_lo = v_min_epu16(sum_lo, max_val1);
                v_store(dst + col, sum_lo);
            }
            for (; col < width; col++) {
                int v = tmp[col]             * coef_y[0]
                      + tmp[col + i_tmp]     * coef_y[1]
                      + tmp[col + 2 * i_tmp] * coef_y[2]
                      + tmp[col + 3 * i_tmp] * coef_y[3];
                v = (v + add2) >> shift2;
                dst[col] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
            }
            dst += i_dst;
            tmp += i_tmp;
        }
    }
}

/* ---- 融合 MC+avg: 色度双向预测第二路 ---- */
static void ip_filter_chroma_ext_10bit_avg_neon(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y, int max_val)
{
    AVS2_ALIGN16(short tmp_res[(32 + 3) * 32]);
    short *tmp = tmp_res;
    const int i_tmp = 32;
    int shift1, shift2, add1, add2;
    int row, col;
    v128_t max_val1 = v_set1_epi16((int16_t)max_val);
    v128_t one16 = v_set1_epi16(1);
    v128_t mAddOffset;
    v128_t mCoef = vmovl_s8(vreinterpret_s8_s32(vdup_n_s32(*(const int32_t*)coef_x)));

    shift1 = 2;
    shift2 = 10;
    add1 = (1 << shift1) >> 1;
    add2 = 1 << (shift2 - 1);

    /* ---- 第一级: 水平滤波 (与普通版相同) ---- */
    mAddOffset = v_set1_epi32(add1);
    src = src - i_src - 1;

    for (row = -1; row < height + 2; row++) {
        const pel_t *p = src;
        for (col = 0; col + 7 < width; col += 8) {
            uint16x8_t s;
            uint16x4_t lo1, lo2;
            v128_t s1, s2, m1, m2, sum_lo;
            v128_t s3, s4, m3, m4, sum_hi;

            s = vld1q_u16(p + col);
            lo1 = vget_low_u16(vextq_u16(s, s, 1));
            lo2 = vget_low_u16(vextq_u16(s, s, 3));
            s1 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(s), lo1));
            s2 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(vextq_u16(s, s, 2)), lo2));
            m1 = v_madd_epi16(s1, mCoef);
            m2 = v_madd_epi16(s2, mCoef);
            sum_lo = v_hadd_epi32(m1, m2);

            s = vld1q_u16(p + col + 4);
            lo1 = vget_low_u16(vextq_u16(s, s, 1));
            lo2 = vget_low_u16(vextq_u16(s, s, 3));
            s3 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(s), lo1));
            s4 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(vextq_u16(s, s, 2)), lo2));
            m3 = v_madd_epi16(s3, mCoef);
            m4 = v_madd_epi16(s4, mCoef);
            sum_hi = v_hadd_epi32(m3, m4);

            sum_lo = v_add_epi32(sum_lo, mAddOffset);
            sum_hi = v_add_epi32(sum_hi, mAddOffset);
            sum_lo = v_srai_epi32(sum_lo, shift1);
            sum_hi = v_srai_epi32(sum_hi, shift1);
            sum_lo = v_packs_epi32(sum_lo, sum_hi);
            v_store(tmp + col, sum_lo);
        }
        for (; col < width; col++) {
            int v = src[col]     * coef_x[0] + src[col + 1] * coef_x[1]
                  + src[col + 2] * coef_x[2] + src[col + 3] * coef_x[3];
            v = (v + add1) >> shift1;
            tmp[col] = (short)v;
        }
        src += i_src;
        tmp += i_tmp;
    }

    /* ---- 第二级: 垂直滤波 + 融合平均 ---- */
    tmp = tmp_res;
    mAddOffset = v_set1_epi32(add2);

    {
        v128_t coeffy0 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 0))));
        v128_t coeffy1 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 2))));

        for (row = 0; row < height; row++) {
            for (col = 0; col + 7 < width; col += 8) {
                v128_t S0, S1, S2, S3;
                v128_t T0_lo, T0_hi, T1_lo, T1_hi;
                v128_t N0_lo, N0_hi, N1_lo, N1_hi;
                v128_t sum_lo, sum_hi, vdst, vsum;

                S0 = v_load(tmp + col);
                S1 = v_load(tmp + col + i_tmp);
                S2 = v_load(tmp + col + 2 * i_tmp);
                S3 = v_load(tmp + col + 3 * i_tmp);

                T0_lo = v_unpacklo_epi16(S0, S1);
                T0_hi = v_unpackhi_epi16(S0, S1);
                T1_lo = v_unpacklo_epi16(S2, S3);
                T1_hi = v_unpackhi_epi16(S2, S3);

                N0_lo = v_madd_epi16(T0_lo, coeffy0);
                N0_hi = v_madd_epi16(T0_hi, coeffy0);
                N1_lo = v_madd_epi16(T1_lo, coeffy1);
                N1_hi = v_madd_epi16(T1_hi, coeffy1);

                sum_lo = v_add_epi32(N0_lo, N1_lo);
                sum_hi = v_add_epi32(N0_hi, N1_hi);

                sum_lo = v_add_epi32(sum_lo, mAddOffset);
                sum_hi = v_add_epi32(sum_hi, mAddOffset);
                sum_lo = v_srai_epi32(sum_lo, shift2);
                sum_hi = v_srai_epi32(sum_hi, shift2);
                sum_lo = v_packus_epi32(sum_lo, sum_hi);
                sum_lo = v_min_epu16(sum_lo, max_val1);

                /* 融合平均: dst[col] = (dst[col] + result[col] + 1) >> 1 */
                vdst = v_load(dst + col);
                vsum = v_add_epi16(vdst, sum_lo);
                vsum = v_add_epi16(vsum, one16);
                vsum = v_srli_epi16(vsum, 1);
                v_store(dst + col, vsum);
            }
            for (; col < width; col++) {
                int v = tmp[col]             * coef_y[0]
                      + tmp[col + i_tmp]     * coef_y[1]
                      + tmp[col + 2 * i_tmp] * coef_y[2]
                      + tmp[col + 3 * i_tmp] * coef_y[3];
                v = (v + add2) >> shift2;
                v = v < 0 ? 0 : (v > max_val ? max_val : v);
                dst[col] = (pel_t)((dst[col] + v + 1) >> 1);
            }
            dst += i_dst;
            tmp += i_tmp;
        }
    }
}

/* ===========================================================================
 * 整像素块平均 / 双向平均 / 块填充 (10-bit NEON)
 * =========================================================================== */

/* dst[i] = (dst[i] + src[i] + 1) >> 1 */
static void block_avg_10bit_neon(pel_t *dst, int i_dst,
                                 const pel_t *src, int i_src,
                                 int width, int height)
{
    const uint16x8_t one = vdupq_n_u16(1);
    int y, x;

    for (y = 0; y < height; y++) {
        pel_t *d = dst;
        const pel_t *s = src;
        x = 0;
        for (; x + 7 < width; x += 8) {
            uint16x8_t vd = vld1q_u16(d + x);
            uint16x8_t vs = vld1q_u16(s + x);
            uint16x8_t sum = vaddq_u16(vd, vs);
            sum = vaddq_u16(sum, one);
            sum = vshrq_n_u16(sum, 1);
            vst1q_u16(d + x, sum);
        }
        for (; x < width; x++) {
            int v = d[x] + s[x];
            d[x] = (pel_t)((v + 1) >> 1);
        }
        dst += i_dst;
        src += i_src;
    }
}

/* 10-bit 双向平均: dst16[x] = (dst16[x] + pred2[x] + 1) >> 1 */
static void bi_avg_10bit_neon(uint16_t *p16, int stride16,
                              const int16_t *pred2, int pred2_stride,
                              int w, int h)
{
    const uint16x8_t one = vdupq_n_u16(1);
    int y, x;
    for (y = 0; y < h; y++) {
        uint16_t *d = p16 + y * stride16;
        const int16_t *p = pred2 + y * pred2_stride;
        x = 0;
        for (; x + 7 < w; x += 8) {
            uint16x8_t vd = vld1q_u16(d + x);
            uint16x8_t vp = vreinterpretq_u16_s16(vld1q_s16(p + x));
            vd = vaddq_u16(vd, vp);
            vd = vaddq_u16(vd, one);
            vd = vshrq_n_u16(vd, 1);
            vst1q_u16(d + x, vd);
        }
        for (; x < w; x++) {
            int v = d[x] + p[x];
            d[x] = (uint16_t)((v + 1) >> 1);
        }
    }
}

/* 10-bit 双源双向平均: dst16[x] = (pred1[x] + pred2[x] + 1) >> 1 */
static void bi_avg_2src_10bit_neon(uint16_t *p16, int stride16,
                                   const int16_t *pred1, int pred1_stride,
                                   const int16_t *pred2, int pred2_stride,
                                   int w, int h)
{
    const uint16x8_t one = vdupq_n_u16(1);
    int y, x;
    for (y = 0; y < h; y++) {
        uint16_t *d = p16 + y * stride16;
        const int16_t *p1 = pred1 + y * pred1_stride;
        const int16_t *p2 = pred2 + y * pred2_stride;
        x = 0;
        for (; x + 7 < w; x += 8) {
            uint16x8_t v1 = vreinterpretq_u16_s16(vld1q_s16(p1 + x));
            uint16x8_t v2 = vreinterpretq_u16_s16(vld1q_s16(p2 + x));
            uint16x8_t sum = vaddq_u16(v1, v2);
            sum = vaddq_u16(sum, one);
            sum = vshrq_n_u16(sum, 1);
            vst1q_u16(d + x, sum);
        }
        for (; x < w; x++) {
            int v = p1[x] + p2[x];
            d[x] = (uint16_t)((v + 1) >> 1);
        }
    }
}

/* 10-bit 块填充 */
static void fill_block_10bit_neon(uint16_t *p16, int stride16,
                                  int w, int h, int fill_val)
{
    const uint16x8_t vfill = vdupq_n_u16((uint16_t)fill_val);
    int y, x;
    for (y = 0; y < h; y++) {
        uint16_t *d = p16 + y * stride16;
        x = 0;
        for (; x + 7 < w; x += 8) {
            vst1q_u16(d + x, vfill);
        }
        for (; x < w; x++) {
            d[x] = (uint16_t)fill_val;
        }
    }
}

/* ===========================================================================
 * 8-bit NEON 运动补偿实现
 *
 * 策略: 将 uint8_t 像素加载后宽展为 int16 进行滤波运算 (与 10-bit 同构),
 * 结果饱和缩窄回 uint8 存储. 所有移位/舍入与 C 参考一致:
 *   - hor/ver: shift=6, offset=32, clip [0,255]
 *   - ext: shift1=0, shift2=12, add1=0, add2=2048
 *     (8-bit 时 shift1=bit_depth-8=0, 中间结果不移位, 范围 ±32640 fits int16)
 * =========================================================================== */

/* 加载 8 个 uint8 像素并宽展为 int16x8 (类比 10-bit 的 v_load) */
static inline v128_t v_load_u8(const uint8_t *p) {
    return vreinterpretq_s16_u16(vmovl_u8(vld1_u8(p)));
}

/* 将 int16x8 饱和缩窄为 uint8x8 并存储 8 字节 */
static inline void v_store_u8(uint8_t *p, v128_t v) {
    vst1_u8(p, vqmovn_u16(vreinterpretq_u16_s16(v)));
}

/* 8-bit 块拷贝: 16 字节/次 */
static void block_copy_8bit_neon(uint8_t *dst, int i_dst,
                                  const uint8_t *src, int i_src,
                                  int width, int height)
{
    int y, x;
    for (y = 0; y < height; y++) {
        for (x = 0; x + 15 < width; x += 16) {
            vst1q_u8(dst + x, vld1q_u8(src + x));
        }
        for (; x + 7 < width; x += 8) {
            vst1_u8(dst + x, vld1_u8(src + x));
        }
        for (; x < width; x++) {
            dst[x] = src[x];
        }
        src += i_src;
        dst += i_dst;
    }
}

/* 8-bit 亮度 8 抽头水平插值 */
static void ip_filter_luma_hor_8bit_neon(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height, const int8_t *coeff)
{
    int j, i;
    v128_t max_val1 = v_set1_epi16(255);
    v128_t offset = v_set1_epi32(32);
    v128_t mCoef = vmovl_s8(vld1_s8(coeff));

    src -= 3;

    for (j = 0; j < height; j++) {
        const uint8_t *p = src;
        for (i = 0; i + 7 < width; i += 8) {
            v128_t T0, T1, T2, T3, T4, T5, T6, T7;
            v128_t M0, M1, M2, M3, M4, M5, M6, M7;
            v128_t A0, A1, A2, A3, B0, B1;

            T0 = v_load_u8(p++);
            T1 = v_load_u8(p++);
            T2 = v_load_u8(p++);
            T3 = v_load_u8(p++);
            T4 = v_load_u8(p++);
            T5 = v_load_u8(p++);
            T6 = v_load_u8(p++);
            T7 = v_load_u8(p++);

            M0 = v_madd_epi16(T0, mCoef);
            M1 = v_madd_epi16(T1, mCoef);
            M2 = v_madd_epi16(T2, mCoef);
            M3 = v_madd_epi16(T3, mCoef);
            M4 = v_madd_epi16(T4, mCoef);
            M5 = v_madd_epi16(T5, mCoef);
            M6 = v_madd_epi16(T6, mCoef);
            M7 = v_madd_epi16(T7, mCoef);

            A0 = v_hadd_epi32(M0, M1);
            A1 = v_hadd_epi32(M2, M3);
            A2 = v_hadd_epi32(M4, M5);
            A3 = v_hadd_epi32(M6, M7);

            B0 = v_hadd_epi32(A0, A1);
            B1 = v_hadd_epi32(A2, A3);

            B0 = v_add_epi32(B0, offset);
            B1 = v_add_epi32(B1, offset);
            B0 = v_srai_epi32(B0, 6);
            B1 = v_srai_epi32(B1, 6);
            B0 = v_packus_epi32(B0, B1);
            B0 = v_min_epu16(B0, max_val1);
            v_store_u8(dst + i, B0);
        }
        for (; i < width; i++) {
            int v = src[i]     * coeff[0] + src[i + 1] * coeff[1]
                  + src[i + 2] * coeff[2] + src[i + 3] * coeff[3]
                  + src[i + 4] * coeff[4] + src[i + 5] * coeff[5]
                  + src[i + 6] * coeff[6] + src[i + 7] * coeff[7];
            v = (v + 32) >> 6;
            dst[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        dst += i_dst;
        src += i_src;
    }
}

/* 8-bit 亮度 8 抽头垂直插值 */
static void ip_filter_luma_ver_8bit_neon(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height, const int8_t *coeff)
{
    int i, j;
    v128_t max_val1 = v_set1_epi16(255);
    v128_t mAddOffset = v_set1_epi32(32);

    v128_t coeff00 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coeff + 0))));
    v128_t coeff01 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coeff + 2))));
    v128_t coeff02 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coeff + 4))));
    v128_t coeff03 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coeff + 6))));

    src -= 3 * i_src;

    for (j = 0; j < height; j++) {
        const uint8_t *p = src;
        for (i = 0; i + 7 < width; i += 8) {
            v128_t T0, T1, T2, T3, T4, T5, T6, T7;
            v128_t M0_lo, M0_hi, M1_lo, M1_hi, M2_lo, M2_hi, M3_lo, M3_hi;
            v128_t N0_lo, N0_hi, N1_lo, N1_hi, N2_lo, N2_hi, N3_lo, N3_hi;
            v128_t sum_lo, sum_hi;

            T0 = v_load_u8(p);
            T1 = v_load_u8(p + i_src);
            T2 = v_load_u8(p + 2 * i_src);
            T3 = v_load_u8(p + 3 * i_src);
            T4 = v_load_u8(p + 4 * i_src);
            T5 = v_load_u8(p + 5 * i_src);
            T6 = v_load_u8(p + 6 * i_src);
            T7 = v_load_u8(p + 7 * i_src);

            M0_lo = v_unpacklo_epi16(T0, T1);
            M0_hi = v_unpackhi_epi16(T0, T1);
            M1_lo = v_unpacklo_epi16(T2, T3);
            M1_hi = v_unpackhi_epi16(T2, T3);
            M2_lo = v_unpacklo_epi16(T4, T5);
            M2_hi = v_unpackhi_epi16(T4, T5);
            M3_lo = v_unpacklo_epi16(T6, T7);
            M3_hi = v_unpackhi_epi16(T6, T7);

            N0_lo = v_madd_epi16(M0_lo, coeff00);
            N0_hi = v_madd_epi16(M0_hi, coeff00);
            N1_lo = v_madd_epi16(M1_lo, coeff01);
            N1_hi = v_madd_epi16(M1_hi, coeff01);
            N2_lo = v_madd_epi16(M2_lo, coeff02);
            N2_hi = v_madd_epi16(M2_hi, coeff02);
            N3_lo = v_madd_epi16(M3_lo, coeff03);
            N3_hi = v_madd_epi16(M3_hi, coeff03);

            sum_lo = v_add_epi32(v_add_epi32(N0_lo, N1_lo), v_add_epi32(N2_lo, N3_lo));
            sum_hi = v_add_epi32(v_add_epi32(N0_hi, N1_hi), v_add_epi32(N2_hi, N3_hi));

            sum_lo = v_add_epi32(sum_lo, mAddOffset);
            sum_hi = v_add_epi32(sum_hi, mAddOffset);
            sum_lo = v_srai_epi32(sum_lo, 6);
            sum_hi = v_srai_epi32(sum_hi, 6);

            sum_lo = v_packus_epi32(sum_lo, sum_hi);
            sum_lo = v_min_epu16(sum_lo, max_val1);
            v_store_u8(dst + i, sum_lo);

            p += 8;
        }
        for (; i < width; i++) {
            int v = src[i]              * coeff[0]
                  + src[i + i_src]      * coeff[1]
                  + src[i + 2 * i_src]  * coeff[2]
                  + src[i + 3 * i_src]  * coeff[3]
                  + src[i + 4 * i_src]  * coeff[4]
                  + src[i + 5 * i_src]  * coeff[5]
                  + src[i + 6 * i_src]  * coeff[6]
                  + src[i + 7 * i_src]  * coeff[7];
            v = (v + 32) >> 6;
            dst[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        src += i_src;
        dst += i_dst;
    }
}

/* 8-bit 亮度 8 抽头双向插值: 先水平 (shift1=0) 后垂直 (shift2=12) */
static void ip_filter_luma_ext_8bit_neon(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y)
{
    AVS2_ALIGN16(short tmp_res[(64 + 7) * 64]);
    short *tmp = tmp_res;
    const int i_tmp = MC_TMP_STRIDE;
    const int shift2 = 12;  /* 20 - bit_depth */
    const int add2 = 1 << (shift2 - 1);
    int i, j;
    v128_t offset;
    v128_t max_val1 = v_set1_epi16(255);
    v128_t mCoef = vmovl_s8(vld1_s8(coef_x));

    /* ---- 第一级: 水平滤波 (shift1=0, 不移位, 中间结果存 int16) ---- */
    src += -3 * i_src - 3;

    for (j = -3; j < height + 4; j++) {
        const uint8_t *p = src;
        for (i = 0; i + 7 < width; i += 8) {
            v128_t T0, T1, T2, T3, T4, T5, T6, T7;
            v128_t M0, M1, M2, M3, M4, M5, M6, M7;
            v128_t A0, A1, A2, A3, B0, B1;

            T0 = v_load_u8(p++);
            T1 = v_load_u8(p++);
            T2 = v_load_u8(p++);
            T3 = v_load_u8(p++);
            T4 = v_load_u8(p++);
            T5 = v_load_u8(p++);
            T6 = v_load_u8(p++);
            T7 = v_load_u8(p++);

            M0 = v_madd_epi16(T0, mCoef);
            M1 = v_madd_epi16(T1, mCoef);
            M2 = v_madd_epi16(T2, mCoef);
            M3 = v_madd_epi16(T3, mCoef);
            M4 = v_madd_epi16(T4, mCoef);
            M5 = v_madd_epi16(T5, mCoef);
            M6 = v_madd_epi16(T6, mCoef);
            M7 = v_madd_epi16(T7, mCoef);

            A0 = v_hadd_epi32(M0, M1);
            A1 = v_hadd_epi32(M2, M3);
            A2 = v_hadd_epi32(M4, M5);
            A3 = v_hadd_epi32(M6, M7);

            B0 = v_hadd_epi32(A0, A1);
            B1 = v_hadd_epi32(A2, A3);

            /* shift1=0: 仅加 add1=0, 不移位, 有符号饱和到 int16 */
            B0 = v_packs_epi32(B0, B1);
            v_store(tmp + i, B0);
        }
        for (; i < width; i++) {
            int v = src[i]     * coef_x[0] + src[i + 1] * coef_x[1]
                  + src[i + 2] * coef_x[2] + src[i + 3] * coef_x[3]
                  + src[i + 4] * coef_x[4] + src[i + 5] * coef_x[5]
                  + src[i + 6] * coef_x[6] + src[i + 7] * coef_x[7];
            tmp[i] = (short)v;
        }
        tmp += i_tmp;
        src += i_src;
    }

    /* ---- 第二级: 垂直滤波 ---- */
    offset = v_set1_epi32(add2);
    tmp = tmp_res;

    {
        v128_t coeff00 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 0))));
        v128_t coeff01 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 2))));
        v128_t coeff02 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 4))));
        v128_t coeff03 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 6))));

        for (j = 0; j < height; j++) {
            const short *p = tmp;
            for (i = 0; i + 7 < width; i += 8) {
                v128_t T0, T1, T2, T3, T4, T5, T6, T7;
                v128_t M0_lo, M0_hi, M1_lo, M1_hi, M2_lo, M2_hi, M3_lo, M3_hi;
                v128_t N0_lo, N0_hi, N1_lo, N1_hi, N2_lo, N2_hi, N3_lo, N3_hi;
                v128_t sum_lo, sum_hi;

                T0 = v_load(p);
                T1 = v_load(p + i_tmp);
                T2 = v_load(p + 2 * i_tmp);
                T3 = v_load(p + 3 * i_tmp);
                T4 = v_load(p + 4 * i_tmp);
                T5 = v_load(p + 5 * i_tmp);
                T6 = v_load(p + 6 * i_tmp);
                T7 = v_load(p + 7 * i_tmp);

                M0_lo = v_unpacklo_epi16(T0, T1);
                M0_hi = v_unpackhi_epi16(T0, T1);
                M1_lo = v_unpacklo_epi16(T2, T3);
                M1_hi = v_unpackhi_epi16(T2, T3);
                M2_lo = v_unpacklo_epi16(T4, T5);
                M2_hi = v_unpackhi_epi16(T4, T5);
                M3_lo = v_unpacklo_epi16(T6, T7);
                M3_hi = v_unpackhi_epi16(T6, T7);

                N0_lo = v_madd_epi16(M0_lo, coeff00);
                N0_hi = v_madd_epi16(M0_hi, coeff00);
                N1_lo = v_madd_epi16(M1_lo, coeff01);
                N1_hi = v_madd_epi16(M1_hi, coeff01);
                N2_lo = v_madd_epi16(M2_lo, coeff02);
                N2_hi = v_madd_epi16(M2_hi, coeff02);
                N3_lo = v_madd_epi16(M3_lo, coeff03);
                N3_hi = v_madd_epi16(M3_hi, coeff03);

                sum_lo = v_add_epi32(v_add_epi32(N0_lo, N1_lo), v_add_epi32(N2_lo, N3_lo));
                sum_hi = v_add_epi32(v_add_epi32(N0_hi, N1_hi), v_add_epi32(N2_hi, N3_hi));

                sum_lo = v_add_epi32(sum_lo, offset);
                sum_hi = v_add_epi32(sum_hi, offset);
                sum_lo = v_srai_epi32(sum_lo, shift2);
                sum_hi = v_srai_epi32(sum_hi, shift2);

                sum_lo = v_packus_epi32(sum_lo, sum_hi);
                sum_lo = v_min_epu16(sum_lo, max_val1);
                v_store_u8(dst + i, sum_lo);

                p += 8;
            }
            for (; i < width; i++) {
                int v = tmp[i]              * coef_y[0]
                      + tmp[i + i_tmp]      * coef_y[1]
                      + tmp[i + 2 * i_tmp]  * coef_y[2]
                      + tmp[i + 3 * i_tmp]  * coef_y[3]
                      + tmp[i + 4 * i_tmp]  * coef_y[4]
                      + tmp[i + 5 * i_tmp]  * coef_y[5]
                      + tmp[i + 6 * i_tmp]  * coef_y[6]
                      + tmp[i + 7 * i_tmp]  * coef_y[7];
                v = (v + add2) >> shift2;
                dst[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
            dst += i_dst;
            tmp += i_tmp;
        }
    }
}

/* 8-bit 融合 MC+avg: 亮度双向第二路 (垂直滤波后与 dst 平均) */
static void ip_filter_luma_ext_8bit_avg_neon(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y)
{
    AVS2_ALIGN16(short tmp_res[(64 + 7) * 64]);
    short *tmp = tmp_res;
    const int i_tmp = MC_TMP_STRIDE;
    const int shift2 = 12;
    const int add2 = 1 << (shift2 - 1);
    int i, j;
    v128_t offset;
    v128_t max_val1 = v_set1_epi16(255);
    v128_t one = v_set1_epi16(1);
    v128_t mCoef = vmovl_s8(vld1_s8(coef_x));

    /* ---- 第一级: 水平滤波 (shift1=0) ---- */
    src += -3 * i_src - 3;
    offset = v_set1_epi32(0);

    for (j = -3; j < height + 4; j++) {
        const uint8_t *p = src;
        for (i = 0; i + 7 < width; i += 8) {
            v128_t T0, T1, T2, T3, T4, T5, T6, T7;
            v128_t M0, M1, M2, M3, M4, M5, M6, M7;
            v128_t A0, A1, A2, A3, B0, B1;

            T0 = v_load_u8(p++);
            T1 = v_load_u8(p++);
            T2 = v_load_u8(p++);
            T3 = v_load_u8(p++);
            T4 = v_load_u8(p++);
            T5 = v_load_u8(p++);
            T6 = v_load_u8(p++);
            T7 = v_load_u8(p++);

            M0 = v_madd_epi16(T0, mCoef);
            M1 = v_madd_epi16(T1, mCoef);
            M2 = v_madd_epi16(T2, mCoef);
            M3 = v_madd_epi16(T3, mCoef);
            M4 = v_madd_epi16(T4, mCoef);
            M5 = v_madd_epi16(T5, mCoef);
            M6 = v_madd_epi16(T6, mCoef);
            M7 = v_madd_epi16(T7, mCoef);

            A0 = v_hadd_epi32(M0, M1);
            A1 = v_hadd_epi32(M2, M3);
            A2 = v_hadd_epi32(M4, M5);
            A3 = v_hadd_epi32(M6, M7);

            B0 = v_hadd_epi32(A0, A1);
            B1 = v_hadd_epi32(A2, A3);

            B0 = v_packs_epi32(B0, B1);
            v_store(tmp + i, B0);
        }
        for (; i < width; i++) {
            int v = src[i]     * coef_x[0] + src[i + 1] * coef_x[1]
                  + src[i + 2] * coef_x[2] + src[i + 3] * coef_x[3]
                  + src[i + 4] * coef_x[4] + src[i + 5] * coef_x[5]
                  + src[i + 6] * coef_x[6] + src[i + 7] * coef_x[7];
            tmp[i] = (short)v;
        }
        tmp += i_tmp;
        src += i_src;
    }

    /* ---- 第二级: 垂直滤波 + 融合平均 ---- */
    offset = v_set1_epi32(add2);
    tmp = tmp_res;

    {
        v128_t coeff00 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 0))));
        v128_t coeff01 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 2))));
        v128_t coeff02 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 4))));
        v128_t coeff03 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 6))));

        for (j = 0; j < height; j++) {
            const short *p = tmp;
            for (i = 0; i + 7 < width; i += 8) {
                v128_t T0, T1, T2, T3, T4, T5, T6, T7;
                v128_t M0_lo, M0_hi, M1_lo, M1_hi, M2_lo, M2_hi, M3_lo, M3_hi;
                v128_t N0_lo, N0_hi, N1_lo, N1_hi, N2_lo, N2_hi, N3_lo, N3_hi;
                v128_t sum_lo, sum_hi, vdst, vsum;

                T0 = v_load(p);
                T1 = v_load(p + i_tmp);
                T2 = v_load(p + 2 * i_tmp);
                T3 = v_load(p + 3 * i_tmp);
                T4 = v_load(p + 4 * i_tmp);
                T5 = v_load(p + 5 * i_tmp);
                T6 = v_load(p + 6 * i_tmp);
                T7 = v_load(p + 7 * i_tmp);

                M0_lo = v_unpacklo_epi16(T0, T1);
                M0_hi = v_unpackhi_epi16(T0, T1);
                M1_lo = v_unpacklo_epi16(T2, T3);
                M1_hi = v_unpackhi_epi16(T2, T3);
                M2_lo = v_unpacklo_epi16(T4, T5);
                M2_hi = v_unpackhi_epi16(T4, T5);
                M3_lo = v_unpacklo_epi16(T6, T7);
                M3_hi = v_unpackhi_epi16(T6, T7);

                N0_lo = v_madd_epi16(M0_lo, coeff00);
                N0_hi = v_madd_epi16(M0_hi, coeff00);
                N1_lo = v_madd_epi16(M1_lo, coeff01);
                N1_hi = v_madd_epi16(M1_hi, coeff01);
                N2_lo = v_madd_epi16(M2_lo, coeff02);
                N2_hi = v_madd_epi16(M2_hi, coeff02);
                N3_lo = v_madd_epi16(M3_lo, coeff03);
                N3_hi = v_madd_epi16(M3_hi, coeff03);

                sum_lo = v_add_epi32(v_add_epi32(N0_lo, N1_lo), v_add_epi32(N2_lo, N3_lo));
                sum_hi = v_add_epi32(v_add_epi32(N0_hi, N1_hi), v_add_epi32(N2_hi, N3_hi));

                sum_lo = v_add_epi32(sum_lo, offset);
                sum_hi = v_add_epi32(sum_hi, offset);
                sum_lo = v_srai_epi32(sum_lo, shift2);
                sum_hi = v_srai_epi32(sum_hi, shift2);

                sum_lo = v_packus_epi32(sum_lo, sum_hi);
                sum_lo = v_min_epu16(sum_lo, max_val1);

                /* 融合平均: dst[i] = (dst[i] + result[i] + 1) >> 1 */
                vdst = v_load_u8(dst + i);
                vsum = v_add_epi16(vdst, sum_lo);
                vsum = v_add_epi16(vsum, one);
                vsum = v_srli_epi16(vsum, 1);
                v_store_u8(dst + i, vsum);

                p += 8;
            }
            for (; i < width; i++) {
                int v = tmp[i]              * coef_y[0]
                      + tmp[i + i_tmp]      * coef_y[1]
                      + tmp[i + 2 * i_tmp]  * coef_y[2]
                      + tmp[i + 3 * i_tmp]  * coef_y[3]
                      + tmp[i + 4 * i_tmp]  * coef_y[4]
                      + tmp[i + 5 * i_tmp]  * coef_y[5]
                      + tmp[i + 6 * i_tmp]  * coef_y[6]
                      + tmp[i + 7 * i_tmp]  * coef_y[7];
                v = (v + add2) >> shift2;
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                dst[i] = (uint8_t)((dst[i] + v + 1) >> 1);
            }
            dst += i_dst;
            tmp += i_tmp;
        }
    }
}

/* 8-bit 色度 4 抽头水平插值 */
static void ip_filter_chroma_hor_8bit_neon(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height, const int8_t *coeff)
{
    int row, col;
    const int offset_val = 32;
    const int shift = 6;

    v128_t mCoef = vmovl_s8(vreinterpret_s8_s32(vdup_n_s32(*(const int32_t*)coeff)));
    v128_t mAddOffset = v_set1_epi32(offset_val);
    v128_t max_val1 = v_set1_epi16(255);

    src -= 1;

    for (row = 0; row < height; row++) {
        const uint8_t *p = src;
        for (col = 0; col + 7 < width; col += 8) {
            uint16x8_t s;
            uint16x4_t lo1, lo2;
            v128_t s1, s2, m1, m2, sum_lo;
            v128_t s3, s4, m3, m4, sum_hi;

            /* 加载 8 uint8 并宽展为 uint16x8 (与 10-bit vld1q_u16 等价) */
            s = vmovl_u8(vld1_u8(p + col));
            lo1 = vget_low_u16(vextq_u16(s, s, 1));
            lo2 = vget_low_u16(vextq_u16(s, s, 3));
            s1 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(s), lo1));
            s2 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(vextq_u16(s, s, 2)), lo2));
            m1 = v_madd_epi16(s1, mCoef);
            m2 = v_madd_epi16(s2, mCoef);
            sum_lo = v_hadd_epi32(m1, m2);

            s = vmovl_u8(vld1_u8(p + col + 4));
            lo1 = vget_low_u16(vextq_u16(s, s, 1));
            lo2 = vget_low_u16(vextq_u16(s, s, 3));
            s3 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(s), lo1));
            s4 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(vextq_u16(s, s, 2)), lo2));
            m3 = v_madd_epi16(s3, mCoef);
            m4 = v_madd_epi16(s4, mCoef);
            sum_hi = v_hadd_epi32(m3, m4);

            sum_lo = v_add_epi32(sum_lo, mAddOffset);
            sum_hi = v_add_epi32(sum_hi, mAddOffset);
            sum_lo = v_srai_epi32(sum_lo, shift);
            sum_hi = v_srai_epi32(sum_hi, shift);
            sum_lo = v_packus_epi32(sum_lo, sum_hi);
            sum_lo = v_min_epu16(sum_lo, max_val1);
            v_store_u8(dst + col, sum_lo);
        }
        for (; col < width; col++) {
            int v = src[col]     * coeff[0] + src[col + 1] * coeff[1]
                  + src[col + 2] * coeff[2] + src[col + 3] * coeff[3];
            v = (v + offset_val) >> shift;
            dst[col] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        src += i_src;
        dst += i_dst;
    }
}

/* 8-bit 色度 4 抽头垂直插值 */
static void ip_filter_chroma_ver_8bit_neon(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height, const int8_t *coeff)
{
    int row, col;
    const int offset_val = 32;
    const int shift = 6;
    v128_t mAddOffset = v_set1_epi32(offset_val);
    v128_t max_val1 = v_set1_epi16(255);
    const int i_src2 = i_src * 2;
    const int i_src3 = i_src * 3;

    v128_t coeff0 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coeff + 0))));
    v128_t coeff1 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coeff + 2))));

    src -= i_src;

    for (row = 0; row < height; row++) {
        const uint8_t *p = src;
        for (col = 0; col + 7 < width; col += 8) {
            v128_t S0, S1, S2, S3;
            v128_t T0_lo, T0_hi, T1_lo, T1_hi;
            v128_t N0_lo, N0_hi, N1_lo, N1_hi;
            v128_t sum_lo, sum_hi;

            S0 = v_load_u8(p + col);
            S1 = v_load_u8(p + col + i_src);
            S2 = v_load_u8(p + col + i_src2);
            S3 = v_load_u8(p + col + i_src3);

            T0_lo = v_unpacklo_epi16(S0, S1);
            T0_hi = v_unpackhi_epi16(S0, S1);
            T1_lo = v_unpacklo_epi16(S2, S3);
            T1_hi = v_unpackhi_epi16(S2, S3);

            N0_lo = v_madd_epi16(T0_lo, coeff0);
            N0_hi = v_madd_epi16(T0_hi, coeff0);
            N1_lo = v_madd_epi16(T1_lo, coeff1);
            N1_hi = v_madd_epi16(T1_hi, coeff1);

            sum_lo = v_add_epi32(N0_lo, N1_lo);
            sum_hi = v_add_epi32(N0_hi, N1_hi);

            sum_lo = v_add_epi32(sum_lo, mAddOffset);
            sum_hi = v_add_epi32(sum_hi, mAddOffset);
            sum_lo = v_srai_epi32(sum_lo, shift);
            sum_hi = v_srai_epi32(sum_hi, shift);
            sum_lo = v_packus_epi32(sum_lo, sum_hi);
            sum_lo = v_min_epu16(sum_lo, max_val1);
            v_store_u8(dst + col, sum_lo);
        }
        for (; col < width; col++) {
            int v = src[col]             * coeff[0]
                  + src[col + i_src]     * coeff[1]
                  + src[col + i_src2]    * coeff[2]
                  + src[col + i_src3]    * coeff[3];
            v = (v + offset_val) >> shift;
            dst[col] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
        src += i_src;
        dst += i_dst;
    }
}

/* 8-bit 色度 4 抽头双向插值: 先水平 (shift1=0) 后垂直 (shift2=12) */
static void ip_filter_chroma_ext_8bit_neon(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y)
{
    AVS2_ALIGN16(short tmp_res[(32 + 3) * 32]);
    short *tmp = tmp_res;
    const int i_tmp = 32;
    const int shift2 = 12;
    const int add2 = 1 << (shift2 - 1);
    int row, col;
    v128_t max_val1 = v_set1_epi16(255);
    v128_t mAddOffset;
    v128_t mCoef = vmovl_s8(vreinterpret_s8_s32(vdup_n_s32(*(const int32_t*)coef_x)));

    /* ---- 第一级: 水平滤波 (shift1=0) ---- */
    src = src - i_src - 1;

    for (row = -1; row < height + 2; row++) {
        const uint8_t *p = src;
        for (col = 0; col + 7 < width; col += 8) {
            uint16x8_t s;
            uint16x4_t lo1, lo2;
            v128_t s1, s2, m1, m2, sum_lo;
            v128_t s3, s4, m3, m4, sum_hi;

            s = vmovl_u8(vld1_u8(p + col));
            lo1 = vget_low_u16(vextq_u16(s, s, 1));
            lo2 = vget_low_u16(vextq_u16(s, s, 3));
            s1 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(s), lo1));
            s2 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(vextq_u16(s, s, 2)), lo2));
            m1 = v_madd_epi16(s1, mCoef);
            m2 = v_madd_epi16(s2, mCoef);
            sum_lo = v_hadd_epi32(m1, m2);

            s = vmovl_u8(vld1_u8(p + col + 4));
            lo1 = vget_low_u16(vextq_u16(s, s, 1));
            lo2 = vget_low_u16(vextq_u16(s, s, 3));
            s3 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(s), lo1));
            s4 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(vextq_u16(s, s, 2)), lo2));
            m3 = v_madd_epi16(s3, mCoef);
            m4 = v_madd_epi16(s4, mCoef);
            sum_hi = v_hadd_epi32(m3, m4);

            sum_lo = v_packs_epi32(sum_lo, sum_hi);
            v_store(tmp + col, sum_lo);
        }
        for (; col < width; col++) {
            int v = src[col]     * coef_x[0] + src[col + 1] * coef_x[1]
                  + src[col + 2] * coef_x[2] + src[col + 3] * coef_x[3];
            tmp[col] = (short)v;
        }
        src += i_src;
        tmp += i_tmp;
    }

    /* ---- 第二级: 垂直滤波 ---- */
    tmp = tmp_res;
    mAddOffset = v_set1_epi32(add2);

    {
        v128_t coeffy0 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 0))));
        v128_t coeffy1 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 2))));

        for (row = 0; row < height; row++) {
            for (col = 0; col + 7 < width; col += 8) {
                v128_t S0, S1, S2, S3;
                v128_t T0_lo, T0_hi, T1_lo, T1_hi;
                v128_t N0_lo, N0_hi, N1_lo, N1_hi;
                v128_t sum_lo, sum_hi;

                S0 = v_load(tmp + col);
                S1 = v_load(tmp + col + i_tmp);
                S2 = v_load(tmp + col + 2 * i_tmp);
                S3 = v_load(tmp + col + 3 * i_tmp);

                T0_lo = v_unpacklo_epi16(S0, S1);
                T0_hi = v_unpackhi_epi16(S0, S1);
                T1_lo = v_unpacklo_epi16(S2, S3);
                T1_hi = v_unpackhi_epi16(S2, S3);

                N0_lo = v_madd_epi16(T0_lo, coeffy0);
                N0_hi = v_madd_epi16(T0_hi, coeffy0);
                N1_lo = v_madd_epi16(T1_lo, coeffy1);
                N1_hi = v_madd_epi16(T1_hi, coeffy1);

                sum_lo = v_add_epi32(N0_lo, N1_lo);
                sum_hi = v_add_epi32(N0_hi, N1_hi);

                sum_lo = v_add_epi32(sum_lo, mAddOffset);
                sum_hi = v_add_epi32(sum_hi, mAddOffset);
                sum_lo = v_srai_epi32(sum_lo, shift2);
                sum_hi = v_srai_epi32(sum_hi, shift2);
                sum_lo = v_packus_epi32(sum_lo, sum_hi);
                sum_lo = v_min_epu16(sum_lo, max_val1);
                v_store_u8(dst + col, sum_lo);
            }
            for (; col < width; col++) {
                int v = tmp[col]             * coef_y[0]
                      + tmp[col + i_tmp]     * coef_y[1]
                      + tmp[col + 2 * i_tmp] * coef_y[2]
                      + tmp[col + 3 * i_tmp] * coef_y[3];
                v = (v + add2) >> shift2;
                dst[col] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
            }
            dst += i_dst;
            tmp += i_tmp;
        }
    }
}

/* 8-bit 融合 MC+avg: 色度双向第二路 */
static void ip_filter_chroma_ext_8bit_avg_neon(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y)
{
    AVS2_ALIGN16(short tmp_res[(32 + 3) * 32]);
    short *tmp = tmp_res;
    const int i_tmp = 32;
    const int shift2 = 12;
    const int add2 = 1 << (shift2 - 1);
    int row, col;
    v128_t max_val1 = v_set1_epi16(255);
    v128_t one16 = v_set1_epi16(1);
    v128_t mAddOffset;
    v128_t mCoef = vmovl_s8(vreinterpret_s8_s32(vdup_n_s32(*(const int32_t*)coef_x)));

    /* ---- 第一级: 水平滤波 (shift1=0) ---- */
    src = src - i_src - 1;

    for (row = -1; row < height + 2; row++) {
        const uint8_t *p = src;
        for (col = 0; col + 7 < width; col += 8) {
            uint16x8_t s;
            uint16x4_t lo1, lo2;
            v128_t s1, s2, m1, m2, sum_lo;
            v128_t s3, s4, m3, m4, sum_hi;

            s = vmovl_u8(vld1_u8(p + col));
            lo1 = vget_low_u16(vextq_u16(s, s, 1));
            lo2 = vget_low_u16(vextq_u16(s, s, 3));
            s1 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(s), lo1));
            s2 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(vextq_u16(s, s, 2)), lo2));
            m1 = v_madd_epi16(s1, mCoef);
            m2 = v_madd_epi16(s2, mCoef);
            sum_lo = v_hadd_epi32(m1, m2);

            s = vmovl_u8(vld1_u8(p + col + 4));
            lo1 = vget_low_u16(vextq_u16(s, s, 1));
            lo2 = vget_low_u16(vextq_u16(s, s, 3));
            s3 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(s), lo1));
            s4 = vreinterpretq_s16_u16(vcombine_u16(vget_low_u16(vextq_u16(s, s, 2)), lo2));
            m3 = v_madd_epi16(s3, mCoef);
            m4 = v_madd_epi16(s4, mCoef);
            sum_hi = v_hadd_epi32(m3, m4);

            sum_lo = v_packs_epi32(sum_lo, sum_hi);
            v_store(tmp + col, sum_lo);
        }
        for (; col < width; col++) {
            int v = src[col]     * coef_x[0] + src[col + 1] * coef_x[1]
                  + src[col + 2] * coef_x[2] + src[col + 3] * coef_x[3];
            tmp[col] = (short)v;
        }
        src += i_src;
        tmp += i_tmp;
    }

    /* ---- 第二级: 垂直滤波 + 融合平均 ---- */
    tmp = tmp_res;
    mAddOffset = v_set1_epi32(add2);

    {
        v128_t coeffy0 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 0))));
        v128_t coeffy1 = vmovl_s8(vreinterpret_s8_s16(vdup_n_s16(*(const int16_t*)(coef_y + 2))));

        for (row = 0; row < height; row++) {
            for (col = 0; col + 7 < width; col += 8) {
                v128_t S0, S1, S2, S3;
                v128_t T0_lo, T0_hi, T1_lo, T1_hi;
                v128_t N0_lo, N0_hi, N1_lo, N1_hi;
                v128_t sum_lo, sum_hi, vdst, vsum;

                S0 = v_load(tmp + col);
                S1 = v_load(tmp + col + i_tmp);
                S2 = v_load(tmp + col + 2 * i_tmp);
                S3 = v_load(tmp + col + 3 * i_tmp);

                T0_lo = v_unpacklo_epi16(S0, S1);
                T0_hi = v_unpackhi_epi16(S0, S1);
                T1_lo = v_unpacklo_epi16(S2, S3);
                T1_hi = v_unpackhi_epi16(S2, S3);

                N0_lo = v_madd_epi16(T0_lo, coeffy0);
                N0_hi = v_madd_epi16(T0_hi, coeffy0);
                N1_lo = v_madd_epi16(T1_lo, coeffy1);
                N1_hi = v_madd_epi16(T1_hi, coeffy1);

                sum_lo = v_add_epi32(N0_lo, N1_lo);
                sum_hi = v_add_epi32(N0_hi, N1_hi);

                sum_lo = v_add_epi32(sum_lo, mAddOffset);
                sum_hi = v_add_epi32(sum_hi, mAddOffset);
                sum_lo = v_srai_epi32(sum_lo, shift2);
                sum_hi = v_srai_epi32(sum_hi, shift2);
                sum_lo = v_packus_epi32(sum_lo, sum_hi);
                sum_lo = v_min_epu16(sum_lo, max_val1);

                /* 融合平均: dst[col] = (dst[col] + result[col] + 1) >> 1 */
                vdst = v_load_u8(dst + col);
                vsum = v_add_epi16(vdst, sum_lo);
                vsum = v_add_epi16(vsum, one16);
                vsum = v_srli_epi16(vsum, 1);
                v_store_u8(dst + col, vsum);
            }
            for (; col < width; col++) {
                int v = tmp[col]             * coef_y[0]
                      + tmp[col + i_tmp]     * coef_y[1]
                      + tmp[col + 2 * i_tmp] * coef_y[2]
                      + tmp[col + 3 * i_tmp] * coef_y[3];
                v = (v + add2) >> shift2;
                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                dst[col] = (uint8_t)((dst[col] + v + 1) >> 1);
            }
            dst += i_dst;
            tmp += i_tmp;
        }
    }
}

/* 8-bit 整像素块平均: dst[i] = (dst[i] + src[i] + 1) >> 1 */
static void block_avg_8bit_neon(uint8_t *dst, int i_dst,
                                 const uint8_t *src, int i_src,
                                 int width, int height)
{
    int y, x;
    for (y = 0; y < height; y++) {
        for (x = 0; x + 15 < width; x += 16) {
            uint8x16_t vd = vld1q_u8(dst + x);
            uint8x16_t vs = vld1q_u8(src + x);
            vst1q_u8(dst + x, vrhaddq_u8(vd, vs));
        }
        for (; x + 7 < width; x += 8) {
            uint8x8_t vd = vld1_u8(dst + x);
            uint8x8_t vs = vld1_u8(src + x);
            vst1_u8(dst + x, vrhadd_u8(vd, vs));
        }
        for (; x < width; x++) {
            int v = dst[x] + src[x];
            dst[x] = (uint8_t)((v + 1) >> 1);
        }
        dst += i_dst;
        src += i_src;
    }
}

/* 8-bit 双向平均: dst[x] = (dst[x] + pred2[x] + 1) >> 1
 * pred2 为 int16_t[], 值域 [0,255], 取低字节与 dst 平均 */
static void bi_avg_8bit_neon(uint8_t *dst, int dst_stride,
                              const int16_t *pred2, int pred2_stride,
                              int w, int h)
{
    int y, x;
    for (y = 0; y < h; y++) {
        uint8_t *d = dst + y * dst_stride;
        const int16_t *p = pred2 + y * pred2_stride;
        x = 0;
        for (; x + 7 < w; x += 8) {
            uint8x8_t vd = vld1_u8(d + x);
            uint16x8_t vp16 = vreinterpretq_u16_s16(vld1q_s16(p + x));
            uint8x8_t vp = vqmovn_u16(vp16);
            vst1_u8(d + x, vrhadd_u8(vd, vp));
        }
        for (; x < w; x++) {
            int v = d[x] + (uint8_t)p[x];
            d[x] = (uint8_t)((v + 1) >> 1);
        }
    }
}

/* 8-bit 双源双向平均: dst[x] = (pred1[x] + pred2[x] + 1) >> 1
 * pred1/pred2 为 int16_t[], 值域 [0,255], 按无符号求和平均后缩窄 */
static void bi_avg_2src_8bit_neon(uint8_t *dst, int dst_stride,
                                   const int16_t *pred1, int pred1_stride,
                                   const int16_t *pred2, int pred2_stride,
                                   int w, int h)
{
    const uint16x8_t one = vdupq_n_u16(1);
    int y, x;
    for (y = 0; y < h; y++) {
        uint8_t *d = dst + y * dst_stride;
        const int16_t *p1 = pred1 + y * pred1_stride;
        const int16_t *p2 = pred2 + y * pred2_stride;
        x = 0;
        for (; x + 7 < w; x += 8) {
            uint16x8_t v1 = vreinterpretq_u16_s16(vld1q_s16(p1 + x));
            uint16x8_t v2 = vreinterpretq_u16_s16(vld1q_s16(p2 + x));
            uint16x8_t sum = vaddq_u16(v1, v2);
            sum = vaddq_u16(sum, one);
            sum = vshrq_n_u16(sum, 1);
            vst1_u8(d + x, vqmovn_u16(sum));
        }
        for (; x < w; x++) {
            int v = p1[x] + p2[x];
            d[x] = (uint8_t)((v + 1) >> 1);
        }
    }
}

/* 8-bit 块填充 */
static void fill_block_8bit_neon(uint8_t *dst, int dst_stride,
                                  int w, int h, int fill_val)
{
    const uint8x16_t vfill16 = vdupq_n_u8((uint8_t)fill_val);
    const uint8x8_t vfill8 = vdup_n_u8((uint8_t)fill_val);
    int y, x;
    for (y = 0; y < h; y++) {
        uint8_t *d = dst + y * dst_stride;
        x = 0;
        for (; x + 15 < w; x += 16) {
            vst1q_u8(d + x, vfill16);
        }
        for (; x + 7 < w; x += 8) {
            vst1_u8(d + x, vfill8);
        }
        for (; x < w; x++) {
            d[x] = (uint8_t)fill_val;
        }
    }
}

/* ===========================================================================
 * 入口函数 (NEON): 10-bit 和 8-bit 均使用原生 NEON
 * =========================================================================== */
static void mc_luma_neon(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                         ptrdiff_t dstride, int w, int h, int mx, int my,
                         int bit_depth)
{
    if (bit_depth > 8) {
        int dx = mx & 3;
        int dy = my & 3;
        int max_val = (1 << bit_depth) - 1;
        pel_t *dst16 = (pel_t*)(void *)dst;
        const pel_t *src16 = (const pel_t*)(const void *)src;
        int i_dst = (int)(dstride >> 1);
        int i_src = (int)(sstride >> 1);

        if (dx == 0 && dy == 0) {
            block_copy_10bit_neon(dst16, i_dst, src16, i_src, w, h);
        } else if (dx == 0) {
            ip_filter_luma_ver_10bit_neon(dst16, i_dst, src16, i_src,
                                          w, h, avs2_intpl_filters[dy], max_val);
        } else if (dy == 0) {
            ip_filter_luma_hor_10bit_neon(dst16, i_dst, src16, i_src,
                                          w, h, avs2_intpl_filters[dx], max_val);
        } else {
            ip_filter_luma_ext_10bit_neon(dst16, i_dst, src16, i_src,
                                          w, h,
                                          avs2_intpl_filters[dx],
                                          avs2_intpl_filters[dy], max_val);
        }
    } else {
        int dx = mx & 3;
        int dy = my & 3;
        int i_dst = (int)dstride;
        int i_src = (int)sstride;

        if (dx == 0 && dy == 0) {
            block_copy_8bit_neon(dst, i_dst, src, i_src, w, h);
        } else if (dx == 0) {
            ip_filter_luma_ver_8bit_neon(dst, i_dst, src, i_src,
                                         w, h, avs2_intpl_filters[dy]);
        } else if (dy == 0) {
            ip_filter_luma_hor_8bit_neon(dst, i_dst, src, i_src,
                                         w, h, avs2_intpl_filters[dx]);
        } else {
            ip_filter_luma_ext_8bit_neon(dst, i_dst, src, i_src,
                                         w, h,
                                         avs2_intpl_filters[dx],
                                         avs2_intpl_filters[dy]);
        }
    }
}

static void mc_chroma_neon(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                           ptrdiff_t dstride, int w, int h, int mx, int my,
                           int bit_depth)
{
    if (bit_depth > 8) {
        int dx = mx & 7;
        int dy = my & 7;
        int max_val = (1 << bit_depth) - 1;
        pel_t *dst16 = (pel_t*)(void *)dst;
        const pel_t *src16 = (const pel_t*)(const void *)src;
        int i_dst = (int)(dstride >> 1);
        int i_src = (int)(sstride >> 1);

        if (dx == 0 && dy == 0) {
            block_copy_10bit_neon(dst16, i_dst, src16, i_src, w, h);
        } else if (dx == 0) {
            ip_filter_chroma_ver_10bit_neon(dst16, i_dst, src16, i_src,
                                            w, h, avs2_intpl_filters_c[dy], max_val);
        } else if (dy == 0) {
            ip_filter_chroma_hor_10bit_neon(dst16, i_dst, src16, i_src,
                                            w, h, avs2_intpl_filters_c[dx], max_val);
        } else {
            ip_filter_chroma_ext_10bit_neon(dst16, i_dst, src16, i_src,
                                            w, h,
                                            avs2_intpl_filters_c[dx],
                                            avs2_intpl_filters_c[dy], max_val);
        }
    } else {
        int dx = mx & 7;
        int dy = my & 7;
        int i_dst = (int)dstride;
        int i_src = (int)sstride;

        if (dx == 0 && dy == 0) {
            block_copy_8bit_neon(dst, i_dst, src, i_src, w, h);
        } else if (dx == 0) {
            ip_filter_chroma_ver_8bit_neon(dst, i_dst, src, i_src,
                                           w, h, avs2_intpl_filters_c[dy]);
        } else if (dy == 0) {
            ip_filter_chroma_hor_8bit_neon(dst, i_dst, src, i_src,
                                           w, h, avs2_intpl_filters_c[dx]);
        } else {
            ip_filter_chroma_ext_8bit_neon(dst, i_dst, src, i_src,
                                           w, h,
                                           avs2_intpl_filters_c[dx],
                                           avs2_intpl_filters_c[dy]);
        }
    }
}

static void bi_avg_neon(uint8_t *dst, ptrdiff_t dst_stride, const int16_t *pred2,
                        int pred2_stride, int w, int h, int bit_depth)
{
    if (bit_depth > 8) {
        bi_avg_10bit_neon((uint16_t*)(void *)dst, (int)(dst_stride >> 1),
                          pred2, pred2_stride, w, h);
    } else {
        bi_avg_8bit_neon(dst, (int)dst_stride, pred2, pred2_stride, w, h);
    }
}

static void bi_avg_2src_neon(uint8_t *dst, ptrdiff_t dst_stride,
                             const int16_t *pred1, int pred1_stride,
                             const int16_t *pred2, int pred2_stride,
                             int w, int h, int bit_depth)
{
    if (bit_depth > 8) {
        bi_avg_2src_10bit_neon((uint16_t*)(void *)dst, (int)(dst_stride >> 1),
                               pred1, pred1_stride, pred2, pred2_stride, w, h);
    } else {
        bi_avg_2src_8bit_neon(dst, (int)dst_stride,
                              pred1, pred1_stride, pred2, pred2_stride, w, h);
    }
}

static void fill_block_neon(uint8_t *dst, ptrdiff_t dst_stride, int w, int h,
                            int fill_val, int bit_depth)
{
    if (bit_depth > 8) {
        fill_block_10bit_neon((uint16_t*)(void *)dst, (int)(dst_stride >> 1),
                              w, h, fill_val);
    } else {
        fill_block_8bit_neon(dst, (int)dst_stride, w, h, fill_val);
    }
}

/* MC + 双向平均: 亮度 (NEON) */
static void mc_luma_avg_neon(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                             ptrdiff_t dstride, int w, int h, int mx, int my,
                             int bit_depth)
{
    if (bit_depth > 8) {
        int dx = mx & 3;
        int dy = my & 3;
        int max_val = (1 << bit_depth) - 1;
        pel_t *dst16 = (pel_t *)(void *)dst;
        const pel_t *src16 = (const pel_t *)(const void *)src;
        int i_dst = (int)(dstride >> 1);
        int i_src = (int)(sstride >> 1);

        if (dx == 0 && dy == 0) {
            block_avg_10bit_neon(dst16, i_dst, src16, i_src, w, h);
        } else if (dx == 0) {
            AVS2_ALIGN16(short pred2[64 * 64]);
            ip_filter_luma_ver_10bit_neon((pel_t *)pred2, w, src16, i_src,
                                          w, h, avs2_intpl_filters[dy], max_val);
            block_avg_10bit_neon(dst16, i_dst, (const pel_t *)pred2, w, w, h);
        } else if (dy == 0) {
            AVS2_ALIGN16(short pred2[64 * 64]);
            ip_filter_luma_hor_10bit_neon((pel_t *)pred2, w, src16, i_src,
                                          w, h, avs2_intpl_filters[dx], max_val);
            block_avg_10bit_neon(dst16, i_dst, (const pel_t *)pred2, w, w, h);
        } else {
            ip_filter_luma_ext_10bit_avg_neon(dst16, i_dst, src16, i_src,
                                              w, h,
                                              avs2_intpl_filters[dx],
                                              avs2_intpl_filters[dy], max_val);
        }
    } else {
        int dx = mx & 3;
        int dy = my & 3;
        int i_dst = (int)dstride;
        int i_src = (int)sstride;

        if (dx == 0 && dy == 0) {
            block_avg_8bit_neon(dst, i_dst, src, i_src, w, h);
        } else if (dx == 0) {
            AVS2_ALIGN16(uint8_t pred2[64 * 64]);
            ip_filter_luma_ver_8bit_neon(pred2, w, src, i_src,
                                         w, h, avs2_intpl_filters[dy]);
            block_avg_8bit_neon(dst, i_dst, pred2, w, w, h);
        } else if (dy == 0) {
            AVS2_ALIGN16(uint8_t pred2[64 * 64]);
            ip_filter_luma_hor_8bit_neon(pred2, w, src, i_src,
                                         w, h, avs2_intpl_filters[dx]);
            block_avg_8bit_neon(dst, i_dst, pred2, w, w, h);
        } else {
            ip_filter_luma_ext_8bit_avg_neon(dst, i_dst, src, i_src,
                                             w, h,
                                             avs2_intpl_filters[dx],
                                             avs2_intpl_filters[dy]);
        }
    }
}

/* MC + 双向平均: 色度 (NEON) */
static void mc_chroma_avg_neon(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                               ptrdiff_t dstride, int w, int h, int mx, int my,
                               int bit_depth)
{
    if (bit_depth > 8) {
        int dx = mx & 7;
        int dy = my & 7;
        int max_val = (1 << bit_depth) - 1;
        pel_t *dst16 = (pel_t *)(void *)dst;
        const pel_t *src16 = (const pel_t *)(const void *)src;
        int i_dst = (int)(dstride >> 1);
        int i_src = (int)(sstride >> 1);

        if (dx == 0 && dy == 0) {
            block_avg_10bit_neon(dst16, i_dst, src16, i_src, w, h);
        } else if (dx == 0) {
            AVS2_ALIGN16(short pred2[32 * 32]);
            ip_filter_chroma_ver_10bit_neon((pel_t *)pred2, w, src16, i_src,
                                            w, h, avs2_intpl_filters_c[dy], max_val);
            block_avg_10bit_neon(dst16, i_dst, (const pel_t *)pred2, w, w, h);
        } else if (dy == 0) {
            AVS2_ALIGN16(short pred2[32 * 32]);
            ip_filter_chroma_hor_10bit_neon((pel_t *)pred2, w, src16, i_src,
                                            w, h, avs2_intpl_filters_c[dx], max_val);
            block_avg_10bit_neon(dst16, i_dst, (const pel_t *)pred2, w, w, h);
        } else {
            ip_filter_chroma_ext_10bit_avg_neon(dst16, i_dst, src16, i_src,
                                                w, h,
                                                avs2_intpl_filters_c[dx],
                                                avs2_intpl_filters_c[dy], max_val);
        }
    } else {
        int dx = mx & 7;
        int dy = my & 7;
        int i_dst = (int)dstride;
        int i_src = (int)sstride;

        if (dx == 0 && dy == 0) {
            block_avg_8bit_neon(dst, i_dst, src, i_src, w, h);
        } else if (dx == 0) {
            AVS2_ALIGN16(uint8_t pred2[32 * 32]);
            ip_filter_chroma_ver_8bit_neon(pred2, w, src, i_src,
                                           w, h, avs2_intpl_filters_c[dy]);
            block_avg_8bit_neon(dst, i_dst, pred2, w, w, h);
        } else if (dy == 0) {
            AVS2_ALIGN16(uint8_t pred2[32 * 32]);
            ip_filter_chroma_hor_8bit_neon(pred2, w, src, i_src,
                                           w, h, avs2_intpl_filters_c[dx]);
            block_avg_8bit_neon(dst, i_dst, pred2, w, w, h);
        } else {
            ip_filter_chroma_ext_8bit_avg_neon(dst, i_dst, src, i_src,
                                               w, h,
                                               avs2_intpl_filters_c[dx],
                                               avs2_intpl_filters_c[dy]);
        }
    }
}

/* ===========================================================================
 * DSP 初始化 (NEON)
 * =========================================================================== */
void avs2_mc_init_neon(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.mc_luma       = mc_luma_neon;
    avs2_dsp_table.mc_chroma     = mc_chroma_neon;
    avs2_dsp_table.mc_luma_avg   = mc_luma_avg_neon;
    avs2_dsp_table.mc_chroma_avg = mc_chroma_avg_neon;
    avs2_dsp_table.bi_avg        = bi_avg_neon;
    avs2_dsp_table.bi_avg_2src   = bi_avg_2src_neon;
    avs2_dsp_table.fill_block    = fill_block_neon;
}

/* 非 NEON 平台: SSE/AVX 接口空实现以供链接 */
void avs2_mc_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_mc_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_mc_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }

#else /* 其它平台: 回退到 C */

/* NEON 路径回退到 C, 保留空实现以供链接 */
void avs2_mc_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_mc_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_mc_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_mc_init_neon(const avs2_cpu_flags *flags) { (void)flags; }

#endif