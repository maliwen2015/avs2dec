/*
 * mc_simd.c - 运动补偿 SIMD 实现 (x86)
 *
 * 当前实现:
 *   - AVX2: 10-bit 亮度 8 抽头 / 色度 4 抽头插值 (参考 libudavs2 10bit AVX2)
 *   - SSE4.1: 8-bit 块拷贝 (block_copy) / 双向预测平均 (bi_avg) / 块填充 (fill_block)
 *     及 8-bit 子像素插值 (亮度 8 抽头 / 色度 4 抽头)
 *
 * 对齐要求:
 *   - avs2_mem_alloc/allocz 统一返回 32 字节对齐内存
 *   - 参考帧/目标帧数据由 avs2_frame_alloc 分配, 保证 32 字节对齐
 *   - 中间缓冲区用 AVS2_ALIGN32 声明, 步长 (MC_TMP_STRIDE=64) 保证行起始 32 字节对齐,
 *     因此对中间缓冲区的访问使用 _mm256_load_si256 / _mm256_store_si256 (aligned)
 *   - src (参考帧, 运动矢量偏移): MV 整数部分可为任意值, src 处于任意像素偏移,
 *     无法保证对齐, 因此使用 _mm256_loadu_si256 / _mm_loadu_si128 (unaligned)
 *   - dst (目标帧, CU 位置): 10-bit 下 CU 位置为 8 像素倍数 = 16 字节对齐,
 *     128-bit (16 字节) 存储使用 _mm_store_si128 (aligned);
 *     256-bit (32 字节) 存储因 dst 仅 16 字节对齐, 保持 _mm256_storeu_si256 (unaligned)
 *     8-bit 下 dst 仅 8 字节对齐, 128-bit 存储使用 _mm_storeu_si128 (unaligned)
 *   - 系数表为 int8_t, 使用 _mm256_cvtepi8_epi16 扩展为 16-bit
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

#include <immintrin.h>

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
 * 第一部分: 块拷贝 (10-bit AVX2)
 * =========================================================================== */

/* 整像素块拷贝 (10-bit, 每像素 2 字节)
 * width 取值: 4,8,16,32,64 (亮度); 2,4,8,16,32 (色度)
 * 使用 _mm256_loadu/storeu_si256 (32 字节 = 16 个 uint16)
 */
static void block_copy_10bit_avx2(pel_t *dst, int i_dst,
                                   const pel_t *src, int i_src,
                                   int width, int height)
{
    int y, x;
    /* 按 16 像素 (32 字节) 对齐处理 */
    if ((width & 15) == 0) {
        for (y = 0; y < height; y++) {
            if (y + 8 < height)
                _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
            for (x = 0; x < width; x += 16) {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + x));
                /* dst 仅 16 字节对齐, 256-bit 存储保持 unaligned */
                _mm256_storeu_si256((__m256i*)(dst + x), v);
            }
            src += i_src;
            dst += i_dst;
        }
    } else if ((width & 7) == 0) {
        /* 8 像素对齐 (128-bit) */
        for (y = 0; y < height; y++) {
            if (y + 8 < height)
                _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
            for (x = 0; x < width; x += 8) {
                __m128i v = _mm_loadu_si128((const __m128i*)(src + x));
                _mm_store_si128((__m128i*)(dst + x), v);
            }
            src += i_src;
            dst += i_dst;
        }
    } else if ((width & 3) == 0) {
        /* 4 像素 (64-bit) */
        for (y = 0; y < height; y++) {
            if (y + 8 < height)
                _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
            for (x = 0; x < width; x += 4) {
                __m128i v = _mm_loadl_epi64((const __m128i*)(src + x));
                _mm_storel_epi64((__m128i*)(dst + x), v);
            }
            src += i_src;
            dst += i_dst;
        }
    } else {
        /* width=2 (色度 4:2:0 最小块): 用 32-bit 拷贝避免越界 */
        for (y = 0; y < height; y++) {
            if (y + 8 < height)
                _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
            *(int*)(dst) = *(const int*)(src);
            src += i_src;
            dst += i_dst;
        }
    }
}

/* 整像素块拷贝 (8-bit, 每像素 1 字节) -- SSE4.1
 * width 取值: 4,8,16,32,64 (亮度); 2,4,8,16,32 (色度)
 * src 处于 MV 整数偏移, 无法保证对齐, 使用 _mm_loadu_si128 / _mm_loadl_epi64 (unaligned)
 * dst 为 CU 位置, 8-bit 下仅 8 字节对齐, 故 128-bit 存储使用 _mm_storeu_si128 (unaligned)
 */
static void block_copy_8bit_sse41(uint8_t *dst, int i_dst,
                                    const uint8_t *src, int i_src,
                                    int width, int height)
{
    int y, x;

    /* 按 16 像素 (16 字节) 对齐处理 */
    if ((width & 15) == 0) {
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 16) {
                __m128i v = _mm_loadu_si128((const __m128i*)(src + x));
                /* dst 仅 8 字节对齐, 128-bit 存储保持 unaligned */
                _mm_storeu_si128((__m128i*)(dst + x), v);
            }
            src += i_src;
            dst += i_dst;
        }
    } else if ((width & 7) == 0) {
        /* 8 像素 (64-bit) */
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 8) {
                __m128i v = _mm_loadl_epi64((const __m128i*)(src + x));
                _mm_storel_epi64((__m128i*)(dst + x), v);
            }
            src += i_src;
            dst += i_dst;
        }
    } else if ((width & 3) == 0) {
        /* 4 像素 (32-bit): 精确拷贝 4 字节, 避免越界写 */
        for (y = 0; y < height; y++) {
            for (x = 0; x < width; x += 4) {
                *(uint32_t*)(dst + x) = *(const uint32_t*)(src + x);
            }
            src += i_src;
            dst += i_dst;
        }
    } else {
        /* 其他宽度 (如 width=2): 标量拷贝避免越界 */
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
 * 第二部分: 亮度 8 抽头水平插值 (10-bit AVX2)
 *
 * 参考: libudavs2 com_if_filter_hor_8_w16_sse256_10bit
 * 算法: 对每个输出像素, 取 src[x-3..x+4] 共 8 个像素,
 *       与 8 个 int8_t 系数相乘累加, 加 32 后右移 6 位, 裁剪到 [0, max_val]
 *
 * 优化: 一次加载 16 个 uint16 像素 (256-bit), 用 p++ 滑动产生 8 组
 *       每组 4 个像素与系数的 madd, 共 8 组 -> 2 级 hadd -> 8 个结果
 * =========================================================================== */
static void ip_filter_luma_hor_10bit_avx2(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height, const int8_t *coeff, int max_val)
{
    int j, i;
    __m256i max_val1 = _mm256_set1_epi16((short)max_val);
    __m256i offset = _mm256_set1_epi32(32);
    /* 系数: 8 个 int8_t -> 4 个 int16 (每两个相邻系数为一组, 符号扩展)
     * mCoef 包含 [c0,c1,c0,c1,c0,c1,c0,c1,c0,c1,c0,c1,c0,c1,c0,c1] (16 个 int16) */
    const int32_t *coef32 = (const int32_t*)coeff;
    __m128i mCoef0 = _mm_setr_epi32(coef32[0], coef32[1], coef32[0], coef32[1]);
    __m256i mCoef = _mm256_cvtepi8_epi16(mCoef0);

    src -= 3;  /* 水平滤波需要左侧 3 个像素 */

    for (j = 0; j < height; j++) {
        const pel_t *p = src;
        /* Prefetch 8 rows ahead to hide DRAM latency (~100ns) */
        if (j + 8 < height)
            _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
        /* 主循环: 每次处理 16 个输出像素 */
        for (i = 0; i + 15 < width; i += 16) {
            __m256i T0, T1, T2, T3, T4, T5, T6, T7;
            __m256i M0, M1, M2, M3, M4, M5, M6, M7;

            /* 加载 8 组连续的 16-pixel 窗口 (p++ 滑动 1 个像素)
             * 每组与 mCoef 做 madd_epi16 (4 个 int32 结果)
             * 8 组共 32 个 int32, 经 3 级 hadd 得到 8 个 int32 (对应 8 个输出)
             * 但这里一次处理 16 个输出, 需要两轮 */
            T0 = _mm256_loadu_si256((__m256i*)p++);
            T1 = _mm256_loadu_si256((__m256i*)p++);
            T2 = _mm256_loadu_si256((__m256i*)p++);
            T3 = _mm256_loadu_si256((__m256i*)p++);
            T4 = _mm256_loadu_si256((__m256i*)p++);
            T5 = _mm256_loadu_si256((__m256i*)p++);
            T6 = _mm256_loadu_si256((__m256i*)p++);
            T7 = _mm256_loadu_si256((__m256i*)p++);

            M0 = _mm256_madd_epi16(T0, mCoef);
            M1 = _mm256_madd_epi16(T1, mCoef);
            M2 = _mm256_madd_epi16(T2, mCoef);
            M3 = _mm256_madd_epi16(T3, mCoef);
            M4 = _mm256_madd_epi16(T4, mCoef);
            M5 = _mm256_madd_epi16(T5, mCoef);
            M6 = _mm256_madd_epi16(T6, mCoef);
            M7 = _mm256_madd_epi16(T7, mCoef);

            M0 = _mm256_hadd_epi32(M0, M1);
            M1 = _mm256_hadd_epi32(M2, M3);
            M2 = _mm256_hadd_epi32(M4, M5);
            M3 = _mm256_hadd_epi32(M6, M7);

            M0 = _mm256_hadd_epi32(M0, M1);
            M1 = _mm256_hadd_epi32(M2, M3);

            M2 = _mm256_add_epi32(M0, offset);
            M3 = _mm256_add_epi32(M1, offset);
            M2 = _mm256_srai_epi32(M2, 6);
            M3 = _mm256_srai_epi32(M3, 6);
            M2 = _mm256_packus_epi32(M2, M3);
            M2 = _mm256_min_epu16(M2, max_val1);
            /* dst 仅 16 字节对齐, 256-bit 存储保持 unaligned */
            _mm256_storeu_si256((__m256i*)(dst + i), M2);

            p += 8;  /* 已滑动 8 次, 加上前面 8 次 p++, 共 16 次滑动对应下一轮起点 */
        }
        /* 处理剩余像素 (width 不是 16 的倍数)
         * src 已 -= 3, 故 src[i..i+7] 对应原始 src[i-3..i+4] */
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
 * 第三部分: 亮度 8 抽头垂直插值 (10-bit AVX2)
 *
 * 参考: libudavs2 com_if_filter_ver_8_w16_sse256_10bit
 * 算法: 对每个输出像素, 取 src[y-3..y+4] 共 8 行同一列像素,
 *       与 8 个 int8_t 系数相乘累加
 *
 * 优化: 利用系数对称性 (coeff[1]==coeff[6] 时合并加法减少乘法)
 *       一次处理 16 个像素 (一行)
 * =========================================================================== */
static void ip_filter_luma_ver_10bit_avx2(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height, const int8_t *coeff, int max_val)
{
    int i, j;
    __m256i max_val1 = _mm256_set1_epi16((short)max_val);
    __m256i mAddOffset = _mm256_set1_epi32(32);
    __m256i T0, T1, T2, T3, T4, T5, T6, T7;
    __m256i M0, M1, M2, M3, M4, M5, M6, M7;
    __m256i N0, N1, N2, N3, N4, N5, N6, N7;

    src -= 3 * i_src;

    if (coeff[3] != coeff[4]) {
        /* 非对称系数: 4 组系数各不相同 */
        __m128i c0 = _mm_set1_epi16(*(const short*)(coeff + 0));
        __m128i c1 = _mm_set1_epi16(*(const short*)(coeff + 2));
        __m128i c2 = _mm_set1_epi16(*(const short*)(coeff + 4));
        __m128i c3 = _mm_set1_epi16(*(const short*)(coeff + 6));
        __m256i coeff00 = _mm256_cvtepi8_epi16(c0);
        __m256i coeff01 = _mm256_cvtepi8_epi16(c1);
        __m256i coeff02 = _mm256_cvtepi8_epi16(c2);
        __m256i coeff03 = _mm256_cvtepi8_epi16(c3);

        for (j = 0; j < height; j++) {
            const pel_t *p = src;
            for (i = 0; i + 15 < width; i += 16) {
                /* Prefetch rows entering the 8-row window 8 iterations ahead.
                 * At ~100 cycles/iter, this gives ~800 cycles lead time to hide
                 * DRAM latency (~100ns = 400 cycles at 4GHz). */
                _mm_prefetch((const char*)(p + 16 * i_src), _MM_HINT_T0);
                _mm_prefetch((const char*)(p + 18 * i_src), _MM_HINT_T0);

                T0 = _mm256_loadu_si256((__m256i*)(p));
                T1 = _mm256_loadu_si256((__m256i*)(p + i_src));
                T2 = _mm256_loadu_si256((__m256i*)(p + 2 * i_src));
                T3 = _mm256_loadu_si256((__m256i*)(p + 3 * i_src));
                T4 = _mm256_loadu_si256((__m256i*)(p + 4 * i_src));
                T5 = _mm256_loadu_si256((__m256i*)(p + 5 * i_src));
                T6 = _mm256_loadu_si256((__m256i*)(p + 6 * i_src));
                T7 = _mm256_loadu_si256((__m256i*)(p + 7 * i_src));

                M0 = _mm256_unpacklo_epi16(T0, T1);
                M1 = _mm256_unpacklo_epi16(T2, T3);
                M2 = _mm256_unpacklo_epi16(T4, T5);
                M3 = _mm256_unpacklo_epi16(T6, T7);
                M4 = _mm256_unpackhi_epi16(T0, T1);
                M5 = _mm256_unpackhi_epi16(T2, T3);
                M6 = _mm256_unpackhi_epi16(T4, T5);
                M7 = _mm256_unpackhi_epi16(T6, T7);

                N0 = _mm256_madd_epi16(M0, coeff00);
                N1 = _mm256_madd_epi16(M1, coeff01);
                N2 = _mm256_madd_epi16(M2, coeff02);
                N3 = _mm256_madd_epi16(M3, coeff03);
                N4 = _mm256_madd_epi16(M4, coeff00);
                N5 = _mm256_madd_epi16(M5, coeff01);
                N6 = _mm256_madd_epi16(M6, coeff02);
                N7 = _mm256_madd_epi16(M7, coeff03);

                N0 = _mm256_add_epi32(N0, N1);
                N1 = _mm256_add_epi32(N2, N3);
                N2 = _mm256_add_epi32(N4, N5);
                N3 = _mm256_add_epi32(N6, N7);

                N0 = _mm256_add_epi32(N0, N1);
                N1 = _mm256_add_epi32(N2, N3);

                N0 = _mm256_add_epi32(N0, mAddOffset);
                N1 = _mm256_add_epi32(N1, mAddOffset);
                N0 = _mm256_srai_epi32(N0, 6);
                N1 = _mm256_srai_epi32(N1, 6);
                N0 = _mm256_packus_epi32(N0, N1);
                N0 = _mm256_min_epu16(N0, max_val1);
                /* dst 仅 16 字节对齐, 256-bit 存储保持 unaligned */
                _mm256_storeu_si256((__m256i*)(dst + i), N0);

                p += 16;
            }
            /* 剩余像素 */
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
    } else {
        /* 对称系数: coeff[3]==coeff[4], 合并 S3+S4 减少一次乘法
         * 此分支暂不特殊优化, 走通用路径 (与上面相同)
         * 注: 为简化代码, 这里仍使用 4 组系数, 性能差异可忽略 */
        __m128i c0 = _mm_set1_epi16(*(const short*)(coeff + 0));
        __m128i c1 = _mm_set1_epi16(*(const short*)(coeff + 2));
        __m128i c2 = _mm_set1_epi16(*(const short*)(coeff + 4));
        __m128i c3 = _mm_set1_epi16(*(const short*)(coeff + 6));
        __m256i coeff00 = _mm256_cvtepi8_epi16(c0);
        __m256i coeff01 = _mm256_cvtepi8_epi16(c1);
        __m256i coeff02 = _mm256_cvtepi8_epi16(c2);
        __m256i coeff03 = _mm256_cvtepi8_epi16(c3);

        for (j = 0; j < height; j++) {
            const pel_t *p = src;
            for (i = 0; i + 15 < width; i += 16) {
                /* Prefetch rows entering the 8-row window 8 iterations ahead.
                 * At ~100 cycles/iter, this gives ~800 cycles lead time to hide
                 * DRAM latency (~100ns = 400 cycles at 4GHz). */
                _mm_prefetch((const char*)(p + 16 * i_src), _MM_HINT_T0);
                _mm_prefetch((const char*)(p + 18 * i_src), _MM_HINT_T0);

                T0 = _mm256_loadu_si256((__m256i*)(p));
                T1 = _mm256_loadu_si256((__m256i*)(p + i_src));
                T2 = _mm256_loadu_si256((__m256i*)(p + 2 * i_src));
                T3 = _mm256_loadu_si256((__m256i*)(p + 3 * i_src));
                T4 = _mm256_loadu_si256((__m256i*)(p + 4 * i_src));
                T5 = _mm256_loadu_si256((__m256i*)(p + 5 * i_src));
                T6 = _mm256_loadu_si256((__m256i*)(p + 6 * i_src));
                T7 = _mm256_loadu_si256((__m256i*)(p + 7 * i_src));

                M0 = _mm256_unpacklo_epi16(T0, T1);
                M1 = _mm256_unpacklo_epi16(T2, T3);
                M2 = _mm256_unpacklo_epi16(T4, T5);
                M3 = _mm256_unpacklo_epi16(T6, T7);
                M4 = _mm256_unpackhi_epi16(T0, T1);
                M5 = _mm256_unpackhi_epi16(T2, T3);
                M6 = _mm256_unpackhi_epi16(T4, T5);
                M7 = _mm256_unpackhi_epi16(T6, T7);

                N0 = _mm256_madd_epi16(M0, coeff00);
                N1 = _mm256_madd_epi16(M1, coeff01);
                N2 = _mm256_madd_epi16(M2, coeff02);
                N3 = _mm256_madd_epi16(M3, coeff03);
                N4 = _mm256_madd_epi16(M4, coeff00);
                N5 = _mm256_madd_epi16(M5, coeff01);
                N6 = _mm256_madd_epi16(M6, coeff02);
                N7 = _mm256_madd_epi16(M7, coeff03);

                N0 = _mm256_add_epi32(N0, N1);
                N1 = _mm256_add_epi32(N2, N3);
                N2 = _mm256_add_epi32(N4, N5);
                N3 = _mm256_add_epi32(N6, N7);

                N0 = _mm256_add_epi32(N0, N1);
                N1 = _mm256_add_epi32(N2, N3);

                N0 = _mm256_add_epi32(N0, mAddOffset);
                N1 = _mm256_add_epi32(N1, mAddOffset);
                N0 = _mm256_srai_epi32(N0, 6);
                N1 = _mm256_srai_epi32(N1, 6);
                N0 = _mm256_packus_epi32(N0, N1);
                N0 = _mm256_min_epu16(N0, max_val1);
                /* dst 仅 16 字节对齐, 256-bit 存储保持 unaligned */
                _mm256_storeu_si256((__m256i*)(dst + i), N0);

                p += 16;
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
}

/* ===========================================================================
 * 第四部分: 亮度 8 抽头双向插值 (10-bit AVX2)
 *
 * 参考: libudavs2 com_if_filter_hor_ver_8_w16_sse256_10bit
 * 算法: 先水平滤波到中间缓冲 (shift1), 再垂直滤波到目标 (shift2)
 *       shift1 = bit_depth - 8 = 2 (10-bit)
 *       shift2 = 20 - bit_depth = 10 (10-bit)
 * =========================================================================== */
static void ip_filter_luma_ext_10bit_avx2(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y, int max_val)
{
    /* 中间缓冲: (height+7) 行 * width 列, 16-bit
     * 最大块 64x64 -> (64+7)*64*2 = 9216 字节, 栈分配 */
    AVS2_ALIGN32(short tmp_res[(64 + 7) * 64]);
    short *tmp = tmp_res;
    const int i_tmp = MC_TMP_STRIDE;
    int add1, shift1, add2, shift2;
    int i, j;
    __m256i offset;
    __m256i T0, T1, T2, T3, T4, T5, T6, T7;
    __m256i M0, M1, M2, M3, M4, M5, M6, M7;
    __m256i N0, N1, N2, N3, N4, N5, N6, N7;
    __m256i max_val1 = _mm256_set1_epi16((short)max_val);
    const int32_t *coef32 = (const int32_t*)coef_x;
    __m128i mCoef0 = _mm_setr_epi32(coef32[0], coef32[1], coef32[0], coef32[1]);
    __m256i mCoef = _mm256_cvtepi8_epi16(mCoef0);
    __m128i c0, c1, c2, c3;
    __m256i coeff00, coeff01, coeff02, coeff03;

    /* 10-bit: shift1=2, shift2=10 */
    shift1 = 2;
    shift2 = 10;
    add1 = (1 << shift1) >> 1;       /* = 2 */
    add2 = 1 << (shift2 - 1);        /* = 512 */

    /* ---- 第一级: 水平滤波 ---- */
    src += -3 * i_src - 3;
    offset = _mm256_set1_epi32(add1);

    for (j = -3; j < height + 4; j++) {
        const pel_t *p = src;
        /* Prefetch 8 rows ahead to hide DRAM latency */
        if (j + 8 < height + 4)
            _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
        for (i = 0; i + 15 < width; i += 16) {
            T0 = _mm256_loadu_si256((__m256i*)p++);
            T1 = _mm256_loadu_si256((__m256i*)p++);
            T2 = _mm256_loadu_si256((__m256i*)p++);
            T3 = _mm256_loadu_si256((__m256i*)p++);
            T4 = _mm256_loadu_si256((__m256i*)p++);
            T5 = _mm256_loadu_si256((__m256i*)p++);
            T6 = _mm256_loadu_si256((__m256i*)p++);
            T7 = _mm256_loadu_si256((__m256i*)p++);

            M0 = _mm256_madd_epi16(T0, mCoef);
            M1 = _mm256_madd_epi16(T1, mCoef);
            M2 = _mm256_madd_epi16(T2, mCoef);
            M3 = _mm256_madd_epi16(T3, mCoef);
            M4 = _mm256_madd_epi16(T4, mCoef);
            M5 = _mm256_madd_epi16(T5, mCoef);
            M6 = _mm256_madd_epi16(T6, mCoef);
            M7 = _mm256_madd_epi16(T7, mCoef);

            M0 = _mm256_hadd_epi32(M0, M1);
            M1 = _mm256_hadd_epi32(M2, M3);
            M2 = _mm256_hadd_epi32(M4, M5);
            M3 = _mm256_hadd_epi32(M6, M7);

            M0 = _mm256_hadd_epi32(M0, M1);
            M1 = _mm256_hadd_epi32(M2, M3);

            M2 = _mm256_add_epi32(M0, offset);
            M3 = _mm256_add_epi32(M1, offset);
            M2 = _mm256_srai_epi32(M2, shift1);
            M3 = _mm256_srai_epi32(M3, shift1);
            /* 中间结果是 signed 16-bit, 用 packs (有符号饱和) */
            M2 = _mm256_packs_epi32(M2, M3);
            /* tmp 缓冲区已 32 字节对齐 (AVS2_ALIGN32), i 为 16 的倍数 = 32 字节步进 */
            _mm256_store_si256((__m256i*)(tmp + i), M2);

            p += 8;
        }
        /* 剩余像素 (标量): src 已 -= 3, 故 src[i..i+7] 对应原始 src[i-3..i+4] */
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
    offset = _mm256_set1_epi32(add2);
    tmp = tmp_res;

    c0 = _mm_set1_epi16(*(const short*)(coef_y + 0));
    c1 = _mm_set1_epi16(*(const short*)(coef_y + 2));
    c2 = _mm_set1_epi16(*(const short*)(coef_y + 4));
    c3 = _mm_set1_epi16(*(const short*)(coef_y + 6));
    coeff00 = _mm256_cvtepi8_epi16(c0);
    coeff01 = _mm256_cvtepi8_epi16(c1);
    coeff02 = _mm256_cvtepi8_epi16(c2);
    coeff03 = _mm256_cvtepi8_epi16(c3);

    for (j = 0; j < height; j++) {
        const short *p = tmp;
        for (i = 0; i + 15 < width; i += 16) {
            /* tmp 缓冲区已 32 字节对齐, i 为 16 的倍数, i_tmp=64 元素=128 字节,
             * 故 p 及 p+k*i_tmp 均为 32 字节对齐, 使用 aligned 加载 */
            T0 = _mm256_load_si256((__m256i*)(p));
            T1 = _mm256_load_si256((__m256i*)(p + i_tmp));
            T2 = _mm256_load_si256((__m256i*)(p + 2 * i_tmp));
            T3 = _mm256_load_si256((__m256i*)(p + 3 * i_tmp));
            T4 = _mm256_load_si256((__m256i*)(p + 4 * i_tmp));
            T5 = _mm256_load_si256((__m256i*)(p + 5 * i_tmp));
            T6 = _mm256_load_si256((__m256i*)(p + 6 * i_tmp));
            T7 = _mm256_load_si256((__m256i*)(p + 7 * i_tmp));

            M0 = _mm256_unpacklo_epi16(T0, T1);
            M1 = _mm256_unpacklo_epi16(T2, T3);
            M2 = _mm256_unpacklo_epi16(T4, T5);
            M3 = _mm256_unpacklo_epi16(T6, T7);
            M4 = _mm256_unpackhi_epi16(T0, T1);
            M5 = _mm256_unpackhi_epi16(T2, T3);
            M6 = _mm256_unpackhi_epi16(T4, T5);
            M7 = _mm256_unpackhi_epi16(T6, T7);

            N0 = _mm256_madd_epi16(M0, coeff00);
            N1 = _mm256_madd_epi16(M1, coeff01);
            N2 = _mm256_madd_epi16(M2, coeff02);
            N3 = _mm256_madd_epi16(M3, coeff03);
            N4 = _mm256_madd_epi16(M4, coeff00);
            N5 = _mm256_madd_epi16(M5, coeff01);
            N6 = _mm256_madd_epi16(M6, coeff02);
            N7 = _mm256_madd_epi16(M7, coeff03);

            N0 = _mm256_add_epi32(N0, N1);
            N1 = _mm256_add_epi32(N2, N3);
            N2 = _mm256_add_epi32(N4, N5);
            N3 = _mm256_add_epi32(N6, N7);

            N0 = _mm256_add_epi32(N0, N1);
            N1 = _mm256_add_epi32(N2, N3);

            N0 = _mm256_add_epi32(N0, offset);
            N1 = _mm256_add_epi32(N1, offset);
            N0 = _mm256_srai_epi32(N0, shift2);
            N1 = _mm256_srai_epi32(N1, shift2);
            N0 = _mm256_packus_epi32(N0, N1);
            N0 = _mm256_min_epu16(N0, max_val1);
            /* dst 仅 16 字节对齐, 256-bit 存储保持 unaligned */
            _mm256_storeu_si256((__m256i*)(dst + i), N0);

            p += 16;
        }
        /* 剩余像素 (标量) */
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

/* ---- 融合 MC+avg: 双向预测第二路, dst[i] = (dst[i] + mc(src)[i] + 1) >> 1 ----
 * 与 ip_filter_luma_ext_10bit_avx2 相同, 仅垂直滤波最终存储改为与 dst 取平均,
 * 省去 pred2 中间缓冲的写+读. */
static void ip_filter_luma_ext_10bit_avg_avx2(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y, int max_val)
{
    AVS2_ALIGN32(short tmp_res[(64 + 7) * 64]);
    short *tmp = tmp_res;
    const int i_tmp = MC_TMP_STRIDE;
    int add1, shift1, add2, shift2;
    int i, j;
    __m256i offset;
    __m256i T0, T1, T2, T3, T4, T5, T6, T7;
    __m256i M0, M1, M2, M3, M4, M5, M6, M7;
    __m256i N0, N1, N2, N3, N4, N5, N6, N7;
    __m256i max_val1 = _mm256_set1_epi16((short)max_val);
    __m256i one = _mm256_set1_epi16(1);
    const int32_t *coef32 = (const int32_t*)coef_x;
    __m128i mCoef0 = _mm_setr_epi32(coef32[0], coef32[1], coef32[0], coef32[1]);
    __m256i mCoef = _mm256_cvtepi8_epi16(mCoef0);
    __m128i c0, c1, c2, c3;
    __m256i coeff00, coeff01, coeff02, coeff03;

    shift1 = 2;
    shift2 = 10;
    add1 = (1 << shift1) >> 1;
    add2 = 1 << (shift2 - 1);

    /* ---- 第一级: 水平滤波 (与普通版完全相同) ---- */
    src += -3 * i_src - 3;
    offset = _mm256_set1_epi32(add1);

    for (j = -3; j < height + 4; j++) {
        const pel_t *p = src;
        if (j + 8 < height + 4)
            _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
        for (i = 0; i + 15 < width; i += 16) {
            T0 = _mm256_loadu_si256((__m256i*)p++);
            T1 = _mm256_loadu_si256((__m256i*)p++);
            T2 = _mm256_loadu_si256((__m256i*)p++);
            T3 = _mm256_loadu_si256((__m256i*)p++);
            T4 = _mm256_loadu_si256((__m256i*)p++);
            T5 = _mm256_loadu_si256((__m256i*)p++);
            T6 = _mm256_loadu_si256((__m256i*)p++);
            T7 = _mm256_loadu_si256((__m256i*)p++);

            M0 = _mm256_madd_epi16(T0, mCoef);
            M1 = _mm256_madd_epi16(T1, mCoef);
            M2 = _mm256_madd_epi16(T2, mCoef);
            M3 = _mm256_madd_epi16(T3, mCoef);
            M4 = _mm256_madd_epi16(T4, mCoef);
            M5 = _mm256_madd_epi16(T5, mCoef);
            M6 = _mm256_madd_epi16(T6, mCoef);
            M7 = _mm256_madd_epi16(T7, mCoef);

            M0 = _mm256_hadd_epi32(M0, M1);
            M1 = _mm256_hadd_epi32(M2, M3);
            M2 = _mm256_hadd_epi32(M4, M5);
            M3 = _mm256_hadd_epi32(M6, M7);

            M0 = _mm256_hadd_epi32(M0, M1);
            M1 = _mm256_hadd_epi32(M2, M3);

            M2 = _mm256_add_epi32(M0, offset);
            M3 = _mm256_add_epi32(M1, offset);
            M2 = _mm256_srai_epi32(M2, shift1);
            M3 = _mm256_srai_epi32(M3, shift1);
            M2 = _mm256_packs_epi32(M2, M3);
            _mm256_store_si256((__m256i*)(tmp + i), M2);

            p += 8;
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
    offset = _mm256_set1_epi32(add2);
    tmp = tmp_res;

    c0 = _mm_set1_epi16(*(const short*)(coef_y + 0));
    c1 = _mm_set1_epi16(*(const short*)(coef_y + 2));
    c2 = _mm_set1_epi16(*(const short*)(coef_y + 4));
    c3 = _mm_set1_epi16(*(const short*)(coef_y + 6));
    coeff00 = _mm256_cvtepi8_epi16(c0);
    coeff01 = _mm256_cvtepi8_epi16(c1);
    coeff02 = _mm256_cvtepi8_epi16(c2);
    coeff03 = _mm256_cvtepi8_epi16(c3);

    for (j = 0; j < height; j++) {
        const short *p = tmp;
        for (i = 0; i + 15 < width; i += 16) {
            T0 = _mm256_load_si256((__m256i*)(p));
            T1 = _mm256_load_si256((__m256i*)(p + i_tmp));
            T2 = _mm256_load_si256((__m256i*)(p + 2 * i_tmp));
            T3 = _mm256_load_si256((__m256i*)(p + 3 * i_tmp));
            T4 = _mm256_load_si256((__m256i*)(p + 4 * i_tmp));
            T5 = _mm256_load_si256((__m256i*)(p + 5 * i_tmp));
            T6 = _mm256_load_si256((__m256i*)(p + 6 * i_tmp));
            T7 = _mm256_load_si256((__m256i*)(p + 7 * i_tmp));

            M0 = _mm256_unpacklo_epi16(T0, T1);
            M1 = _mm256_unpacklo_epi16(T2, T3);
            M2 = _mm256_unpacklo_epi16(T4, T5);
            M3 = _mm256_unpacklo_epi16(T6, T7);
            M4 = _mm256_unpackhi_epi16(T0, T1);
            M5 = _mm256_unpackhi_epi16(T2, T3);
            M6 = _mm256_unpackhi_epi16(T4, T5);
            M7 = _mm256_unpackhi_epi16(T6, T7);

            N0 = _mm256_madd_epi16(M0, coeff00);
            N1 = _mm256_madd_epi16(M1, coeff01);
            N2 = _mm256_madd_epi16(M2, coeff02);
            N3 = _mm256_madd_epi16(M3, coeff03);
            N4 = _mm256_madd_epi16(M4, coeff00);
            N5 = _mm256_madd_epi16(M5, coeff01);
            N6 = _mm256_madd_epi16(M6, coeff02);
            N7 = _mm256_madd_epi16(M7, coeff03);

            N0 = _mm256_add_epi32(N0, N1);
            N1 = _mm256_add_epi32(N2, N3);
            N2 = _mm256_add_epi32(N4, N5);
            N3 = _mm256_add_epi32(N6, N7);

            N0 = _mm256_add_epi32(N0, N1);
            N1 = _mm256_add_epi32(N2, N3);

            N0 = _mm256_add_epi32(N0, offset);
            N1 = _mm256_add_epi32(N1, offset);
            N0 = _mm256_srai_epi32(N0, shift2);
            N1 = _mm256_srai_epi32(N1, shift2);
            N0 = _mm256_packus_epi32(N0, N1);
            N0 = _mm256_min_epu16(N0, max_val1);
            /* 融合平均: dst[i] = (dst[i] + N0[i] + 1) >> 1 */
            {
                __m256i vdst = _mm256_loadu_si256((const __m256i*)(dst + i));
                __m256i sum = _mm256_add_epi16(vdst, N0);
                sum = _mm256_add_epi16(sum, one);
                sum = _mm256_srli_epi16(sum, 1);
                _mm256_storeu_si256((__m256i*)(dst + i), sum);
            }

            p += 16;
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
 * 第五部分: 色度 4 抽头水平插值 (10-bit AVX2)
 *
 * 参考: libudavs2 com_if_filter_hor_4_w8_sse256_10bit
 * 算法: 4 抽头, shift=6, offset=32
 * =========================================================================== */
static void ip_filter_chroma_hor_10bit_avx2(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height, const int8_t *coeff, int max_val)
{
    int row, col;
    const int offset = 32;
    const int shift = 6;

    /* mSwitch1/2 用于从 8 个连续像素中取出 4 组 4-tap 窗口
     * 256-bit 可容纳 16 个 uint16, 一次处理 8 个输出 (用 low 128 + high 128) */
    __m128i mCoef1 = _mm_set1_epi32(*(const int32_t*)coeff);
    __m256i mCoef = _mm256_cvtepi8_epi16(mCoef1);
    __m256i mSwitch1 = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9);
    __m256i mSwitch2 = _mm256_setr_epi8(
        4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13,
        4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13);
    __m256i mAddOffset = _mm256_set1_epi32(offset);
    __m256i max_val1 = _mm256_set1_epi16((short)max_val);

    src -= 1;

    for (row = 0; row < height; row++) {
        if (row + 8 < height)
            _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
        for (col = 0; col + 7 < width; col += 8) {
            __m256i S = _mm256_loadu_si256((__m256i*)(src + col));
            __m256i S0 = _mm256_permute4x64_epi64(S, 0x94);
            __m256i T0 = _mm256_madd_epi16(_mm256_shuffle_epi8(S0, mSwitch1), mCoef);
            __m256i T1 = _mm256_madd_epi16(_mm256_shuffle_epi8(S0, mSwitch2), mCoef);
            __m256i sum = _mm256_hadd_epi32(T0, T1);

            sum = _mm256_add_epi32(sum, mAddOffset);
            sum = _mm256_srai_epi32(sum, shift);
            sum = _mm256_packus_epi32(sum, sum);
            sum = _mm256_permute4x64_epi64(sum, 0xd8);
            sum = _mm256_min_epu16(sum, max_val1);
            _mm_store_si128((__m128i*)(dst + col), _mm256_castsi256_si128(sum));
        }
        /* 剩余像素: src 已 -= 1, 故 src[col..col+3] 对应原始 src[col-1..col+2] */
        for (; col < width; col++) {
            int v = src[col]     * coeff[0] + src[col + 1] * coeff[1]
                  + src[col + 2] * coeff[2] + src[col + 3] * coeff[3];
            v = (v + offset) >> shift;
            dst[col] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
        }
        src += i_src;
        dst += i_dst;
    }
}

/* ===========================================================================
 * 第六部分: 色度 4 抽头垂直插值 (10-bit AVX2)
 *
 * 参考: libudavs2 com_if_filter_ver_4_w16_sse256_10bit
 * =========================================================================== */
static void ip_filter_chroma_ver_10bit_avx2(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height, const int8_t *coeff, int max_val)
{
    int row, col;
    const int offset = 32;
    const int shift = 6;
    int bsym = (coeff[1] == coeff[2]);
    __m256i mAddOffset = _mm256_set1_epi32(offset);
    __m256i max_val1 = _mm256_set1_epi16((short)max_val);
    const int i_src2 = i_src * 2;
    const int i_src3 = i_src * 3;

    src -= i_src;

    if (bsym) {
        /* 对称: coeff[1]==coeff[2], 合并 S1+S2 */
        __m128i c1 = _mm_set1_epi16(*(const short*)coeff);
        __m256i coeff0 = _mm256_cvtepi8_epi16(c1);

        for (row = 0; row < height; row++) {
            for (col = 0; col + 15 < width; col += 16) {
                /* Prefetch rows entering the 4-row window 8 iterations ahead */
                _mm_prefetch((const char*)(src + col + 12 * i_src), _MM_HINT_T0);
                _mm_prefetch((const char*)(src + col + 14 * i_src), _MM_HINT_T0);

                __m256i S0 = _mm256_loadu_si256((__m256i*)(src + col));
                __m256i S1 = _mm256_loadu_si256((__m256i*)(src + col + i_src));
                __m256i S2 = _mm256_loadu_si256((__m256i*)(src + col + i_src2));
                __m256i S3 = _mm256_loadu_si256((__m256i*)(src + col + i_src3));

                __m256i R0 = _mm256_add_epi16(S0, S3);
                __m256i R1 = _mm256_add_epi16(S1, S2);

                __m256i T0 = _mm256_unpacklo_epi16(R0, R1);
                __m256i T1 = _mm256_unpackhi_epi16(R0, R1);

                T0 = _mm256_madd_epi16(T0, coeff0);
                T1 = _mm256_madd_epi16(T1, coeff0);

                T0 = _mm256_add_epi32(T0, mAddOffset);
                T1 = _mm256_add_epi32(T1, mAddOffset);
                T0 = _mm256_srai_epi32(T0, shift);
                T1 = _mm256_srai_epi32(T1, shift);
                T0 = _mm256_packus_epi32(T0, T1);
                T0 = _mm256_min_epu16(T0, max_val1);
                /* dst 仅 16 字节对齐, 256-bit 存储保持 unaligned */
                _mm256_storeu_si256((__m256i*)(dst + col), T0);
            }
            for (; col < width; col++) {
                int v = src[col]             * coeff[0]
                      + src[col + i_src]     * coeff[1]
                      + src[col + i_src2]    * coeff[2]
                      + src[col + i_src3]    * coeff[3];
                v = (v + offset) >> shift;
                dst[col] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
            }
            src += i_src;
            dst += i_dst;
        }
    } else {
        __m128i c0 = _mm_set1_epi16(*(const short*)(coeff + 0));
        __m128i c1 = _mm_set1_epi16(*(const short*)(coeff + 2));
        __m256i coeff0 = _mm256_cvtepi8_epi16(c0);
        __m256i coeff1 = _mm256_cvtepi8_epi16(c1);

        for (row = 0; row < height; row++) {
            for (col = 0; col + 15 < width; col += 16) {
                /* Prefetch rows entering the 4-row window 8 iterations ahead */
                _mm_prefetch((const char*)(src + col + 12 * i_src), _MM_HINT_T0);
                _mm_prefetch((const char*)(src + col + 14 * i_src), _MM_HINT_T0);

                __m256i S0 = _mm256_loadu_si256((__m256i*)(src + col));
                __m256i S1 = _mm256_loadu_si256((__m256i*)(src + col + i_src));
                __m256i S2 = _mm256_loadu_si256((__m256i*)(src + col + i_src2));
                __m256i S3 = _mm256_loadu_si256((__m256i*)(src + col + i_src3));

                __m256i T0 = _mm256_unpacklo_epi16(S0, S1);
                __m256i T1 = _mm256_unpackhi_epi16(S0, S1);
                __m256i T2 = _mm256_unpacklo_epi16(S2, S3);
                __m256i T3 = _mm256_unpackhi_epi16(S2, S3);

                T0 = _mm256_madd_epi16(T0, coeff0);
                T1 = _mm256_madd_epi16(T1, coeff0);
                T2 = _mm256_madd_epi16(T2, coeff1);
                T3 = _mm256_madd_epi16(T3, coeff1);

                __m256i N0 = _mm256_add_epi32(T0, T2);
                __m256i N1 = _mm256_add_epi32(T1, T3);

                N0 = _mm256_add_epi32(N0, mAddOffset);
                N1 = _mm256_add_epi32(N1, mAddOffset);
                N0 = _mm256_srai_epi32(N0, shift);
                N1 = _mm256_srai_epi32(N1, shift);
                N0 = _mm256_packus_epi32(N0, N1);
                N0 = _mm256_min_epu16(N0, max_val1);
                /* dst 仅 16 字节对齐, 256-bit 存储保持 unaligned */
                _mm256_storeu_si256((__m256i*)(dst + col), N0);
            }
            for (; col < width; col++) {
                int v = src[col]             * coeff[0]
                      + src[col + i_src]     * coeff[1]
                      + src[col + i_src2]    * coeff[2]
                      + src[col + i_src3]    * coeff[3];
                v = (v + offset) >> shift;
                dst[col] = (pel_t)(v < 0 ? 0 : (v > max_val ? max_val : v));
            }
            src += i_src;
            dst += i_dst;
        }
    }
}

/* ===========================================================================
 * 第七部分: 色度 4 抽头双向插值 (10-bit AVX2)
 *
 * 参考: libudavs2 com_if_filter_hor_ver_4_w16_sse256_10bit
 * =========================================================================== */
static void ip_filter_chroma_ext_10bit_avx2(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y, int max_val)
{
    AVS2_ALIGN32(short tmp_res[(32 + 3) * 32]);
    short *tmp = tmp_res;
    const int i_tmp = 32;
    int shift1, shift2, add1, add2;
    int row, col;
    __m256i mask8 = _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
    __m256i max_val1 = _mm256_set1_epi16((short)max_val);
    __m256i mAddOffset;
    __m128i mCoef1 = _mm_set1_epi32(*(const int32_t*)coef_x);
    __m256i mCoef = _mm256_cvtepi8_epi16(mCoef1);
    __m256i mSwitch1 = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9);
    __m256i mSwitch2 = _mm256_setr_epi8(
        4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13,
        4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13);
    __m128i cy0, cy1;
    __m256i coeffy0, coeffy1;

    /* 10-bit: shift1=2, shift2=10 */
    shift1 = 2;
    shift2 = 10;
    add1 = (1 << shift1) >> 1;
    add2 = 1 << (shift2 - 1);

    /* ---- 第一级: 水平滤波 ---- */
    mAddOffset = _mm256_set1_epi32(add1);
    src = src - i_src - 1;

    for (row = -1; row < height + 2; row++) {
        if (row + 8 < height + 2)
            _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
        for (col = 0; col + 7 < width; col += 8) {
            __m256i S = _mm256_loadu_si256((__m256i*)(src + col));
            __m256i S0 = _mm256_permute4x64_epi64(S, 0x94);
            __m256i T0 = _mm256_madd_epi16(_mm256_shuffle_epi8(S0, mSwitch1), mCoef);
            __m256i T1 = _mm256_madd_epi16(_mm256_shuffle_epi8(S0, mSwitch2), mCoef);
            __m256i sum = _mm256_hadd_epi32(T0, T1);

            sum = _mm256_add_epi32(sum, mAddOffset);
            sum = _mm256_srai_epi32(sum, shift1);
            /* 中间结果可为负, 用 packs (有符号饱和) 保留, 与 C 参考一致 */
            sum = _mm256_packs_epi32(sum, sum);
            sum = _mm256_permute4x64_epi64(sum, 0xd8);
            /* 用 maskstore 写 8 个 int16 (低 128-bit) */
            _mm256_maskstore_epi32((int*)(tmp + col), mask8, sum);
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
    mAddOffset = _mm256_set1_epi32(add2);
    cy0 = _mm_set1_epi16(*(const short*)(coef_y + 0));
    cy1 = _mm_set1_epi16(*(const short*)(coef_y + 2));
    coeffy0 = _mm256_cvtepi8_epi16(cy0);
    coeffy1 = _mm256_cvtepi8_epi16(cy1);

    for (row = 0; row < height; row++) {
        for (col = 0; col + 7 < width; col += 8) {
            /* col 以 8 (16 字节) 步进, 偶数 col 为 32 字节对齐但奇数 col 仅 16 字节对齐,
             * 故保持 unaligned 加载 */
            __m256i S0 = _mm256_loadu_si256((__m256i*)(tmp + col));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(tmp + col + i_tmp));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(tmp + col + 2 * i_tmp));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(tmp + col + 3 * i_tmp));

            __m256i T0 = _mm256_unpacklo_epi16(S0, S1);
            __m256i T1 = _mm256_unpackhi_epi16(S0, S1);
            __m256i T2 = _mm256_unpacklo_epi16(S2, S3);
            __m256i T3 = _mm256_unpackhi_epi16(S2, S3);

            T0 = _mm256_madd_epi16(T0, coeffy0);
            T1 = _mm256_madd_epi16(T1, coeffy0);
            T2 = _mm256_madd_epi16(T2, coeffy1);
            T3 = _mm256_madd_epi16(T3, coeffy1);

            __m256i N0 = _mm256_add_epi32(T0, T2);
            __m256i N1 = _mm256_add_epi32(T1, T3);

            N0 = _mm256_add_epi32(N0, mAddOffset);
            N1 = _mm256_add_epi32(N1, mAddOffset);
            N0 = _mm256_srai_epi32(N0, shift2);
            N1 = _mm256_srai_epi32(N1, shift2);
            N0 = _mm256_packus_epi32(N0, N1);
            N0 = _mm256_min_epu16(N0, max_val1);
            _mm_store_si128((__m128i*)(dst + col), _mm256_castsi256_si128(N0));
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

/* ---- 融合 MC+avg: 色度双向预测第二路, dst[i] = (dst[i] + mc(src)[i] + 1) >> 1 ---- */
static void ip_filter_chroma_ext_10bit_avg_avx2(
        pel_t *dst, int i_dst, const pel_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y, int max_val)
{
    AVS2_ALIGN32(short tmp_res[(32 + 3) * 32]);
    short *tmp = tmp_res;
    const int i_tmp = 32;
    int shift1, shift2, add1, add2;
    int row, col;
    __m256i mask8 = _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0);
    __m256i max_val1 = _mm256_set1_epi16((short)max_val);
    __m128i one16 = _mm_set1_epi16(1);
    __m256i mAddOffset;
    __m128i mCoef1 = _mm_set1_epi32(*(const int32_t*)coef_x);
    __m256i mCoef = _mm256_cvtepi8_epi16(mCoef1);
    __m256i mSwitch1 = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9,
        0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9);
    __m256i mSwitch2 = _mm256_setr_epi8(
        4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13,
        4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13);
    __m128i cy0, cy1;
    __m256i coeffy0, coeffy1;

    shift1 = 2;
    shift2 = 10;
    add1 = (1 << shift1) >> 1;
    add2 = 1 << (shift2 - 1);

    /* ---- 第一级: 水平滤波 (与普通版相同) ---- */
    mAddOffset = _mm256_set1_epi32(add1);
    src = src - i_src - 1;

    for (row = -1; row < height + 2; row++) {
        if (row + 8 < height + 2)
            _mm_prefetch((const char*)(src + 8 * i_src), _MM_HINT_T0);
        for (col = 0; col + 7 < width; col += 8) {
            __m256i S = _mm256_loadu_si256((__m256i*)(src + col));
            __m256i S0 = _mm256_permute4x64_epi64(S, 0x94);
            __m256i T0 = _mm256_madd_epi16(_mm256_shuffle_epi8(S0, mSwitch1), mCoef);
            __m256i T1 = _mm256_madd_epi16(_mm256_shuffle_epi8(S0, mSwitch2), mCoef);
            __m256i sum = _mm256_hadd_epi32(T0, T1);

            sum = _mm256_add_epi32(sum, mAddOffset);
            sum = _mm256_srai_epi32(sum, shift1);
            sum = _mm256_packs_epi32(sum, sum);
            sum = _mm256_permute4x64_epi64(sum, 0xd8);
            _mm256_maskstore_epi32((int*)(tmp + col), mask8, sum);
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
    mAddOffset = _mm256_set1_epi32(add2);
    cy0 = _mm_set1_epi16(*(const short*)(coef_y + 0));
    cy1 = _mm_set1_epi16(*(const short*)(coef_y + 2));
    coeffy0 = _mm256_cvtepi8_epi16(cy0);
    coeffy1 = _mm256_cvtepi8_epi16(cy1);

    for (row = 0; row < height; row++) {
        for (col = 0; col + 7 < width; col += 8) {
            __m256i S0 = _mm256_loadu_si256((__m256i*)(tmp + col));
            __m256i S1 = _mm256_loadu_si256((__m256i*)(tmp + col + i_tmp));
            __m256i S2 = _mm256_loadu_si256((__m256i*)(tmp + col + 2 * i_tmp));
            __m256i S3 = _mm256_loadu_si256((__m256i*)(tmp + col + 3 * i_tmp));

            __m256i T0 = _mm256_unpacklo_epi16(S0, S1);
            __m256i T1 = _mm256_unpackhi_epi16(S0, S1);
            __m256i T2 = _mm256_unpacklo_epi16(S2, S3);
            __m256i T3 = _mm256_unpackhi_epi16(S2, S3);

            T0 = _mm256_madd_epi16(T0, coeffy0);
            T1 = _mm256_madd_epi16(T1, coeffy0);
            T2 = _mm256_madd_epi16(T2, coeffy1);
            T3 = _mm256_madd_epi16(T3, coeffy1);

            __m256i N0 = _mm256_add_epi32(T0, T2);
            __m256i N1 = _mm256_add_epi32(T1, T3);

            N0 = _mm256_add_epi32(N0, mAddOffset);
            N1 = _mm256_add_epi32(N1, mAddOffset);
            N0 = _mm256_srai_epi32(N0, shift2);
            N1 = _mm256_srai_epi32(N1, shift2);
            N0 = _mm256_packus_epi32(N0, N1);
            N0 = _mm256_min_epu16(N0, max_val1);
            /* 融合平均: dst[col] = (dst[col] + N0[col] + 1) >> 1 */
            {
                __m128i vdst = _mm_load_si128((const __m128i*)(dst + col));
                __m128i vsrc = _mm256_castsi256_si128(N0);
                __m128i sum = _mm_add_epi16(vdst, vsrc);
                sum = _mm_add_epi16(sum, one16);
                sum = _mm_srli_epi16(sum, 1);
                _mm_store_si128((__m128i*)(dst + col), sum);
            }
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
 * 第七部分(续): 8-bit 子像素插值 (SSE4.1)
 *
 * 参考: libudavs2 com_if_filter_*_sse128 (8-bit SSE128)
 *   - 亮度: 8 抽头, shift=6, offset=32
 *   - 色度: 4 抽头, shift=6, offset=32
 *   - 双向 (ext): shift1=0 (bit_depth-8), shift2=12 (20-bit_depth)
 *
 * 数据类型: 像素 uint8_t, 系数 int8_t
 *   - 水平滤波: _mm_shuffle_epi8 重排 + _mm_maddubs_epi16 (无符号*有符号)
 *   - 垂直滤波: _mm_unpacklo_epi8 交错 + _mm_maddubs_epi16
 *   - ext 垂直: 中间结果 int16, 用 _mm_madd_epi16 (有符号*有符号)
 *
 * 尾部处理: width 不是 8 的倍数时用标量循环 (不用 maskmove, 简化代码)
 * =========================================================================== */

/* ---- 亮度 8 抽头水平插值 (8-bit SSE4.1) ----
 * 一次处理 8 个输出像素: 加载 16 字节, 用 4 组 shuffle 产生 4 对输出,
 * 每对经 maddubs 得到 4 个 int16 部分和, 3 级 hadd 合并为完整 8 个 int16 结果,
 * 加 32 右移 6 位, packus 饱和到 8-bit, 存储 8 字节 */
static void ip_filter_luma_hor_8bit_sse41(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height, const int8_t *coeff)
{
    int j, i;
    /* shuffle 掩码: 每组从 16 字节中取 2 个 8 字节窗口, 步进 2 */
    __m128i m_switch1 = _mm_setr_epi8(0,1,2,3,4,5,6,7, 1,2,3,4,5,6,7,8);
    __m128i m_switch2 = _mm_setr_epi8(2,3,4,5,6,7,8,9, 3,4,5,6,7,8,9,10);
    __m128i m_switch3 = _mm_setr_epi8(4,5,6,7,8,9,10,11, 5,6,7,8,9,10,11,12);
    __m128i m_switch4 = _mm_setr_epi8(6,7,8,9,10,11,12,13, 7,8,9,10,11,12,13,14);
    /* 系数: 8 个 int8_t 复制两份填满 16 字节 */
    __m128i m_coef = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)coeff),
                                         _mm_loadl_epi64((const __m128i*)coeff));
    __m128i offset = _mm_set1_epi16(32);

    src -= 3;  /* 水平滤波需要左侧 3 个像素 */

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

            /* dst 仅 8 字节对齐, 用 _mm_storel_epi64 存储 8 字节 */
            _mm_storel_epi64((__m128i*)(dst + i), sum);

            p += 8;
        }
        /* 尾部标量 */
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

/* ---- 亮度 8 抽头垂直插值 (8-bit SSE4.1) ----
 * 一次处理 8 个像素宽: 加载 8 行各 8 字节,
 * _mm_unpacklo_epi8 交错相邻行, maddubs 与系数对相乘累加,
 * 4 组部分和相加得完整 8 个 int16 结果 */
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

    src -= 3 * i_src;  /* 垂直滤波需要上方 3 行 */

    for (j = 0; j < height; j++) {
        const uint8_t *p = src;
        for (i = 0; i + 7 < width; i += 8) {
            __m128i s0, s1, s2, s3, s4, s5, s6, s7;
            __m128i m0, m1, m2, m3;
            __m128i sum;

            s0 = _mm_loadl_epi64((const __m128i*)(p + i));
            s1 = _mm_loadl_epi64((const __m128i*)(p + i + i_src));
            s2 = _mm_loadl_epi64((const __m128i*)(p + i + 2 * i_src));
            s3 = _mm_loadl_epi64((const __m128i*)(p + i + 3 * i_src));
            s4 = _mm_loadl_epi64((const __m128i*)(p + i + 4 * i_src));
            s5 = _mm_loadl_epi64((const __m128i*)(p + i + 5 * i_src));
            s6 = _mm_loadl_epi64((const __m128i*)(p + i + 6 * i_src));
            s7 = _mm_loadl_epi64((const __m128i*)(p + i + 7 * i_src));

            /* 交错相邻行: [s_r0[c], s_r1[c], s_r0[c+1], s_r1[c+1], ...] */
            m0 = _mm_unpacklo_epi8(s0, s1);
            m1 = _mm_unpacklo_epi8(s2, s3);
            m2 = _mm_unpacklo_epi8(s4, s5);
            m3 = _mm_unpacklo_epi8(s6, s7);

            /* maddubs: 无符号像素 * 有符号系数, 8 个 int16 部分和 */
            sum = _mm_maddubs_epi16(m0, m_coef01);
            sum = _mm_add_epi16(sum, _mm_maddubs_epi16(m1, m_coef23));
            sum = _mm_add_epi16(sum, _mm_maddubs_epi16(m2, m_coef45));
            sum = _mm_add_epi16(sum, _mm_maddubs_epi16(m3, m_coef67));

            sum = _mm_add_epi16(sum, offset);
            sum = _mm_srai_epi16(sum, 6);
            sum = _mm_packus_epi16(sum, sum);

            _mm_storel_epi64((__m128i*)(dst + i), sum);
        }
        /* 尾部标量 */
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

/* ---- 亮度 8 抽头双向插值 (8-bit SSE4.1) ----
 * 两级滤波: 先水平滤波到中间缓冲 (int16, shift1=0), 再垂直滤波到目标 (shift2=12)
 * 中间缓冲: (64+7)*64 int16, ALIGNED_16, 用 _mm_store_si128 对齐存储
 * 垂直级用 _mm_madd_epi16 (有符号*有符号), 系数经 _mm_cvtepi8_epi16 符号扩展 */
static void ip_filter_luma_ext_8bit_sse41(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y)
{
    AVS2_ALIGN16(short tmp_res[(64 + 7) * 64]);
    short *tmp = tmp_res;
    const int i_tmp = MC_TMP_STRIDE;
    /* 8-bit: shift1=0, add1=0; shift2=12, add2=2048 */
    const int shift2 = 12;
    const int add2 = 1 << (shift2 - 1);
    int i, j;
    /* 水平级 shuffle 掩码和系数 */
    __m128i m_switch1 = _mm_setr_epi8(0,1,2,3,4,5,6,7, 1,2,3,4,5,6,7,8);
    __m128i m_switch2 = _mm_setr_epi8(2,3,4,5,6,7,8,9, 3,4,5,6,7,8,9,10);
    __m128i m_switch3 = _mm_setr_epi8(4,5,6,7,8,9,10,11, 5,6,7,8,9,10,11,12);
    __m128i m_switch4 = _mm_setr_epi8(6,7,8,9,10,11,12,13, 7,8,9,10,11,12,13,14);
    __m128i m_coef = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)coef_x),
                                         _mm_loadl_epi64((const __m128i*)coef_x));
    /* 垂直级系数: int8 -> int16 (有符号扩展) */
    __m128i cy0 = _mm_cvtepi8_epi16(_mm_set1_epi16(*(const short*)(coef_y + 0)));
    __m128i cy1 = _mm_cvtepi8_epi16(_mm_set1_epi16(*(const short*)(coef_y + 2)));
    __m128i cy2 = _mm_cvtepi8_epi16(_mm_set1_epi16(*(const short*)(coef_y + 4)));
    __m128i cy3 = _mm_cvtepi8_epi16(_mm_set1_epi16(*(const short*)(coef_y + 6)));
    __m128i offset2 = _mm_set1_epi32(add2);

    /* ---- 第一级: 水平滤波 (shift1=0, 直接存 int16 原始和) ---- */
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

            /* 中间缓冲区 ALIGNED_16, i 为 8 的倍数 = 16 字节步进, 用对齐存储 */
            _mm_store_si128((__m128i*)(tmp + i), sum);

            p += 8;
        }
        /* 尾部标量: shift1=0, add1=0, 直接存原始和 */
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

    /* ---- 第二级: 垂直滤波 ---- */
    tmp = tmp_res;

    for (j = 0; j < height; j++) {
        const short *p = tmp;
        for (i = 0; i + 7 < width; i += 8) {
            __m128i t0, t1, t2, t3, t4, t5, t6, t7;
            __m128i m0_lo, m0_hi, m1_lo, m1_hi, m2_lo, m2_hi, m3_lo, m3_hi;
            __m128i n0_lo, n0_hi, n1_lo, n1_hi, n2_lo, n2_hi, n3_lo, n3_hi;
            __m128i sum_lo, sum_hi, result;

            /* tmp 缓冲区 ALIGNED_16, i_tmp=64 元素=128 字节, p 及 p+k*i_tmp 均 16 字节对齐 */
            t0 = _mm_load_si128((__m128i*)(p));
            t1 = _mm_load_si128((__m128i*)(p + i_tmp));
            t2 = _mm_load_si128((__m128i*)(p + 2 * i_tmp));
            t3 = _mm_load_si128((__m128i*)(p + 3 * i_tmp));
            t4 = _mm_load_si128((__m128i*)(p + 4 * i_tmp));
            t5 = _mm_load_si128((__m128i*)(p + 5 * i_tmp));
            t6 = _mm_load_si128((__m128i*)(p + 6 * i_tmp));
            t7 = _mm_load_si128((__m128i*)(p + 7 * i_tmp));

            /* 交错相邻行, 分高低半 */
            m0_lo = _mm_unpacklo_epi16(t0, t1);
            m0_hi = _mm_unpackhi_epi16(t0, t1);
            m1_lo = _mm_unpacklo_epi16(t2, t3);
            m1_hi = _mm_unpackhi_epi16(t2, t3);
            m2_lo = _mm_unpacklo_epi16(t4, t5);
            m2_hi = _mm_unpackhi_epi16(t4, t5);
            m3_lo = _mm_unpacklo_epi16(t6, t7);
            m3_hi = _mm_unpackhi_epi16(t6, t7);

            /* madd_epi16: 有符号 int16 * 有符号 int16, 4 个 int32 结果 */
            n0_lo = _mm_madd_epi16(m0_lo, cy0);
            n0_hi = _mm_madd_epi16(m0_hi, cy0);
            n1_lo = _mm_madd_epi16(m1_lo, cy1);
            n1_hi = _mm_madd_epi16(m1_hi, cy1);
            n2_lo = _mm_madd_epi16(m2_lo, cy2);
            n2_hi = _mm_madd_epi16(m2_hi, cy2);
            n3_lo = _mm_madd_epi16(m3_lo, cy3);
            n3_hi = _mm_madd_epi16(m3_hi, cy3);

            /* 4 组部分和相加 */
            sum_lo = _mm_add_epi32(n0_lo, n1_lo);
            sum_lo = _mm_add_epi32(sum_lo, n2_lo);
            sum_lo = _mm_add_epi32(sum_lo, n3_lo);

            sum_hi = _mm_add_epi32(n0_hi, n1_hi);
            sum_hi = _mm_add_epi32(sum_hi, n2_hi);
            sum_hi = _mm_add_epi32(sum_hi, n3_hi);

            /* 加 add2, 右移 shift2 */
            sum_lo = _mm_add_epi32(sum_lo, offset2);
            sum_hi = _mm_add_epi32(sum_hi, offset2);
            sum_lo = _mm_srai_epi32(sum_lo, shift2);
            sum_hi = _mm_srai_epi32(sum_hi, shift2);

            /* int32 -> uint16 (packus_epi32) -> uint8 (packus_epi16), 两次饱和 */
            result = _mm_packus_epi32(sum_lo, sum_hi);
            result = _mm_packus_epi16(result, result);

            _mm_storel_epi64((__m128i*)(dst + i), result);

            p += 8;
        }
        /* 尾部标量 */
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

/* ---- 色度 4 抽头水平插值 (8-bit SSE4.1) ----
 * 一次处理 8 个输出像素: 2 组 shuffle 产生 2 对输出,
 * 每对经 maddubs 得到 8 个 int16 部分和, 相加得完整结果 */
static void ip_filter_chroma_hor_4bit_sse41(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height, const int8_t *coeff)
{
    int row, col;
    /* 4 抽头: 2 组 shuffle, 每组取 2 个相邻像素与系数对 (c0,c1) / (c2,c3) 配对 */
    __m128i m_switch1 = _mm_setr_epi8(0,1, 1,2, 2,3, 3,4, 4,5, 5,6, 6,7, 7,8);
    __m128i m_switch2 = _mm_setr_epi8(2,3, 3,4, 4,5, 5,6, 6,7, 7,8, 8,9, 9,10);
    __m128i m_coef1 = _mm_set1_epi16(*(const short*)(coeff + 0));
    __m128i m_coef2 = _mm_set1_epi16(*(const short*)(coeff + 2));
    __m128i offset = _mm_set1_epi16(32);

    src -= 1;  /* 4 抽头需要左侧 1 个像素 */

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
        /* 尾部标量 */
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

/* ---- 色度 4 抽头垂直插值 (8-bit SSE4.1) ----
 * 一次处理 8 个像素宽: 加载 4 行各 8 字节,
 * 交错后 maddubs 与系数对相乘, 2 组部分和相加 */
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

    src -= i_src;  /* 4 抽头需要上方 1 行 */

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
        /* 尾部标量 */
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

/* ---- 色度 4 抽头双向插值 (8-bit SSE4.1) ----
 * 两级滤波: 先水平滤波到中间缓冲 (int16, shift1=0), 再垂直滤波到目标 (shift2=12)
 * 中间缓冲: (32+3)*32 int16, ALIGNED_16 */
static void ip_filter_chroma_ext_4bit_sse41(
        uint8_t *dst, int i_dst, const uint8_t *src, int i_src,
        int width, int height,
        const int8_t *coef_x, const int8_t *coef_y)
{
    AVS2_ALIGN16(short tmp_res[(32 + 3) * 32]);
    short *tmp = tmp_res;
    const int i_tmp = 32;
    /* 8-bit: shift1=0, add1=0; shift2=12, add2=2048 */
    const int shift2 = 12;
    const int add2 = 1 << (shift2 - 1);
    int row, col;
    __m128i m_switch1 = _mm_setr_epi8(0,1, 1,2, 2,3, 3,4, 4,5, 5,6, 6,7, 7,8);
    __m128i m_switch2 = _mm_setr_epi8(2,3, 3,4, 4,5, 5,6, 6,7, 7,8, 8,9, 9,10);
    __m128i m_coef1 = _mm_set1_epi16(*(const short*)(coef_x + 0));
    __m128i m_coef2 = _mm_set1_epi16(*(const short*)(coef_x + 2));
    /* 垂直级系数: int8 -> int16 (有符号扩展) */
    __m128i cy0 = _mm_cvtepi8_epi16(_mm_set1_epi16(*(const short*)(coef_y + 0)));
    __m128i cy1 = _mm_cvtepi8_epi16(_mm_set1_epi16(*(const short*)(coef_y + 2)));
    __m128i offset2 = _mm_set1_epi32(add2);

    /* ---- 第一级: 水平滤波 (shift1=0, 直接存 int16 原始和) ---- */
    src = src - i_src - 1;

    for (row = -1; row < height + 2; row++) {
        const uint8_t *p = src;
        for (col = 0; col + 7 < width; col += 8) {
            __m128i s = _mm_loadu_si128((const __m128i*)p);
            __m128i t1 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch1), m_coef1);
            __m128i t2 = _mm_maddubs_epi16(_mm_shuffle_epi8(s, m_switch2), m_coef2);
            __m128i sum = _mm_add_epi16(t1, t2);

            /* 中间缓冲区 ALIGNED_16, col 为 8 的倍数 = 16 字节步进, 用对齐存储 */
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

    /* ---- 第二级: 垂直滤波 ---- */
    tmp = tmp_res;

    for (row = 0; row < height; row++) {
        const short *p = tmp;
        for (col = 0; col + 7 < width; col += 8) {
            __m128i t0, t1, t2, t3;
            __m128i m0_lo, m0_hi, m1_lo, m1_hi;
            __m128i n0_lo, n0_hi, n1_lo, n1_hi;
            __m128i sum_lo, sum_hi, result;

            /* tmp 缓冲区 ALIGNED_16, i_tmp=32 元素=64 字节, p 及 p+k*i_tmp 均 16 字节对齐 */
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
        /* 尾部标量 */
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
 * 第八部分: mc_luma_avx2 / mc_chroma_avx2 入口
 * =========================================================================== */

/* mc_luma_avx2: 亮度运动补偿入口
 * 10-bit 用 AVX2; 8-bit 用 SSE4.1 (整像素拷贝 + 子像素插值)
 */
static void mc_luma_avx2(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                         ptrdiff_t dstride, int w, int h, int mx, int my,
                         int bit_depth)
{
    if (bit_depth > 8) {
        int dx = mx & 3;
        int dy = my & 3;
        int max_val = (1 << bit_depth) - 1;
        pel_t *dst16 = (pel_t*)(void *)dst;
        const pel_t *src16 = (const pel_t*)(const void *)src;
        int i_dst = (int)(dstride >> 1);  /* 元素步长 */
        int i_src = (int)(sstride >> 1);

        if (dx == 0 && dy == 0) {
            block_copy_10bit_avx2(dst16, i_dst, src16, i_src, w, h);
        } else if (dx == 0) {
            ip_filter_luma_ver_10bit_avx2(dst16, i_dst, src16, i_src,
                                           w, h, avs2_intpl_filters[dy], max_val);
        } else if (dy == 0) {
            ip_filter_luma_hor_10bit_avx2(dst16, i_dst, src16, i_src,
                                           w, h, avs2_intpl_filters[dx], max_val);
        } else {
            ip_filter_luma_ext_10bit_avx2(dst16, i_dst, src16, i_src,
                                           w, h,
                                           avs2_intpl_filters[dx],
                                           avs2_intpl_filters[dy], max_val);
        }
    } else {
        /* 8-bit: SSE4.1 子像素插值 */
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

/* mc_chroma_avx2: 色度运动补偿入口
 * 10-bit 用 AVX2; 8-bit 用 SSE4.1 (整像素拷贝 + 子像素插值) */
static void mc_chroma_avx2(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
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
            block_copy_10bit_avx2(dst16, i_dst, src16, i_src, w, h);
        } else if (dx == 0) {
            ip_filter_chroma_ver_10bit_avx2(dst16, i_dst, src16, i_src,
                                             w, h, avs2_intpl_filters_c[dy], max_val);
        } else if (dy == 0) {
            ip_filter_chroma_hor_10bit_avx2(dst16, i_dst, src16, i_src,
                                             w, h, avs2_intpl_filters_c[dx], max_val);
        } else {
            ip_filter_chroma_ext_10bit_avx2(dst16, i_dst, src16, i_src,
                                             w, h,
                                             avs2_intpl_filters_c[dx],
                                             avs2_intpl_filters_c[dy], max_val);
        }
    } else {
        /* 8-bit: SSE4.1 子像素插值 */
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
 * 第八部分(续): 双向预测平均 / 块填充 AVX2 实现
 * ===========================================================================
 * bi_avg: dst[i] = (dst[i] + pred2[i] + 1) >> 1, 10-bit 路径用 AVX2.
 * fill_block: dst[i] = fill_val, 10-bit 路径用 AVX2.
 * 8-bit 路径使用 SSE4.1 (block_copy / bi_avg / fill_block).
 * 两个 10-bit 值相加最大 4092, 不溢出 int16_t (32767). */

/* 双向预测平均 (8-bit, SSE4.1)
 * dst 为 8-bit 像素, pred2 为 16-bit 存储的第二路预测 (值域 [0,255])
 * 算法: dst[i] = (dst[i] + (uint8_t)pred2[i] + 1) >> 1
 * pred2 经 _mm_packus_epi16 饱和到 [0,255], 与 (uint8_t) 强制转换在值域内等价
 * 用 _mm_avg_epu8 完成无符号 8-bit 带舍入平均: (a + b + 1) >> 1 */
static void bi_avg_8bit_sse41(uint8_t *dst, int i_dst,
                                const int16_t *pred2, int pred2_stride,
                                int width, int height)
{
    int y, x;

    for (y = 0; y < height; y++) {
        uint8_t *d = dst;
        const int16_t *p = pred2;
        x = 0;
        /* 主循环: 每次处理 16 个像素 (16 字节 dst + 32 字节 pred2) */
        for (; x + 15 < width; x += 16) {
            __m128i vd = _mm_loadu_si128((const __m128i*)(d + x));
            __m128i vp_lo = _mm_loadu_si128((const __m128i*)(p + x));
            __m128i vp_hi = _mm_loadu_si128((const __m128i*)(p + x + 8));
            /* pred2 (int16) 饱和转 uint8: 低 8 个 + 高 8 个 -> 16 个 uint8 */
            __m128i vp = _mm_packus_epi16(vp_lo, vp_hi);
            /* 无符号 8-bit 平均: (a + b + 1) >> 1 */
            __m128i avg = _mm_avg_epu8(vd, vp);
            /* dst 仅 8 字节对齐, 128-bit 存储保持 unaligned */
            _mm_storeu_si128((__m128i*)(d + x), avg);
        }
        /* 8 像素 (64-bit) */
        for (; x + 7 < width; x += 8) {
            __m128i vd = _mm_loadl_epi64((const __m128i*)(d + x));
            __m128i vp = _mm_loadu_si128((const __m128i*)(p + x));
            vp = _mm_packus_epi16(vp, _mm_setzero_si128());
            __m128i avg = _mm_avg_epu8(vd, vp);
            _mm_storel_epi64((__m128i*)(d + x), avg);
        }
        /* 尾部标量 */
        for (; x < width; x++) {
            int v = d[x] + (uint8_t)p[x];
            d[x] = (uint8_t)((v + 1) >> 1);
        }
        dst += i_dst;
        pred2 += pred2_stride;
    }
}

/* 块填充 (8-bit, SSE4.1)
 * 用 _mm_set1_epi8 广播填充值, _mm_storeu_si128 每次 16 字节
 * dst 仅 8 字节对齐, 故 128-bit 存储使用 _mm_storeu_si128 (unaligned) */
static void fill_block_8bit_sse41(uint8_t *dst, int i_dst,
                                    int width, int height, int fill_val)
{
    int y, x;
    __m128i vfill = _mm_set1_epi8((char)fill_val);

    for (y = 0; y < height; y++) {
        uint8_t *d = dst;
        x = 0;
        /* 主循环: 每次填充 16 字节 (16 像素) */
        for (; x + 16 <= width; x += 16) {
            _mm_storeu_si128((__m128i*)(d + x), vfill);
        }
        /* 尾部标量 */
        for (; x < width; x++) {
            d[x] = (uint8_t)fill_val;
        }
        dst += i_dst;
    }
}

static void bi_avg_avx2(uint8_t *dst, ptrdiff_t dst_stride, const int16_t *pred2,
                        int pred2_stride, int w, int h, int bit_depth)
{
    if (bit_depth > 8) {
        uint16_t *p16 = (uint16_t *)(void *)dst;
        int stride16 = (int)(dst_stride >> 1);
        const __m256i one = _mm256_set1_epi16(1);
        int y, x;

        for (y = 0; y < h; y++) {
            uint16_t *d = p16 + y * stride16;
            const int16_t *p = pred2 + y * pred2_stride;
            x = 0;
            /* AVX2: 每次处理 16 个像素 */
            for (; x + 15 < w; x += 16) {
                __m256i vd = _mm256_loadu_si256((const __m256i *)(d + x));
                __m256i vp = _mm256_loadu_si256((const __m256i *)(p + x));
                __m256i sum = _mm256_add_epi16(vd, vp);
                sum = _mm256_add_epi16(sum, one);
                sum = _mm256_srli_epi16(sum, 1);
                /* dst 仅 16 字节对齐, 256-bit 存储/加载保持 unaligned */
                _mm256_storeu_si256((__m256i *)(d + x), sum);
            }
            /* 尾部标量 */
            for (; x < w; x++) {
                int v = d[x] + p[x];
                d[x] = (uint16_t)((v + 1) >> 1);
            }
        }
    } else {
        /* 8-bit: SSE4.1 双向预测平均 */
        bi_avg_8bit_sse41(dst, (int)dst_stride, pred2, pred2_stride, w, h);
    }
}

/* bi_avg_2src: dst[i] = (pred1[i] + pred2[i] + 1) >> 1.
 * 两个预测均来自栈缓冲 (L1), 避免帧缓冲 cache 往返.
 * 10-bit 路径用 AVX2; 8-bit 将 int16 预测截断后用 SSE4.1 pavgb. */
static void bi_avg_2src_avx2(uint8_t *dst, ptrdiff_t dst_stride,
                             const int16_t *pred1, int pred1_stride,
                             const int16_t *pred2, int pred2_stride,
                             int w, int h, int bit_depth)
{
    if (bit_depth > 8) {
        uint16_t *p16 = (uint16_t *)(void *)dst;
        int stride16 = (int)(dst_stride >> 1);
        const __m256i one = _mm256_set1_epi16(1);
        int y, x;

        for (y = 0; y < h; y++) {
            uint16_t *d = p16 + y * stride16;
            const int16_t *p1 = pred1 + y * pred1_stride;
            const int16_t *p2 = pred2 + y * pred2_stride;
            x = 0;
            for (; x + 15 < w; x += 16) {
                __m256i v1 = _mm256_loadu_si256((const __m256i *)(p1 + x));
                __m256i v2 = _mm256_loadu_si256((const __m256i *)(p2 + x));
                __m256i sum = _mm256_add_epi16(v1, v2);
                sum = _mm256_add_epi16(sum, one);
                sum = _mm256_srli_epi16(sum, 1);
                _mm256_storeu_si256((__m256i *)(d + x), sum);
            }
            for (; x < w; x++) {
                int v = p1[x] + p2[x];
                d[x] = (uint16_t)((v + 1) >> 1);
            }
        }
    } else {
        /* 8-bit: pred1/pred2 为 int16, 需截断为 uint8 后用 pavgb */
        int y, x;
        for (y = 0; y < h; y++) {
            uint8_t *d = dst + y * dst_stride;
            const int16_t *p1 = pred1 + y * pred1_stride;
            const int16_t *p2 = pred2 + y * pred2_stride;
            x = 0;
            for (; x + 15 < w; x += 16) {
                __m128i v1 = _mm_packs_epi16(
                    _mm_loadu_si128((const __m128i*)(p1 + x)),
                    _mm_loadu_si128((const __m128i*)(p1 + x + 8)));
                __m128i v2 = _mm_packs_epi16(
                    _mm_loadu_si128((const __m128i*)(p2 + x)),
                    _mm_loadu_si128((const __m128i*)(p2 + x + 8)));
                __m128i avg = _mm_avg_epu8(v1, v2);
                _mm_storeu_si128((__m128i*)(d + x), avg);
            }
            for (; x < w; x++) {
                d[x] = (uint8_t)((p1[x] + p2[x] + 1) >> 1);
            }
        }
    }
}

static void fill_block_avx2(uint8_t *dst, ptrdiff_t dst_stride, int w, int h,
                            int fill_val, int bit_depth)
{
    if (bit_depth > 8) {
        uint16_t *p16 = (uint16_t *)(void *)dst;
        int stride16 = (int)(dst_stride >> 1);
        __m256i vfill = _mm256_set1_epi16((short)fill_val);
        int y, x;

        for (y = 0; y < h; y++) {
            uint16_t *d = p16 + y * stride16;
            x = 0;
            for (; x + 15 < w; x += 16) {
                /* dst 仅 16 字节对齐, 256-bit 存储保持 unaligned */
                _mm256_storeu_si256((__m256i *)(d + x), vfill);
            }
            /* 尾部标量 */
            for (; x < w; x++) {
                d[x] = (uint16_t)fill_val;
            }
        }
    } else {
        /* 8-bit: SSE4.1 块填充 */
        fill_block_8bit_sse41(dst, (int)dst_stride, w, h, fill_val);
    }
}

/* ===========================================================================
 * 第九部分: 融合 MC+avg 入口 (mc_luma_avg / mc_chroma_avg)
 * ===========================================================================
 * 双向预测第二路: dst[i] = (dst[i] + mc(src,mv)[i] + 1) >> 1
 * - ext 路径 (dx,dy 均非零, B 帧主路径): 融合版, 省去 pred2 中间缓冲
 * - 整像素路径 (dx=dy=0): block_avg 直接平均
 * - hor/ver 单方向路径: pred2 中间缓冲 + bi_avg (较少见, 不单独融合)
 * - 8-bit: 回退到 C 实现 (测试码流为 10-bit) */

/* 整像素块平均 (10-bit): dst[i] = (dst[i] + src[i] + 1) >> 1 */
static void block_avg_10bit_avx2(pel_t *dst, int i_dst,
                                  const pel_t *src, int i_src,
                                  int width, int height)
{
    const __m256i one = _mm256_set1_epi16(1);
    int y, x;
    for (y = 0; y < height; y++) {
        for (x = 0; x + 15 < width; x += 16) {
            __m256i vd = _mm256_loadu_si256((const __m256i*)(dst + x));
            __m256i vs = _mm256_loadu_si256((const __m256i*)(src + x));
            __m256i sum = _mm256_add_epi16(vd, vs);
            sum = _mm256_add_epi16(sum, one);
            sum = _mm256_srli_epi16(sum, 1);
            _mm256_storeu_si256((__m256i*)(dst + x), sum);
        }
        for (; x < width; x++) {
            dst[x] = (pel_t)((dst[x] + src[x] + 1) >> 1);
        }
        dst += i_dst;
        src += i_src;
    }
}

/* mc_luma_avg_avx2: 亮度 MC + 双向平均 */
static void mc_luma_avg_avx2(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
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
            /* 整像素: 直接平均 */
            block_avg_10bit_avx2(dst16, i_dst, src16, i_src, w, h);
        } else if (dx != 0 && dy != 0) {
            /* 双向插值: 融合版 (B 帧主路径) */
            ip_filter_luma_ext_10bit_avg_avx2(dst16, i_dst, src16, i_src,
                                              w, h,
                                              avs2_intpl_filters[dx],
                                              avs2_intpl_filters[dy], max_val);
        } else {
            /* 单方向插值: pred2 中间缓冲 + bi_avg */
            AVS2_ALIGN32(int16_t pred2[64 * 64]);
            if (dx == 0) {
                ip_filter_luma_ver_10bit_avx2((pel_t*)pred2, w, src16, i_src,
                                              w, h, avs2_intpl_filters[dy], max_val);
            } else {
                ip_filter_luma_hor_10bit_avx2((pel_t*)pred2, w, src16, i_src,
                                              w, h, avs2_intpl_filters[dx], max_val);
            }
            bi_avg_avx2(dst, dstride, pred2, w, w, h, bit_depth);
        }
    } else {
        /* 8-bit: 回退到 C 实现 */
        mc_luma_avg_c(src, sstride, dst, dstride, w, h, mx, my, bit_depth);
    }
}

/* mc_chroma_avg_avx2: 色度 MC + 双向平均 */
static void mc_chroma_avg_avx2(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
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
            block_avg_10bit_avx2(dst16, i_dst, src16, i_src, w, h);
        } else if (dx != 0 && dy != 0) {
            ip_filter_chroma_ext_10bit_avg_avx2(dst16, i_dst, src16, i_src,
                                                w, h,
                                                avs2_intpl_filters_c[dx],
                                                avs2_intpl_filters_c[dy], max_val);
        } else {
            AVS2_ALIGN32(int16_t pred2[32 * 32]);
            if (dx == 0) {
                ip_filter_chroma_ver_10bit_avx2((pel_t*)pred2, w, src16, i_src,
                                                w, h, avs2_intpl_filters_c[dy], max_val);
            } else {
                ip_filter_chroma_hor_10bit_avx2((pel_t*)pred2, w, src16, i_src,
                                                w, h, avs2_intpl_filters_c[dx], max_val);
            }
            bi_avg_avx2(dst, dstride, pred2, w, w, h, bit_depth);
        }
    } else {
        mc_chroma_avg_c(src, sstride, dst, dstride, w, h, mx, my, bit_depth);
    }
}

/* ===========================================================================
 * 第十部分: 注册函数
 * =========================================================================== */

/* SSE4.1: 8-bit 函数由 AVX2 分发器调用 (block_copy / bi_avg / fill_block),
 * 此处无需单独注册 */
void avs2_mc_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }

/* AVX2: 注册运动补偿 (10-bit 用 AVX2, 8-bit 用 SSE4.1) */
void avs2_mc_init_avx2(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.mc_luma    = mc_luma_avx2;
    avs2_dsp_table.mc_chroma  = mc_chroma_avx2;
    avs2_dsp_table.mc_luma_avg   = mc_luma_avg_avx2;
    avs2_dsp_table.mc_chroma_avg = mc_chroma_avg_avx2;
    avs2_dsp_table.bi_avg     = bi_avg_avx2;
    avs2_dsp_table.bi_avg_2src = bi_avg_2src_avx2;
    avs2_dsp_table.fill_block = fill_block_avx2;
}

/* AVX512 预留 */
void avs2_mc_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }

#else  /* 非 x86 平台 */

void avs2_mc_init_sse41(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_mc_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }
void avs2_mc_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }

#endif
