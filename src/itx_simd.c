/*
 * itx_simd.c - 反变换 SIMD 实现 (x86)
 *
 * 当前实现:
 *   - SSE4.1: 4x4 IDCT (从 davs2 intrinsic_idct.cc 移植)
 *   - AVX2:   8x8 IDCT (从 davs2 intrinsic_idct_avx2.cc 移植)
 *   - 16x16/32x32/64x64: 暂用 C 回退
 *
 * 预留: AVX512 接口 (暂空)
 *
 * 与 davs2 的差异:
 *   - davs2 使用全局 g_bit_depth, 本实现通过参数传递 bit_depth
 *   - davs2 使用 coeff_t (int16_t 别名), 本实现直接使用 int16_t
 *   - davs2 使用 ALIGN32 宏, 本实现自定义 AVS2_ALIGN32 宏
 *
 * 就地变换: src == dst 时安全 (所有读取在写入之前完成)
 *
 * 对齐要求: avs2_mem_alloc/allocz 统一返回 32 字节对齐内存.
 *   - 4x4 SSE4.1: coeff 基地址 16 字节对齐, 用 _mm_load/store_si128
 *   - 8x8 AVX2:   coeff 基地址 32 字节对齐, 行起始至少 16 字节对齐,
 *                 用 _mm_load/store_si128 (128-bit) + _mm256_set_m128i 组合
 *                 系数表用 AVS2_ALIGN32 声明配合 _mm256_load_si256
 */

#include "internal.h"

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

/* ---- C 回退函数声明 (在 itx.c 中定义) ---- */
extern void idct_4x4_c(const int16_t *src, int16_t *dst, int i_dst, int bit_depth);
extern void idct_8x8_c(const int16_t *src, int16_t *dst, int i_dst, int bit_depth);

/* ===========================================================================
 * 4x4 反 DCT (SSE4.1)
 * 参考: davs2 source/common/vec/intrinsic_idct.cc idct_4x4_sse128
 *
 * 系数矩阵 g_t4[4][4]:
 *   { 32,  32,  32,  32 }
 *   { 42,  17, -17, -42 }
 *   { 32, -32, -32,  32 }
 *   { 17, -42,  42, -17 }
 *
 * 常量打包为 32 位 (两个 int16_t):
 *   c16_p32_p32 = {32, 32, 32, 32, ...}  — g_t4[0] 与 g_t4[2] 偶数项
 *   c16_n32_p32 = {-32, 32, -32, 32, ...} — g_t4[2] 奇偶分离
 *   c16_p17_p42 = {42, 17, 42, 17, ...}  — g_t4[1] 与 g_t4[3] 奇数项
 *   c16_n42_p17 = {-42, 17, -42, 17, ...}
 *
 * shift1=5, shift2=20-bit_depth, clip2=bit_depth+1
 *
 * 对齐: coeff 基地址 32 字节对齐 (avs2_mem_allocz 保证),
 *       coeff+0 与 coeff+8 均为 16 字节对齐, 可用 _mm_load/store_si128
 * ===========================================================================
 */
static void idct_4x4_sse41(int16_t *coeff, int w, int h, int bit_depth)
{
    const int shift1 = 5;
    const int shift2 = 20 - bit_depth;
    const int clip_depth2 = bit_depth + 1;

    const __m128i c16_p17_p42 = _mm_set1_epi32(0x0011002A);  /* {42, 17} */
    const __m128i c16_n42_p17 = _mm_set1_epi32(0xFFD60011);  /* {-42, 17} */
    const __m128i c16_n32_p32 = _mm_set1_epi32(0xFFE00020);  /* {-32, 32} */
    const __m128i c16_p32_p32 = _mm_set1_epi32(0x00200020);  /* {32, 32} */

    __m128i c32_rnd = _mm_set1_epi32(1 << (shift1 - 1));
    __m128i S0, S1;
    __m128i T0, T1;
    __m128i E0, E1, O0, O1;

    (void)w; (void)h;

    /* 加载 4x4 块 (2 个 128-bit 寄存器, 每个存 2 行) — 对齐加载 */
    S0 = _mm_load_si128((const __m128i*)(coeff + 0));   /* row0, row1 */
    S1 = _mm_load_si128((const __m128i*)(coeff + 8));   /* row2, row3 */

    /* ---- 第一遍: 水平反变换 ---- */
    T0 = _mm_unpacklo_epi16(S0, S1);
    E0 = _mm_add_epi32(_mm_madd_epi16(T0, c16_p32_p32), c32_rnd);
    E1 = _mm_add_epi32(_mm_madd_epi16(T0, c16_n32_p32), c32_rnd);

    T1 = _mm_unpackhi_epi16(S0, S1);
    O0 = _mm_madd_epi16(T1, c16_p17_p42);
    O1 = _mm_madd_epi16(T1, c16_n42_p17);

    S0 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E0, O0), shift1),
                          _mm_srai_epi32(_mm_sub_epi32(E1, O1), shift1));
    S1 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E1, O1), shift1),
                          _mm_srai_epi32(_mm_sub_epi32(E0, O0), shift1));

    /* ---- 转置 4x4 ---- */
    T0 = _mm_unpacklo_epi16(S0, S1);
    T1 = _mm_unpackhi_epi16(S0, S1);
    S0 = _mm_unpacklo_epi32(T0, T1);
    S1 = _mm_unpackhi_epi32(T0, T1);

    /* ---- 第二遍: 垂直反变换 ---- */
    c32_rnd = _mm_set1_epi32(shift2 > 0 ? (1 << (shift2 - 1)) : 0);

    T0 = _mm_unpacklo_epi16(S0, S1);
    E0 = _mm_add_epi32(_mm_madd_epi16(T0, c16_p32_p32), c32_rnd);
    E1 = _mm_add_epi32(_mm_madd_epi16(T0, c16_n32_p32), c32_rnd);

    T1 = _mm_unpackhi_epi16(S0, S1);
    O0 = _mm_madd_epi16(T1, c16_p17_p42);
    O1 = _mm_madd_epi16(T1, c16_n42_p17);

    S0 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E0, O0), shift2),
                          _mm_srai_epi32(_mm_sub_epi32(E1, O1), shift2));
    S1 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E1, O1), shift2),
                          _mm_srai_epi32(_mm_sub_epi32(E0, O0), shift2));

    /* ---- 转置 4x4 ---- */
    T0 = _mm_unpacklo_epi16(S0, S1);
    T1 = _mm_unpackhi_epi16(S0, S1);
    S0 = _mm_unpacklo_epi32(T0, T1);
    S1 = _mm_unpackhi_epi32(T0, T1);

    /* ---- 裁剪到 [-(1<<(clip_depth2-1)), (1<<(clip_depth2-1))-1] ---- */
    {
        const __m128i max_val = _mm_set1_epi16((short)((1 << (clip_depth2 - 1)) - 1));
        const __m128i min_val = _mm_set1_epi16((short)(-(1 << (clip_depth2 - 1))));

        S0 = _mm_max_epi16(_mm_min_epi16(S0, max_val), min_val);
        S1 = _mm_max_epi16(_mm_min_epi16(S1, max_val), min_val);
    }

    /* ---- 存储 (就地, 连续 4x4) — 对齐存储 ---- */
    _mm_store_si128((__m128i*)(coeff + 0), S0);
    _mm_store_si128((__m128i*)(coeff + 8), S1);
}

/* ===========================================================================
 * 8x8 反 DCT (AVX2)
 * 参考: davs2 source/common/vec/intrinsic_idct_avx2.cc idct_8x8_avx2
 *
 * 系数表 tab_idct_8x8_256[12][16]:
 *   [0..7] 为奇数行系数对 (用于计算 O0..O3)
 *   [8..11] 为偶数行系数对 (用于计算 EE0,EE1,EO0,EO1)
 *
 * shift1=5, shift2=20-bit_depth, clip2=bit_depth+1
 *
 * 对齐: coeff 基地址 32 字节对齐 (avs2_mem_allocz 保证),
 *       8x8 块各行起始 (coeff+8*k) 至少 16 字节对齐, 可用 _mm_load/store_si128.
 *       256-bit 系数表用 AVS2_ALIGN32 声明, 配合 _mm256_load_si256.
 * ===========================================================================
 */
AVS2_ALIGN32(static const int16_t tab_idct_8x8_256[12][16]) =
{
    { 44, 38, 44, 38, 44, 38, 44, 38, 44, 38, 44, 38, 44, 38, 44, 38 },
    { 25, 9, 25, 9, 25, 9, 25, 9, 25, 9, 25, 9, 25, 9, 25, 9 },
    { 38, -9, 38, -9, 38, -9, 38, -9, 38, -9, 38, -9, 38, -9, 38, -9 },
    { -44, -25, -44, -25, -44, -25, -44, -25, -44, -25, -44, -25, -44, -25, -44, -25 },
    { 25, -44, 25, -44, 25, -44, 25, -44, 25, -44, 25, -44, 25, -44, 25, -44 },
    { 9, 38, 9, 38, 9, 38, 9, 38, 9, 38, 9, 38, 9, 38, 9, 38 },
    { 9, -25, 9, -25, 9, -25, 9, -25, 9, -25, 9, -25, 9, -25, 9, -25 },
    { 38, -44, 38, -44, 38, -44, 38, -44, 38, -44, 38, -44, 38, -44, 38, -44 },
    { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 },
    { 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32, 32, -32 },
    { 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17, 42, 17 },
    { 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42, 17, -42 }
};

static void idct_8x8_avx2(int16_t *coeff, int w, int h, int bit_depth)
{
    const int SHIFT1 = 5;
    const int SHIFT2 = 20 - bit_depth;
    const int CLIP2 = bit_depth + 1;

    __m256i mAdd;
    __m256i S1S5, S3S7;
    __m256i T0, T1, T2, T3;
    __m256i E0, E1, E2, E3, O0, O1, O2, O3;
    __m256i EE0, EE1, EO0, EO1;
    __m256i S0, S1, S2, S3, S4, S5, S6, S7;
    __m256i C00, C01, C02, C03, C04, C05, C06, C07;
    __m256i max_val, min_val;

    (void)w; (void)h;

    /* ===================================================================
     * 第一遍: 水平反变换 (每行 8 点 -> 8 点)
     * =================================================================== */

    /* 加载奇数行 (1,3,5,7), 交织成 T2/T3 供 madd 使用 — 对齐加载 */
    S1S5 = _mm256_set_m128i(_mm_load_si128((const __m128i*)&coeff[40]), _mm_load_si128((const __m128i*)&coeff[ 8]));
    S3S7 = _mm256_set_m128i(_mm_load_si128((const __m128i*)&coeff[56]), _mm_load_si128((const __m128i*)&coeff[24]));

    T0 = _mm256_unpacklo_epi16(S1S5, S3S7);
    T1 = _mm256_unpackhi_epi16(S1S5, S3S7);
    T2 = _mm256_permute2x128_si256(T0, T1, 0x20);
    T3 = _mm256_permute2x128_si256(T0, T1, 0x31);

    /* O[k] = sum(g_t8[1][k]*S1 + g_t8[3][k]*S3 + g_t8[5][k]*S5 + g_t8[7][k]*S7) */
    O0 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[0]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[1]))));
    O1 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[2]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[3]))));
    O2 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[4]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[5]))));
    O3 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[6]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[7]))));

    /* 加载偶数行 (0,2,4,6) — 对齐加载 */
    S1S5 = _mm256_set_m128i(_mm_load_si128((const __m128i*)&coeff[16]), _mm_load_si128((const __m128i*)&coeff[ 0]));
    S3S7 = _mm256_set_m128i(_mm_load_si128((const __m128i*)&coeff[48]), _mm_load_si128((const __m128i*)&coeff[32]));

    T0 = _mm256_unpacklo_epi16(S1S5, S3S7);
    T1 = _mm256_unpackhi_epi16(S1S5, S3S7);
    T2 = _mm256_permute2x128_si256(T0, T1, 0x20);
    T3 = _mm256_permute2x128_si256(T0, T1, 0x31);

    EE0 = _mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[8])));
    EE1 = _mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[9])));
    EO0 = _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[10])));
    EO1 = _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[11])));

    /* 合并偶数项与奇数项 */
    mAdd = _mm256_set1_epi32(1 << (SHIFT1 - 1));

    E0 = _mm256_add_epi32(EE0, EO0);
    E1 = _mm256_add_epi32(EE1, EO1);
    E3 = _mm256_sub_epi32(EE0, EO0);
    E2 = _mm256_sub_epi32(EE1, EO1);
    E0 = _mm256_add_epi32(E0, mAdd);
    E1 = _mm256_add_epi32(E1, mAdd);
    E2 = _mm256_add_epi32(E2, mAdd);
    E3 = _mm256_add_epi32(E3, mAdd);

    S0 = _mm256_srai_epi32(_mm256_add_epi32(E0, O0), SHIFT1);
    S7 = _mm256_srai_epi32(_mm256_sub_epi32(E0, O0), SHIFT1);
    S1 = _mm256_srai_epi32(_mm256_add_epi32(E1, O1), SHIFT1);
    S6 = _mm256_srai_epi32(_mm256_sub_epi32(E1, O1), SHIFT1);
    S2 = _mm256_srai_epi32(_mm256_add_epi32(E2, O2), SHIFT1);
    S5 = _mm256_srai_epi32(_mm256_sub_epi32(E2, O2), SHIFT1);
    S3 = _mm256_srai_epi32(_mm256_add_epi32(E3, O3), SHIFT1);
    S4 = _mm256_srai_epi32(_mm256_sub_epi32(E3, O3), SHIFT1);

    /* packs_epi32: 32-bit -> 16-bit (带符号饱和, 等效于 clip 到 LIMIT_BIT=16) */
    C00 = _mm256_permute2x128_si256(S0, S4, 0x20);
    C01 = _mm256_permute2x128_si256(S0, S4, 0x31);
    C02 = _mm256_permute2x128_si256(S1, S5, 0x20);
    C03 = _mm256_permute2x128_si256(S1, S5, 0x31);
    C04 = _mm256_permute2x128_si256(S2, S6, 0x20);
    C05 = _mm256_permute2x128_si256(S2, S6, 0x31);
    C06 = _mm256_permute2x128_si256(S3, S7, 0x20);
    C07 = _mm256_permute2x128_si256(S3, S7, 0x31);

    S0 = _mm256_packs_epi32(C00, C01);
    S1 = _mm256_packs_epi32(C02, C03);
    S2 = _mm256_packs_epi32(C04, C05);
    S3 = _mm256_packs_epi32(C06, C07);

    /* ---- 转置 8x8 (int16) ---- */
    S4 = _mm256_unpacklo_epi16(S0, S1);
    S5 = _mm256_unpacklo_epi16(S2, S3);
    S6 = _mm256_unpackhi_epi16(S0, S1);
    S7 = _mm256_unpackhi_epi16(S2, S3);

    C00 = _mm256_unpacklo_epi32(S4, S5);
    C01 = _mm256_unpacklo_epi32(S6, S7);
    C02 = _mm256_unpackhi_epi32(S4, S5);
    C03 = _mm256_unpackhi_epi32(S6, S7);

    C04 = _mm256_permute2x128_si256(C00, C02, 0x20);
    C05 = _mm256_permute2x128_si256(C00, C02, 0x31);
    C06 = _mm256_permute2x128_si256(C01, C03, 0x20);
    C07 = _mm256_permute2x128_si256(C01, C03, 0x31);

    S0 = _mm256_unpacklo_epi64(C04, C05);
    S1 = _mm256_unpacklo_epi64(C06, C07);
    S2 = _mm256_unpackhi_epi64(C04, C05);
    S3 = _mm256_unpackhi_epi64(C06, C07);

    S4 = _mm256_permute2x128_si256(S2, S3, 0x20);
    S5 = _mm256_permute2x128_si256(S2, S3, 0x31);

    /* ===================================================================
     * 第二遍: 垂直反变换 (每列 8 点 -> 8 点)
     * =================================================================== */

    /* 奇数行 */
    T0 = _mm256_unpacklo_epi16(S4, S5);
    T1 = _mm256_unpackhi_epi16(S4, S5);
    T2 = _mm256_permute2x128_si256(T0, T1, 0x20);
    T3 = _mm256_permute2x128_si256(T0, T1, 0x31);

    O0 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[0]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[1]))));
    O1 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[2]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[3]))));
    O2 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[4]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[5]))));
    O3 = _mm256_add_epi32(_mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[6]))),
                          _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[7]))));

    /* 偶数行 */
    T0 = _mm256_unpacklo_epi16(S0, S1);
    T1 = _mm256_unpackhi_epi16(S0, S1);
    T2 = _mm256_permute2x128_si256(T0, T1, 0x20);
    T3 = _mm256_permute2x128_si256(T0, T1, 0x31);

    EE0 = _mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[8])));
    EE1 = _mm256_madd_epi16(T2, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[9])));
    EO0 = _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[10])));
    EO1 = _mm256_madd_epi16(T3, _mm256_load_si256((__m256i*)(tab_idct_8x8_256[11])));

    mAdd = _mm256_set1_epi32(SHIFT2 > 0 ? (1 << (SHIFT2 - 1)) : 0);

    E0 = _mm256_add_epi32(EE0, EO0);
    E1 = _mm256_add_epi32(EE1, EO1);
    E3 = _mm256_sub_epi32(EE0, EO0);
    E2 = _mm256_sub_epi32(EE1, EO1);
    E0 = _mm256_add_epi32(E0, mAdd);
    E1 = _mm256_add_epi32(E1, mAdd);
    E2 = _mm256_add_epi32(E2, mAdd);
    E3 = _mm256_add_epi32(E3, mAdd);

    S0 = _mm256_srai_epi32(_mm256_add_epi32(E0, O0), SHIFT2);
    S7 = _mm256_srai_epi32(_mm256_sub_epi32(E0, O0), SHIFT2);
    S1 = _mm256_srai_epi32(_mm256_add_epi32(E1, O1), SHIFT2);
    S6 = _mm256_srai_epi32(_mm256_sub_epi32(E1, O1), SHIFT2);
    S2 = _mm256_srai_epi32(_mm256_add_epi32(E2, O2), SHIFT2);
    S5 = _mm256_srai_epi32(_mm256_sub_epi32(E2, O2), SHIFT2);
    S3 = _mm256_srai_epi32(_mm256_add_epi32(E3, O3), SHIFT2);
    S4 = _mm256_srai_epi32(_mm256_sub_epi32(E3, O3), SHIFT2);

    /* packs_epi32: 32-bit -> 16-bit */
    C00 = _mm256_permute2x128_si256(S0, S4, 0x20);
    C01 = _mm256_permute2x128_si256(S0, S4, 0x31);
    C02 = _mm256_permute2x128_si256(S1, S5, 0x20);
    C03 = _mm256_permute2x128_si256(S1, S5, 0x31);
    C04 = _mm256_permute2x128_si256(S2, S6, 0x20);
    C05 = _mm256_permute2x128_si256(S2, S6, 0x31);
    C06 = _mm256_permute2x128_si256(S3, S7, 0x20);
    C07 = _mm256_permute2x128_si256(S3, S7, 0x31);

    S0 = _mm256_packs_epi32(C00, C01);
    S1 = _mm256_packs_epi32(C02, C03);
    S2 = _mm256_packs_epi32(C04, C05);
    S3 = _mm256_packs_epi32(C06, C07);

    /* ---- 转置 8x8 (int16) ---- */
    S4 = _mm256_unpacklo_epi16(S0, S1);
    S5 = _mm256_unpacklo_epi16(S2, S3);
    S6 = _mm256_unpackhi_epi16(S0, S1);
    S7 = _mm256_unpackhi_epi16(S2, S3);

    C00 = _mm256_unpacklo_epi32(S4, S5);
    C01 = _mm256_unpacklo_epi32(S6, S7);
    C02 = _mm256_unpackhi_epi32(S4, S5);
    C03 = _mm256_unpackhi_epi32(S6, S7);

    C04 = _mm256_permute2x128_si256(C00, C02, 0x20);
    C05 = _mm256_permute2x128_si256(C00, C02, 0x31);
    C06 = _mm256_permute2x128_si256(C01, C03, 0x20);
    C07 = _mm256_permute2x128_si256(C01, C03, 0x31);

    S0 = _mm256_unpacklo_epi64(C04, C05);
    S1 = _mm256_unpacklo_epi64(C06, C07);
    S2 = _mm256_unpackhi_epi64(C04, C05);
    S3 = _mm256_unpackhi_epi64(C06, C07);

    /* ---- 裁剪到 [-(1<<(CLIP2-1)), (1<<(CLIP2-1))-1] ---- */
    max_val = _mm256_set1_epi16((short)((1 << (CLIP2 - 1)) - 1));
    min_val = _mm256_set1_epi16((short)(-(1 << (CLIP2 - 1))));

    S0 = _mm256_max_epi16(_mm256_min_epi16(S0, max_val), min_val);
    S1 = _mm256_max_epi16(_mm256_min_epi16(S1, max_val), min_val);
    S2 = _mm256_max_epi16(_mm256_min_epi16(S2, max_val), min_val);
    S3 = _mm256_max_epi16(_mm256_min_epi16(S3, max_val), min_val);

    /* ---- 存储 (就地, 连续 8x8) — 对齐存储 ---- */
    _mm_store_si128((__m128i*)&coeff[ 0], _mm256_castsi256_si128(S0));
    _mm_store_si128((__m128i*)&coeff[16], _mm256_extracti128_si256(S0, 1));
    _mm_store_si128((__m128i*)&coeff[32], _mm256_castsi256_si128(S1));
    _mm_store_si128((__m128i*)&coeff[48], _mm256_extracti128_si256(S1, 1));
    _mm_store_si128((__m128i*)&coeff[ 8], _mm256_castsi256_si128(S2));
    _mm_store_si128((__m128i*)&coeff[24], _mm256_extracti128_si256(S2, 1));
    _mm_store_si128((__m128i*)&coeff[40], _mm256_castsi256_si128(S3));
    _mm_store_si128((__m128i*)&coeff[56], _mm256_extracti128_si256(S3, 1));
}

/* ===========================================================================
 * 16x16 反 DCT (AVX2)
 * 参考: davs2 source/common/vec/intrinsic_idct_avx2.cc idct_16x16_avx2
 *
 * 4 层蝶形分解: EEE/EEO/EO/O, A/B 双轨设计
 * for(pass) 循环复用两遍变换代码, 中间 TRANSPOSE_16x16_16BIT
 *
 * 对齐: coeff 基地址 32 字节对齐, 每行 16*2=32 字节,
 *       所有行起始地址 32 字节对齐, 用 _mm256_load/store_si256
 * ===========================================================================
 */
static void idct_16x16_avx2(int16_t *coeff, int w, int h, int bit_depth)
{
    const int shift = 20 - bit_depth;
    const int clip = bit_depth + 1;

    const __m256i c16_p43_p45 = _mm256_set1_epi32(0x002B002D);
    const __m256i c16_p35_p40 = _mm256_set1_epi32(0x00230028);
    const __m256i c16_p21_p29 = _mm256_set1_epi32(0x0015001D);
    const __m256i c16_p04_p13 = _mm256_set1_epi32(0x0004000D);
    const __m256i c16_p29_p43 = _mm256_set1_epi32(0x001D002B);
    const __m256i c16_n21_p04 = _mm256_set1_epi32(0xFFEB0004);
    const __m256i c16_n45_n40 = _mm256_set1_epi32(0xFFD3FFD8);
    const __m256i c16_n13_n35 = _mm256_set1_epi32(0xFFF3FFDD);
    const __m256i c16_p04_p40 = _mm256_set1_epi32(0x00040028);
    const __m256i c16_n43_n35 = _mm256_set1_epi32(0xFFD5FFDD);
    const __m256i c16_p29_n13 = _mm256_set1_epi32(0x001DFFF3);
    const __m256i c16_p21_p45 = _mm256_set1_epi32(0x0015002D);
    const __m256i c16_n21_p35 = _mm256_set1_epi32(0xFFEB0023);
    const __m256i c16_p04_n43 = _mm256_set1_epi32(0x0004FFD5);
    const __m256i c16_p13_p45 = _mm256_set1_epi32(0x000D002D);
    const __m256i c16_n29_n40 = _mm256_set1_epi32(0xFFE3FFD8);
    const __m256i c16_n40_p29 = _mm256_set1_epi32(0xFFD8001D);
    const __m256i c16_p45_n13 = _mm256_set1_epi32(0x002DFFF3);
    const __m256i c16_n43_n04 = _mm256_set1_epi32(0xFFD5FFFC);
    const __m256i c16_p35_p21 = _mm256_set1_epi32(0x00230015);
    const __m256i c16_n45_p21 = _mm256_set1_epi32(0xFFD30015);
    const __m256i c16_p13_p29 = _mm256_set1_epi32(0x000D001D);
    const __m256i c16_p35_n43 = _mm256_set1_epi32(0x0023FFD5);
    const __m256i c16_n40_p04 = _mm256_set1_epi32(0xFFD80004);
    const __m256i c16_n35_p13 = _mm256_set1_epi32(0xFFDD000D);
    const __m256i c16_n40_p45 = _mm256_set1_epi32(0xFFD8002D);
    const __m256i c16_p04_p21 = _mm256_set1_epi32(0x00040015);
    const __m256i c16_p43_n29 = _mm256_set1_epi32(0x002BFFE3);
    const __m256i c16_n13_p04 = _mm256_set1_epi32(0xFFF30004);
    const __m256i c16_n29_p21 = _mm256_set1_epi32(0xFFE30015);
    const __m256i c16_n40_p35 = _mm256_set1_epi32(0xFFD80023);
    const __m256i c16_n45_p43 = _mm256_set1_epi32(0xFFD3002B);

    const __m256i c16_p38_p44 = _mm256_set1_epi32(0x0026002C);
    const __m256i c16_p09_p25 = _mm256_set1_epi32(0x00090019);
    const __m256i c16_n09_p38 = _mm256_set1_epi32(0xFFF70026);
    const __m256i c16_n25_n44 = _mm256_set1_epi32(0xFFE7FFD4);
    const __m256i c16_n44_p25 = _mm256_set1_epi32(0xFFD40019);
    const __m256i c16_p38_p09 = _mm256_set1_epi32(0x00260009);
    const __m256i c16_n25_p09 = _mm256_set1_epi32(0xFFE70009);
    const __m256i c16_n44_p38 = _mm256_set1_epi32(0xFFD40026);

    const __m256i c16_p17_p42 = _mm256_set1_epi32(0x0011002A);
    const __m256i c16_n42_p17 = _mm256_set1_epi32(0xFFD60011);

    const __m256i c16_n32_p32 = _mm256_set1_epi32(0xFFE00020);
    const __m256i c16_p32_p32 = _mm256_set1_epi32(0x00200020);

    __m256i max_val, min_val;
    __m256i c32_rnd = _mm256_set1_epi32(16); /* 第一遍舍入值 1<<(5-1) */

    int nShift = 5;
    int pass;

    __m256i in00, in01, in02, in03, in04, in05, in06, in07;
    __m256i in08, in09, in10, in11, in12, in13, in14, in15;
    __m256i res00, res01, res02, res03, res04, res05, res06, res07;
    __m256i res08, res09, res10, res11, res12, res13, res14, res15;

    (void)w; (void)h;

    /* 加载 16 行 — 对齐加载 (每行 32 字节, 32 字节对齐) */
    in00 = _mm256_load_si256((const __m256i*)&coeff[0 * 16]);
    in01 = _mm256_load_si256((const __m256i*)&coeff[1 * 16]);
    in02 = _mm256_load_si256((const __m256i*)&coeff[2 * 16]);
    in03 = _mm256_load_si256((const __m256i*)&coeff[3 * 16]);
    in04 = _mm256_load_si256((const __m256i*)&coeff[4 * 16]);
    in05 = _mm256_load_si256((const __m256i*)&coeff[5 * 16]);
    in06 = _mm256_load_si256((const __m256i*)&coeff[6 * 16]);
    in07 = _mm256_load_si256((const __m256i*)&coeff[7 * 16]);
    in08 = _mm256_load_si256((const __m256i*)&coeff[8 * 16]);
    in09 = _mm256_load_si256((const __m256i*)&coeff[9 * 16]);
    in10 = _mm256_load_si256((const __m256i*)&coeff[10 * 16]);
    in11 = _mm256_load_si256((const __m256i*)&coeff[11 * 16]);
    in12 = _mm256_load_si256((const __m256i*)&coeff[12 * 16]);
    in13 = _mm256_load_si256((const __m256i*)&coeff[13 * 16]);
    in14 = _mm256_load_si256((const __m256i*)&coeff[14 * 16]);
    in15 = _mm256_load_si256((const __m256i*)&coeff[15 * 16]);

    for (pass = 0; pass < 2; pass++) {
        const __m256i T_00_00A = _mm256_unpacklo_epi16(in01, in03);
        const __m256i T_00_00B = _mm256_unpackhi_epi16(in01, in03);
        const __m256i T_00_01A = _mm256_unpacklo_epi16(in05, in07);
        const __m256i T_00_01B = _mm256_unpackhi_epi16(in05, in07);
        const __m256i T_00_02A = _mm256_unpacklo_epi16(in09, in11);
        const __m256i T_00_02B = _mm256_unpackhi_epi16(in09, in11);
        const __m256i T_00_03A = _mm256_unpacklo_epi16(in13, in15);
        const __m256i T_00_03B = _mm256_unpackhi_epi16(in13, in15);
        const __m256i T_00_04A = _mm256_unpacklo_epi16(in02, in06);
        const __m256i T_00_04B = _mm256_unpackhi_epi16(in02, in06);
        const __m256i T_00_05A = _mm256_unpacklo_epi16(in10, in14);
        const __m256i T_00_05B = _mm256_unpackhi_epi16(in10, in14);
        const __m256i T_00_06A = _mm256_unpacklo_epi16(in04, in12);
        const __m256i T_00_06B = _mm256_unpackhi_epi16(in04, in12);
        const __m256i T_00_07A = _mm256_unpacklo_epi16(in00, in08);
        const __m256i T_00_07B = _mm256_unpackhi_epi16(in00, in08);

        __m256i O0A, O1A, O2A, O3A, O4A, O5A, O6A, O7A;
        __m256i O0B, O1B, O2B, O3B, O4B, O5B, O6B, O7B;
        __m256i EO0A, EO1A, EO2A, EO3A;
        __m256i EO0B, EO1B, EO2B, EO3B;
        __m256i EEO0A, EEO1A;
        __m256i EEO0B, EEO1B;
        __m256i EEE0A, EEE1A;
        __m256i EEE0B, EEE1B;

    {
        __m256i T00, T01;
#define COMPUTE_ROW(row0103, row0507, row0911, row1315, c0103, c0507, c0911, c1315, row) \
    T00 = _mm256_add_epi32(_mm256_madd_epi16(row0103, c0103), _mm256_madd_epi16(row0507, c0507)); \
    T01 = _mm256_add_epi32(_mm256_madd_epi16(row0911, c0911), _mm256_madd_epi16(row1315, c1315)); \
    row = _mm256_add_epi32(T00, T01);

        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_p43_p45, c16_p35_p40, c16_p21_p29, c16_p04_p13, O0A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_p29_p43, c16_n21_p04, c16_n45_n40, c16_n13_n35, O1A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_p04_p40, c16_n43_n35, c16_p29_n13, c16_p21_p45, O2A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n21_p35, c16_p04_n43, c16_p13_p45, c16_n29_n40, O3A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n40_p29, c16_p45_n13, c16_n43_n04, c16_p35_p21, O4A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n45_p21, c16_p13_p29, c16_p35_n43, c16_n40_p04, O5A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n35_p13, c16_n40_p45, c16_p04_p21, c16_p43_n29, O6A)
        COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, c16_n13_p04, c16_n29_p21, c16_n40_p35, c16_n45_p43, O7A)

        COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_p43_p45, c16_p35_p40, c16_p21_p29, c16_p04_p13, O0B)
        COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_p29_p43, c16_n21_p04, c16_n45_n40, c16_n13_n35, O1B)
        COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_p04_p40, c16_n43_n35, c16_p29_n13, c16_p21_p45, O2B)
        COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_n21_p35, c16_p04_n43, c16_p13_p45, c16_n29_n40, O3B)
        COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_n40_p29, c16_p45_n13, c16_n43_n04, c16_p35_p21, O4B)
        COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_n45_p21, c16_p13_p29, c16_p35_n43, c16_n40_p04, O5B)
        COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_n35_p13, c16_n40_p45, c16_p04_p21, c16_p43_n29, O6B)
        COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, c16_n13_p04, c16_n29_p21, c16_n40_p35, c16_n45_p43, O7B)
#undef COMPUTE_ROW
    }

    EO0A = _mm256_add_epi32(_mm256_madd_epi16(T_00_04A, c16_p38_p44), _mm256_madd_epi16(T_00_05A, c16_p09_p25));
    EO0B = _mm256_add_epi32(_mm256_madd_epi16(T_00_04B, c16_p38_p44), _mm256_madd_epi16(T_00_05B, c16_p09_p25));
    EO1A = _mm256_add_epi32(_mm256_madd_epi16(T_00_04A, c16_n09_p38), _mm256_madd_epi16(T_00_05A, c16_n25_n44));
    EO1B = _mm256_add_epi32(_mm256_madd_epi16(T_00_04B, c16_n09_p38), _mm256_madd_epi16(T_00_05B, c16_n25_n44));
    EO2A = _mm256_add_epi32(_mm256_madd_epi16(T_00_04A, c16_n44_p25), _mm256_madd_epi16(T_00_05A, c16_p38_p09));
    EO2B = _mm256_add_epi32(_mm256_madd_epi16(T_00_04B, c16_n44_p25), _mm256_madd_epi16(T_00_05B, c16_p38_p09));
    EO3A = _mm256_add_epi32(_mm256_madd_epi16(T_00_04A, c16_n25_p09), _mm256_madd_epi16(T_00_05A, c16_n44_p38));
    EO3B = _mm256_add_epi32(_mm256_madd_epi16(T_00_04B, c16_n25_p09), _mm256_madd_epi16(T_00_05B, c16_n44_p38));

    EEO0A = _mm256_madd_epi16(T_00_06A, c16_p17_p42);
    EEO0B = _mm256_madd_epi16(T_00_06B, c16_p17_p42);
    EEO1A = _mm256_madd_epi16(T_00_06A, c16_n42_p17);
    EEO1B = _mm256_madd_epi16(T_00_06B, c16_n42_p17);

    EEE0A = _mm256_madd_epi16(T_00_07A, c16_p32_p32);
    EEE0B = _mm256_madd_epi16(T_00_07B, c16_p32_p32);
    EEE1A = _mm256_madd_epi16(T_00_07A, c16_n32_p32);
    EEE1B = _mm256_madd_epi16(T_00_07B, c16_n32_p32);
    {
        const __m256i EE0A = _mm256_add_epi32(EEE0A, EEO0A);
        const __m256i EE0B = _mm256_add_epi32(EEE0B, EEO0B);
        const __m256i EE1A = _mm256_add_epi32(EEE1A, EEO1A);
        const __m256i EE1B = _mm256_add_epi32(EEE1B, EEO1B);
        const __m256i EE3A = _mm256_sub_epi32(EEE0A, EEO0A);
        const __m256i EE3B = _mm256_sub_epi32(EEE0B, EEO0B);
        const __m256i EE2A = _mm256_sub_epi32(EEE1A, EEO1A);
        const __m256i EE2B = _mm256_sub_epi32(EEE1B, EEO1B);

        const __m256i E0A = _mm256_add_epi32(EE0A, EO0A);
        const __m256i E0B = _mm256_add_epi32(EE0B, EO0B);
        const __m256i E1A = _mm256_add_epi32(EE1A, EO1A);
        const __m256i E1B = _mm256_add_epi32(EE1B, EO1B);
        const __m256i E2A = _mm256_add_epi32(EE2A, EO2A);
        const __m256i E2B = _mm256_add_epi32(EE2B, EO2B);
        const __m256i E3A = _mm256_add_epi32(EE3A, EO3A);
        const __m256i E3B = _mm256_add_epi32(EE3B, EO3B);
        const __m256i E7A = _mm256_sub_epi32(EE0A, EO0A);
        const __m256i E7B = _mm256_sub_epi32(EE0B, EO0B);
        const __m256i E6A = _mm256_sub_epi32(EE1A, EO1A);
        const __m256i E6B = _mm256_sub_epi32(EE1B, EO1B);
        const __m256i E5A = _mm256_sub_epi32(EE2A, EO2A);
        const __m256i E5B = _mm256_sub_epi32(EE2B, EO2B);
        const __m256i E4A = _mm256_sub_epi32(EE3A, EO3A);
        const __m256i E4B = _mm256_sub_epi32(EE3B, EO3B);

        const __m256i T10A = _mm256_add_epi32(E0A, c32_rnd);
        const __m256i T10B = _mm256_add_epi32(E0B, c32_rnd);
        const __m256i T11A = _mm256_add_epi32(E1A, c32_rnd);
        const __m256i T11B = _mm256_add_epi32(E1B, c32_rnd);
        const __m256i T12A = _mm256_add_epi32(E2A, c32_rnd);
        const __m256i T12B = _mm256_add_epi32(E2B, c32_rnd);
        const __m256i T13A = _mm256_add_epi32(E3A, c32_rnd);
        const __m256i T13B = _mm256_add_epi32(E3B, c32_rnd);
        const __m256i T14A = _mm256_add_epi32(E4A, c32_rnd);
        const __m256i T14B = _mm256_add_epi32(E4B, c32_rnd);
        const __m256i T15A = _mm256_add_epi32(E5A, c32_rnd);
        const __m256i T15B = _mm256_add_epi32(E5B, c32_rnd);
        const __m256i T16A = _mm256_add_epi32(E6A, c32_rnd);
        const __m256i T16B = _mm256_add_epi32(E6B, c32_rnd);
        const __m256i T17A = _mm256_add_epi32(E7A, c32_rnd);
        const __m256i T17B = _mm256_add_epi32(E7B, c32_rnd);

        const __m256i T20A = _mm256_add_epi32(T10A, O0A);
        const __m256i T20B = _mm256_add_epi32(T10B, O0B);
        const __m256i T21A = _mm256_add_epi32(T11A, O1A);
        const __m256i T21B = _mm256_add_epi32(T11B, O1B);
        const __m256i T22A = _mm256_add_epi32(T12A, O2A);
        const __m256i T22B = _mm256_add_epi32(T12B, O2B);
        const __m256i T23A = _mm256_add_epi32(T13A, O3A);
        const __m256i T23B = _mm256_add_epi32(T13B, O3B);
        const __m256i T24A = _mm256_add_epi32(T14A, O4A);
        const __m256i T24B = _mm256_add_epi32(T14B, O4B);
        const __m256i T25A = _mm256_add_epi32(T15A, O5A);
        const __m256i T25B = _mm256_add_epi32(T15B, O5B);
        const __m256i T26A = _mm256_add_epi32(T16A, O6A);
        const __m256i T26B = _mm256_add_epi32(T16B, O6B);
        const __m256i T27A = _mm256_add_epi32(T17A, O7A);
        const __m256i T27B = _mm256_add_epi32(T17B, O7B);
        const __m256i T2FA = _mm256_sub_epi32(T10A, O0A);
        const __m256i T2FB = _mm256_sub_epi32(T10B, O0B);
        const __m256i T2EA = _mm256_sub_epi32(T11A, O1A);
        const __m256i T2EB = _mm256_sub_epi32(T11B, O1B);
        const __m256i T2DA = _mm256_sub_epi32(T12A, O2A);
        const __m256i T2DB = _mm256_sub_epi32(T12B, O2B);
        const __m256i T2CA = _mm256_sub_epi32(T13A, O3A);
        const __m256i T2CB = _mm256_sub_epi32(T13B, O3B);
        const __m256i T2BA = _mm256_sub_epi32(T14A, O4A);
        const __m256i T2BB = _mm256_sub_epi32(T14B, O4B);
        const __m256i T2AA = _mm256_sub_epi32(T15A, O5A);
        const __m256i T2AB = _mm256_sub_epi32(T15B, O5B);
        const __m256i T29A = _mm256_sub_epi32(T16A, O6A);
        const __m256i T29B = _mm256_sub_epi32(T16B, O6B);
        const __m256i T28A = _mm256_sub_epi32(T17A, O7A);
        const __m256i T28B = _mm256_sub_epi32(T17B, O7B);

        const __m256i T30A = _mm256_srai_epi32(T20A, nShift);
        const __m256i T30B = _mm256_srai_epi32(T20B, nShift);
        const __m256i T31A = _mm256_srai_epi32(T21A, nShift);
        const __m256i T31B = _mm256_srai_epi32(T21B, nShift);
        const __m256i T32A = _mm256_srai_epi32(T22A, nShift);
        const __m256i T32B = _mm256_srai_epi32(T22B, nShift);
        const __m256i T33A = _mm256_srai_epi32(T23A, nShift);
        const __m256i T33B = _mm256_srai_epi32(T23B, nShift);
        const __m256i T34A = _mm256_srai_epi32(T24A, nShift);
        const __m256i T34B = _mm256_srai_epi32(T24B, nShift);
        const __m256i T35A = _mm256_srai_epi32(T25A, nShift);
        const __m256i T35B = _mm256_srai_epi32(T25B, nShift);
        const __m256i T36A = _mm256_srai_epi32(T26A, nShift);
        const __m256i T36B = _mm256_srai_epi32(T26B, nShift);
        const __m256i T37A = _mm256_srai_epi32(T27A, nShift);
        const __m256i T37B = _mm256_srai_epi32(T27B, nShift);

        const __m256i T38A = _mm256_srai_epi32(T28A, nShift);
        const __m256i T38B = _mm256_srai_epi32(T28B, nShift);
        const __m256i T39A = _mm256_srai_epi32(T29A, nShift);
        const __m256i T39B = _mm256_srai_epi32(T29B, nShift);
        const __m256i T3AA = _mm256_srai_epi32(T2AA, nShift);
        const __m256i T3AB = _mm256_srai_epi32(T2AB, nShift);
        const __m256i T3BA = _mm256_srai_epi32(T2BA, nShift);
        const __m256i T3BB = _mm256_srai_epi32(T2BB, nShift);
        const __m256i T3CA = _mm256_srai_epi32(T2CA, nShift);
        const __m256i T3CB = _mm256_srai_epi32(T2CB, nShift);
        const __m256i T3DA = _mm256_srai_epi32(T2DA, nShift);
        const __m256i T3DB = _mm256_srai_epi32(T2DB, nShift);
        const __m256i T3EA = _mm256_srai_epi32(T2EA, nShift);
        const __m256i T3EB = _mm256_srai_epi32(T2EB, nShift);
        const __m256i T3FA = _mm256_srai_epi32(T2FA, nShift);
        const __m256i T3FB = _mm256_srai_epi32(T2FB, nShift);

        res00 = _mm256_packs_epi32(T30A, T30B);
        res01 = _mm256_packs_epi32(T31A, T31B);
        res02 = _mm256_packs_epi32(T32A, T32B);
        res03 = _mm256_packs_epi32(T33A, T33B);
        res04 = _mm256_packs_epi32(T34A, T34B);
        res05 = _mm256_packs_epi32(T35A, T35B);
        res06 = _mm256_packs_epi32(T36A, T36B);
        res07 = _mm256_packs_epi32(T37A, T37B);

        res08 = _mm256_packs_epi32(T38A, T38B);
        res09 = _mm256_packs_epi32(T39A, T39B);
        res10 = _mm256_packs_epi32(T3AA, T3AB);
        res11 = _mm256_packs_epi32(T3BA, T3BB);
        res12 = _mm256_packs_epi32(T3CA, T3CB);
        res13 = _mm256_packs_epi32(T3DA, T3DB);
        res14 = _mm256_packs_epi32(T3EA, T3EB);
        res15 = _mm256_packs_epi32(T3FA, T3FB);
    }

        /* ---- 转置 16x16 (int16) ---- */
        {
            __m256i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
            __m256i tr0_8, tr0_9, tr0_10, tr0_11, tr0_12, tr0_13, tr0_14, tr0_15;
#define TRANSPOSE_16x16_16BIT(I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15) \
        tr0_0 = _mm256_unpacklo_epi16(I0, I1); \
        tr0_1 = _mm256_unpacklo_epi16(I2, I3); \
        tr0_2 = _mm256_unpacklo_epi16(I4, I5); \
        tr0_3 = _mm256_unpacklo_epi16(I6, I7); \
        tr0_4 = _mm256_unpacklo_epi16(I8, I9); \
        tr0_5 = _mm256_unpacklo_epi16(I10, I11); \
        tr0_6 = _mm256_unpacklo_epi16(I12, I13); \
        tr0_7 = _mm256_unpacklo_epi16(I14, I15); \
        tr0_8 = _mm256_unpackhi_epi16(I0, I1); \
        tr0_9 = _mm256_unpackhi_epi16(I2, I3); \
        tr0_10 = _mm256_unpackhi_epi16(I4, I5); \
        tr0_11 = _mm256_unpackhi_epi16(I6, I7); \
        tr0_12 = _mm256_unpackhi_epi16(I8, I9); \
        tr0_13 = _mm256_unpackhi_epi16(I10, I11); \
        tr0_14 = _mm256_unpackhi_epi16(I12, I13); \
        tr0_15 = _mm256_unpackhi_epi16(I14, I15); \
        O0 = _mm256_unpacklo_epi32(tr0_0, tr0_1); \
        O1 = _mm256_unpacklo_epi32(tr0_2, tr0_3); \
        O2 = _mm256_unpacklo_epi32(tr0_4, tr0_5); \
        O3 = _mm256_unpacklo_epi32(tr0_6, tr0_7); \
        O4 = _mm256_unpackhi_epi32(tr0_0, tr0_1); \
        O5 = _mm256_unpackhi_epi32(tr0_2, tr0_3); \
        O6 = _mm256_unpackhi_epi32(tr0_4, tr0_5); \
        O7 = _mm256_unpackhi_epi32(tr0_6, tr0_7); \
        O8 = _mm256_unpacklo_epi32(tr0_8, tr0_9); \
        O9 = _mm256_unpacklo_epi32(tr0_10, tr0_11); \
        O10 = _mm256_unpacklo_epi32(tr0_12, tr0_13); \
        O11 = _mm256_unpacklo_epi32(tr0_14, tr0_15); \
        O12 = _mm256_unpackhi_epi32(tr0_8, tr0_9); \
        O13 = _mm256_unpackhi_epi32(tr0_10, tr0_11); \
        O14 = _mm256_unpackhi_epi32(tr0_12, tr0_13); \
        O15 = _mm256_unpackhi_epi32(tr0_14, tr0_15); \
        tr0_0 = _mm256_unpacklo_epi64(O0, O1); \
        tr0_1 = _mm256_unpacklo_epi64(O2, O3); \
        tr0_2 = _mm256_unpackhi_epi64(O0, O1); \
        tr0_3 = _mm256_unpackhi_epi64(O2, O3); \
        tr0_4 = _mm256_unpacklo_epi64(O4, O5); \
        tr0_5 = _mm256_unpacklo_epi64(O6, O7); \
        tr0_6 = _mm256_unpackhi_epi64(O4, O5); \
        tr0_7 = _mm256_unpackhi_epi64(O6, O7); \
        tr0_8 = _mm256_unpacklo_epi64(O8, O9); \
        tr0_9 = _mm256_unpacklo_epi64(O10, O11); \
        tr0_10 = _mm256_unpackhi_epi64(O8, O9); \
        tr0_11 = _mm256_unpackhi_epi64(O10, O11); \
        tr0_12 = _mm256_unpacklo_epi64(O12, O13); \
        tr0_13 = _mm256_unpacklo_epi64(O14, O15); \
        tr0_14 = _mm256_unpackhi_epi64(O12, O13); \
        tr0_15 = _mm256_unpackhi_epi64(O14, O15); \
        O0 = _mm256_permute2x128_si256(tr0_0, tr0_1, 0x20); \
        O1 = _mm256_permute2x128_si256(tr0_2, tr0_3, 0x20); \
        O2 = _mm256_permute2x128_si256(tr0_4, tr0_5, 0x20); \
        O3 = _mm256_permute2x128_si256(tr0_6, tr0_7, 0x20); \
        O4 = _mm256_permute2x128_si256(tr0_8, tr0_9, 0x20); \
        O5 = _mm256_permute2x128_si256(tr0_10, tr0_11, 0x20); \
        O6 = _mm256_permute2x128_si256(tr0_12, tr0_13, 0x20); \
        O7 = _mm256_permute2x128_si256(tr0_14, tr0_15, 0x20); \
        O8 = _mm256_permute2x128_si256(tr0_0, tr0_1, 0x31); \
        O9 = _mm256_permute2x128_si256(tr0_2, tr0_3, 0x31); \
        O10 = _mm256_permute2x128_si256(tr0_4, tr0_5, 0x31); \
        O11 = _mm256_permute2x128_si256(tr0_6, tr0_7, 0x31); \
        O12 = _mm256_permute2x128_si256(tr0_8, tr0_9, 0x31); \
        O13 = _mm256_permute2x128_si256(tr0_10, tr0_11, 0x31); \
        O14 = _mm256_permute2x128_si256(tr0_12, tr0_13, 0x31); \
        O15 = _mm256_permute2x128_si256(tr0_14, tr0_15, 0x31);

            TRANSPOSE_16x16_16BIT(res00, res01, res02, res03, res04, res05, res06, res07, res08, res09, res10, res11, res12, res13, res14, res15, in00, in01, in02, in03, in04, in05, in06, in07, in08, in09, in10, in11, in12, in13, in14, in15)
#undef TRANSPOSE_16x16_16BIT
        }

        nShift = shift;
        c32_rnd = _mm256_set1_epi32(shift ? (1 << (shift - 1)) : 0);
    }

    /* ---- 裁剪到 [-(1<<(clip-1)), (1<<(clip-1))-1] ---- */
    max_val = _mm256_set1_epi16((1 << (clip - 1)) - 1);
    min_val = _mm256_set1_epi16(-(1 << (clip - 1)));

    in00 = _mm256_max_epi16(_mm256_min_epi16(in00, max_val), min_val);
    in01 = _mm256_max_epi16(_mm256_min_epi16(in01, max_val), min_val);
    in02 = _mm256_max_epi16(_mm256_min_epi16(in02, max_val), min_val);
    in03 = _mm256_max_epi16(_mm256_min_epi16(in03, max_val), min_val);
    in04 = _mm256_max_epi16(_mm256_min_epi16(in04, max_val), min_val);
    in05 = _mm256_max_epi16(_mm256_min_epi16(in05, max_val), min_val);
    in06 = _mm256_max_epi16(_mm256_min_epi16(in06, max_val), min_val);
    in07 = _mm256_max_epi16(_mm256_min_epi16(in07, max_val), min_val);
    in08 = _mm256_max_epi16(_mm256_min_epi16(in08, max_val), min_val);
    in09 = _mm256_max_epi16(_mm256_min_epi16(in09, max_val), min_val);
    in10 = _mm256_max_epi16(_mm256_min_epi16(in10, max_val), min_val);
    in11 = _mm256_max_epi16(_mm256_min_epi16(in11, max_val), min_val);
    in12 = _mm256_max_epi16(_mm256_min_epi16(in12, max_val), min_val);
    in13 = _mm256_max_epi16(_mm256_min_epi16(in13, max_val), min_val);
    in14 = _mm256_max_epi16(_mm256_min_epi16(in14, max_val), min_val);
    in15 = _mm256_max_epi16(_mm256_min_epi16(in15, max_val), min_val);

    /* ---- 存储 (就地, 连续 16x16) — 对齐存储 ---- */
    _mm256_store_si256((__m256i*)&coeff[0 * 16], in00);
    _mm256_store_si256((__m256i*)&coeff[1 * 16], in01);
    _mm256_store_si256((__m256i*)&coeff[2 * 16], in02);
    _mm256_store_si256((__m256i*)&coeff[3 * 16], in03);
    _mm256_store_si256((__m256i*)&coeff[4 * 16], in04);
    _mm256_store_si256((__m256i*)&coeff[5 * 16], in05);
    _mm256_store_si256((__m256i*)&coeff[6 * 16], in06);
    _mm256_store_si256((__m256i*)&coeff[7 * 16], in07);
    _mm256_store_si256((__m256i*)&coeff[8 * 16], in08);
    _mm256_store_si256((__m256i*)&coeff[9 * 16], in09);
    _mm256_store_si256((__m256i*)&coeff[10 * 16], in10);
    _mm256_store_si256((__m256i*)&coeff[11 * 16], in11);
    _mm256_store_si256((__m256i*)&coeff[12 * 16], in12);
    _mm256_store_si256((__m256i*)&coeff[13 * 16], in13);
    _mm256_store_si256((__m256i*)&coeff[14 * 16], in14);
    _mm256_store_si256((__m256i*)&coeff[15 * 16], in15);
}

/* ===========================================================================
 * 32x32 反 DCT (AVX2)
 * 参考: davs2 source/common/vec/intrinsic_idct_avx2.cc idct_32x32_avx2
 *
 * 蝶形分解层次:
 *   EEEE0/1 (in00,in16) -> EEE0..3 -> EE0..7 -> E0..7,EF..E8
 *   EEEO0/1 (in08,in24)
 *   EEO0..3 (in04,in12,in20,in28)
 *   EO0..7  (in02,in06,in10,in14,in18,in22,in26,in30)
 *   O0..15  (in01,in03,in05,...,in31)
 *
 * 两遍处理:
 *   pass 0: 水平反变换, nShift=5, c32_rnd=16
 *   pass 1: 垂直反变换, nShift=20-bit_depth, c32_rnd=1<<(shift-1)
 * 之后裁剪到 [-2^(clip-1), 2^(clip-1)-1] 并写回.
 * =========================================================================== */
static void idct_32x32_avx2(int16_t *coeff, int w, int h, int bit_depth)
{
    int shift = 20 - bit_depth;            /* 不加 i_dst&1 (始终为 0) */
    int clip = bit_depth + 1;              /* 不加 i_dst&1 (始终为 0) */
    int k, i;
    __m256i max_val, min_val;
    __m256i EEO0A, EEO1A, EEO2A, EEO3A, EEO0B, EEO1B, EEO2B, EEO3B;
    __m256i EEEO0A, EEEO0B, EEEO1A, EEEO1B;
    __m256i EEEE0A, EEEE0B, EEEE1A, EEEE1B;
    __m256i EEE0A, EEE0B, EEE1A, EEE1B, EEE3A, EEE3B, EEE2A, EEE2B;
    __m256i EE0A, EE0B, EE1A, EE1B, EE2A, EE2B, EE3A, EE3B, EE7A, EE7B, EE6A, EE6B, EE5A, EE5B, EE4A, EE4B;
    __m256i E0A, E0B, E1A, E1B, E2A, E2B, E3A, E3B, E4A, E4B, E5A, E5B, E6A, E6B, E7A, E7B, EFA, EFB, EEA, EEB, EDA, EDB, ECA, ECB, EBA, EBB, EAA, EAB, E9A, E9B, E8A, E8B;
    __m256i T10A, T10B, T11A, T11B, T12A, T12B, T13A, T13B, T14A, T14B, T15A, T15B, T16A, T16B, T17A, T17B, T18A, T18B, T19A, T19B, T1AA, T1AB, T1BA, T1BB, T1CA, T1CB, T1DA, T1DB, T1EA, T1EB, T1FA, T1FB;
    __m256i T2_00A, T2_00B, T2_01A, T2_01B, T2_02A, T2_02B, T2_03A, T2_03B, T2_04A, T2_04B, T2_05A, T2_05B, T2_06A, T2_06B, T2_07A, T2_07B, T2_08A, T2_08B, T2_09A, T2_09B, T2_10A, T2_10B, T2_11A, T2_11B, T2_12A, T2_12B, T2_13A, T2_13B, T2_14A, T2_14B, T2_15A, T2_15B, T2_31A, T2_31B, T2_30A, T2_30B, T2_29A, T2_29B, T2_28A, T2_28B, T2_27A, T2_27B, T2_26A, T2_26B, T2_25A, T2_25B, T2_24A, T2_24B, T2_23A, T2_23B, T2_22A, T2_22B, T2_21A, T2_21B, T2_20A, T2_20B, T2_19A, T2_19B, T2_18A, T2_18B, T2_17A, T2_17B, T2_16A, T2_16B;
    __m256i T3_00A, T3_00B, T3_01A, T3_01B, T3_02A, T3_02B, T3_03A, T3_03B, T3_04A, T3_04B, T3_05A, T3_05B, T3_06A, T3_06B, T3_07A, T3_07B, T3_08A, T3_08B, T3_09A, T3_09B, T3_10A, T3_10B, T3_11A, T3_11B, T3_12A, T3_12B, T3_13A, T3_13B, T3_14A, T3_14B, T3_15A, T3_15B;
    __m256i T3_16A, T3_16B, T3_17A, T3_17B, T3_18A, T3_18B, T3_19A, T3_19B, T3_20A, T3_20B, T3_21A, T3_21B, T3_22A, T3_22B, T3_23A, T3_23B, T3_24A, T3_24B, T3_25A, T3_25B, T3_26A, T3_26B, T3_27A, T3_27B, T3_28A, T3_28B, T3_29A, T3_29B, T3_30A, T3_30B, T3_31A, T3_31B;

    /* ---- O 行系数 (8 对 madd) ---- */
    const __m256i c16_p45_p45 = _mm256_set1_epi32(0x002D002D);
    const __m256i c16_p43_p44 = _mm256_set1_epi32(0x002B002C);
    const __m256i c16_p39_p41 = _mm256_set1_epi32(0x00270029);
    const __m256i c16_p34_p36 = _mm256_set1_epi32(0x00220024);
    const __m256i c16_p27_p30 = _mm256_set1_epi32(0x001B001E);
    const __m256i c16_p19_p23 = _mm256_set1_epi32(0x00130017);
    const __m256i c16_p11_p15 = _mm256_set1_epi32(0x000B000F);
    const __m256i c16_p02_p07 = _mm256_set1_epi32(0x00020007);
    const __m256i c16_p41_p45 = _mm256_set1_epi32(0x0029002D);
    const __m256i c16_p23_p34 = _mm256_set1_epi32(0x00170022);
    const __m256i c16_n02_p11 = _mm256_set1_epi32(0xFFFE000B);
    const __m256i c16_n27_n15 = _mm256_set1_epi32(0xFFE5FFF1);
    const __m256i c16_n43_n36 = _mm256_set1_epi32(0xFFD5FFDC);
    const __m256i c16_n44_n45 = _mm256_set1_epi32(0xFFD4FFD3);
    const __m256i c16_n30_n39 = _mm256_set1_epi32(0xFFE2FFD9);
    const __m256i c16_n07_n19 = _mm256_set1_epi32(0xFFF9FFED);
    const __m256i c16_p34_p44 = _mm256_set1_epi32(0x0022002C);
    const __m256i c16_n07_p15 = _mm256_set1_epi32(0xFFF9000F);
    const __m256i c16_n41_n27 = _mm256_set1_epi32(0xFFD7FFE5);
    const __m256i c16_n39_n45 = _mm256_set1_epi32(0xFFD9FFD3);
    const __m256i c16_n02_n23 = _mm256_set1_epi32(0xFFFEFFE9);
    const __m256i c16_p36_p19 = _mm256_set1_epi32(0x00240013);
    const __m256i c16_p43_p45 = _mm256_set1_epi32(0x002B002D);
    const __m256i c16_p11_p30 = _mm256_set1_epi32(0x000B001E);
    const __m256i c16_p23_p43 = _mm256_set1_epi32(0x0017002B);
    const __m256i c16_n34_n07 = _mm256_set1_epi32(0xFFDEFFF9);
    const __m256i c16_n36_n45 = _mm256_set1_epi32(0xFFDCFFD3);
    const __m256i c16_p19_n11 = _mm256_set1_epi32(0x0013FFF5);
    const __m256i c16_p44_p41 = _mm256_set1_epi32(0x002C0029);
    const __m256i c16_n02_p27 = _mm256_set1_epi32(0xFFFE001B);
    const __m256i c16_n45_n30 = _mm256_set1_epi32(0xFFD3FFE2);
    const __m256i c16_n15_n39 = _mm256_set1_epi32(0xFFF1FFD9);
    const __m256i c16_p11_p41 = _mm256_set1_epi32(0x000B0029);
    const __m256i c16_n45_n27 = _mm256_set1_epi32(0xFFD3FFE5);
    const __m256i c16_p07_n30 = _mm256_set1_epi32(0x0007FFE2);
    const __m256i c16_p43_p39 = _mm256_set1_epi32(0x002B0027);
    const __m256i c16_n23_p15 = _mm256_set1_epi32(0xFFE9000F);
    const __m256i c16_n34_n45 = _mm256_set1_epi32(0xFFDEFFD3);
    const __m256i c16_p36_p02 = _mm256_set1_epi32(0x00240002);
    const __m256i c16_p19_p44 = _mm256_set1_epi32(0x0013002C);
    const __m256i c16_n02_p39 = _mm256_set1_epi32(0xFFFE0027);
    const __m256i c16_n36_n41 = _mm256_set1_epi32(0xFFDCFFD7);
    const __m256i c16_p43_p07 = _mm256_set1_epi32(0x002B0007);
    const __m256i c16_n11_p34 = _mm256_set1_epi32(0xFFF50022);
    const __m256i c16_n30_n44 = _mm256_set1_epi32(0xFFE2FFD4);
    const __m256i c16_p45_p15 = _mm256_set1_epi32(0x002D000F);
    const __m256i c16_n19_p27 = _mm256_set1_epi32(0xFFED001B);
    const __m256i c16_n23_n45 = _mm256_set1_epi32(0xFFE9FFD3);
    const __m256i c16_n15_p36 = _mm256_set1_epi32(0xFFF10024);
    const __m256i c16_n11_n45 = _mm256_set1_epi32(0xFFF5FFD3);
    const __m256i c16_p34_p39 = _mm256_set1_epi32(0x00220027);
    const __m256i c16_n45_n19 = _mm256_set1_epi32(0xFFD3FFED);
    const __m256i c16_p41_n07 = _mm256_set1_epi32(0x0029FFF9);
    const __m256i c16_n23_p30 = _mm256_set1_epi32(0xFFE9001E);
    const __m256i c16_n02_n44 = _mm256_set1_epi32(0xFFFEFFD4);
    const __m256i c16_p27_p43 = _mm256_set1_epi32(0x001B002B);
    const __m256i c16_n27_p34 = _mm256_set1_epi32(0xFFE50022);
    const __m256i c16_p19_n39 = _mm256_set1_epi32(0x0013FFD9);
    const __m256i c16_n11_p43 = _mm256_set1_epi32(0xFFF5002B);
    const __m256i c16_p02_n45 = _mm256_set1_epi32(0x0002FFD3);
    const __m256i c16_p07_p45 = _mm256_set1_epi32(0x0007002D);
    const __m256i c16_n15_n44 = _mm256_set1_epi32(0xFFF1FFD4);
    const __m256i c16_p23_p41 = _mm256_set1_epi32(0x00170029);
    const __m256i c16_n30_n36 = _mm256_set1_epi32(0xFFE2FFDC);
    const __m256i c16_n36_p30 = _mm256_set1_epi32(0xFFDC001E);
    const __m256i c16_p41_n23 = _mm256_set1_epi32(0x0029FFE9);
    const __m256i c16_n44_p15 = _mm256_set1_epi32(0xFFD4000F);
    const __m256i c16_p45_n07 = _mm256_set1_epi32(0x002DFFF9);
    const __m256i c16_n45_n02 = _mm256_set1_epi32(0xFFD3FFFE);
    const __m256i c16_p43_p11 = _mm256_set1_epi32(0x002B000B);
    const __m256i c16_n39_n19 = _mm256_set1_epi32(0xFFD9FFED);
    const __m256i c16_p34_p27 = _mm256_set1_epi32(0x0022001B);
    const __m256i c16_n43_p27 = _mm256_set1_epi32(0xFFD5001B);
    const __m256i c16_p44_n02 = _mm256_set1_epi32(0x002CFFFE);
    const __m256i c16_n30_n23 = _mm256_set1_epi32(0xFFE2FFE9);
    const __m256i c16_p07_p41 = _mm256_set1_epi32(0x00070029);
    const __m256i c16_p19_n45 = _mm256_set1_epi32(0x0013FFD3);
    const __m256i c16_n39_p34 = _mm256_set1_epi32(0xFFD90022);
    const __m256i c16_p45_n11 = _mm256_set1_epi32(0x002DFFF5);
    const __m256i c16_n36_n15 = _mm256_set1_epi32(0xFFDCFFF1);
    const __m256i c16_n45_p23 = _mm256_set1_epi32(0xFFD30017);
    const __m256i c16_p27_p19 = _mm256_set1_epi32(0x001B0013);
    const __m256i c16_p15_n45 = _mm256_set1_epi32(0x000FFFD3);
    const __m256i c16_n44_p30 = _mm256_set1_epi32(0xFFD4001E);
    const __m256i c16_p34_p11 = _mm256_set1_epi32(0x0022000B);
    const __m256i c16_p07_n43 = _mm256_set1_epi32(0x0007FFD5);
    const __m256i c16_n41_p36 = _mm256_set1_epi32(0xFFD70024);
    const __m256i c16_p39_p02 = _mm256_set1_epi32(0x00270002);
    const __m256i c16_n44_p19 = _mm256_set1_epi32(0xFFD40013);
    const __m256i c16_n02_p36 = _mm256_set1_epi32(0xFFFE0024);
    const __m256i c16_p45_n34 = _mm256_set1_epi32(0x002DFFDE);
    const __m256i c16_n15_n23 = _mm256_set1_epi32(0xFFF1FFE9);
    const __m256i c16_n39_p43 = _mm256_set1_epi32(0xFFD9002B);
    const __m256i c16_p30_p07 = _mm256_set1_epi32(0x001E0007);
    const __m256i c16_p27_n45 = _mm256_set1_epi32(0x001BFFD3);
    const __m256i c16_n41_p11 = _mm256_set1_epi32(0xFFD7000B);
    const __m256i c16_n39_p15 = _mm256_set1_epi32(0xFFD9000F);
    const __m256i c16_n30_p45 = _mm256_set1_epi32(0xFFE2002D);
    const __m256i c16_p27_p02 = _mm256_set1_epi32(0x001B0002);
    const __m256i c16_p41_n44 = _mm256_set1_epi32(0x0029FFD4);
    const __m256i c16_n11_n19 = _mm256_set1_epi32(0xFFF5FFED);
    const __m256i c16_n45_p36 = _mm256_set1_epi32(0xFFD30024);
    const __m256i c16_n07_p34 = _mm256_set1_epi32(0xFFF90022);
    const __m256i c16_p43_n23 = _mm256_set1_epi32(0x002BFFE9);
    const __m256i c16_n30_p11 = _mm256_set1_epi32(0xFFE2000B);
    const __m256i c16_n45_p43 = _mm256_set1_epi32(0xFFD3002B);
    const __m256i c16_n19_p36 = _mm256_set1_epi32(0xFFED0024);
    const __m256i c16_p23_n02 = _mm256_set1_epi32(0x0017FFFE);
    const __m256i c16_p45_n39 = _mm256_set1_epi32(0x002DFFD9);
    const __m256i c16_p27_n41 = _mm256_set1_epi32(0x001BFFD7);
    const __m256i c16_n15_n07 = _mm256_set1_epi32(0xFFF1FFF9);
    const __m256i c16_n44_p34 = _mm256_set1_epi32(0xFFD40022);
    const __m256i c16_n19_p07 = _mm256_set1_epi32(0xFFED0007);
    const __m256i c16_n39_p30 = _mm256_set1_epi32(0xFFD9001E);
    const __m256i c16_n45_p44 = _mm256_set1_epi32(0xFFD3002C);
    const __m256i c16_n36_p43 = _mm256_set1_epi32(0xFFDC002B);
    const __m256i c16_n15_p27 = _mm256_set1_epi32(0xFFF1001B);
    const __m256i c16_p11_p02 = _mm256_set1_epi32(0x000B0002);
    const __m256i c16_p34_n23 = _mm256_set1_epi32(0x0022FFE9);
    const __m256i c16_p45_n41 = _mm256_set1_epi32(0x002DFFD7);
    const __m256i c16_n07_p02 = _mm256_set1_epi32(0xFFF90002);
    const __m256i c16_n15_p11 = _mm256_set1_epi32(0xFFF1000B);
    const __m256i c16_n23_p19 = _mm256_set1_epi32(0xFFE90013);
    const __m256i c16_n30_p27 = _mm256_set1_epi32(0xFFE2001B);
    const __m256i c16_n36_p34 = _mm256_set1_epi32(0xFFDC0022);
    const __m256i c16_n41_p39 = _mm256_set1_epi32(0xFFD70027);
    const __m256i c16_n44_p43 = _mm256_set1_epi32(0xFFD4002B);
    const __m256i c16_n45_p45 = _mm256_set1_epi32(0xFFD3002D);

    /* ---- EO 行系数 (4 对 madd) ---- */
    /* const __m256i c16_p43_p45 = _mm256_set1_epi32(0x002B002D);  // 与上面重复, 已定义 */
    const __m256i c16_p35_p40 = _mm256_set1_epi32(0x00230028);
    const __m256i c16_p21_p29 = _mm256_set1_epi32(0x0015001D);
    const __m256i c16_p04_p13 = _mm256_set1_epi32(0x0004000D);
    const __m256i c16_p29_p43 = _mm256_set1_epi32(0x001D002B);
    const __m256i c16_n21_p04 = _mm256_set1_epi32(0xFFEB0004);
    const __m256i c16_n45_n40 = _mm256_set1_epi32(0xFFD3FFD8);
    const __m256i c16_n13_n35 = _mm256_set1_epi32(0xFFF3FFDD);
    const __m256i c16_p04_p40 = _mm256_set1_epi32(0x00040028);
    const __m256i c16_n43_n35 = _mm256_set1_epi32(0xFFD5FFDD);
    const __m256i c16_p29_n13 = _mm256_set1_epi32(0x001DFFF3);
    const __m256i c16_p21_p45 = _mm256_set1_epi32(0x0015002D);
    const __m256i c16_n21_p35 = _mm256_set1_epi32(0xFFEB0023);
    const __m256i c16_p04_n43 = _mm256_set1_epi32(0x0004FFD5);
    const __m256i c16_p13_p45 = _mm256_set1_epi32(0x000D002D);
    const __m256i c16_n29_n40 = _mm256_set1_epi32(0xFFE3FFD8);
    const __m256i c16_n40_p29 = _mm256_set1_epi32(0xFFD8001D);
    const __m256i c16_p45_n13 = _mm256_set1_epi32(0x002DFFF3);
    const __m256i c16_n43_n04 = _mm256_set1_epi32(0xFFD5FFFC);
    const __m256i c16_p35_p21 = _mm256_set1_epi32(0x00230015);
    const __m256i c16_n45_p21 = _mm256_set1_epi32(0xFFD30015);
    const __m256i c16_p13_p29 = _mm256_set1_epi32(0x000D001D);
    const __m256i c16_p35_n43 = _mm256_set1_epi32(0x0023FFD5);
    const __m256i c16_n40_p04 = _mm256_set1_epi32(0xFFD80004);
    const __m256i c16_n35_p13 = _mm256_set1_epi32(0xFFDD000D);
    const __m256i c16_n40_p45 = _mm256_set1_epi32(0xFFD8002D);
    const __m256i c16_p04_p21 = _mm256_set1_epi32(0x00040015);
    const __m256i c16_p43_n29 = _mm256_set1_epi32(0x002BFFE3);
    const __m256i c16_n13_p04 = _mm256_set1_epi32(0xFFF30004);
    const __m256i c16_n29_p21 = _mm256_set1_epi32(0xFFE30015);
    const __m256i c16_n40_p35 = _mm256_set1_epi32(0xFFD80023);
    /* const __m256i c16_n45_p43 = _mm256_set1_epi32(0xFFD3002B);  // 与上面重复, 已定义 */

    /* ---- EEO 行系数 (2 对 madd) ---- */
    const __m256i c16_p38_p44 = _mm256_set1_epi32(0x0026002C);
    const __m256i c16_p09_p25 = _mm256_set1_epi32(0x00090019);
    const __m256i c16_n09_p38 = _mm256_set1_epi32(0xFFF70026);
    const __m256i c16_n25_n44 = _mm256_set1_epi32(0xFFE7FFD4);

    const __m256i c16_n44_p25 = _mm256_set1_epi32(0xFFD40019);
    const __m256i c16_p38_p09 = _mm256_set1_epi32(0x00260009);
    const __m256i c16_n25_p09 = _mm256_set1_epi32(0xFFE70009);
    const __m256i c16_n44_p38 = _mm256_set1_epi32(0xFFD40026);

    /* ---- EEEO 行系数 (1 对 madd) ---- */
    const __m256i c16_p17_p42 = _mm256_set1_epi32(0x0011002A);
    const __m256i c16_n42_p17 = _mm256_set1_epi32(0xFFD60011);

    /* ---- EEEE 行系数 (1 对 madd) ---- */
    const __m256i c16_p32_p32 = _mm256_set1_epi32(0x00200020);
    const __m256i c16_n32_p32 = _mm256_set1_epi32(0xFFE00020);

    __m256i c32_rnd = _mm256_set1_epi32(16);   /* pass 0 的舍入值: 1<<(5-1) */
    int nShift = 5;                             /* pass 0 的移位值 */

    /* 就地变换的输入/输出缓冲 (低/高半部各一个 __m256i) */
    __m256i in00[2], in01[2], in02[2], in03[2], in04[2], in05[2], in06[2], in07[2], in08[2], in09[2], in10[2], in11[2], in12[2], in13[2], in14[2], in15[2];
    __m256i in16[2], in17[2], in18[2], in19[2], in20[2], in21[2], in22[2], in23[2], in24[2], in25[2], in26[2], in27[2], in28[2], in29[2], in30[2], in31[2];
    __m256i res00[2], res01[2], res02[2], res03[2], res04[2], res05[2], res06[2], res07[2], res08[2], res09[2], res10[2], res11[2], res12[2], res13[2], res14[2], res15[2];
    __m256i res16[2], res17[2], res18[2], res19[2], res20[2], res21[2], res22[2], res23[2], res24[2], res25[2], res26[2], res27[2], res28[2], res29[2], res30[2], res31[2];

    int pass, part;

    (void)w; (void)h;

    /* ---- 加载 32x32 块 (对齐加载: coeff 基地址 + 每半行起始均 32 字节对齐) ---- */
    for (i = 0; i < 2; i++) {
        const int offset = (i << 4);
        in00[i] = _mm256_load_si256((const __m256i*)&coeff[0 * 32 + offset]);
        in01[i] = _mm256_load_si256((const __m256i*)&coeff[1 * 32 + offset]);
        in02[i] = _mm256_load_si256((const __m256i*)&coeff[2 * 32 + offset]);
        in03[i] = _mm256_load_si256((const __m256i*)&coeff[3 * 32 + offset]);
        in04[i] = _mm256_load_si256((const __m256i*)&coeff[4 * 32 + offset]);
        in05[i] = _mm256_load_si256((const __m256i*)&coeff[5 * 32 + offset]);
        in06[i] = _mm256_load_si256((const __m256i*)&coeff[6 * 32 + offset]);
        in07[i] = _mm256_load_si256((const __m256i*)&coeff[7 * 32 + offset]);
        in08[i] = _mm256_load_si256((const __m256i*)&coeff[8 * 32 + offset]);
        in09[i] = _mm256_load_si256((const __m256i*)&coeff[9 * 32 + offset]);
        in10[i] = _mm256_load_si256((const __m256i*)&coeff[10 * 32 + offset]);
        in11[i] = _mm256_load_si256((const __m256i*)&coeff[11 * 32 + offset]);
        in12[i] = _mm256_load_si256((const __m256i*)&coeff[12 * 32 + offset]);
        in13[i] = _mm256_load_si256((const __m256i*)&coeff[13 * 32 + offset]);
        in14[i] = _mm256_load_si256((const __m256i*)&coeff[14 * 32 + offset]);
        in15[i] = _mm256_load_si256((const __m256i*)&coeff[15 * 32 + offset]);
        in16[i] = _mm256_load_si256((const __m256i*)&coeff[16 * 32 + offset]);
        in17[i] = _mm256_load_si256((const __m256i*)&coeff[17 * 32 + offset]);
        in18[i] = _mm256_load_si256((const __m256i*)&coeff[18 * 32 + offset]);
        in19[i] = _mm256_load_si256((const __m256i*)&coeff[19 * 32 + offset]);
        in20[i] = _mm256_load_si256((const __m256i*)&coeff[20 * 32 + offset]);
        in21[i] = _mm256_load_si256((const __m256i*)&coeff[21 * 32 + offset]);
        in22[i] = _mm256_load_si256((const __m256i*)&coeff[22 * 32 + offset]);
        in23[i] = _mm256_load_si256((const __m256i*)&coeff[23 * 32 + offset]);
        in24[i] = _mm256_load_si256((const __m256i*)&coeff[24 * 32 + offset]);
        in25[i] = _mm256_load_si256((const __m256i*)&coeff[25 * 32 + offset]);
        in26[i] = _mm256_load_si256((const __m256i*)&coeff[26 * 32 + offset]);
        in27[i] = _mm256_load_si256((const __m256i*)&coeff[27 * 32 + offset]);
        in28[i] = _mm256_load_si256((const __m256i*)&coeff[28 * 32 + offset]);
        in29[i] = _mm256_load_si256((const __m256i*)&coeff[29 * 32 + offset]);
        in30[i] = _mm256_load_si256((const __m256i*)&coeff[30 * 32 + offset]);
        in31[i] = _mm256_load_si256((const __m256i*)&coeff[31 * 32 + offset]);
    }

    /* ---- 两遍反变换: pass 0 水平, pass 1 垂直 ---- */
    for (pass = 0; pass < 2; pass++) {
        for (part = 0; part < 2; part++) {
            /* 奇数行配对 unpack (in01/03, in05/07, ..., in29/31) -> T_00_00A/B .. T_00_07A/B */
            const __m256i T_00_00A = _mm256_unpacklo_epi16(in01[part], in03[part]);       /* [33 13 32 12 31 11 30 10] */
            const __m256i T_00_00B = _mm256_unpackhi_epi16(in01[part], in03[part]);       /* [37 17 36 16 35 15 34 14] */
            const __m256i T_00_01A = _mm256_unpacklo_epi16(in05[part], in07[part]);
            const __m256i T_00_01B = _mm256_unpackhi_epi16(in05[part], in07[part]);
            const __m256i T_00_02A = _mm256_unpacklo_epi16(in09[part], in11[part]);
            const __m256i T_00_02B = _mm256_unpackhi_epi16(in09[part], in11[part]);
            const __m256i T_00_03A = _mm256_unpacklo_epi16(in13[part], in15[part]);
            const __m256i T_00_03B = _mm256_unpackhi_epi16(in13[part], in15[part]);
            const __m256i T_00_04A = _mm256_unpacklo_epi16(in17[part], in19[part]);
            const __m256i T_00_04B = _mm256_unpackhi_epi16(in17[part], in19[part]);
            const __m256i T_00_05A = _mm256_unpacklo_epi16(in21[part], in23[part]);
            const __m256i T_00_05B = _mm256_unpackhi_epi16(in21[part], in23[part]);
            const __m256i T_00_06A = _mm256_unpacklo_epi16(in25[part], in27[part]);
            const __m256i T_00_06B = _mm256_unpackhi_epi16(in25[part], in27[part]);
            const __m256i T_00_07A = _mm256_unpacklo_epi16(in29[part], in31[part]);
            const __m256i T_00_07B = _mm256_unpackhi_epi16(in29[part], in31[part]);

            /* EO 行配对 unpack (in02/06, in10/14, in18/22, in26/30) -> T_00_08A/B .. T_00_11A/B */
            const __m256i T_00_08A = _mm256_unpacklo_epi16(in02[part], in06[part]);
            const __m256i T_00_08B = _mm256_unpackhi_epi16(in02[part], in06[part]);
            const __m256i T_00_09A = _mm256_unpacklo_epi16(in10[part], in14[part]);
            const __m256i T_00_09B = _mm256_unpackhi_epi16(in10[part], in14[part]);
            const __m256i T_00_10A = _mm256_unpacklo_epi16(in18[part], in22[part]);
            const __m256i T_00_10B = _mm256_unpackhi_epi16(in18[part], in22[part]);
            const __m256i T_00_11A = _mm256_unpacklo_epi16(in26[part], in30[part]);
            const __m256i T_00_11B = _mm256_unpackhi_epi16(in26[part], in30[part]);

            /* EEO 行配对 unpack (in04/12, in20/28) -> T_00_12A/B, T_00_13A/B */
            const __m256i T_00_12A = _mm256_unpacklo_epi16(in04[part], in12[part]);
            const __m256i T_00_12B = _mm256_unpackhi_epi16(in04[part], in12[part]);
            const __m256i T_00_13A = _mm256_unpacklo_epi16(in20[part], in28[part]);
            const __m256i T_00_13B = _mm256_unpackhi_epi16(in20[part], in28[part]);

            /* EEEO 行配对 unpack (in08/24) -> T_00_14A/B */
            const __m256i T_00_14A = _mm256_unpacklo_epi16(in08[part], in24[part]);
            const __m256i T_00_14B = _mm256_unpackhi_epi16(in08[part], in24[part]);

            /* EEEE 行配对 unpack (in00/16) -> T_00_15A/B */
            const __m256i T_00_15A = _mm256_unpacklo_epi16(in00[part], in16[part]);
            const __m256i T_00_15B = _mm256_unpackhi_epi16(in00[part], in16[part]);

            __m256i O00A, O01A, O02A, O03A, O04A, O05A, O06A, O07A, O08A, O09A, O10A, O11A, O12A, O13A, O14A, O15A;
            __m256i O00B, O01B, O02B, O03B, O04B, O05B, O06B, O07B, O08B, O09B, O10B, O11B, O12B, O13B, O14B, O15B;
            __m256i EO0A, EO1A, EO2A, EO3A, EO4A, EO5A, EO6A, EO7A;
            __m256i EO0B, EO1B, EO2B, EO3B, EO4B, EO5B, EO6B, EO7B;
            {
                __m256i T00, T01, T02, T03;
                /* O 行: 8 对 madd 求和 (奇数行 01/03/05/.../31) */
#define     COMPUTE_ROW(r0103, r0507, r0911, r1315, r1719, r2123, r2527, r2931, c0103, c0507, c0911, c1315, c1719, c2123, c2527, c2931, row) \
            T00 = _mm256_add_epi32(_mm256_madd_epi16(r0103, c0103), _mm256_madd_epi16(r0507, c0507)); \
            T01 = _mm256_add_epi32(_mm256_madd_epi16(r0911, c0911), _mm256_madd_epi16(r1315, c1315)); \
            T02 = _mm256_add_epi32(_mm256_madd_epi16(r1719, c1719), _mm256_madd_epi16(r2123, c2123)); \
            T03 = _mm256_add_epi32(_mm256_madd_epi16(r2527, c2527), _mm256_madd_epi16(r2931, c2931)); \
            row = _mm256_add_epi32(_mm256_add_epi32(T00, T01), _mm256_add_epi32(T02, T03));

                COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_p45_p45, c16_p43_p44, c16_p39_p41, c16_p34_p36, c16_p27_p30, c16_p19_p23, c16_p11_p15, c16_p02_p07, O00A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_p41_p45, c16_p23_p34, c16_n02_p11, c16_n27_n15, c16_n43_n36, c16_n44_n45, c16_n30_n39, c16_n07_n19, O01A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_p34_p44, c16_n07_p15, c16_n41_n27, c16_n39_n45, c16_n02_n23, c16_p36_p19, c16_p43_p45, c16_p11_p30, O02A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_p23_p43, c16_n34_n07, c16_n36_n45, c16_p19_n11, c16_p44_p41, c16_n02_p27, c16_n45_n30, c16_n15_n39, O03A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_p11_p41, c16_n45_n27, c16_p07_n30, c16_p43_p39, c16_n23_p15, c16_n34_n45, c16_p36_p02, c16_p19_p44, O04A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_n02_p39, c16_n36_n41, c16_p43_p07, c16_n11_p34, c16_n30_n44, c16_p45_p15, c16_n19_p27, c16_n23_n45, O05A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_n15_p36, c16_n11_n45, c16_p34_p39, c16_n45_n19, c16_p41_n07, c16_n23_p30, c16_n02_n44, c16_p27_p43, O06A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_n27_p34, c16_p19_n39, c16_n11_p43, c16_p02_n45, c16_p07_p45, c16_n15_n44, c16_p23_p41, c16_n30_n36, O07A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_n36_p30, c16_p41_n23, c16_n44_p15, c16_p45_n07, c16_n45_n02, c16_p43_p11, c16_n39_n19, c16_p34_p27, O08A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_n43_p27, c16_p44_n02, c16_n30_n23, c16_p07_p41, c16_p19_n45, c16_n39_p34, c16_p45_n11, c16_n36_n15, O09A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_n45_p23, c16_p27_p19, c16_p15_n45, c16_n44_p30, c16_p34_p11, c16_p07_n43, c16_n41_p36, c16_p39_p02, O10A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_n44_p19, c16_n02_p36, c16_p45_n34, c16_n15_n23, c16_n39_p43, c16_p30_p07, c16_p27_n45, c16_n41_p11, O11A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_n39_p15, c16_n30_p45, c16_p27_p02, c16_p41_n44, c16_n11_n19, c16_n45_p36, c16_n07_p34, c16_p43_n23, O12A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_n30_p11, c16_n45_p43, c16_n19_p36, c16_p23_n02, c16_p45_n39, c16_p27_n41, c16_n15_n07, c16_n44_p34, O13A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_n19_p07, c16_n39_p30, c16_n45_p44, c16_n36_p43, c16_n15_p27, c16_p11_p02, c16_p34_n23, c16_p45_n41, O14A)
                    COMPUTE_ROW(T_00_00A, T_00_01A, T_00_02A, T_00_03A, T_00_04A, T_00_05A, T_00_06A, T_00_07A, \
                    c16_n07_p02, c16_n15_p11, c16_n23_p19, c16_n30_p27, c16_n36_p34, c16_n41_p39, c16_n44_p43, c16_n45_p45, O15A)

                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_p45_p45, c16_p43_p44, c16_p39_p41, c16_p34_p36, c16_p27_p30, c16_p19_p23, c16_p11_p15, c16_p02_p07, O00B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_p41_p45, c16_p23_p34, c16_n02_p11, c16_n27_n15, c16_n43_n36, c16_n44_n45, c16_n30_n39, c16_n07_n19, O01B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_p34_p44, c16_n07_p15, c16_n41_n27, c16_n39_n45, c16_n02_n23, c16_p36_p19, c16_p43_p45, c16_p11_p30, O02B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_p23_p43, c16_n34_n07, c16_n36_n45, c16_p19_n11, c16_p44_p41, c16_n02_p27, c16_n45_n30, c16_n15_n39, O03B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_p11_p41, c16_n45_n27, c16_p07_n30, c16_p43_p39, c16_n23_p15, c16_n34_n45, c16_p36_p02, c16_p19_p44, O04B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_n02_p39, c16_n36_n41, c16_p43_p07, c16_n11_p34, c16_n30_n44, c16_p45_p15, c16_n19_p27, c16_n23_n45, O05B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_n15_p36, c16_n11_n45, c16_p34_p39, c16_n45_n19, c16_p41_n07, c16_n23_p30, c16_n02_n44, c16_p27_p43, O06B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_n27_p34, c16_p19_n39, c16_n11_p43, c16_p02_n45, c16_p07_p45, c16_n15_n44, c16_p23_p41, c16_n30_n36, O07B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_n36_p30, c16_p41_n23, c16_n44_p15, c16_p45_n07, c16_n45_n02, c16_p43_p11, c16_n39_n19, c16_p34_p27, O08B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_n43_p27, c16_p44_n02, c16_n30_n23, c16_p07_p41, c16_p19_n45, c16_n39_p34, c16_p45_n11, c16_n36_n15, O09B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_n45_p23, c16_p27_p19, c16_p15_n45, c16_n44_p30, c16_p34_p11, c16_p07_n43, c16_n41_p36, c16_p39_p02, O10B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_n44_p19, c16_n02_p36, c16_p45_n34, c16_n15_n23, c16_n39_p43, c16_p30_p07, c16_p27_n45, c16_n41_p11, O11B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_n39_p15, c16_n30_p45, c16_p27_p02, c16_p41_n44, c16_n11_n19, c16_n45_p36, c16_n07_p34, c16_p43_n23, O12B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_n30_p11, c16_n45_p43, c16_n19_p36, c16_p23_n02, c16_p45_n39, c16_p27_n41, c16_n15_n07, c16_n44_p34, O13B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_n19_p07, c16_n39_p30, c16_n45_p44, c16_n36_p43, c16_n15_p27, c16_p11_p02, c16_p34_n23, c16_p45_n41, O14B)
                    COMPUTE_ROW(T_00_00B, T_00_01B, T_00_02B, T_00_03B, T_00_04B, T_00_05B, T_00_06B, T_00_07B, \
                    c16_n07_p02, c16_n15_p11, c16_n23_p19, c16_n30_p27, c16_n36_p34, c16_n41_p39, c16_n44_p43, c16_n45_p45, O15B)

#undef      COMPUTE_ROW
            }


            {
                __m256i T00, T01;
                /* EO 行: 4 对 madd 求和 (偶数行 02/06/10/14/18/22/26/30) */
#define     COMPUTE_ROW(row0206, row1014, row1822, row2630, c0206, c1014, c1822, c2630, row) \
            T00 = _mm256_add_epi32(_mm256_madd_epi16(row0206, c0206), _mm256_madd_epi16(row1014, c1014)); \
            T01 = _mm256_add_epi32(_mm256_madd_epi16(row1822, c1822), _mm256_madd_epi16(row2630, c2630)); \
            row = _mm256_add_epi32(T00, T01);

                COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_p43_p45, c16_p35_p40, c16_p21_p29, c16_p04_p13, EO0A)
                    COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_p29_p43, c16_n21_p04, c16_n45_n40, c16_n13_n35, EO1A)
                    COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_p04_p40, c16_n43_n35, c16_p29_n13, c16_p21_p45, EO2A)
                    COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_n21_p35, c16_p04_n43, c16_p13_p45, c16_n29_n40, EO3A)
                    COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_n40_p29, c16_p45_n13, c16_n43_n04, c16_p35_p21, EO4A)
                    COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_n45_p21, c16_p13_p29, c16_p35_n43, c16_n40_p04, EO5A)
                    COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_n35_p13, c16_n40_p45, c16_p04_p21, c16_p43_n29, EO6A)
                    COMPUTE_ROW(T_00_08A, T_00_09A, T_00_10A, T_00_11A, c16_n13_p04, c16_n29_p21, c16_n40_p35, c16_n45_p43, EO7A)

                    COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_p43_p45, c16_p35_p40, c16_p21_p29, c16_p04_p13, EO0B)
                    COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_p29_p43, c16_n21_p04, c16_n45_n40, c16_n13_n35, EO1B)
                    COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_p04_p40, c16_n43_n35, c16_p29_n13, c16_p21_p45, EO2B)
                    COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_n21_p35, c16_p04_n43, c16_p13_p45, c16_n29_n40, EO3B)
                    COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_n40_p29, c16_p45_n13, c16_n43_n04, c16_p35_p21, EO4B)
                    COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_n45_p21, c16_p13_p29, c16_p35_n43, c16_n40_p04, EO5B)
                    COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_n35_p13, c16_n40_p45, c16_p04_p21, c16_p43_n29, EO6B)
                    COMPUTE_ROW(T_00_08B, T_00_09B, T_00_10B, T_00_11B, c16_n13_p04, c16_n29_p21, c16_n40_p35, c16_n45_p43, EO7B)
#undef      COMPUTE_ROW
            }

            /* EEO 行: 2 对 madd (in04/12, in20/28) */
            EEO0A = _mm256_add_epi32(_mm256_madd_epi16(T_00_12A, c16_p38_p44), _mm256_madd_epi16(T_00_13A, c16_p09_p25));
            EEO1A = _mm256_add_epi32(_mm256_madd_epi16(T_00_12A, c16_n09_p38), _mm256_madd_epi16(T_00_13A, c16_n25_n44));
            EEO2A = _mm256_add_epi32(_mm256_madd_epi16(T_00_12A, c16_n44_p25), _mm256_madd_epi16(T_00_13A, c16_p38_p09));
            EEO3A = _mm256_add_epi32(_mm256_madd_epi16(T_00_12A, c16_n25_p09), _mm256_madd_epi16(T_00_13A, c16_n44_p38));
            EEO0B = _mm256_add_epi32(_mm256_madd_epi16(T_00_12B, c16_p38_p44), _mm256_madd_epi16(T_00_13B, c16_p09_p25));
            EEO1B = _mm256_add_epi32(_mm256_madd_epi16(T_00_12B, c16_n09_p38), _mm256_madd_epi16(T_00_13B, c16_n25_n44));
            EEO2B = _mm256_add_epi32(_mm256_madd_epi16(T_00_12B, c16_n44_p25), _mm256_madd_epi16(T_00_13B, c16_p38_p09));
            EEO3B = _mm256_add_epi32(_mm256_madd_epi16(T_00_12B, c16_n25_p09), _mm256_madd_epi16(T_00_13B, c16_n44_p38));

            /* EEEO 行: 1 对 madd (in08/24) */
            EEEO0A = _mm256_madd_epi16(T_00_14A, c16_p17_p42);
            EEEO0B = _mm256_madd_epi16(T_00_14B, c16_p17_p42);
            EEEO1A = _mm256_madd_epi16(T_00_14A, c16_n42_p17);
            EEEO1B = _mm256_madd_epi16(T_00_14B, c16_n42_p17);

            /* EEEE 行: 1 对 madd (in00/16) */
            EEEE0A = _mm256_madd_epi16(T_00_15A, c16_p32_p32);
            EEEE0B = _mm256_madd_epi16(T_00_15B, c16_p32_p32);
            EEEE1A = _mm256_madd_epi16(T_00_15A, c16_n32_p32);
            EEEE1B = _mm256_madd_epi16(T_00_15B, c16_n32_p32);

            /* EEE 层: EEE = EEEE +/- EEEO */
            EEE0A = _mm256_add_epi32(EEEE0A, EEEO0A);          /* EEE0 = EEEE0 + EEEO0 */
            EEE0B = _mm256_add_epi32(EEEE0B, EEEO0B);
            EEE1A = _mm256_add_epi32(EEEE1A, EEEO1A);          /* EEE1 = EEEE1 + EEEO1 */
            EEE1B = _mm256_add_epi32(EEEE1B, EEEO1B);
            EEE3A = _mm256_sub_epi32(EEEE0A, EEEO0A);          /* EEE3 = EEEE0 - EEEO0 */
            EEE3B = _mm256_sub_epi32(EEEE0B, EEEO0B);
            EEE2A = _mm256_sub_epi32(EEEE1A, EEEO1A);          /* EEE2 = EEEE1 - EEEO1 */
            EEE2B = _mm256_sub_epi32(EEEE1B, EEEO1B);

            /* EE 层: EE = EEE +/- EEO */
            EE0A = _mm256_add_epi32(EEE0A, EEO0A);          /* EE0 = EEE0 + EEO0 */
            EE0B = _mm256_add_epi32(EEE0B, EEO0B);
            EE1A = _mm256_add_epi32(EEE1A, EEO1A);          /* EE1 = EEE1 + EEO1 */
            EE1B = _mm256_add_epi32(EEE1B, EEO1B);
            EE2A = _mm256_add_epi32(EEE2A, EEO2A);          /* EE2 = EEE2 + EEO2 */
            EE2B = _mm256_add_epi32(EEE2B, EEO2B);
            EE3A = _mm256_add_epi32(EEE3A, EEO3A);          /* EE3 = EEE3 + EEO3 */
            EE3B = _mm256_add_epi32(EEE3B, EEO3B);
            EE7A = _mm256_sub_epi32(EEE0A, EEO0A);          /* EE7 = EEE0 - EEO0 */
            EE7B = _mm256_sub_epi32(EEE0B, EEO0B);
            EE6A = _mm256_sub_epi32(EEE1A, EEO1A);          /* EE6 = EEE1 - EEO1 */
            EE6B = _mm256_sub_epi32(EEE1B, EEO1B);
            EE5A = _mm256_sub_epi32(EEE2A, EEO2A);          /* EE5 = EEE2 - EEO2 */
            EE5B = _mm256_sub_epi32(EEE2B, EEO2B);
            EE4A = _mm256_sub_epi32(EEE3A, EEO3A);          /* EE4 = EEE3 - EEO3 */
            EE4B = _mm256_sub_epi32(EEE3B, EEO3B);

            /* E 层: E = EE +/- EO */
            E0A = _mm256_add_epi32(EE0A, EO0A);          /* E0 = EE0 + EO0 */
            E0B = _mm256_add_epi32(EE0B, EO0B);
            E1A = _mm256_add_epi32(EE1A, EO1A);          /* E1 = EE1 + EO1 */
            E1B = _mm256_add_epi32(EE1B, EO1B);
            E2A = _mm256_add_epi32(EE2A, EO2A);          /* E2 = EE2 + EO2 */
            E2B = _mm256_add_epi32(EE2B, EO2B);
            E3A = _mm256_add_epi32(EE3A, EO3A);          /* E3 = EE3 + EO3 */
            E3B = _mm256_add_epi32(EE3B, EO3B);
            E4A = _mm256_add_epi32(EE4A, EO4A);          /* E4 = EE4 + EO4 */
            E4B = _mm256_add_epi32(EE4B, EO4B);
            E5A = _mm256_add_epi32(EE5A, EO5A);          /* E5 = EE5 + EO5 */
            E5B = _mm256_add_epi32(EE5B, EO5B);
            E6A = _mm256_add_epi32(EE6A, EO6A);          /* E6 = EE6 + EO6 */
            E6B = _mm256_add_epi32(EE6B, EO6B);
            E7A = _mm256_add_epi32(EE7A, EO7A);          /* E7 = EE7 + EO7 */
            E7B = _mm256_add_epi32(EE7B, EO7B);
            EFA = _mm256_sub_epi32(EE0A, EO0A);          /* EF = EE0 - EO0 */
            EFB = _mm256_sub_epi32(EE0B, EO0B);
            EEA = _mm256_sub_epi32(EE1A, EO1A);          /* EE = EE1 - EO1 */
            EEB = _mm256_sub_epi32(EE1B, EO1B);
            EDA = _mm256_sub_epi32(EE2A, EO2A);          /* ED = EE2 - EO2 */
            EDB = _mm256_sub_epi32(EE2B, EO2B);
            ECA = _mm256_sub_epi32(EE3A, EO3A);          /* EC = EE3 - EO3 */
            ECB = _mm256_sub_epi32(EE3B, EO3B);
            EBA = _mm256_sub_epi32(EE4A, EO4A);          /* EB = EE4 - EO4 */
            EBB = _mm256_sub_epi32(EE4B, EO4B);
            EAA = _mm256_sub_epi32(EE5A, EO5A);          /* EA = EE5 - EO5 */
            EAB = _mm256_sub_epi32(EE5B, EO5B);
            E9A = _mm256_sub_epi32(EE6A, EO6A);          /* E9 = EE6 - EO6 */
            E9B = _mm256_sub_epi32(EE6B, EO6B);
            E8A = _mm256_sub_epi32(EE7A, EO7A);          /* E8 = EE7 - EO7 */
            E8B = _mm256_sub_epi32(EE7B, EO7B);

            /* 加舍入值 */
            T10A = _mm256_add_epi32(E0A, c32_rnd);         /* E0 + rnd */
            T10B = _mm256_add_epi32(E0B, c32_rnd);
            T11A = _mm256_add_epi32(E1A, c32_rnd);         /* E1 + rnd */
            T11B = _mm256_add_epi32(E1B, c32_rnd);
            T12A = _mm256_add_epi32(E2A, c32_rnd);         /* E2 + rnd */
            T12B = _mm256_add_epi32(E2B, c32_rnd);
            T13A = _mm256_add_epi32(E3A, c32_rnd);         /* E3 + rnd */
            T13B = _mm256_add_epi32(E3B, c32_rnd);
            T14A = _mm256_add_epi32(E4A, c32_rnd);         /* E4 + rnd */
            T14B = _mm256_add_epi32(E4B, c32_rnd);
            T15A = _mm256_add_epi32(E5A, c32_rnd);         /* E5 + rnd */
            T15B = _mm256_add_epi32(E5B, c32_rnd);
            T16A = _mm256_add_epi32(E6A, c32_rnd);         /* E6 + rnd */
            T16B = _mm256_add_epi32(E6B, c32_rnd);
            T17A = _mm256_add_epi32(E7A, c32_rnd);         /* E7 + rnd */
            T17B = _mm256_add_epi32(E7B, c32_rnd);
            T18A = _mm256_add_epi32(E8A, c32_rnd);         /* E8 + rnd */
            T18B = _mm256_add_epi32(E8B, c32_rnd);
            T19A = _mm256_add_epi32(E9A, c32_rnd);         /* E9 + rnd */
            T19B = _mm256_add_epi32(E9B, c32_rnd);
            T1AA = _mm256_add_epi32(EAA, c32_rnd);         /* E10 + rnd */
            T1AB = _mm256_add_epi32(EAB, c32_rnd);
            T1BA = _mm256_add_epi32(EBA, c32_rnd);         /* E11 + rnd */
            T1BB = _mm256_add_epi32(EBB, c32_rnd);
            T1CA = _mm256_add_epi32(ECA, c32_rnd);         /* E12 + rnd */
            T1CB = _mm256_add_epi32(ECB, c32_rnd);
            T1DA = _mm256_add_epi32(EDA, c32_rnd);         /* E13 + rnd */
            T1DB = _mm256_add_epi32(EDB, c32_rnd);
            T1EA = _mm256_add_epi32(EEA, c32_rnd);         /* E14 + rnd */
            T1EB = _mm256_add_epi32(EEB, c32_rnd);
            T1FA = _mm256_add_epi32(EFA, c32_rnd);         /* E15 + rnd */
            T1FB = _mm256_add_epi32(EFB, c32_rnd);

            /* T2 = E +/- O (加 rnd 后) */
            T2_00A = _mm256_add_epi32(T10A, O00A);          /* E0 + O0 + rnd */
            T2_00B = _mm256_add_epi32(T10B, O00B);
            T2_01A = _mm256_add_epi32(T11A, O01A);          /* E1 + O1 + rnd */
            T2_01B = _mm256_add_epi32(T11B, O01B);
            T2_02A = _mm256_add_epi32(T12A, O02A);          /* E2 + O2 + rnd */
            T2_02B = _mm256_add_epi32(T12B, O02B);
            T2_03A = _mm256_add_epi32(T13A, O03A);          /* E3 + O3 + rnd */
            T2_03B = _mm256_add_epi32(T13B, O03B);
            T2_04A = _mm256_add_epi32(T14A, O04A);          /* E4 + O4 + rnd */
            T2_04B = _mm256_add_epi32(T14B, O04B);
            T2_05A = _mm256_add_epi32(T15A, O05A);          /* E5 + O5 + rnd */
            T2_05B = _mm256_add_epi32(T15B, O05B);
            T2_06A = _mm256_add_epi32(T16A, O06A);          /* E6 + O6 + rnd */
            T2_06B = _mm256_add_epi32(T16B, O06B);
            T2_07A = _mm256_add_epi32(T17A, O07A);          /* E7 + O7 + rnd */
            T2_07B = _mm256_add_epi32(T17B, O07B);
            T2_08A = _mm256_add_epi32(T18A, O08A);          /* E8 + O8 + rnd */
            T2_08B = _mm256_add_epi32(T18B, O08B);
            T2_09A = _mm256_add_epi32(T19A, O09A);          /* E9 + O9 + rnd */
            T2_09B = _mm256_add_epi32(T19B, O09B);
            T2_10A = _mm256_add_epi32(T1AA, O10A);          /* E10 + O10 + rnd */
            T2_10B = _mm256_add_epi32(T1AB, O10B);
            T2_11A = _mm256_add_epi32(T1BA, O11A);          /* E11 + O11 + rnd */
            T2_11B = _mm256_add_epi32(T1BB, O11B);
            T2_12A = _mm256_add_epi32(T1CA, O12A);          /* E12 + O12 + rnd */
            T2_12B = _mm256_add_epi32(T1CB, O12B);
            T2_13A = _mm256_add_epi32(T1DA, O13A);          /* E13 + O13 + rnd */
            T2_13B = _mm256_add_epi32(T1DB, O13B);
            T2_14A = _mm256_add_epi32(T1EA, O14A);          /* E14 + O14 + rnd */
            T2_14B = _mm256_add_epi32(T1EB, O14B);
            T2_15A = _mm256_add_epi32(T1FA, O15A);          /* E15 + O15 + rnd */
            T2_15B = _mm256_add_epi32(T1FB, O15B);
            T2_31A = _mm256_sub_epi32(T10A, O00A);          /* E0 - O0 + rnd */
            T2_31B = _mm256_sub_epi32(T10B, O00B);
            T2_30A = _mm256_sub_epi32(T11A, O01A);          /* E1 - O1 + rnd */
            T2_30B = _mm256_sub_epi32(T11B, O01B);
            T2_29A = _mm256_sub_epi32(T12A, O02A);          /* E2 - O2 + rnd */
            T2_29B = _mm256_sub_epi32(T12B, O02B);
            T2_28A = _mm256_sub_epi32(T13A, O03A);          /* E3 - O3 + rnd */
            T2_28B = _mm256_sub_epi32(T13B, O03B);
            T2_27A = _mm256_sub_epi32(T14A, O04A);          /* E4 - O4 + rnd */
            T2_27B = _mm256_sub_epi32(T14B, O04B);
            T2_26A = _mm256_sub_epi32(T15A, O05A);          /* E5 - O5 + rnd */
            T2_26B = _mm256_sub_epi32(T15B, O05B);
            T2_25A = _mm256_sub_epi32(T16A, O06A);          /* E6 - O6 + rnd */
            T2_25B = _mm256_sub_epi32(T16B, O06B);
            T2_24A = _mm256_sub_epi32(T17A, O07A);          /* E7 - O7 + rnd */
            T2_24B = _mm256_sub_epi32(T17B, O07B);
            T2_23A = _mm256_sub_epi32(T18A, O08A);          /* E8 - O8 + rnd */
            T2_23B = _mm256_sub_epi32(T18B, O08B);
            T2_22A = _mm256_sub_epi32(T19A, O09A);          /* E9 - O9 + rnd */
            T2_22B = _mm256_sub_epi32(T19B, O09B);
            T2_21A = _mm256_sub_epi32(T1AA, O10A);          /* E10 - O10 + rnd */
            T2_21B = _mm256_sub_epi32(T1AB, O10B);
            T2_20A = _mm256_sub_epi32(T1BA, O11A);          /* E11 - O11 + rnd */
            T2_20B = _mm256_sub_epi32(T1BB, O11B);
            T2_19A = _mm256_sub_epi32(T1CA, O12A);          /* E12 - O12 + rnd */
            T2_19B = _mm256_sub_epi32(T1CB, O12B);
            T2_18A = _mm256_sub_epi32(T1DA, O13A);          /* E13 - O13 + rnd */
            T2_18B = _mm256_sub_epi32(T1DB, O13B);
            T2_17A = _mm256_sub_epi32(T1EA, O14A);          /* E14 - O14 + rnd */
            T2_17B = _mm256_sub_epi32(T1EB, O14B);
            T2_16A = _mm256_sub_epi32(T1FA, O15A);          /* E15 - O15 + rnd */
            T2_16B = _mm256_sub_epi32(T1FB, O15B);

            /* 移位 (右移 nShift) */
            T3_00A = _mm256_srai_epi32(T2_00A, nShift);             /* [30 20 10 00] */
            T3_00B = _mm256_srai_epi32(T2_00B, nShift);             /* [70 60 50 40] */
            T3_01A = _mm256_srai_epi32(T2_01A, nShift);             /* [31 21 11 01] */
            T3_01B = _mm256_srai_epi32(T2_01B, nShift);             /* [71 61 51 41] */
            T3_02A = _mm256_srai_epi32(T2_02A, nShift);             /* [32 22 12 02] */
            T3_02B = _mm256_srai_epi32(T2_02B, nShift);             /* [72 62 52 42] */
            T3_03A = _mm256_srai_epi32(T2_03A, nShift);             /* [33 23 13 03] */
            T3_03B = _mm256_srai_epi32(T2_03B, nShift);             /* [73 63 53 43] */
            T3_04A = _mm256_srai_epi32(T2_04A, nShift);             /* [34 24 14 04] */
            T3_04B = _mm256_srai_epi32(T2_04B, nShift);             /* [74 64 54 44] */
            T3_05A = _mm256_srai_epi32(T2_05A, nShift);             /* [35 25 15 05] */
            T3_05B = _mm256_srai_epi32(T2_05B, nShift);             /* [75 65 55 45] */
            T3_06A = _mm256_srai_epi32(T2_06A, nShift);             /* [36 26 16 06] */
            T3_06B = _mm256_srai_epi32(T2_06B, nShift);             /* [76 66 56 46] */
            T3_07A = _mm256_srai_epi32(T2_07A, nShift);             /* [37 27 17 07] */
            T3_07B = _mm256_srai_epi32(T2_07B, nShift);             /* [77 67 57 47] */
            T3_08A = _mm256_srai_epi32(T2_08A, nShift);             /* [30 20 10 00] x8 */
            T3_08B = _mm256_srai_epi32(T2_08B, nShift);             /* [70 60 50 40] */
            T3_09A = _mm256_srai_epi32(T2_09A, nShift);             /* [31 21 11 01] x9 */
            T3_09B = _mm256_srai_epi32(T2_09B, nShift);             /* [71 61 51 41] */
            T3_10A = _mm256_srai_epi32(T2_10A, nShift);             /* [32 22 12 02] xA */
            T3_10B = _mm256_srai_epi32(T2_10B, nShift);             /* [72 62 52 42] */
            T3_11A = _mm256_srai_epi32(T2_11A, nShift);             /* [33 23 13 03] xB */
            T3_11B = _mm256_srai_epi32(T2_11B, nShift);             /* [73 63 53 43] */
            T3_12A = _mm256_srai_epi32(T2_12A, nShift);             /* [34 24 14 04] xC */
            T3_12B = _mm256_srai_epi32(T2_12B, nShift);             /* [74 64 54 44] */
            T3_13A = _mm256_srai_epi32(T2_13A, nShift);             /* [35 25 15 05] xD */
            T3_13B = _mm256_srai_epi32(T2_13B, nShift);             /* [75 65 55 45] */
            T3_14A = _mm256_srai_epi32(T2_14A, nShift);             /* [36 26 16 06] xE */
            T3_14B = _mm256_srai_epi32(T2_14B, nShift);             /* [76 66 56 46] */
            T3_15A = _mm256_srai_epi32(T2_15A, nShift);             /* [37 27 17 07] xF */
            T3_15B = _mm256_srai_epi32(T2_15B, nShift);             /* [77 67 57 47] */

            T3_16A = _mm256_srai_epi32(T2_16A, nShift);             /* [30 20 10 00] */
            T3_16B = _mm256_srai_epi32(T2_16B, nShift);             /* [70 60 50 40] */
            T3_17A = _mm256_srai_epi32(T2_17A, nShift);             /* [31 21 11 01] */
            T3_17B = _mm256_srai_epi32(T2_17B, nShift);             /* [71 61 51 41] */
            T3_18A = _mm256_srai_epi32(T2_18A, nShift);             /* [32 22 12 02] */
            T3_18B = _mm256_srai_epi32(T2_18B, nShift);             /* [72 62 52 42] */
            T3_19A = _mm256_srai_epi32(T2_19A, nShift);             /* [33 23 13 03] */
            T3_19B = _mm256_srai_epi32(T2_19B, nShift);             /* [73 63 53 43] */
            T3_20A = _mm256_srai_epi32(T2_20A, nShift);             /* [34 24 14 04] */
            T3_20B = _mm256_srai_epi32(T2_20B, nShift);             /* [74 64 54 44] */
            T3_21A = _mm256_srai_epi32(T2_21A, nShift);             /* [35 25 15 05] */
            T3_21B = _mm256_srai_epi32(T2_21B, nShift);             /* [75 65 55 45] */
            T3_22A = _mm256_srai_epi32(T2_22A, nShift);             /* [36 26 16 06] */
            T3_22B = _mm256_srai_epi32(T2_22B, nShift);             /* [76 66 56 46] */
            T3_23A = _mm256_srai_epi32(T2_23A, nShift);             /* [37 27 17 07] */
            T3_23B = _mm256_srai_epi32(T2_23B, nShift);             /* [77 67 57 47] */
            T3_24A = _mm256_srai_epi32(T2_24A, nShift);             /* [30 20 10 00] x8 */
            T3_24B = _mm256_srai_epi32(T2_24B, nShift);             /* [70 60 50 40] */
            T3_25A = _mm256_srai_epi32(T2_25A, nShift);             /* [31 21 11 01] x9 */
            T3_25B = _mm256_srai_epi32(T2_25B, nShift);             /* [71 61 51 41] */
            T3_26A = _mm256_srai_epi32(T2_26A, nShift);             /* [32 22 12 02] xA */
            T3_26B = _mm256_srai_epi32(T2_26B, nShift);             /* [72 62 52 42] */
            T3_27A = _mm256_srai_epi32(T2_27A, nShift);             /* [33 23 13 03] xB */
            T3_27B = _mm256_srai_epi32(T2_27B, nShift);             /* [73 63 53 43] */
            T3_28A = _mm256_srai_epi32(T2_28A, nShift);             /* [34 24 14 04] xC */
            T3_28B = _mm256_srai_epi32(T2_28B, nShift);             /* [74 64 54 44] */
            T3_29A = _mm256_srai_epi32(T2_29A, nShift);             /* [35 25 15 05] xD */
            T3_29B = _mm256_srai_epi32(T2_29B, nShift);             /* [75 65 55 45] */
            T3_30A = _mm256_srai_epi32(T2_30A, nShift);             /* [36 26 16 06] xE */
            T3_30B = _mm256_srai_epi32(T2_30B, nShift);             /* [76 66 56 46] */
            T3_31A = _mm256_srai_epi32(T2_31A, nShift);             /* [37 27 17 07] xF */
            T3_31B = _mm256_srai_epi32(T2_31B, nShift);             /* [77 67 57 47] */

            /* 打包 32-bit -> 16-bit (有符号饱和) */
            res00[part] = _mm256_packs_epi32(T3_00A, T3_00B);        /* [70 60 50 40 30 20 10 00] */
            res01[part] = _mm256_packs_epi32(T3_01A, T3_01B);        /* [71 61 51 41 31 21 11 01] */
            res02[part] = _mm256_packs_epi32(T3_02A, T3_02B);        /* [72 62 52 42 32 22 12 02] */
            res03[part] = _mm256_packs_epi32(T3_03A, T3_03B);        /* [73 63 53 43 33 23 13 03] */
            res04[part] = _mm256_packs_epi32(T3_04A, T3_04B);        /* [74 64 54 44 34 24 14 04] */
            res05[part] = _mm256_packs_epi32(T3_05A, T3_05B);        /* [75 65 55 45 35 25 15 05] */
            res06[part] = _mm256_packs_epi32(T3_06A, T3_06B);        /* [76 66 56 46 36 26 16 06] */
            res07[part] = _mm256_packs_epi32(T3_07A, T3_07B);        /* [77 67 57 47 37 27 17 07] */
            res08[part] = _mm256_packs_epi32(T3_08A, T3_08B);        /* [A0 ... 80] */
            res09[part] = _mm256_packs_epi32(T3_09A, T3_09B);        /* [A1 ... 81] */
            res10[part] = _mm256_packs_epi32(T3_10A, T3_10B);        /* [A2 ... 82] */
            res11[part] = _mm256_packs_epi32(T3_11A, T3_11B);        /* [A3 ... 83] */
            res12[part] = _mm256_packs_epi32(T3_12A, T3_12B);        /* [A4 ... 84] */
            res13[part] = _mm256_packs_epi32(T3_13A, T3_13B);        /* [A5 ... 85] */
            res14[part] = _mm256_packs_epi32(T3_14A, T3_14B);        /* [A6 ... 86] */
            res15[part] = _mm256_packs_epi32(T3_15A, T3_15B);        /* [A7 ... 87] */
            res16[part] = _mm256_packs_epi32(T3_16A, T3_16B);
            res17[part] = _mm256_packs_epi32(T3_17A, T3_17B);
            res18[part] = _mm256_packs_epi32(T3_18A, T3_18B);
            res19[part] = _mm256_packs_epi32(T3_19A, T3_19B);
            res20[part] = _mm256_packs_epi32(T3_20A, T3_20B);
            res21[part] = _mm256_packs_epi32(T3_21A, T3_21B);
            res22[part] = _mm256_packs_epi32(T3_22A, T3_22B);
            res23[part] = _mm256_packs_epi32(T3_23A, T3_23B);
            res24[part] = _mm256_packs_epi32(T3_24A, T3_24B);
            res25[part] = _mm256_packs_epi32(T3_25A, T3_25B);
            res26[part] = _mm256_packs_epi32(T3_26A, T3_26B);
            res27[part] = _mm256_packs_epi32(T3_27A, T3_27B);
            res28[part] = _mm256_packs_epi32(T3_28A, T3_28B);
            res29[part] = _mm256_packs_epi32(T3_29A, T3_29B);
            res30[part] = _mm256_packs_epi32(T3_30A, T3_30B);
            res31[part] = _mm256_packs_epi32(T3_31A, T3_31B);

        }

        /* ---- 32x32 矩阵转置 (分 4 块 16x16 转置) ----
         * 转置后结果写回 in00..in31, 作为下一遍 (或最终输出) 的输入.
         * 4 块对应 32x32 的四个 16x16 象限:
         *   [res00..15[0] | res00..15[1]]   ->  [in00..15[0] | in00..15[1]]
         *   [res16..31[0] | res16..31[1]]       [in16..31[0] | in16..31[1]]
         */
        {
            __m256i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7, tr0_8, tr0_9, tr0_10, tr0_11, tr0_12, tr0_13, tr0_14, tr0_15;
#define TRANSPOSE_16x16_16BIT(I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14, O15) \
        tr0_0 = _mm256_unpacklo_epi16(I0, I1); \
        tr0_1 = _mm256_unpacklo_epi16(I2, I3); \
        tr0_2 = _mm256_unpacklo_epi16(I4, I5); \
        tr0_3 = _mm256_unpacklo_epi16(I6, I7); \
        tr0_4 = _mm256_unpacklo_epi16(I8, I9); \
        tr0_5 = _mm256_unpacklo_epi16(I10, I11); \
        tr0_6 = _mm256_unpacklo_epi16(I12, I13); \
        tr0_7 = _mm256_unpacklo_epi16(I14, I15); \
        tr0_8 = _mm256_unpackhi_epi16(I0, I1); \
        tr0_9 = _mm256_unpackhi_epi16(I2, I3); \
        tr0_10 = _mm256_unpackhi_epi16(I4, I5); \
        tr0_11 = _mm256_unpackhi_epi16(I6, I7); \
        tr0_12 = _mm256_unpackhi_epi16(I8, I9); \
        tr0_13 = _mm256_unpackhi_epi16(I10, I11); \
        tr0_14 = _mm256_unpackhi_epi16(I12, I13); \
        tr0_15 = _mm256_unpackhi_epi16(I14, I15); \
        O0 = _mm256_unpacklo_epi32(tr0_0, tr0_1); \
        O1 = _mm256_unpacklo_epi32(tr0_2, tr0_3); \
        O2 = _mm256_unpacklo_epi32(tr0_4, tr0_5); \
        O3 = _mm256_unpacklo_epi32(tr0_6, tr0_7); \
        O4 = _mm256_unpackhi_epi32(tr0_0, tr0_1); \
        O5 = _mm256_unpackhi_epi32(tr0_2, tr0_3); \
        O6 = _mm256_unpackhi_epi32(tr0_4, tr0_5); \
        O7 = _mm256_unpackhi_epi32(tr0_6, tr0_7); \
        O8 = _mm256_unpacklo_epi32(tr0_8, tr0_9); \
        O9 = _mm256_unpacklo_epi32(tr0_10, tr0_11); \
        O10 = _mm256_unpacklo_epi32(tr0_12, tr0_13); \
        O11 = _mm256_unpacklo_epi32(tr0_14, tr0_15); \
        O12 = _mm256_unpackhi_epi32(tr0_8, tr0_9); \
        O13 = _mm256_unpackhi_epi32(tr0_10, tr0_11); \
        O14 = _mm256_unpackhi_epi32(tr0_12, tr0_13); \
        O15 = _mm256_unpackhi_epi32(tr0_14, tr0_15); \
        tr0_0 = _mm256_unpacklo_epi64(O0, O1); \
        tr0_1 = _mm256_unpacklo_epi64(O2, O3); \
        tr0_2 = _mm256_unpackhi_epi64(O0, O1); \
        tr0_3 = _mm256_unpackhi_epi64(O2, O3); \
        tr0_4 = _mm256_unpacklo_epi64(O4, O5); \
        tr0_5 = _mm256_unpacklo_epi64(O6, O7); \
        tr0_6 = _mm256_unpackhi_epi64(O4, O5); \
        tr0_7 = _mm256_unpackhi_epi64(O6, O7); \
        tr0_8 = _mm256_unpacklo_epi64(O8, O9); \
        tr0_9 = _mm256_unpacklo_epi64(O10, O11); \
        tr0_10 = _mm256_unpackhi_epi64(O8, O9); \
        tr0_11 = _mm256_unpackhi_epi64(O10, O11); \
        tr0_12 = _mm256_unpacklo_epi64(O12, O13); \
        tr0_13 = _mm256_unpacklo_epi64(O14, O15); \
        tr0_14 = _mm256_unpackhi_epi64(O12, O13); \
        tr0_15 = _mm256_unpackhi_epi64(O14, O15); \
        O0 = _mm256_permute2x128_si256(tr0_0, tr0_1, 0x20); \
        O1 = _mm256_permute2x128_si256(tr0_2, tr0_3, 0x20); \
        O2 = _mm256_permute2x128_si256(tr0_4, tr0_5, 0x20); \
        O3 = _mm256_permute2x128_si256(tr0_6, tr0_7, 0x20); \
        O4 = _mm256_permute2x128_si256(tr0_8, tr0_9, 0x20); \
        O5 = _mm256_permute2x128_si256(tr0_10, tr0_11, 0x20); \
        O6 = _mm256_permute2x128_si256(tr0_12, tr0_13, 0x20); \
        O7 = _mm256_permute2x128_si256(tr0_14, tr0_15, 0x20); \
        O8 = _mm256_permute2x128_si256(tr0_0, tr0_1, 0x31); \
        O9 = _mm256_permute2x128_si256(tr0_2, tr0_3, 0x31); \
        O10 = _mm256_permute2x128_si256(tr0_4, tr0_5, 0x31); \
        O11 = _mm256_permute2x128_si256(tr0_6, tr0_7, 0x31); \
        O12 = _mm256_permute2x128_si256(tr0_8, tr0_9, 0x31); \
        O13 = _mm256_permute2x128_si256(tr0_10, tr0_11, 0x31); \
        O14 = _mm256_permute2x128_si256(tr0_12, tr0_13, 0x31); \
        O15 = _mm256_permute2x128_si256(tr0_14, tr0_15, 0x31); \

            TRANSPOSE_16x16_16BIT(res00[0], res01[0], res02[0], res03[0], res04[0], res05[0], res06[0], res07[0], res08[0], res09[0], res10[0], res11[0], res12[0], res13[0], res14[0], res15[0], in00[0], in01[0], in02[0], in03[0], in04[0], in05[0], in06[0], in07[0], in08[0], in09[0], in10[0], in11[0], in12[0], in13[0], in14[0], in15[0])
                TRANSPOSE_16x16_16BIT(res16[0], res17[0], res18[0], res19[0], res20[0], res21[0], res22[0], res23[0], res24[0], res25[0], res26[0], res27[0], res28[0], res29[0], res30[0], res31[0], in00[1], in01[1], in02[1], in03[1], in04[1], in05[1], in06[1], in07[1], in08[1], in09[1], in10[1], in11[1], in12[1], in13[1], in14[1], in15[1]);
            TRANSPOSE_16x16_16BIT(res00[1], res01[1], res02[1], res03[1], res04[1], res05[1], res06[1], res07[1], res08[1], res09[1], res10[1], res11[1], res12[1], res13[1], res14[1], res15[1], in16[0], in17[0], in18[0], in19[0], in20[0], in21[0], in22[0], in23[0], in24[0], in25[0], in26[0], in27[0], in28[0], in29[0], in30[0], in31[0]);
            TRANSPOSE_16x16_16BIT(res16[1], res17[1], res18[1], res19[1], res20[1], res21[1], res22[1], res23[1], res24[1], res25[1], res26[1], res27[1], res28[1], res29[1], res30[1], res31[1], in16[1], in17[1], in18[1], in19[1], in20[1], in21[1], in22[1], in23[1], in24[1], in25[1], in26[1], in27[1], in28[1], in29[1], in30[1], in31[1]);

#undef  TRANSPOSE_16x16_16BIT

        }

        /* 设置 pass 1 的舍入值与移位值 (pass 0 用初始值 16/5) */
        c32_rnd = _mm256_set1_epi32(shift ? (1 << (shift - 1)) : 0);   /* pass == 1 第二遍输入舍入 */
        nShift = shift;
    }

    /* ---- 裁剪到 [-2^(clip-1), 2^(clip-1)-1] ---- */
    max_val = _mm256_set1_epi16((1 << (clip - 1)) - 1);
    min_val = _mm256_set1_epi16(-(1 << (clip - 1)));

    for (k = 0; k < 2; k++) {
        in00[k] = _mm256_max_epi16(_mm256_min_epi16(in00[k], max_val), min_val);
        in01[k] = _mm256_max_epi16(_mm256_min_epi16(in01[k], max_val), min_val);
        in02[k] = _mm256_max_epi16(_mm256_min_epi16(in02[k], max_val), min_val);
        in03[k] = _mm256_max_epi16(_mm256_min_epi16(in03[k], max_val), min_val);
        in04[k] = _mm256_max_epi16(_mm256_min_epi16(in04[k], max_val), min_val);
        in05[k] = _mm256_max_epi16(_mm256_min_epi16(in05[k], max_val), min_val);
        in06[k] = _mm256_max_epi16(_mm256_min_epi16(in06[k], max_val), min_val);
        in07[k] = _mm256_max_epi16(_mm256_min_epi16(in07[k], max_val), min_val);
        in08[k] = _mm256_max_epi16(_mm256_min_epi16(in08[k], max_val), min_val);
        in09[k] = _mm256_max_epi16(_mm256_min_epi16(in09[k], max_val), min_val);
        in10[k] = _mm256_max_epi16(_mm256_min_epi16(in10[k], max_val), min_val);
        in11[k] = _mm256_max_epi16(_mm256_min_epi16(in11[k], max_val), min_val);
        in12[k] = _mm256_max_epi16(_mm256_min_epi16(in12[k], max_val), min_val);
        in13[k] = _mm256_max_epi16(_mm256_min_epi16(in13[k], max_val), min_val);
        in14[k] = _mm256_max_epi16(_mm256_min_epi16(in14[k], max_val), min_val);
        in15[k] = _mm256_max_epi16(_mm256_min_epi16(in15[k], max_val), min_val);
        in16[k] = _mm256_max_epi16(_mm256_min_epi16(in16[k], max_val), min_val);
        in17[k] = _mm256_max_epi16(_mm256_min_epi16(in17[k], max_val), min_val);
        in18[k] = _mm256_max_epi16(_mm256_min_epi16(in18[k], max_val), min_val);
        in19[k] = _mm256_max_epi16(_mm256_min_epi16(in19[k], max_val), min_val);
        in20[k] = _mm256_max_epi16(_mm256_min_epi16(in20[k], max_val), min_val);
        in21[k] = _mm256_max_epi16(_mm256_min_epi16(in21[k], max_val), min_val);
        in22[k] = _mm256_max_epi16(_mm256_min_epi16(in22[k], max_val), min_val);
        in23[k] = _mm256_max_epi16(_mm256_min_epi16(in23[k], max_val), min_val);
        in24[k] = _mm256_max_epi16(_mm256_min_epi16(in24[k], max_val), min_val);
        in25[k] = _mm256_max_epi16(_mm256_min_epi16(in25[k], max_val), min_val);
        in26[k] = _mm256_max_epi16(_mm256_min_epi16(in26[k], max_val), min_val);
        in27[k] = _mm256_max_epi16(_mm256_min_epi16(in27[k], max_val), min_val);
        in28[k] = _mm256_max_epi16(_mm256_min_epi16(in28[k], max_val), min_val);
        in29[k] = _mm256_max_epi16(_mm256_min_epi16(in29[k], max_val), min_val);
        in30[k] = _mm256_max_epi16(_mm256_min_epi16(in30[k], max_val), min_val);
        in31[k] = _mm256_max_epi16(_mm256_min_epi16(in31[k], max_val), min_val);
    }


    /* ---- 存储结果 (对齐存储: coeff 基地址 + 每半行起始均 32 字节对齐) ---- */
    for (i = 0; i < 2; i++) {
        const int offset = (i << 4);
        _mm256_store_si256((__m256i*)&coeff[0 * 32 + offset], in00[i]);
        _mm256_store_si256((__m256i*)&coeff[1 * 32 + offset], in01[i]);
        _mm256_store_si256((__m256i*)&coeff[2 * 32 + offset], in02[i]);
        _mm256_store_si256((__m256i*)&coeff[3 * 32 + offset], in03[i]);
        _mm256_store_si256((__m256i*)&coeff[4 * 32 + offset], in04[i]);
        _mm256_store_si256((__m256i*)&coeff[5 * 32 + offset], in05[i]);
        _mm256_store_si256((__m256i*)&coeff[6 * 32 + offset], in06[i]);
        _mm256_store_si256((__m256i*)&coeff[7 * 32 + offset], in07[i]);
        _mm256_store_si256((__m256i*)&coeff[8 * 32 + offset], in08[i]);
        _mm256_store_si256((__m256i*)&coeff[9 * 32 + offset], in09[i]);
        _mm256_store_si256((__m256i*)&coeff[10 * 32 + offset], in10[i]);
        _mm256_store_si256((__m256i*)&coeff[11 * 32 + offset], in11[i]);
        _mm256_store_si256((__m256i*)&coeff[12 * 32 + offset], in12[i]);
        _mm256_store_si256((__m256i*)&coeff[13 * 32 + offset], in13[i]);
        _mm256_store_si256((__m256i*)&coeff[14 * 32 + offset], in14[i]);
        _mm256_store_si256((__m256i*)&coeff[15 * 32 + offset], in15[i]);
        _mm256_store_si256((__m256i*)&coeff[16 * 32 + offset], in16[i]);
        _mm256_store_si256((__m256i*)&coeff[17 * 32 + offset], in17[i]);
        _mm256_store_si256((__m256i*)&coeff[18 * 32 + offset], in18[i]);
        _mm256_store_si256((__m256i*)&coeff[19 * 32 + offset], in19[i]);
        _mm256_store_si256((__m256i*)&coeff[20 * 32 + offset], in20[i]);
        _mm256_store_si256((__m256i*)&coeff[21 * 32 + offset], in21[i]);
        _mm256_store_si256((__m256i*)&coeff[22 * 32 + offset], in22[i]);
        _mm256_store_si256((__m256i*)&coeff[23 * 32 + offset], in23[i]);
        _mm256_store_si256((__m256i*)&coeff[24 * 32 + offset], in24[i]);
        _mm256_store_si256((__m256i*)&coeff[25 * 32 + offset], in25[i]);
        _mm256_store_si256((__m256i*)&coeff[26 * 32 + offset], in26[i]);
        _mm256_store_si256((__m256i*)&coeff[27 * 32 + offset], in27[i]);
        _mm256_store_si256((__m256i*)&coeff[28 * 32 + offset], in28[i]);
        _mm256_store_si256((__m256i*)&coeff[29 * 32 + offset], in29[i]);
        _mm256_store_si256((__m256i*)&coeff[30 * 32 + offset], in30[i]);
        _mm256_store_si256((__m256i*)&coeff[31 * 32 + offset], in31[i]);
    }

}

/* ===========================================================================
 * 残差叠加: dst[i] = clip(dst[i] + coeff[i], 0, max_val)
 * AVX2 实现: 一次处理 16 个 int16 像素
 * C fallback 在 itx.c 中实现并注册
 * ===========================================================================
 */

/* AVX2: 10-bit 路径 (uint16_t dst + int16_t coeff) */
static void recon_residual_avx2_10bit(uint8_t *dst, ptrdiff_t stride,
                                      const int16_t *coeff, int w, int h,
                                      int bit_depth)
{
    uint16_t *dst16 = (uint16_t *)(void *)dst;
    int stride16 = (int)(stride >> 1);
    int max_val = (1 << bit_depth) - 1;
    __m256i v_zero = _mm256_setzero_si256();
    __m256i v_max = _mm256_set1_epi16((short)max_val);
    /* 计算首行对齐偏移 (stride 64 字节对齐, 各行偏移相同) */
    int misalign = (int)((uintptr_t)&dst16[0] & 31) >> 1;  /* uint16_t 元素数 */
    int head = (16 - misalign) & 15;  /* 对齐到 32 字节边界需跳过的元素数 */
    int y;

    for (y = 0; y < h; y++) {
        int x = 0;
        /* 标量处理未对齐头部 */
        for (; x < head && x < w; x++) {
            int v = dst16[y * stride16 + x] + coeff[y * w + x];
            if (v < 0) v = 0;
            if (v > max_val) v = max_val;
            dst16[y * stride16 + x] = (uint16_t)v;
        }
        /* AVX2: 一次处理 16 个像素 (dst 已对齐) */
        for (; x + 15 < w; x += 16) {
            __m256i d = _mm256_load_si256((const __m256i *)&dst16[y * stride16 + x]);
            __m256i c = _mm256_load_si256((const __m256i *)&coeff[y * w + x]);
            __m256i r = _mm256_add_epi16(d, c);
            /* clip: 先 max(0) 去负值 (有符号比较), 再 min(max_val) (无符号比较) */
            r = _mm256_max_epi16(r, v_zero);
            r = _mm256_min_epu16(r, v_max);
            _mm256_store_si256((__m256i *)&dst16[y * stride16 + x], r);
        }
        /* 标量处理剩余像素 */
        for (; x < w; x++) {
            int v = dst16[y * stride16 + x] + coeff[y * w + x];
            if (v < 0) v = 0;
            if (v > max_val) v = max_val;
            dst16[y * stride16 + x] = (uint16_t)v;
        }
    }
    (void)bit_depth;
}

/* AVX2: 8-bit 路径 (uint8_t dst + int16_t coeff, 需扩展/打包) */
static void recon_residual_avx2_8bit(uint8_t *dst, ptrdiff_t stride,
                                     const int16_t *coeff, int w, int h,
                                     int bit_depth)
{
    int max_val = (1 << bit_depth) - 1;
    __m128i v_zero = _mm_setzero_si128();
    __m128i v_max = _mm_set1_epi8((char)max_val);
    /* 计算首行对齐偏移 (stride 64 字节对齐, 各行偏移相同) */
    int misalign = (int)((uintptr_t)&dst[0] & 15);  /* 字节数 (= uint8_t 元素数) */
    int head = (16 - misalign) & 15;  /* 对齐到 16 字节边界需跳过的元素数 */
    int y;

    for (y = 0; y < h; y++) {
        int x = 0;
        /* 标量处理未对齐头部 */
        for (; x < head && x < w; x++) {
            int v = dst[y * stride + x] + coeff[y * w + x];
            if (v < 0) v = 0;
            if (v > max_val) v = max_val;
            dst[y * stride + x] = (uint8_t)v;
        }
        /* SSE: 一次处理 16 个像素 (8-bit, dst 已对齐) */
        for (; x + 15 < w; x += 16) {
            __m128i d = _mm_load_si128((const __m128i *)&dst[y * stride + x]);
            /* 将 uint8 扩展为 int16 (低 8 + 高 8) */
            __m128i d_lo = _mm_unpacklo_epi8(d, v_zero);
            __m128i d_hi = _mm_unpackhi_epi8(d, v_zero);
            __m128i c_lo = _mm_load_si128((const __m128i *)&coeff[y * w + x]);
            __m128i c_hi = _mm_load_si128((const __m128i *)&coeff[y * w + x + 8]);
            __m128i r_lo = _mm_adds_epi16(d_lo, c_lo);
            __m128i r_hi = _mm_adds_epi16(d_hi, c_hi);
            /* saturated add 已处理下溢 (clip 到 0), 还需 clip 上界 */
            r_lo = _mm_min_epu8(r_lo, v_max);
            r_hi = _mm_min_epu8(r_hi, v_max);
            /* 打包回 uint8 */
            __m128i r = _mm_packus_epi16(r_lo, r_hi);
            _mm_store_si128((__m128i *)&dst[y * stride + x], r);
        }
        /* 标量处理剩余像素 */
        for (; x < w; x++) {
            int v = dst[y * stride + x] + coeff[y * w + x];
            if (v < 0) v = 0;
            if (v > max_val) v = max_val;
            dst[y * stride + x] = (uint8_t)v;
        }
    }
    (void)bit_depth;
}

/* AVX2 dispatcher */
static void recon_residual_avx2(uint8_t *dst, ptrdiff_t stride, const int16_t *coeff,
                                int w, int h, int bit_depth)
{
    if (bit_depth > 8) {
        recon_residual_avx2_10bit(dst, stride, coeff, w, h, bit_depth);
    } else {
        recon_residual_avx2_8bit(dst, stride, coeff, w, h, bit_depth);
    }
}

/* ===========================================================================
 * 注册函数
 * ===========================================================================
 */

/* SSE4.1: 注册 4x4 IDCT */
void avs2_itx_init_sse41(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.itx[0] = idct_4x4_sse41;
}

/* AVX2: 注册 8x8/16x16/32x32 IDCT (4x4 保持 SSE4.1 实现) + 残差叠加 */
void avs2_itx_init_avx2(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.itx[1] = idct_8x8_avx2;
    avs2_dsp_table.itx[2] = idct_16x16_avx2;
    avs2_dsp_table.itx[3] = idct_32x32_avx2;
    avs2_dsp_table.recon_residual = recon_residual_avx2;
}

/* 测试接口: 暴露内部 static 函数供 test_idct.c 调用 */
#ifdef AVS2_TEST_EXPOSE
void idct_8x8_avx2_test(int16_t *coeff, int w, int h, int bit_depth)
{
    idct_8x8_avx2(coeff, w, h, bit_depth);
}
void idct_16x16_avx2_test(int16_t *coeff, int w, int h, int bit_depth)
{
    idct_16x16_avx2(coeff, w, h, bit_depth);
}
void idct_32x32_avx2_test(int16_t *coeff, int w, int h, int bit_depth)
{
    idct_32x32_avx2(coeff, w, h, bit_depth);
}
void idct_4x4_sse41_test(int16_t *coeff, int w, int h, int bit_depth)
{
    idct_4x4_sse41(coeff, w, h, bit_depth);
}
#endif

/* AVX512: 预留接口, 暂用 AVX2 实现 */
void avs2_itx_init_avx512(const avs2_cpu_flags *flags)
{
    (void)flags;
    /* TODO: AVX512 实现可覆盖 8x8/16x16/32x32 */
}

#else /* !x86 */

/* 非 x86 平台: 空实现 (NEON 在单独的文件中) */
void avs2_itx_init_sse41(const avs2_cpu_flags *flags)  { (void)flags; }
void avs2_itx_init_avx2(const avs2_cpu_flags *flags)   { (void)flags; }
void avs2_itx_init_avx512(const avs2_cpu_flags *flags) { (void)flags; }

#endif /* x86 */
