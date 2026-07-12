/*
 * itx_simd.c - 反变换 SIMD 实现 (x86)
 *
 * 当前实现:
 *   - SSE4.1: 4x4/8x8/16x16/32x32 IDCT (从 libudavs2 intrinsic_dct.c 移植)
 *   - 64x64:  暂用 C 回退
 *
 * 预留: AVX512 接口 (暂空)
 *
 * 与 libudavs2 的差异:
 *   - libudavs2 使用 coeff_t (int16_t 别名), 本实现直接使用 int16_t
 *   - libudavs2 使用 ALIGNED_16/32 宏, 本实现自定义 AVS2_ALIGN16/32 宏
 *   - libudavs2 函数签名 (blk, shift, clip) 两遍就地变换,
 *     本实现通过 wrapper 适配 (coeff, w, h, bit_depth) 签名
 *
 * 就地变换: src == dst 时安全 (所有读取在写入之前完成)
 *
 * 对齐要求: avs2_mem_alloc/allocz 统一返回 32 字节对齐内存.
 *   - 4x4/8x8/16x16/32x32 SSE4.1: coeff 基地址 32 字节对齐,
 *     各行起始至少 16 字节对齐, 用 _mm_load/store_si128 (128-bit)
 *     系数表用 AVS2_ALIGN16 声明配合 _mm_load_si128
 */

#include "internal.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <tmmintrin.h>  /* SSSE3 */
#include <smmintrin.h>  /* SSE4.1 */

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
 * 8x8 反 DCT (SSE4.1)
 * 参考: libudavs2 intrinsic_dct.c idct_8x8_sse128
 *
 * 系数表 tab_idct_8x8_sse[12][8]:
 *   行 0-7:  奇数部分 O 层系数 (成对存放, 配合 _mm_madd_epi16)
 *   行 8-9:  偶数部分 EE 层系数
 *   行 10-11: 偶数部分 E0 层系数
 *
 * 两遍就地变换: 第一遍 shift=5 (硬编码), 第二遍 shift=参数传入
 * 最终裁剪到 clip 参数指定的范围
 * ===========================================================================
 */

AVS2_ALIGN16(static const int16_t tab_idct_8x8_sse[12][8]) =
{
    {  44,  38,  44,  38,  44,  38,  44,  38 },
    {  25,   9,  25,   9,  25,   9,  25,   9 },
    {  38,  -9,  38,  -9,  38,  -9,  38,  -9 },
    { -44, -25, -44, -25, -44, -25, -44, -25 },
    {  25, -44,  25, -44,  25, -44,  25, -44 },
    {   9,  38,   9,  38,   9,  38,   9,  38 },
    {   9, -25,   9, -25,   9, -25,   9, -25 },
    {  38, -44,  38, -44,  38, -44,  38, -44 },
    {  32,  32,  32,  32,  32,  32,  32,  32 },
    {  32, -32,  32, -32,  32, -32,  32, -32 },
    {  42,  17,  42,  17,  42,  17,  42,  17 },
    {  17, -42,  17, -42,  17, -42,  17, -42 }
};

static void idct_8x8_sse41_core(int16_t *blk, int shift, int clip)
{
    __m128i m128iS0, m128iS1, m128iS2, m128iS3, m128iS4, m128iS5, m128iS6, m128iS7, m128iAdd, m128Tmp0, m128Tmp1, m128Tmp2, m128Tmp3, E0h, E1h, E2h, E3h, E0l, E1l, E2l, E3l, O0h, O1h, O2h, O3h, O0l, O1l, O2l, O3l, EE0l, EE1l, E00l, E01l, EE0h, EE1h, E00h, E01h;
    __m128i T00, T01, T02, T03, T04, T05, T06, T07;

    m128iAdd = _mm_set1_epi32(16); /* */

    m128iS1 = _mm_load_si128((__m128i*)&blk[8]);
    m128iS3 = _mm_load_si128((__m128i*)&blk[24]);

    m128Tmp0 = _mm_unpacklo_epi16(m128iS1, m128iS3);
    E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[0])));
    m128Tmp1 = _mm_unpackhi_epi16(m128iS1, m128iS3);
    E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[0])));


    m128iS5 = _mm_load_si128((__m128i*)&blk[40]);
    m128iS7 = _mm_load_si128((__m128i*)&blk[56]);

    m128Tmp2 = _mm_unpacklo_epi16(m128iS5, m128iS7);
    E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[1])));
    m128Tmp3 = _mm_unpackhi_epi16(m128iS5, m128iS7);
    E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[1])));
    O0l = _mm_add_epi32(E1l, E2l);
    O0h = _mm_add_epi32(E1h, E2h);

    E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[2])));
    E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[2])));
    E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[3])));
    E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[3])));

    O1l = _mm_add_epi32(E1l, E2l);
    O1h = _mm_add_epi32(E1h, E2h);

    E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[4])));
    E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[4])));
    E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[5])));
    E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[5])));
    O2l = _mm_add_epi32(E1l, E2l);
    O2h = _mm_add_epi32(E1h, E2h);

    E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[6])));
    E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[6])));
    E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[7])));
    E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[7])));
    O3h = _mm_add_epi32(E1h, E2h);
    O3l = _mm_add_epi32(E1l, E2l);

    /*    -------     */


    m128iS0 = _mm_load_si128((__m128i*)&blk[0]);
    m128iS4 = _mm_load_si128((__m128i*)&blk[32]);

    m128Tmp0 = _mm_unpacklo_epi16(m128iS0, m128iS4);
    EE0l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[8])));
    m128Tmp1 = _mm_unpackhi_epi16(m128iS0, m128iS4);
    EE0h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[8])));

    EE1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[9])));
    EE1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[9])));

    /*    -------     */


    m128iS2 = _mm_load_si128((__m128i*)&blk[16]);
    m128iS6 = _mm_load_si128((__m128i*)&blk[48]);

    m128Tmp0 = _mm_unpacklo_epi16(m128iS2, m128iS6);
    E00l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[10])));
    m128Tmp1 = _mm_unpackhi_epi16(m128iS2, m128iS6);
    E00h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[10])));
    E01l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[11])));
    E01h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[11])));
    E0l = _mm_add_epi32(EE0l, E00l);
    E0l = _mm_add_epi32(E0l, m128iAdd);
    E0h = _mm_add_epi32(EE0h, E00h);
    E0h = _mm_add_epi32(E0h, m128iAdd);
    E3l = _mm_sub_epi32(EE0l, E00l);
    E3l = _mm_add_epi32(E3l, m128iAdd);
    E3h = _mm_sub_epi32(EE0h, E00h);
    E3h = _mm_add_epi32(E3h, m128iAdd);

    E1l = _mm_add_epi32(EE1l, E01l);
    E1l = _mm_add_epi32(E1l, m128iAdd);
    E1h = _mm_add_epi32(EE1h, E01h);
    E1h = _mm_add_epi32(E1h, m128iAdd);
    E2l = _mm_sub_epi32(EE1l, E01l);
    E2l = _mm_add_epi32(E2l, m128iAdd);
    E2h = _mm_sub_epi32(EE1h, E01h);
    E2h = _mm_add_epi32(E2h, m128iAdd);
    m128iS0 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E0l, O0l), 5), _mm_srai_epi32(_mm_add_epi32(E0h, O0h), 5)); /* */
    m128iS7 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E0l, O0l), 5), _mm_srai_epi32(_mm_sub_epi32(E0h, O0h), 5));
    m128iS1 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E1l, O1l), 5), _mm_srai_epi32(_mm_add_epi32(E1h, O1h), 5));
    m128iS6 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E1l, O1l), 5), _mm_srai_epi32(_mm_sub_epi32(E1h, O1h), 5));
    m128iS2 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E2l, O2l), 5), _mm_srai_epi32(_mm_add_epi32(E2h, O2h), 5));
    m128iS5 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E2l, O2l), 5), _mm_srai_epi32(_mm_sub_epi32(E2h, O2h), 5));
    m128iS3 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E3l, O3l), 5), _mm_srai_epi32(_mm_add_epi32(E3h, O3h), 5));
    m128iS4 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E3l, O3l), 5), _mm_srai_epi32(_mm_sub_epi32(E3h, O3h), 5));

    /*  Invers matrix   */

    E0l = _mm_unpacklo_epi16(m128iS0, m128iS4);
    E1l = _mm_unpacklo_epi16(m128iS1, m128iS5);
    E2l = _mm_unpacklo_epi16(m128iS2, m128iS6);
    E3l = _mm_unpacklo_epi16(m128iS3, m128iS7);
    O0l = _mm_unpackhi_epi16(m128iS0, m128iS4);
    O1l = _mm_unpackhi_epi16(m128iS1, m128iS5);
    O2l = _mm_unpackhi_epi16(m128iS2, m128iS6);
    O3l = _mm_unpackhi_epi16(m128iS3, m128iS7);
    m128Tmp0 = _mm_unpacklo_epi16(E0l, E2l);
    m128Tmp1 = _mm_unpacklo_epi16(E1l, E3l);
    m128iS0 = _mm_unpacklo_epi16(m128Tmp0, m128Tmp1);
    m128iS1 = _mm_unpackhi_epi16(m128Tmp0, m128Tmp1);
    m128Tmp2 = _mm_unpackhi_epi16(E0l, E2l);
    m128Tmp3 = _mm_unpackhi_epi16(E1l, E3l);
    m128iS2 = _mm_unpacklo_epi16(m128Tmp2, m128Tmp3);
    m128iS3 = _mm_unpackhi_epi16(m128Tmp2, m128Tmp3);
    m128Tmp0 = _mm_unpacklo_epi16(O0l, O2l);
    m128Tmp1 = _mm_unpacklo_epi16(O1l, O3l);
    m128iS4 = _mm_unpacklo_epi16(m128Tmp0, m128Tmp1);
    m128iS5 = _mm_unpackhi_epi16(m128Tmp0, m128Tmp1);
    m128Tmp2 = _mm_unpackhi_epi16(O0l, O2l);
    m128Tmp3 = _mm_unpackhi_epi16(O1l, O3l);
    m128iS6 = _mm_unpacklo_epi16(m128Tmp2, m128Tmp3);
    m128iS7 = _mm_unpackhi_epi16(m128Tmp2, m128Tmp3);

    m128iAdd = _mm_set1_epi32(shift ? (1 << (shift - 1)) : 0); /* */

    m128Tmp0 = _mm_unpacklo_epi16(m128iS1, m128iS3);
    E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[0])));
    m128Tmp1 = _mm_unpackhi_epi16(m128iS1, m128iS3);
    E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[0])));
    m128Tmp2 = _mm_unpacklo_epi16(m128iS5, m128iS7);
    E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[1])));
    m128Tmp3 = _mm_unpackhi_epi16(m128iS5, m128iS7);
    E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[1])));
    O0l = _mm_add_epi32(E1l, E2l);
    O0h = _mm_add_epi32(E1h, E2h);
    E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[2])));
    E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[2])));
    E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[3])));
    E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[3])));
    O1l = _mm_add_epi32(E1l, E2l);
    O1h = _mm_add_epi32(E1h, E2h);
    E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[4])));
    E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[4])));
    E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[5])));
    E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[5])));
    O2l = _mm_add_epi32(E1l, E2l);
    O2h = _mm_add_epi32(E1h, E2h);
    E1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[6])));
    E1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[6])));
    E2l = _mm_madd_epi16(m128Tmp2, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[7])));
    E2h = _mm_madd_epi16(m128Tmp3, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[7])));
    O3h = _mm_add_epi32(E1h, E2h);
    O3l = _mm_add_epi32(E1l, E2l);

    m128Tmp0 = _mm_unpacklo_epi16(m128iS0, m128iS4);
    EE0l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[8])));
    m128Tmp1 = _mm_unpackhi_epi16(m128iS0, m128iS4);
    EE0h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[8])));
    EE1l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[9])));
    EE1h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[9])));

    m128Tmp0 = _mm_unpacklo_epi16(m128iS2, m128iS6);
    E00l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[10])));
    m128Tmp1 = _mm_unpackhi_epi16(m128iS2, m128iS6);
    E00h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[10])));
    E01l = _mm_madd_epi16(m128Tmp0, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[11])));
    E01h = _mm_madd_epi16(m128Tmp1, _mm_load_si128((__m128i*)(tab_idct_8x8_sse[11])));
    E0l = _mm_add_epi32(EE0l, E00l);
    E0l = _mm_add_epi32(E0l, m128iAdd);
    E0h = _mm_add_epi32(EE0h, E00h);
    E0h = _mm_add_epi32(E0h, m128iAdd);
    E3l = _mm_sub_epi32(EE0l, E00l);
    E3l = _mm_add_epi32(E3l, m128iAdd);
    E3h = _mm_sub_epi32(EE0h, E00h);
    E3h = _mm_add_epi32(E3h, m128iAdd);
    E1l = _mm_add_epi32(EE1l, E01l);
    E1l = _mm_add_epi32(E1l, m128iAdd);
    E1h = _mm_add_epi32(EE1h, E01h);
    E1h = _mm_add_epi32(E1h, m128iAdd);
    E2l = _mm_sub_epi32(EE1l, E01l);
    E2l = _mm_add_epi32(E2l, m128iAdd);
    E2h = _mm_sub_epi32(EE1h, E01h);
    E2h = _mm_add_epi32(E2h, m128iAdd);

    m128iS0 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E0l, O0l), shift), _mm_srai_epi32(_mm_add_epi32(E0h, O0h), shift));
    m128iS7 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E0l, O0l), shift), _mm_srai_epi32(_mm_sub_epi32(E0h, O0h), shift));
    m128iS1 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E1l, O1l), shift), _mm_srai_epi32(_mm_add_epi32(E1h, O1h), shift));
    m128iS6 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E1l, O1l), shift), _mm_srai_epi32(_mm_sub_epi32(E1h, O1h), shift));
    m128iS2 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E2l, O2l), shift), _mm_srai_epi32(_mm_add_epi32(E2h, O2h), shift));
    m128iS5 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E2l, O2l), shift), _mm_srai_epi32(_mm_sub_epi32(E2h, O2h), shift));
    m128iS3 = _mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(E3l, O3l), shift), _mm_srai_epi32(_mm_add_epi32(E3h, O3h), shift));
    m128iS4 = _mm_packs_epi32(_mm_srai_epi32(_mm_sub_epi32(E3l, O3l), shift), _mm_srai_epi32(_mm_sub_epi32(E3h, O3h), shift));




    // [07 06 05 04 03 02 01 00]
    // [17 16 15 14 13 12 11 10]
    // [27 26 25 24 23 22 21 20]
    // [37 36 35 34 33 32 31 30]
    // [47 46 45 44 43 42 41 40]
    // [57 56 55 54 53 52 51 50]
    // [67 66 65 64 63 62 61 60]
    // [77 76 75 74 73 72 71 70]

    T00 = _mm_unpacklo_epi16(m128iS0, m128iS1);     // [13 03 12 02 11 01 10 00]
    T01 = _mm_unpackhi_epi16(m128iS0, m128iS1);     // [17 07 16 06 15 05 14 04]
    T02 = _mm_unpacklo_epi16(m128iS2, m128iS3);     // [33 23 32 22 31 21 30 20]
    T03 = _mm_unpackhi_epi16(m128iS2, m128iS3);     // [37 27 36 26 35 25 34 24]
    T04 = _mm_unpacklo_epi16(m128iS4, m128iS5);     // [53 43 52 42 51 41 50 40]
    T05 = _mm_unpackhi_epi16(m128iS4, m128iS5);     // [57 47 56 46 55 45 54 44]
    T06 = _mm_unpacklo_epi16(m128iS6, m128iS7);     // [73 63 72 62 71 61 70 60]
    T07 = _mm_unpackhi_epi16(m128iS6, m128iS7);     // [77 67 76 66 75 65 74 64]

    //clip
    {
        __m128i max_val = _mm_set1_epi16((1 << (clip - 1)) - 1);
        __m128i min_val = _mm_set1_epi16(-(1 << (clip - 1)));
        T00 = _mm_min_epi16(T00, max_val);
        T00 = _mm_max_epi16(T00, min_val);
        T01 = _mm_min_epi16(T01, max_val);
        T01 = _mm_max_epi16(T01, min_val);
        T02 = _mm_min_epi16(T02, max_val);
        T02 = _mm_max_epi16(T02, min_val);
        T03 = _mm_min_epi16(T03, max_val);
        T03 = _mm_max_epi16(T03, min_val);
        T04 = _mm_min_epi16(T04, max_val);
        T04 = _mm_max_epi16(T04, min_val);
        T05 = _mm_min_epi16(T05, max_val);
        T05 = _mm_max_epi16(T05, min_val);
        T06 = _mm_min_epi16(T06, max_val);
        T06 = _mm_max_epi16(T06, min_val);
        T07 = _mm_min_epi16(T07, max_val);
        T07 = _mm_max_epi16(T07, min_val);
    }

    {
        __m128i T10, T11;
        T10 = _mm_unpacklo_epi32(T00, T02);                                     // [31 21 11 01 30 20 10 00]
        T11 = _mm_unpackhi_epi32(T00, T02);                                     // [33 23 13 03 32 22 12 02]
        _mm_storel_epi64((__m128i*)&blk[0 * 8 + 0], T10);                   // [30 20 10 00]
        _mm_storeh_pi((__m64*)&blk[1 * 8 + 0], _mm_castsi128_ps(T10));  // [31 21 11 01]
        _mm_storel_epi64((__m128i*)&blk[2 * 8 + 0], T11);                   // [32 22 12 02]
        _mm_storeh_pi((__m64*)&blk[3 * 8 + 0], _mm_castsi128_ps(T11));  // [33 23 13 03]

        T10 = _mm_unpacklo_epi32(T04, T06);                                     // [71 61 51 41 70 60 50 40]
        T11 = _mm_unpackhi_epi32(T04, T06);                                     // [73 63 53 43 72 62 52 42]
        _mm_storel_epi64((__m128i*)&blk[0 * 8 + 4], T10);
        _mm_storeh_pi((__m64*)&blk[1 * 8 + 4], _mm_castsi128_ps(T10));
        _mm_storel_epi64((__m128i*)&blk[2 * 8 + 4], T11);
        _mm_storeh_pi((__m64*)&blk[3 * 8 + 4], _mm_castsi128_ps(T11));

        T10 = _mm_unpacklo_epi32(T01, T03);                                     // [35 25 15 05 34 24 14 04]
        T11 = _mm_unpackhi_epi32(T01, T03);                                     // [37 27 17 07 36 26 16 06]
        _mm_storel_epi64((__m128i*)&blk[4 * 8 + 0], T10);
        _mm_storeh_pi((__m64*)&blk[5 * 8 + 0], _mm_castsi128_ps(T10));
        _mm_storel_epi64((__m128i*)&blk[6 * 8 + 0], T11);
        _mm_storeh_pi((__m64*)&blk[7 * 8 + 0], _mm_castsi128_ps(T11));

        T10 = _mm_unpacklo_epi32(T05, T07);                                     // [75 65 55 45 74 64 54 44]
        T11 = _mm_unpackhi_epi32(T05, T07);                                     // [77 67 57 47 76 56 46 36]
        _mm_storel_epi64((__m128i*)&blk[4 * 8 + 4], T10);
        _mm_storeh_pi((__m64*)&blk[5 * 8 + 4], _mm_castsi128_ps(T10));
        _mm_storel_epi64((__m128i*)&blk[6 * 8 + 4], T11);
        _mm_storeh_pi((__m64*)&blk[7 * 8 + 4], _mm_castsi128_ps(T11));
    }
}

/* wrapper: 适配 (coeff, w, h, bit_depth) 签名 */
static void idct_8x8_sse41(int16_t *coeff, int w, int h, int bit_depth)
{
    (void)w; (void)h;
    idct_8x8_sse41_core(coeff, 20 - bit_depth, bit_depth + 1);
}

/* ===========================================================================
 * 16x16 反 DCT (SSE4.1)
 * 参考: libudavs2 intrinsic_dct.c idct_16x16_sse128
 *
 * 两遍就地变换: 第一遍 shift=5 (硬编码), 第二遍 shift=参数传入
 * 最终裁剪到 clip 参数指定的范围
 * ===========================================================================
 */

static void idct_16x16_sse41_core(int16_t *blk, int shift, int clip)
{
    const __m128i c16_p43_p45 = _mm_set1_epi32(0x002B002D);		//row0 87high - 90low address
    const __m128i c16_p35_p40 = _mm_set1_epi32(0x00230028);
    const __m128i c16_p21_p29 = _mm_set1_epi32(0x0015001D);
    const __m128i c16_p04_p13 = _mm_set1_epi32(0x0004000D);
    const __m128i c16_p29_p43 = _mm_set1_epi32(0x001D002B);		//row1
    const __m128i c16_n21_p04 = _mm_set1_epi32(0xFFEB0004);
    const __m128i c16_n45_n40 = _mm_set1_epi32(0xFFD3FFD8);
    const __m128i c16_n13_n35 = _mm_set1_epi32(0xFFF3FFDD);
    const __m128i c16_p04_p40 = _mm_set1_epi32(0x00040028);		//row2
    const __m128i c16_n43_n35 = _mm_set1_epi32(0xFFD5FFDD);
    const __m128i c16_p29_n13 = _mm_set1_epi32(0x001DFFF3);
    const __m128i c16_p21_p45 = _mm_set1_epi32(0x0015002D);
    const __m128i c16_n21_p35 = _mm_set1_epi32(0xFFEB0023);		//row3
    const __m128i c16_p04_n43 = _mm_set1_epi32(0x0004FFD5);
    const __m128i c16_p13_p45 = _mm_set1_epi32(0x000D002D);
    const __m128i c16_n29_n40 = _mm_set1_epi32(0xFFE3FFD8);
    const __m128i c16_n40_p29 = _mm_set1_epi32(0xFFD8001D);		//row4
    const __m128i c16_p45_n13 = _mm_set1_epi32(0x002DFFF3);
    const __m128i c16_n43_n04 = _mm_set1_epi32(0xFFD5FFFC);
    const __m128i c16_p35_p21 = _mm_set1_epi32(0x00230015);
    const __m128i c16_n45_p21 = _mm_set1_epi32(0xFFD30015);		//row5
    const __m128i c16_p13_p29 = _mm_set1_epi32(0x000D001D);
    const __m128i c16_p35_n43 = _mm_set1_epi32(0x0023FFD5);
    const __m128i c16_n40_p04 = _mm_set1_epi32(0xFFD80004);
    const __m128i c16_n35_p13 = _mm_set1_epi32(0xFFDD000D);		//row6
    const __m128i c16_n40_p45 = _mm_set1_epi32(0xFFD8002D);
    const __m128i c16_p04_p21 = _mm_set1_epi32(0x00040015);
    const __m128i c16_p43_n29 = _mm_set1_epi32(0x002BFFE3);
    const __m128i c16_n13_p04 = _mm_set1_epi32(0xFFF30004);		//row7
    const __m128i c16_n29_p21 = _mm_set1_epi32(0xFFE30015);
    const __m128i c16_n40_p35 = _mm_set1_epi32(0xFFD80023);
    const __m128i c16_n45_p43 = _mm_set1_epi32(0xFFD3002B);

    const __m128i c16_p38_p44 = _mm_set1_epi32(0x0026002C);
    const __m128i c16_p09_p25 = _mm_set1_epi32(0x00090019);
    const __m128i c16_n09_p38 = _mm_set1_epi32(0xFFF70026);
    const __m128i c16_n25_n44 = _mm_set1_epi32(0xFFE7FFD4);
    const __m128i c16_n44_p25 = _mm_set1_epi32(0xFFD40019);
    const __m128i c16_p38_p09 = _mm_set1_epi32(0x00260009);
    const __m128i c16_n25_p09 = _mm_set1_epi32(0xFFE70009);
    const __m128i c16_n44_p38 = _mm_set1_epi32(0xFFD40026);

    const __m128i c16_p17_p42 = _mm_set1_epi32(0x0011002A);
    const __m128i c16_n42_p17 = _mm_set1_epi32(0xFFD60011);

    const __m128i c16_n32_p32 = _mm_set1_epi32(0xFFE00020);
    const __m128i c16_p32_p32 = _mm_set1_epi32(0x00200020);

    int i, pass, part;

    __m128i c32_rnd = _mm_set1_epi32(16); /* */

    int nShift = 5;

    // DCT1
    __m128i in00[2], in01[2], in02[2], in03[2], in04[2], in05[2], in06[2], in07[2];
    __m128i in08[2], in09[2], in10[2], in11[2], in12[2], in13[2], in14[2], in15[2];
    __m128i res00[2], res01[2], res02[2], res03[2], res04[2], res05[2], res06[2], res07[2];
    __m128i res08[2], res09[2], res10[2], res11[2], res12[2], res13[2], res14[2], res15[2];

    for (i = 0; i < 2; i++)
    {
        const int offset = (i << 3);

        in00[i] = _mm_load_si128((const __m128i*)&blk[0 * 16 + offset]);	// [07 06 05 04 03 02 01 00]
        in01[i] = _mm_load_si128((const __m128i*)&blk[1 * 16 + offset]);  // [17 16 15 14 13 12 11 10]
        in02[i] = _mm_load_si128((const __m128i*)&blk[2 * 16 + offset]); // [27 26 25 24 23 22 21 20]
        in03[i] = _mm_load_si128((const __m128i*)&blk[3 * 16 + offset]);     // [37 36 35 34 33 32 31 30]
        in04[i] = _mm_load_si128((const __m128i*)&blk[4 * 16 + offset]);   // [47 46 45 44 43 42 41 40]
        in05[i] = _mm_load_si128((const __m128i*)&blk[5 * 16 + offset]);     // [57 56 55 54 53 52 51 50]
        in06[i] = _mm_load_si128((const __m128i*)&blk[6 * 16 + offset]);    // [67 66 65 64 63 62 61 60]
        in07[i] = _mm_load_si128((const __m128i*)&blk[7 * 16 + offset]);    // [77 76 75 74 73 72 71 70]
        in08[i] = _mm_load_si128((const __m128i*)&blk[8 * 16 + offset]);
        in09[i] = _mm_load_si128((const __m128i*)&blk[9 * 16 + offset]);
        in10[i] = _mm_load_si128((const __m128i*)&blk[10 * 16 + offset]);
        in11[i] = _mm_load_si128((const __m128i*)&blk[11 * 16 + offset]);
        in12[i] = _mm_load_si128((const __m128i*)&blk[12 * 16 + offset]);
        in13[i] = _mm_load_si128((const __m128i*)&blk[13 * 16 + offset]);
        in14[i] = _mm_load_si128((const __m128i*)&blk[14 * 16 + offset]);
        in15[i] = _mm_load_si128((const __m128i*)&blk[15 * 16 + offset]);
    }

    for (pass = 0; pass < 2; pass++)
    {
        if (pass == 1)
        {
            c32_rnd = _mm_set1_epi32(shift ? (1 << (shift - 1)) : 0); /* */
            nShift = shift;
        }

        for (part = 0; part < 2; part++)
        {
            const __m128i T_00_00A = _mm_unpacklo_epi16(in01[part], in03[part]);       // [33 13 32 12 31 11 30 10]
            const __m128i T_00_00B = _mm_unpackhi_epi16(in01[part], in03[part]);       // [37 17 36 16 35 15 34 14]
            const __m128i T_00_01A = _mm_unpacklo_epi16(in05[part], in07[part]);       // [ ]
            const __m128i T_00_01B = _mm_unpackhi_epi16(in05[part], in07[part]);       // [ ]
            const __m128i T_00_02A = _mm_unpacklo_epi16(in09[part], in11[part]);       // [ ]
            const __m128i T_00_02B = _mm_unpackhi_epi16(in09[part], in11[part]);       // [ ]
            const __m128i T_00_03A = _mm_unpacklo_epi16(in13[part], in15[part]);       // [ ]
            const __m128i T_00_03B = _mm_unpackhi_epi16(in13[part], in15[part]);       // [ ]
            const __m128i T_00_04A = _mm_unpacklo_epi16(in02[part], in06[part]);       // [ ]
            const __m128i T_00_04B = _mm_unpackhi_epi16(in02[part], in06[part]);       // [ ]
            const __m128i T_00_05A = _mm_unpacklo_epi16(in10[part], in14[part]);       // [ ]
            const __m128i T_00_05B = _mm_unpackhi_epi16(in10[part], in14[part]);       // [ ]
            const __m128i T_00_06A = _mm_unpacklo_epi16(in04[part], in12[part]);       // [ ]row
            const __m128i T_00_06B = _mm_unpackhi_epi16(in04[part], in12[part]);       // [ ]
            const __m128i T_00_07A = _mm_unpacklo_epi16(in00[part], in08[part]);       // [83 03 82 02 81 01 81 00] row08 row00
            const __m128i T_00_07B = _mm_unpackhi_epi16(in00[part], in08[part]);       // [87 07 86 06 85 05 84 04]

            __m128i O0A, O1A, O2A, O3A, O4A, O5A, O6A, O7A;
            __m128i O0B, O1B, O2B, O3B, O4B, O5B, O6B, O7B;
            __m128i EO0A, EO1A, EO2A, EO3A;
            __m128i EO0B, EO1B, EO2B, EO3B;
            __m128i EEO0A, EEO1A;
            __m128i EEO0B, EEO1B;
            __m128i EEE0A, EEE1A;
            __m128i EEE0B, EEE1B;
                __m128i T00, T01;
#define COMPUTE_ROW(row0103, row0507, row0911, row1315, c0103, c0507, c0911, c1315, row) \
    T00 = _mm_add_epi32(_mm_madd_epi16(row0103, c0103), _mm_madd_epi16(row0507, c0507)); \
    T01 = _mm_add_epi32(_mm_madd_epi16(row0911, c0911), _mm_madd_epi16(row1315, c1315)); \
    row = _mm_add_epi32(T00, T01);

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
            

            
            EO0A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_p38_p44), _mm_madd_epi16(T_00_05A, c16_p09_p25)); // EO0
            EO0B = _mm_add_epi32(_mm_madd_epi16(T_00_04B, c16_p38_p44), _mm_madd_epi16(T_00_05B, c16_p09_p25));
            EO1A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_n09_p38), _mm_madd_epi16(T_00_05A, c16_n25_n44)); // EO1
            EO1B = _mm_add_epi32(_mm_madd_epi16(T_00_04B, c16_n09_p38), _mm_madd_epi16(T_00_05B, c16_n25_n44));
            EO2A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_n44_p25), _mm_madd_epi16(T_00_05A, c16_p38_p09)); // EO2
            EO2B = _mm_add_epi32(_mm_madd_epi16(T_00_04B, c16_n44_p25), _mm_madd_epi16(T_00_05B, c16_p38_p09));
            EO3A = _mm_add_epi32(_mm_madd_epi16(T_00_04A, c16_n25_p09), _mm_madd_epi16(T_00_05A, c16_n44_p38)); // EO3
            EO3B = _mm_add_epi32(_mm_madd_epi16(T_00_04B, c16_n25_p09), _mm_madd_epi16(T_00_05B, c16_n44_p38));

            
            EEO0A = _mm_madd_epi16(T_00_06A, c16_p17_p42);
            EEO0B = _mm_madd_epi16(T_00_06B, c16_p17_p42);
            EEO1A = _mm_madd_epi16(T_00_06A, c16_n42_p17);
            EEO1B = _mm_madd_epi16(T_00_06B, c16_n42_p17);

            
            EEE0A = _mm_madd_epi16(T_00_07A, c16_p32_p32);
            EEE0B = _mm_madd_epi16(T_00_07B, c16_p32_p32);
            EEE1A = _mm_madd_epi16(T_00_07A, c16_n32_p32);
            EEE1B = _mm_madd_epi16(T_00_07B, c16_n32_p32);

            {
                const __m128i EE0A = _mm_add_epi32(EEE0A, EEO0A);          // EE0 = EEE0 + EEO0
                const __m128i EE0B = _mm_add_epi32(EEE0B, EEO0B);
                const __m128i EE1A = _mm_add_epi32(EEE1A, EEO1A);          // EE1 = EEE1 + EEO1
                const __m128i EE1B = _mm_add_epi32(EEE1B, EEO1B);
                const __m128i EE3A = _mm_sub_epi32(EEE0A, EEO0A);          // EE2 = EEE0 - EEO0
                const __m128i EE3B = _mm_sub_epi32(EEE0B, EEO0B);
                const __m128i EE2A = _mm_sub_epi32(EEE1A, EEO1A);          // EE3 = EEE1 - EEO1
                const __m128i EE2B = _mm_sub_epi32(EEE1B, EEO1B);

                const __m128i E0A = _mm_add_epi32(EE0A, EO0A);          // E0 = EE0 + EO0
                const __m128i E0B = _mm_add_epi32(EE0B, EO0B);
                const __m128i E1A = _mm_add_epi32(EE1A, EO1A);          // E1 = EE1 + EO1
                const __m128i E1B = _mm_add_epi32(EE1B, EO1B);
                const __m128i E2A = _mm_add_epi32(EE2A, EO2A);          // E2 = EE2 + EO2
                const __m128i E2B = _mm_add_epi32(EE2B, EO2B);
                const __m128i E3A = _mm_add_epi32(EE3A, EO3A);          // E3 = EE3 + EO3
                const __m128i E3B = _mm_add_epi32(EE3B, EO3B);
                const __m128i E7A = _mm_sub_epi32(EE0A, EO0A);          // E0 = EE0 - EO0
                const __m128i E7B = _mm_sub_epi32(EE0B, EO0B);
                const __m128i E6A = _mm_sub_epi32(EE1A, EO1A);          // E1 = EE1 - EO1
                const __m128i E6B = _mm_sub_epi32(EE1B, EO1B);
                const __m128i E5A = _mm_sub_epi32(EE2A, EO2A);          // E2 = EE2 - EO2
                const __m128i E5B = _mm_sub_epi32(EE2B, EO2B);
                const __m128i E4A = _mm_sub_epi32(EE3A, EO3A);          // E3 = EE3 - EO3
                const __m128i E4B = _mm_sub_epi32(EE3B, EO3B);

                const __m128i T10A = _mm_add_epi32(E0A, c32_rnd);         // E0 + rnd
                const __m128i T10B = _mm_add_epi32(E0B, c32_rnd);
                const __m128i T11A = _mm_add_epi32(E1A, c32_rnd);         // E1 + rnd
                const __m128i T11B = _mm_add_epi32(E1B, c32_rnd);
                const __m128i T12A = _mm_add_epi32(E2A, c32_rnd);         // E2 + rnd
                const __m128i T12B = _mm_add_epi32(E2B, c32_rnd);
                const __m128i T13A = _mm_add_epi32(E3A, c32_rnd);         // E3 + rnd
                const __m128i T13B = _mm_add_epi32(E3B, c32_rnd);
                const __m128i T14A = _mm_add_epi32(E4A, c32_rnd);         // E4 + rnd
                const __m128i T14B = _mm_add_epi32(E4B, c32_rnd);
                const __m128i T15A = _mm_add_epi32(E5A, c32_rnd);         // E5 + rnd
                const __m128i T15B = _mm_add_epi32(E5B, c32_rnd);
                const __m128i T16A = _mm_add_epi32(E6A, c32_rnd);         // E6 + rnd
                const __m128i T16B = _mm_add_epi32(E6B, c32_rnd);
                const __m128i T17A = _mm_add_epi32(E7A, c32_rnd);         // E7 + rnd
                const __m128i T17B = _mm_add_epi32(E7B, c32_rnd);

                const __m128i T20A = _mm_add_epi32(T10A, O0A);          // E0 + O0 + rnd
                const __m128i T20B = _mm_add_epi32(T10B, O0B);
                const __m128i T21A = _mm_add_epi32(T11A, O1A);          // E1 + O1 + rnd
                const __m128i T21B = _mm_add_epi32(T11B, O1B);
                const __m128i T22A = _mm_add_epi32(T12A, O2A);          // E2 + O2 + rnd
                const __m128i T22B = _mm_add_epi32(T12B, O2B);
                const __m128i T23A = _mm_add_epi32(T13A, O3A);          // E3 + O3 + rnd
                const __m128i T23B = _mm_add_epi32(T13B, O3B);
                const __m128i T24A = _mm_add_epi32(T14A, O4A);          // E4
                const __m128i T24B = _mm_add_epi32(T14B, O4B);
                const __m128i T25A = _mm_add_epi32(T15A, O5A);          // E5
                const __m128i T25B = _mm_add_epi32(T15B, O5B);
                const __m128i T26A = _mm_add_epi32(T16A, O6A);          // E6
                const __m128i T26B = _mm_add_epi32(T16B, O6B);
                const __m128i T27A = _mm_add_epi32(T17A, O7A);          // E7
                const __m128i T27B = _mm_add_epi32(T17B, O7B);
                const __m128i T2FA = _mm_sub_epi32(T10A, O0A);          // E0 - O0 + rnd
                const __m128i T2FB = _mm_sub_epi32(T10B, O0B);
                const __m128i T2EA = _mm_sub_epi32(T11A, O1A);          // E1 - O1 + rnd
                const __m128i T2EB = _mm_sub_epi32(T11B, O1B);
                const __m128i T2DA = _mm_sub_epi32(T12A, O2A);          // E2 - O2 + rnd
                const __m128i T2DB = _mm_sub_epi32(T12B, O2B);
                const __m128i T2CA = _mm_sub_epi32(T13A, O3A);          // E3 - O3 + rnd
                const __m128i T2CB = _mm_sub_epi32(T13B, O3B);
                const __m128i T2BA = _mm_sub_epi32(T14A, O4A);          // E4
                const __m128i T2BB = _mm_sub_epi32(T14B, O4B);
                const __m128i T2AA = _mm_sub_epi32(T15A, O5A);          // E5
                const __m128i T2AB = _mm_sub_epi32(T15B, O5B);
                const __m128i T29A = _mm_sub_epi32(T16A, O6A);          // E6
                const __m128i T29B = _mm_sub_epi32(T16B, O6B);
                const __m128i T28A = _mm_sub_epi32(T17A, O7A);          // E7
                const __m128i T28B = _mm_sub_epi32(T17B, O7B);

                const __m128i T30A = _mm_srai_epi32(T20A, nShift);             // [30 20 10 00]
                const __m128i T30B = _mm_srai_epi32(T20B, nShift);             // [70 60 50 40]
                const __m128i T31A = _mm_srai_epi32(T21A, nShift);             // [31 21 11 01]
                const __m128i T31B = _mm_srai_epi32(T21B, nShift);             // [71 61 51 41]
                const __m128i T32A = _mm_srai_epi32(T22A, nShift);             // [32 22 12 02]
                const __m128i T32B = _mm_srai_epi32(T22B, nShift);             // [72 62 52 42]
                const __m128i T33A = _mm_srai_epi32(T23A, nShift);             // [33 23 13 03]
                const __m128i T33B = _mm_srai_epi32(T23B, nShift);             // [73 63 53 43]
                const __m128i T34A = _mm_srai_epi32(T24A, nShift);             // [33 24 14 04]
                const __m128i T34B = _mm_srai_epi32(T24B, nShift);             // [74 64 54 44]
                const __m128i T35A = _mm_srai_epi32(T25A, nShift);             // [35 25 15 05]
                const __m128i T35B = _mm_srai_epi32(T25B, nShift);             // [75 65 55 45]
                const __m128i T36A = _mm_srai_epi32(T26A, nShift);             // [36 26 16 06]
                const __m128i T36B = _mm_srai_epi32(T26B, nShift);             // [76 66 56 46]
                const __m128i T37A = _mm_srai_epi32(T27A, nShift);             // [37 27 17 07]
                const __m128i T37B = _mm_srai_epi32(T27B, nShift);             // [77 67 57 47]

                const __m128i T38A = _mm_srai_epi32(T28A, nShift);             // [30 20 10 00] x8
                const __m128i T38B = _mm_srai_epi32(T28B, nShift);             // [70 60 50 40]
                const __m128i T39A = _mm_srai_epi32(T29A, nShift);             // [31 21 11 01] x9
                const __m128i T39B = _mm_srai_epi32(T29B, nShift);             // [71 61 51 41]
                const __m128i T3AA = _mm_srai_epi32(T2AA, nShift);             // [32 22 12 02] xA
                const __m128i T3AB = _mm_srai_epi32(T2AB, nShift);             // [72 62 52 42]
                const __m128i T3BA = _mm_srai_epi32(T2BA, nShift);             // [33 23 13 03] xB
                const __m128i T3BB = _mm_srai_epi32(T2BB, nShift);             // [73 63 53 43]
                const __m128i T3CA = _mm_srai_epi32(T2CA, nShift);             // [33 24 14 04] xC
                const __m128i T3CB = _mm_srai_epi32(T2CB, nShift);             // [74 64 54 44]
                const __m128i T3DA = _mm_srai_epi32(T2DA, nShift);             // [35 25 15 05] xD
                const __m128i T3DB = _mm_srai_epi32(T2DB, nShift);             // [75 65 55 45]
                const __m128i T3EA = _mm_srai_epi32(T2EA, nShift);             // [36 26 16 06] xE
                const __m128i T3EB = _mm_srai_epi32(T2EB, nShift);             // [76 66 56 46]
                const __m128i T3FA = _mm_srai_epi32(T2FA, nShift);             // [37 27 17 07] xF
                const __m128i T3FB = _mm_srai_epi32(T2FB, nShift);             // [77 67 57 47]

                res00[part] = _mm_packs_epi32(T30A, T30B);        // [70 60 50 40 30 20 10 00]
                res01[part] = _mm_packs_epi32(T31A, T31B);        // [71 61 51 41 31 21 11 01]
                res02[part] = _mm_packs_epi32(T32A, T32B);        // [72 62 52 42 32 22 12 02]
                res03[part] = _mm_packs_epi32(T33A, T33B);        // [73 63 53 43 33 23 13 03]
                res04[part] = _mm_packs_epi32(T34A, T34B);        // [74 64 54 44 34 24 14 04]
                res05[part] = _mm_packs_epi32(T35A, T35B);        // [75 65 55 45 35 25 15 05]
                res06[part] = _mm_packs_epi32(T36A, T36B);        // [76 66 56 46 36 26 16 06]
                res07[part] = _mm_packs_epi32(T37A, T37B);        // [77 67 57 47 37 27 17 07]

                res08[part] = _mm_packs_epi32(T38A, T38B);        // [A0 ... 80]
                res09[part] = _mm_packs_epi32(T39A, T39B);        // [A1 ... 81]
                res10[part] = _mm_packs_epi32(T3AA, T3AB);        // [A2 ... 82]
                res11[part] = _mm_packs_epi32(T3BA, T3BB);        // [A3 ... 83]
                res12[part] = _mm_packs_epi32(T3CA, T3CB);        // [A4 ... 84]
                res13[part] = _mm_packs_epi32(T3DA, T3DB);        // [A5 ... 85]
                res14[part] = _mm_packs_epi32(T3EA, T3EB);        // [A6 ... 86]
                res15[part] = _mm_packs_epi32(T3FA, T3FB);        // [A7 ... 87]
            }
        }
        //transpose matrix 8x8 16bit.
        {
            __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
            __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;
#define TRANSPOSE_8x8_16BIT(I0, I1, I2, I3, I4, I5, I6, I7, O0, O1, O2, O3, O4, O5, O6, O7) \
    tr0_0 = _mm_unpacklo_epi16(I0, I1); \
    tr0_1 = _mm_unpacklo_epi16(I2, I3); \
    tr0_2 = _mm_unpackhi_epi16(I0, I1); \
    tr0_3 = _mm_unpackhi_epi16(I2, I3); \
    tr0_4 = _mm_unpacklo_epi16(I4, I5); \
    tr0_5 = _mm_unpacklo_epi16(I6, I7); \
    tr0_6 = _mm_unpackhi_epi16(I4, I5); \
    tr0_7 = _mm_unpackhi_epi16(I6, I7); \
    tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1); \
    tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3); \
    tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1); \
    tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3); \
    tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5); \
    tr1_5 = _mm_unpacklo_epi32(tr0_6, tr0_7); \
    tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5); \
    tr1_7 = _mm_unpackhi_epi32(tr0_6, tr0_7); \
    O0 = _mm_unpacklo_epi64(tr1_0, tr1_4); \
    O1 = _mm_unpackhi_epi64(tr1_0, tr1_4); \
    O2 = _mm_unpacklo_epi64(tr1_2, tr1_6); \
    O3 = _mm_unpackhi_epi64(tr1_2, tr1_6); \
    O4 = _mm_unpacklo_epi64(tr1_1, tr1_5); \
    O5 = _mm_unpackhi_epi64(tr1_1, tr1_5); \
    O6 = _mm_unpacklo_epi64(tr1_3, tr1_7); \
    O7 = _mm_unpackhi_epi64(tr1_3, tr1_7); \

            TRANSPOSE_8x8_16BIT(res00[0], res01[0], res02[0], res03[0], res04[0], res05[0], res06[0], res07[0], in00[0], in01[0], in02[0], in03[0], in04[0], in05[0], in06[0], in07[0])
                TRANSPOSE_8x8_16BIT(res08[0], res09[0], res10[0], res11[0], res12[0], res13[0], res14[0], res15[0], in00[1], in01[1], in02[1], in03[1], in04[1], in05[1], in06[1], in07[1])
                TRANSPOSE_8x8_16BIT(res00[1], res01[1], res02[1], res03[1], res04[1], res05[1], res06[1], res07[1], in08[0], in09[0], in10[0], in11[0], in12[0], in13[0], in14[0], in15[0])
                TRANSPOSE_8x8_16BIT(res08[1], res09[1], res10[1], res11[1], res12[1], res13[1], res14[1], res15[1], in08[1], in09[1], in10[1], in11[1], in12[1], in13[1], in14[1], in15[1])

#undef TRANSPOSE_8x8_16BIT
        }
    }

    //clip
    {
        __m128i max_val = _mm_set1_epi16((1 << (clip - 1)) - 1);
        __m128i min_val = _mm_set1_epi16(-(1 << (clip - 1)));
        in00[0] = _mm_min_epi16(in00[0], max_val);
        in00[0] = _mm_max_epi16(in00[0], min_val);
        in00[1] = _mm_min_epi16(in00[1], max_val);
        in00[1] = _mm_max_epi16(in00[1], min_val);

        in01[0] = _mm_min_epi16(in01[0], max_val);
        in01[0] = _mm_max_epi16(in01[0], min_val);
        in01[1] = _mm_min_epi16(in01[1], max_val);
        in01[1] = _mm_max_epi16(in01[1], min_val);

        in02[0] = _mm_min_epi16(in02[0], max_val);
        in02[0] = _mm_max_epi16(in02[0], min_val);
        in02[1] = _mm_min_epi16(in02[1], max_val);
        in02[1] = _mm_max_epi16(in02[1], min_val);

        in03[0] = _mm_min_epi16(in03[0], max_val);
        in03[0] = _mm_max_epi16(in03[0], min_val);
        in03[1] = _mm_min_epi16(in03[1], max_val);
        in03[1] = _mm_max_epi16(in03[1], min_val);

        in04[0] = _mm_min_epi16(in04[0], max_val);
        in04[0] = _mm_max_epi16(in04[0], min_val);
        in04[1] = _mm_min_epi16(in04[1], max_val);
        in04[1] = _mm_max_epi16(in04[1], min_val);

        in05[0] = _mm_min_epi16(in05[0], max_val);
        in05[0] = _mm_max_epi16(in05[0], min_val);
        in05[1] = _mm_min_epi16(in05[1], max_val);
        in05[1] = _mm_max_epi16(in05[1], min_val);

        in06[0] = _mm_min_epi16(in06[0], max_val);
        in06[0] = _mm_max_epi16(in06[0], min_val);
        in06[1] = _mm_min_epi16(in06[1], max_val);
        in06[1] = _mm_max_epi16(in06[1], min_val);

        in07[0] = _mm_min_epi16(in07[0], max_val);
        in07[0] = _mm_max_epi16(in07[0], min_val);
        in07[1] = _mm_min_epi16(in07[1], max_val);
        in07[1] = _mm_max_epi16(in07[1], min_val);

        in08[0] = _mm_min_epi16(in08[0], max_val);
        in08[0] = _mm_max_epi16(in08[0], min_val);
        in08[1] = _mm_min_epi16(in08[1], max_val);
        in08[1] = _mm_max_epi16(in08[1], min_val);

        in09[0] = _mm_min_epi16(in09[0], max_val);
        in09[0] = _mm_max_epi16(in09[0], min_val);
        in09[1] = _mm_min_epi16(in09[1], max_val);
        in09[1] = _mm_max_epi16(in09[1], min_val);

        in10[0] = _mm_min_epi16(in10[0], max_val);
        in10[0] = _mm_max_epi16(in10[0], min_val);
        in10[1] = _mm_min_epi16(in10[1], max_val);
        in10[1] = _mm_max_epi16(in10[1], min_val);

        in11[0] = _mm_min_epi16(in11[0], max_val);
        in11[0] = _mm_max_epi16(in11[0], min_val);
        in11[1] = _mm_min_epi16(in11[1], max_val);
        in11[1] = _mm_max_epi16(in11[1], min_val);

        in12[0] = _mm_min_epi16(in12[0], max_val);
        in12[0] = _mm_max_epi16(in12[0], min_val);
        in12[1] = _mm_min_epi16(in12[1], max_val);
        in12[1] = _mm_max_epi16(in12[1], min_val);

        in13[0] = _mm_min_epi16(in13[0], max_val);
        in13[0] = _mm_max_epi16(in13[0], min_val);
        in13[1] = _mm_min_epi16(in13[1], max_val);
        in13[1] = _mm_max_epi16(in13[1], min_val);

        in14[0] = _mm_min_epi16(in14[0], max_val);
        in14[0] = _mm_max_epi16(in14[0], min_val);
        in14[1] = _mm_min_epi16(in14[1], max_val);
        in14[1] = _mm_max_epi16(in14[1], min_val);

        in15[0] = _mm_min_epi16(in15[0], max_val);
        in15[0] = _mm_max_epi16(in15[0], min_val);
        in15[1] = _mm_min_epi16(in15[1], max_val);
        in15[1] = _mm_max_epi16(in15[1], min_val);
    }



    _mm_store_si128((__m128i*)&blk[0 * 16 + 0], in00[0]);
    _mm_store_si128((__m128i*)&blk[0 * 16 + 8], in00[1]);
    _mm_store_si128((__m128i*)&blk[1 * 16 + 0], in01[0]);
    _mm_store_si128((__m128i*)&blk[1 * 16 + 8], in01[1]);
    _mm_store_si128((__m128i*)&blk[2 * 16 + 0], in02[0]);
    _mm_store_si128((__m128i*)&blk[2 * 16 + 8], in02[1]);
    _mm_store_si128((__m128i*)&blk[3 * 16 + 0], in03[0]);
    _mm_store_si128((__m128i*)&blk[3 * 16 + 8], in03[1]);
    _mm_store_si128((__m128i*)&blk[4 * 16 + 0], in04[0]);
    _mm_store_si128((__m128i*)&blk[4 * 16 + 8], in04[1]);
    _mm_store_si128((__m128i*)&blk[5 * 16 + 0], in05[0]);
    _mm_store_si128((__m128i*)&blk[5 * 16 + 8], in05[1]);
    _mm_store_si128((__m128i*)&blk[6 * 16 + 0], in06[0]);
    _mm_store_si128((__m128i*)&blk[6 * 16 + 8], in06[1]);
    _mm_store_si128((__m128i*)&blk[7 * 16 + 0], in07[0]);
    _mm_store_si128((__m128i*)&blk[7 * 16 + 8], in07[1]);
    _mm_store_si128((__m128i*)&blk[8 * 16 + 0], in08[0]);
    _mm_store_si128((__m128i*)&blk[8 * 16 + 8], in08[1]);
    _mm_store_si128((__m128i*)&blk[9 * 16 + 0], in09[0]);
    _mm_store_si128((__m128i*)&blk[9 * 16 + 8], in09[1]);
    _mm_store_si128((__m128i*)&blk[10 * 16 + 0], in10[0]);
    _mm_store_si128((__m128i*)&blk[10 * 16 + 8], in10[1]);
    _mm_store_si128((__m128i*)&blk[11 * 16 + 0], in11[0]);
    _mm_store_si128((__m128i*)&blk[11 * 16 + 8], in11[1]);
    _mm_store_si128((__m128i*)&blk[12 * 16 + 0], in12[0]);
    _mm_store_si128((__m128i*)&blk[12 * 16 + 8], in12[1]);
    _mm_store_si128((__m128i*)&blk[13 * 16 + 0], in13[0]);
    _mm_store_si128((__m128i*)&blk[13 * 16 + 8], in13[1]);
    _mm_store_si128((__m128i*)&blk[14 * 16 + 0], in14[0]);
    _mm_store_si128((__m128i*)&blk[14 * 16 + 8], in14[1]);
    _mm_store_si128((__m128i*)&blk[15 * 16 + 0], in15[0]);
    _mm_store_si128((__m128i*)&blk[15 * 16 + 8], in15[1]);
}

/* wrapper: 适配 (coeff, w, h, bit_depth) 签名 */
static void idct_16x16_sse41(int16_t *coeff, int w, int h, int bit_depth)
{
    (void)w; (void)h;
    idct_16x16_sse41_core(coeff, 20 - bit_depth, bit_depth + 1);
}

/* ===========================================================================
 * 32x32 反 DCT (SSE4.1)
 * 参考: libudavs2 intrinsic_dct.c idct_32x32_sse128
 *
 * 两遍就地变换: 第一遍 shift=5 (硬编码), 第二遍 shift=参数传入
 * 最终裁剪到 clip 参数指定的范围
 *
 * 注: idct_32x32_c 中的 a_flag (i_dst&1) 在 DSP 函数指针路径下 w=32 (偶数),
 *     a_flag 恒为 0, 故 shift2 = 20 - bit_depth
 * ===========================================================================
 */

static void idct_32x32_sse41_core(int16_t *blk, int shift, int clip)
{
    const __m128i c16_p45_p45 = _mm_set1_epi32(0x002D002D);
    const __m128i c16_p43_p44 = _mm_set1_epi32(0x002B002C);
    const __m128i c16_p39_p41 = _mm_set1_epi32(0x00270029);
    const __m128i c16_p34_p36 = _mm_set1_epi32(0x00220024);
    const __m128i c16_p27_p30 = _mm_set1_epi32(0x001B001E);
    const __m128i c16_p19_p23 = _mm_set1_epi32(0x00130017);
    const __m128i c16_p11_p15 = _mm_set1_epi32(0x000B000F);
    const __m128i c16_p02_p07 = _mm_set1_epi32(0x00020007);
    const __m128i c16_p41_p45 = _mm_set1_epi32(0x0029002D);
    const __m128i c16_p23_p34 = _mm_set1_epi32(0x00170022);
    const __m128i c16_n02_p11 = _mm_set1_epi32(0xFFFE000B);
    const __m128i c16_n27_n15 = _mm_set1_epi32(0xFFE5FFF1);
    const __m128i c16_n43_n36 = _mm_set1_epi32(0xFFD5FFDC);
    const __m128i c16_n44_n45 = _mm_set1_epi32(0xFFD4FFD3);
    const __m128i c16_n30_n39 = _mm_set1_epi32(0xFFE2FFD9);
    const __m128i c16_n07_n19 = _mm_set1_epi32(0xFFF9FFED);
    const __m128i c16_p34_p44 = _mm_set1_epi32(0x0022002C);
    const __m128i c16_n07_p15 = _mm_set1_epi32(0xFFF9000F);
    const __m128i c16_n41_n27 = _mm_set1_epi32(0xFFD7FFE5);
    const __m128i c16_n39_n45 = _mm_set1_epi32(0xFFD9FFD3);
    const __m128i c16_n02_n23 = _mm_set1_epi32(0xFFFEFFE9);
    const __m128i c16_p36_p19 = _mm_set1_epi32(0x00240013);
    const __m128i c16_p43_p45 = _mm_set1_epi32(0x002B002D);
    const __m128i c16_p11_p30 = _mm_set1_epi32(0x000B001E);
    const __m128i c16_p23_p43 = _mm_set1_epi32(0x0017002B);
    const __m128i c16_n34_n07 = _mm_set1_epi32(0xFFDEFFF9);
    const __m128i c16_n36_n45 = _mm_set1_epi32(0xFFDCFFD3);
    const __m128i c16_p19_n11 = _mm_set1_epi32(0x0013FFF5);
    const __m128i c16_p44_p41 = _mm_set1_epi32(0x002C0029);
    const __m128i c16_n02_p27 = _mm_set1_epi32(0xFFFE001B);
    const __m128i c16_n45_n30 = _mm_set1_epi32(0xFFD3FFE2);
    const __m128i c16_n15_n39 = _mm_set1_epi32(0xFFF1FFD9);
    const __m128i c16_p11_p41 = _mm_set1_epi32(0x000B0029);
    const __m128i c16_n45_n27 = _mm_set1_epi32(0xFFD3FFE5);
    const __m128i c16_p07_n30 = _mm_set1_epi32(0x0007FFE2);
    const __m128i c16_p43_p39 = _mm_set1_epi32(0x002B0027);
    const __m128i c16_n23_p15 = _mm_set1_epi32(0xFFE9000F);
    const __m128i c16_n34_n45 = _mm_set1_epi32(0xFFDEFFD3);
    const __m128i c16_p36_p02 = _mm_set1_epi32(0x00240002);
    const __m128i c16_p19_p44 = _mm_set1_epi32(0x0013002C);
    const __m128i c16_n02_p39 = _mm_set1_epi32(0xFFFE0027);
    const __m128i c16_n36_n41 = _mm_set1_epi32(0xFFDCFFD7);
    const __m128i c16_p43_p07 = _mm_set1_epi32(0x002B0007);
    const __m128i c16_n11_p34 = _mm_set1_epi32(0xFFF50022);
    const __m128i c16_n30_n44 = _mm_set1_epi32(0xFFE2FFD4);
    const __m128i c16_p45_p15 = _mm_set1_epi32(0x002D000F);
    const __m128i c16_n19_p27 = _mm_set1_epi32(0xFFED001B);
    const __m128i c16_n23_n45 = _mm_set1_epi32(0xFFE9FFD3);
    const __m128i c16_n15_p36 = _mm_set1_epi32(0xFFF10024);
    const __m128i c16_n11_n45 = _mm_set1_epi32(0xFFF5FFD3);
    const __m128i c16_p34_p39 = _mm_set1_epi32(0x00220027);
    const __m128i c16_n45_n19 = _mm_set1_epi32(0xFFD3FFED);
    const __m128i c16_p41_n07 = _mm_set1_epi32(0x0029FFF9);
    const __m128i c16_n23_p30 = _mm_set1_epi32(0xFFE9001E);
    const __m128i c16_n02_n44 = _mm_set1_epi32(0xFFFEFFD4);
    const __m128i c16_p27_p43 = _mm_set1_epi32(0x001B002B);
    const __m128i c16_n27_p34 = _mm_set1_epi32(0xFFE50022);
    const __m128i c16_p19_n39 = _mm_set1_epi32(0x0013FFD9);
    const __m128i c16_n11_p43 = _mm_set1_epi32(0xFFF5002B);
    const __m128i c16_p02_n45 = _mm_set1_epi32(0x0002FFD3);
    const __m128i c16_p07_p45 = _mm_set1_epi32(0x0007002D);
    const __m128i c16_n15_n44 = _mm_set1_epi32(0xFFF1FFD4);
    const __m128i c16_p23_p41 = _mm_set1_epi32(0x00170029);
    const __m128i c16_n30_n36 = _mm_set1_epi32(0xFFE2FFDC);
    const __m128i c16_n36_p30 = _mm_set1_epi32(0xFFDC001E);
    const __m128i c16_p41_n23 = _mm_set1_epi32(0x0029FFE9);
    const __m128i c16_n44_p15 = _mm_set1_epi32(0xFFD4000F);
    const __m128i c16_p45_n07 = _mm_set1_epi32(0x002DFFF9);
    const __m128i c16_n45_n02 = _mm_set1_epi32(0xFFD3FFFE);
    const __m128i c16_p43_p11 = _mm_set1_epi32(0x002B000B);
    const __m128i c16_n39_n19 = _mm_set1_epi32(0xFFD9FFED);
    const __m128i c16_p34_p27 = _mm_set1_epi32(0x0022001B);
    const __m128i c16_n43_p27 = _mm_set1_epi32(0xFFD5001B);
    const __m128i c16_p44_n02 = _mm_set1_epi32(0x002CFFFE);
    const __m128i c16_n30_n23 = _mm_set1_epi32(0xFFE2FFE9);
    const __m128i c16_p07_p41 = _mm_set1_epi32(0x00070029);
    const __m128i c16_p19_n45 = _mm_set1_epi32(0x0013FFD3);
    const __m128i c16_n39_p34 = _mm_set1_epi32(0xFFD90022);
    const __m128i c16_p45_n11 = _mm_set1_epi32(0x002DFFF5);
    const __m128i c16_n36_n15 = _mm_set1_epi32(0xFFDCFFF1);
    const __m128i c16_n45_p23 = _mm_set1_epi32(0xFFD30017);
    const __m128i c16_p27_p19 = _mm_set1_epi32(0x001B0013);
    const __m128i c16_p15_n45 = _mm_set1_epi32(0x000FFFD3);
    const __m128i c16_n44_p30 = _mm_set1_epi32(0xFFD4001E);
    const __m128i c16_p34_p11 = _mm_set1_epi32(0x0022000B);
    const __m128i c16_p07_n43 = _mm_set1_epi32(0x0007FFD5);
    const __m128i c16_n41_p36 = _mm_set1_epi32(0xFFD70024);
    const __m128i c16_p39_p02 = _mm_set1_epi32(0x00270002);
    const __m128i c16_n44_p19 = _mm_set1_epi32(0xFFD40013);
    const __m128i c16_n02_p36 = _mm_set1_epi32(0xFFFE0024);
    const __m128i c16_p45_n34 = _mm_set1_epi32(0x002DFFDE);
    const __m128i c16_n15_n23 = _mm_set1_epi32(0xFFF1FFE9);
    const __m128i c16_n39_p43 = _mm_set1_epi32(0xFFD9002B);
    const __m128i c16_p30_p07 = _mm_set1_epi32(0x001E0007);
    const __m128i c16_p27_n45 = _mm_set1_epi32(0x001BFFD3);
    const __m128i c16_n41_p11 = _mm_set1_epi32(0xFFD7000B);
    const __m128i c16_n39_p15 = _mm_set1_epi32(0xFFD9000F);
    const __m128i c16_n30_p45 = _mm_set1_epi32(0xFFE2002D);
    const __m128i c16_p27_p02 = _mm_set1_epi32(0x001B0002);
    const __m128i c16_p41_n44 = _mm_set1_epi32(0x0029FFD4);
    const __m128i c16_n11_n19 = _mm_set1_epi32(0xFFF5FFED);
    const __m128i c16_n45_p36 = _mm_set1_epi32(0xFFD30024);
    const __m128i c16_n07_p34 = _mm_set1_epi32(0xFFF90022);
    const __m128i c16_p43_n23 = _mm_set1_epi32(0x002BFFE9);
    const __m128i c16_n30_p11 = _mm_set1_epi32(0xFFE2000B);
    const __m128i c16_n45_p43 = _mm_set1_epi32(0xFFD3002B);
    const __m128i c16_n19_p36 = _mm_set1_epi32(0xFFED0024);
    const __m128i c16_p23_n02 = _mm_set1_epi32(0x0017FFFE);
    const __m128i c16_p45_n39 = _mm_set1_epi32(0x002DFFD9);
    const __m128i c16_p27_n41 = _mm_set1_epi32(0x001BFFD7);
    const __m128i c16_n15_n07 = _mm_set1_epi32(0xFFF1FFF9);
    const __m128i c16_n44_p34 = _mm_set1_epi32(0xFFD40022);
    const __m128i c16_n19_p07 = _mm_set1_epi32(0xFFED0007);
    const __m128i c16_n39_p30 = _mm_set1_epi32(0xFFD9001E);
    const __m128i c16_n45_p44 = _mm_set1_epi32(0xFFD3002C);
    const __m128i c16_n36_p43 = _mm_set1_epi32(0xFFDC002B);
    const __m128i c16_n15_p27 = _mm_set1_epi32(0xFFF1001B);
    const __m128i c16_p11_p02 = _mm_set1_epi32(0x000B0002);
    const __m128i c16_p34_n23 = _mm_set1_epi32(0x0022FFE9);
    const __m128i c16_p45_n41 = _mm_set1_epi32(0x002DFFD7);
    const __m128i c16_n07_p02 = _mm_set1_epi32(0xFFF90002);
    const __m128i c16_n15_p11 = _mm_set1_epi32(0xFFF1000B);
    const __m128i c16_n23_p19 = _mm_set1_epi32(0xFFE90013);
    const __m128i c16_n30_p27 = _mm_set1_epi32(0xFFE2001B);
    const __m128i c16_n36_p34 = _mm_set1_epi32(0xFFDC0022);
    const __m128i c16_n41_p39 = _mm_set1_epi32(0xFFD70027);
    const __m128i c16_n44_p43 = _mm_set1_epi32(0xFFD4002B);
    const __m128i c16_n45_p45 = _mm_set1_epi32(0xFFD3002D);

    //	const __m128i c16_p43_p45 = _mm_set1_epi32(0x002B002D);
    const __m128i c16_p35_p40 = _mm_set1_epi32(0x00230028);
    const __m128i c16_p21_p29 = _mm_set1_epi32(0x0015001D);
    const __m128i c16_p04_p13 = _mm_set1_epi32(0x0004000D);
    const __m128i c16_p29_p43 = _mm_set1_epi32(0x001D002B);
    const __m128i c16_n21_p04 = _mm_set1_epi32(0xFFEB0004);
    const __m128i c16_n45_n40 = _mm_set1_epi32(0xFFD3FFD8);
    const __m128i c16_n13_n35 = _mm_set1_epi32(0xFFF3FFDD);
    const __m128i c16_p04_p40 = _mm_set1_epi32(0x00040028);
    const __m128i c16_n43_n35 = _mm_set1_epi32(0xFFD5FFDD);
    const __m128i c16_p29_n13 = _mm_set1_epi32(0x001DFFF3);
    const __m128i c16_p21_p45 = _mm_set1_epi32(0x0015002D);
    const __m128i c16_n21_p35 = _mm_set1_epi32(0xFFEB0023);
    const __m128i c16_p04_n43 = _mm_set1_epi32(0x0004FFD5);
    const __m128i c16_p13_p45 = _mm_set1_epi32(0x000D002D);
    const __m128i c16_n29_n40 = _mm_set1_epi32(0xFFE3FFD8);
    const __m128i c16_n40_p29 = _mm_set1_epi32(0xFFD8001D);
    const __m128i c16_p45_n13 = _mm_set1_epi32(0x002DFFF3);
    const __m128i c16_n43_n04 = _mm_set1_epi32(0xFFD5FFFC);
    const __m128i c16_p35_p21 = _mm_set1_epi32(0x00230015);
    const __m128i c16_n45_p21 = _mm_set1_epi32(0xFFD30015);
    const __m128i c16_p13_p29 = _mm_set1_epi32(0x000D001D);
    const __m128i c16_p35_n43 = _mm_set1_epi32(0x0023FFD5);
    const __m128i c16_n40_p04 = _mm_set1_epi32(0xFFD80004);
    const __m128i c16_n35_p13 = _mm_set1_epi32(0xFFDD000D);
    const __m128i c16_n40_p45 = _mm_set1_epi32(0xFFD8002D);
    const __m128i c16_p04_p21 = _mm_set1_epi32(0x00040015);
    const __m128i c16_p43_n29 = _mm_set1_epi32(0x002BFFE3);
    const __m128i c16_n13_p04 = _mm_set1_epi32(0xFFF30004);
    const __m128i c16_n29_p21 = _mm_set1_epi32(0xFFE30015);
    const __m128i c16_n40_p35 = _mm_set1_epi32(0xFFD80023);
    //	const __m128i c16_n45_p43 = _mm_set1_epi32(0xFFD3002B);


    const __m128i c16_p38_p44 = _mm_set1_epi32(0x0026002C);
    const __m128i c16_p09_p25 = _mm_set1_epi32(0x00090019);
    const __m128i c16_n09_p38 = _mm_set1_epi32(0xFFF70026);
    const __m128i c16_n25_n44 = _mm_set1_epi32(0xFFE7FFD4);

    const __m128i c16_n44_p25 = _mm_set1_epi32(0xFFD40019);
    const __m128i c16_p38_p09 = _mm_set1_epi32(0x00260009);
    const __m128i c16_n25_p09 = _mm_set1_epi32(0xFFE70009);
    const __m128i c16_n44_p38 = _mm_set1_epi32(0xFFD40026);

    const __m128i c16_p17_p42 = _mm_set1_epi32(0x0011002A);
    const __m128i c16_n42_p17 = _mm_set1_epi32(0xFFD60011);

    const __m128i c16_p32_p32 = _mm_set1_epi32(0x00200020);
    const __m128i c16_n32_p32 = _mm_set1_epi32(0xFFE00020);

    __m128i c32_rnd = _mm_set1_epi32(16);

    int nShift = 5;
    int i, pass, part;

    // DCT1
    __m128i in00[4], in01[4], in02[4], in03[4], in04[4], in05[4], in06[4], in07[4], in08[4], in09[4], in10[4], in11[4], in12[4], in13[4], in14[4], in15[4];
    __m128i in16[4], in17[4], in18[4], in19[4], in20[4], in21[4], in22[4], in23[4], in24[4], in25[4], in26[4], in27[4], in28[4], in29[4], in30[4], in31[4];
    __m128i res00[4], res01[4], res02[4], res03[4], res04[4], res05[4], res06[4], res07[4], res08[4], res09[4], res10[4], res11[4], res12[4], res13[4], res14[4], res15[4];
    __m128i res16[4], res17[4], res18[4], res19[4], res20[4], res21[4], res22[4], res23[4], res24[4], res25[4], res26[4], res27[4], res28[4], res29[4], res30[4], res31[4];

    for (i = 0; i < 4; i++)
    {
        const int offset = (i << 3);

        in00[i] = _mm_load_si128((const __m128i*)&blk[0 * 32 + offset]);
        in01[i] = _mm_load_si128((const __m128i*)&blk[1 * 32 + offset]);
        in02[i] = _mm_load_si128((const __m128i*)&blk[2 * 32 + offset]);
        in03[i] = _mm_load_si128((const __m128i*)&blk[3 * 32 + offset]);
        in04[i] = _mm_load_si128((const __m128i*)&blk[4 * 32 + offset]);
        in05[i] = _mm_load_si128((const __m128i*)&blk[5 * 32 + offset]);
        in06[i] = _mm_load_si128((const __m128i*)&blk[6 * 32 + offset]);
        in07[i] = _mm_load_si128((const __m128i*)&blk[7 * 32 + offset]);
        in08[i] = _mm_load_si128((const __m128i*)&blk[8 * 32 + offset]);
        in09[i] = _mm_load_si128((const __m128i*)&blk[9 * 32 + offset]);
        in10[i] = _mm_load_si128((const __m128i*)&blk[10 * 32 + offset]);
        in11[i] = _mm_load_si128((const __m128i*)&blk[11 * 32 + offset]);
        in12[i] = _mm_load_si128((const __m128i*)&blk[12 * 32 + offset]);
        in13[i] = _mm_load_si128((const __m128i*)&blk[13 * 32 + offset]);
        in14[i] = _mm_load_si128((const __m128i*)&blk[14 * 32 + offset]);
        in15[i] = _mm_load_si128((const __m128i*)&blk[15 * 32 + offset]);
        in16[i] = _mm_load_si128((const __m128i*)&blk[16 * 32 + offset]);
        in17[i] = _mm_load_si128((const __m128i*)&blk[17 * 32 + offset]);
        in18[i] = _mm_load_si128((const __m128i*)&blk[18 * 32 + offset]);
        in19[i] = _mm_load_si128((const __m128i*)&blk[19 * 32 + offset]);
        in20[i] = _mm_load_si128((const __m128i*)&blk[20 * 32 + offset]);
        in21[i] = _mm_load_si128((const __m128i*)&blk[21 * 32 + offset]);
        in22[i] = _mm_load_si128((const __m128i*)&blk[22 * 32 + offset]);
        in23[i] = _mm_load_si128((const __m128i*)&blk[23 * 32 + offset]);
        in24[i] = _mm_load_si128((const __m128i*)&blk[24 * 32 + offset]);
        in25[i] = _mm_load_si128((const __m128i*)&blk[25 * 32 + offset]);
        in26[i] = _mm_load_si128((const __m128i*)&blk[26 * 32 + offset]);
        in27[i] = _mm_load_si128((const __m128i*)&blk[27 * 32 + offset]);
        in28[i] = _mm_load_si128((const __m128i*)&blk[28 * 32 + offset]);
        in29[i] = _mm_load_si128((const __m128i*)&blk[29 * 32 + offset]);
        in30[i] = _mm_load_si128((const __m128i*)&blk[30 * 32 + offset]);
        in31[i] = _mm_load_si128((const __m128i*)&blk[31 * 32 + offset]);
    }

    for (pass = 0; pass < 2; pass++)
    {
        if (pass == 1)
        {
            c32_rnd = _mm_set1_epi32(shift ? (1 << (shift - 1)) : 0); /* */
            nShift = shift;
        }

        for (part = 0; part < 4; part++)
        {
            const __m128i T_00_00A = _mm_unpacklo_epi16(in01[part], in03[part]);       // [33 13 32 12 31 11 30 10]
            const __m128i T_00_00B = _mm_unpackhi_epi16(in01[part], in03[part]);       // [37 17 36 16 35 15 34 14]
            const __m128i T_00_01A = _mm_unpacklo_epi16(in05[part], in07[part]);       // [ ]
            const __m128i T_00_01B = _mm_unpackhi_epi16(in05[part], in07[part]);       // [ ]
            const __m128i T_00_02A = _mm_unpacklo_epi16(in09[part], in11[part]);       // [ ]
            const __m128i T_00_02B = _mm_unpackhi_epi16(in09[part], in11[part]);       // [ ]
            const __m128i T_00_03A = _mm_unpacklo_epi16(in13[part], in15[part]);       // [ ]
            const __m128i T_00_03B = _mm_unpackhi_epi16(in13[part], in15[part]);       // [ ]
            const __m128i T_00_04A = _mm_unpacklo_epi16(in17[part], in19[part]);       // [ ]
            const __m128i T_00_04B = _mm_unpackhi_epi16(in17[part], in19[part]);       // [ ]
            const __m128i T_00_05A = _mm_unpacklo_epi16(in21[part], in23[part]);       // [ ]
            const __m128i T_00_05B = _mm_unpackhi_epi16(in21[part], in23[part]);       // [ ]
            const __m128i T_00_06A = _mm_unpacklo_epi16(in25[part], in27[part]);       // [ ]
            const __m128i T_00_06B = _mm_unpackhi_epi16(in25[part], in27[part]);       // [ ]
            const __m128i T_00_07A = _mm_unpacklo_epi16(in29[part], in31[part]);       //
            const __m128i T_00_07B = _mm_unpackhi_epi16(in29[part], in31[part]);       // [ ]

            const __m128i T_00_08A = _mm_unpacklo_epi16(in02[part], in06[part]);       // [ ]
            const __m128i T_00_08B = _mm_unpackhi_epi16(in02[part], in06[part]);       // [ ]
            const __m128i T_00_09A = _mm_unpacklo_epi16(in10[part], in14[part]);       // [ ]
            const __m128i T_00_09B = _mm_unpackhi_epi16(in10[part], in14[part]);       // [ ]
            const __m128i T_00_10A = _mm_unpacklo_epi16(in18[part], in22[part]);       // [ ]
            const __m128i T_00_10B = _mm_unpackhi_epi16(in18[part], in22[part]);       // [ ]
            const __m128i T_00_11A = _mm_unpacklo_epi16(in26[part], in30[part]);       // [ ]
            const __m128i T_00_11B = _mm_unpackhi_epi16(in26[part], in30[part]);       // [ ]

            const __m128i T_00_12A = _mm_unpacklo_epi16(in04[part], in12[part]);       // [ ]
            const __m128i T_00_12B = _mm_unpackhi_epi16(in04[part], in12[part]);       // [ ]
            const __m128i T_00_13A = _mm_unpacklo_epi16(in20[part], in28[part]);       // [ ]
            const __m128i T_00_13B = _mm_unpackhi_epi16(in20[part], in28[part]);       // [ ]

            const __m128i T_00_14A = _mm_unpacklo_epi16(in08[part], in24[part]);       //
            const __m128i T_00_14B = _mm_unpackhi_epi16(in08[part], in24[part]);       // [ ]
            const __m128i T_00_15A = _mm_unpacklo_epi16(in00[part], in16[part]);       //
            const __m128i T_00_15B = _mm_unpackhi_epi16(in00[part], in16[part]);       // [ ]

            __m128i O00A, O01A, O02A, O03A, O04A, O05A, O06A, O07A, O08A, O09A, O10A, O11A, O12A, O13A, O14A, O15A;
            __m128i O00B, O01B, O02B, O03B, O04B, O05B, O06B, O07B, O08B, O09B, O10B, O11B, O12B, O13B, O14B, O15B;
            __m128i EO0A, EO1A, EO2A, EO3A, EO4A, EO5A, EO6A, EO7A;
            __m128i EO0B, EO1B, EO2B, EO3B, EO4B, EO5B, EO6B, EO7B;
            {
                __m128i T00, T01, T02, T03;
#define COMPUTE_ROW(r0103, r0507, r0911, r1315, r1719, r2123, r2527, r2931, c0103, c0507, c0911, c1315, c1719, c2123, c2527, c2931, row) \
    T00 = _mm_add_epi32(_mm_madd_epi16(r0103, c0103), _mm_madd_epi16(r0507, c0507)); \
    T01 = _mm_add_epi32(_mm_madd_epi16(r0911, c0911), _mm_madd_epi16(r1315, c1315)); \
    T02 = _mm_add_epi32(_mm_madd_epi16(r1719, c1719), _mm_madd_epi16(r2123, c2123)); \
    T03 = _mm_add_epi32(_mm_madd_epi16(r2527, c2527), _mm_madd_epi16(r2931, c2931)); \
    row = _mm_add_epi32(_mm_add_epi32(T00, T01), _mm_add_epi32(T02, T03));

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

#undef COMPUTE_ROW
            }


            {
                __m128i T00, T01;
#define COMPUTE_ROW(row0206, row1014, row1822, row2630, c0206, c1014, c1822, c2630, row) \
    T00 = _mm_add_epi32(_mm_madd_epi16(row0206, c0206), _mm_madd_epi16(row1014, c1014)); \
    T01 = _mm_add_epi32(_mm_madd_epi16(row1822, c1822), _mm_madd_epi16(row2630, c2630)); \
    row = _mm_add_epi32(T00, T01);

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
#undef COMPUTE_ROW
            }
            {
                const __m128i EEO0A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_p38_p44), _mm_madd_epi16(T_00_13A, c16_p09_p25));
                const __m128i EEO1A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_n09_p38), _mm_madd_epi16(T_00_13A, c16_n25_n44));
                const __m128i EEO2A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_n44_p25), _mm_madd_epi16(T_00_13A, c16_p38_p09));
                const __m128i EEO3A = _mm_add_epi32(_mm_madd_epi16(T_00_12A, c16_n25_p09), _mm_madd_epi16(T_00_13A, c16_n44_p38));
                const __m128i EEO0B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_p38_p44), _mm_madd_epi16(T_00_13B, c16_p09_p25));
                const __m128i EEO1B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_n09_p38), _mm_madd_epi16(T_00_13B, c16_n25_n44));
                const __m128i EEO2B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_n44_p25), _mm_madd_epi16(T_00_13B, c16_p38_p09));
                const __m128i EEO3B = _mm_add_epi32(_mm_madd_epi16(T_00_12B, c16_n25_p09), _mm_madd_epi16(T_00_13B, c16_n44_p38));

                const __m128i EEEO0A = _mm_madd_epi16(T_00_14A, c16_p17_p42);
                const __m128i EEEO0B = _mm_madd_epi16(T_00_14B, c16_p17_p42);
                const __m128i EEEO1A = _mm_madd_epi16(T_00_14A, c16_n42_p17);
                const __m128i EEEO1B = _mm_madd_epi16(T_00_14B, c16_n42_p17);

                const __m128i EEEE0A = _mm_madd_epi16(T_00_15A, c16_p32_p32);
                const __m128i EEEE0B = _mm_madd_epi16(T_00_15B, c16_p32_p32);
                const __m128i EEEE1A = _mm_madd_epi16(T_00_15A, c16_n32_p32);
                const __m128i EEEE1B = _mm_madd_epi16(T_00_15B, c16_n32_p32);

                const __m128i EEE0A = _mm_add_epi32(EEEE0A, EEEO0A);          // EEE0 = EEEE0 + EEEO0
                const __m128i EEE0B = _mm_add_epi32(EEEE0B, EEEO0B);
                const __m128i EEE1A = _mm_add_epi32(EEEE1A, EEEO1A);          // EEE1 = EEEE1 + EEEO1
                const __m128i EEE1B = _mm_add_epi32(EEEE1B, EEEO1B);
                const __m128i EEE3A = _mm_sub_epi32(EEEE0A, EEEO0A);          // EEE2 = EEEE0 - EEEO0
                const __m128i EEE3B = _mm_sub_epi32(EEEE0B, EEEO0B);
                const __m128i EEE2A = _mm_sub_epi32(EEEE1A, EEEO1A);          // EEE3 = EEEE1 - EEEO1
                const __m128i EEE2B = _mm_sub_epi32(EEEE1B, EEEO1B);

                const __m128i EE0A = _mm_add_epi32(EEE0A, EEO0A);          // EE0 = EEE0 + EEO0
                const __m128i EE0B = _mm_add_epi32(EEE0B, EEO0B);
                const __m128i EE1A = _mm_add_epi32(EEE1A, EEO1A);          // EE1 = EEE1 + EEO1
                const __m128i EE1B = _mm_add_epi32(EEE1B, EEO1B);
                const __m128i EE2A = _mm_add_epi32(EEE2A, EEO2A);          // EE2 = EEE0 + EEO0
                const __m128i EE2B = _mm_add_epi32(EEE2B, EEO2B);
                const __m128i EE3A = _mm_add_epi32(EEE3A, EEO3A);          // EE3 = EEE1 + EEO1
                const __m128i EE3B = _mm_add_epi32(EEE3B, EEO3B);
                const __m128i EE7A = _mm_sub_epi32(EEE0A, EEO0A);          // EE7 = EEE0 - EEO0
                const __m128i EE7B = _mm_sub_epi32(EEE0B, EEO0B);
                const __m128i EE6A = _mm_sub_epi32(EEE1A, EEO1A);          // EE6 = EEE1 - EEO1
                const __m128i EE6B = _mm_sub_epi32(EEE1B, EEO1B);
                const __m128i EE5A = _mm_sub_epi32(EEE2A, EEO2A);          // EE5 = EEE0 - EEO0
                const __m128i EE5B = _mm_sub_epi32(EEE2B, EEO2B);
                const __m128i EE4A = _mm_sub_epi32(EEE3A, EEO3A);          // EE4 = EEE1 - EEO1
                const __m128i EE4B = _mm_sub_epi32(EEE3B, EEO3B);

                const __m128i E0A = _mm_add_epi32(EE0A, EO0A);          // E0 = EE0 + EO0
                const __m128i E0B = _mm_add_epi32(EE0B, EO0B);
                const __m128i E1A = _mm_add_epi32(EE1A, EO1A);          // E1 = EE1 + EO1
                const __m128i E1B = _mm_add_epi32(EE1B, EO1B);
                const __m128i E2A = _mm_add_epi32(EE2A, EO2A);          // E2 = EE2 + EO2
                const __m128i E2B = _mm_add_epi32(EE2B, EO2B);
                const __m128i E3A = _mm_add_epi32(EE3A, EO3A);          // E3 = EE3 + EO3
                const __m128i E3B = _mm_add_epi32(EE3B, EO3B);
                const __m128i E4A = _mm_add_epi32(EE4A, EO4A);          // E4 =
                const __m128i E4B = _mm_add_epi32(EE4B, EO4B);
                const __m128i E5A = _mm_add_epi32(EE5A, EO5A);          // E5 =
                const __m128i E5B = _mm_add_epi32(EE5B, EO5B);
                const __m128i E6A = _mm_add_epi32(EE6A, EO6A);          // E6 =
                const __m128i E6B = _mm_add_epi32(EE6B, EO6B);
                const __m128i E7A = _mm_add_epi32(EE7A, EO7A);          // E7 =
                const __m128i E7B = _mm_add_epi32(EE7B, EO7B);
                const __m128i EFA = _mm_sub_epi32(EE0A, EO0A);          // EF = EE0 - EO0
                const __m128i EFB = _mm_sub_epi32(EE0B, EO0B);
                const __m128i EEA = _mm_sub_epi32(EE1A, EO1A);          // EE = EE1 - EO1
                const __m128i EEB = _mm_sub_epi32(EE1B, EO1B);
                const __m128i EDA = _mm_sub_epi32(EE2A, EO2A);          // ED = EE2 - EO2
                const __m128i EDB = _mm_sub_epi32(EE2B, EO2B);
                const __m128i ECA = _mm_sub_epi32(EE3A, EO3A);          // EC = EE3 - EO3
                const __m128i ECB = _mm_sub_epi32(EE3B, EO3B);
                const __m128i EBA = _mm_sub_epi32(EE4A, EO4A);          // EB =
                const __m128i EBB = _mm_sub_epi32(EE4B, EO4B);
                const __m128i EAA = _mm_sub_epi32(EE5A, EO5A);          // EA =
                const __m128i EAB = _mm_sub_epi32(EE5B, EO5B);
                const __m128i E9A = _mm_sub_epi32(EE6A, EO6A);          // E9 =
                const __m128i E9B = _mm_sub_epi32(EE6B, EO6B);
                const __m128i E8A = _mm_sub_epi32(EE7A, EO7A);          // E8 =
                const __m128i E8B = _mm_sub_epi32(EE7B, EO7B);

                const __m128i T10A = _mm_add_epi32(E0A, c32_rnd);         // E0 + rnd
                const __m128i T10B = _mm_add_epi32(E0B, c32_rnd);
                const __m128i T11A = _mm_add_epi32(E1A, c32_rnd);         // E1 + rnd
                const __m128i T11B = _mm_add_epi32(E1B, c32_rnd);
                const __m128i T12A = _mm_add_epi32(E2A, c32_rnd);         // E2 + rnd
                const __m128i T12B = _mm_add_epi32(E2B, c32_rnd);
                const __m128i T13A = _mm_add_epi32(E3A, c32_rnd);         // E3 + rnd
                const __m128i T13B = _mm_add_epi32(E3B, c32_rnd);
                const __m128i T14A = _mm_add_epi32(E4A, c32_rnd);         // E4 + rnd
                const __m128i T14B = _mm_add_epi32(E4B, c32_rnd);
                const __m128i T15A = _mm_add_epi32(E5A, c32_rnd);         // E5 + rnd
                const __m128i T15B = _mm_add_epi32(E5B, c32_rnd);
                const __m128i T16A = _mm_add_epi32(E6A, c32_rnd);         // E6 + rnd
                const __m128i T16B = _mm_add_epi32(E6B, c32_rnd);
                const __m128i T17A = _mm_add_epi32(E7A, c32_rnd);         // E7 + rnd
                const __m128i T17B = _mm_add_epi32(E7B, c32_rnd);
                const __m128i T18A = _mm_add_epi32(E8A, c32_rnd);         // E8 + rnd
                const __m128i T18B = _mm_add_epi32(E8B, c32_rnd);
                const __m128i T19A = _mm_add_epi32(E9A, c32_rnd);         // E9 + rnd
                const __m128i T19B = _mm_add_epi32(E9B, c32_rnd);
                const __m128i T1AA = _mm_add_epi32(EAA, c32_rnd);         // E10 + rnd
                const __m128i T1AB = _mm_add_epi32(EAB, c32_rnd);
                const __m128i T1BA = _mm_add_epi32(EBA, c32_rnd);         // E11 + rnd
                const __m128i T1BB = _mm_add_epi32(EBB, c32_rnd);
                const __m128i T1CA = _mm_add_epi32(ECA, c32_rnd);         // E12 + rnd
                const __m128i T1CB = _mm_add_epi32(ECB, c32_rnd);
                const __m128i T1DA = _mm_add_epi32(EDA, c32_rnd);         // E13 + rnd
                const __m128i T1DB = _mm_add_epi32(EDB, c32_rnd);
                const __m128i T1EA = _mm_add_epi32(EEA, c32_rnd);         // E14 + rnd
                const __m128i T1EB = _mm_add_epi32(EEB, c32_rnd);
                const __m128i T1FA = _mm_add_epi32(EFA, c32_rnd);         // E15 + rnd
                const __m128i T1FB = _mm_add_epi32(EFB, c32_rnd);

                const __m128i T2_00A = _mm_add_epi32(T10A, O00A);          // E0 + O0 + rnd
                const __m128i T2_00B = _mm_add_epi32(T10B, O00B);
                const __m128i T2_01A = _mm_add_epi32(T11A, O01A);          // E1 + O1 + rnd
                const __m128i T2_01B = _mm_add_epi32(T11B, O01B);
                const __m128i T2_02A = _mm_add_epi32(T12A, O02A);          // E2 + O2 + rnd
                const __m128i T2_02B = _mm_add_epi32(T12B, O02B);
                const __m128i T2_03A = _mm_add_epi32(T13A, O03A);          // E3 + O3 + rnd
                const __m128i T2_03B = _mm_add_epi32(T13B, O03B);
                const __m128i T2_04A = _mm_add_epi32(T14A, O04A);          // E4
                const __m128i T2_04B = _mm_add_epi32(T14B, O04B);
                const __m128i T2_05A = _mm_add_epi32(T15A, O05A);          // E5
                const __m128i T2_05B = _mm_add_epi32(T15B, O05B);
                const __m128i T2_06A = _mm_add_epi32(T16A, O06A);          // E6
                const __m128i T2_06B = _mm_add_epi32(T16B, O06B);
                const __m128i T2_07A = _mm_add_epi32(T17A, O07A);          // E7
                const __m128i T2_07B = _mm_add_epi32(T17B, O07B);
                const __m128i T2_08A = _mm_add_epi32(T18A, O08A);          // E8
                const __m128i T2_08B = _mm_add_epi32(T18B, O08B);
                const __m128i T2_09A = _mm_add_epi32(T19A, O09A);          // E9
                const __m128i T2_09B = _mm_add_epi32(T19B, O09B);
                const __m128i T2_10A = _mm_add_epi32(T1AA, O10A);          // E10
                const __m128i T2_10B = _mm_add_epi32(T1AB, O10B);
                const __m128i T2_11A = _mm_add_epi32(T1BA, O11A);          // E11
                const __m128i T2_11B = _mm_add_epi32(T1BB, O11B);
                const __m128i T2_12A = _mm_add_epi32(T1CA, O12A);          // E12
                const __m128i T2_12B = _mm_add_epi32(T1CB, O12B);
                const __m128i T2_13A = _mm_add_epi32(T1DA, O13A);          // E13
                const __m128i T2_13B = _mm_add_epi32(T1DB, O13B);
                const __m128i T2_14A = _mm_add_epi32(T1EA, O14A);          // E14
                const __m128i T2_14B = _mm_add_epi32(T1EB, O14B);
                const __m128i T2_15A = _mm_add_epi32(T1FA, O15A);          // E15
                const __m128i T2_15B = _mm_add_epi32(T1FB, O15B);
                const __m128i T2_31A = _mm_sub_epi32(T10A, O00A);          // E0 - O0 + rnd
                const __m128i T2_31B = _mm_sub_epi32(T10B, O00B);
                const __m128i T2_30A = _mm_sub_epi32(T11A, O01A);          // E1 - O1 + rnd
                const __m128i T2_30B = _mm_sub_epi32(T11B, O01B);
                const __m128i T2_29A = _mm_sub_epi32(T12A, O02A);          // E2 - O2 + rnd
                const __m128i T2_29B = _mm_sub_epi32(T12B, O02B);
                const __m128i T2_28A = _mm_sub_epi32(T13A, O03A);          // E3 - O3 + rnd
                const __m128i T2_28B = _mm_sub_epi32(T13B, O03B);
                const __m128i T2_27A = _mm_sub_epi32(T14A, O04A);          // E4
                const __m128i T2_27B = _mm_sub_epi32(T14B, O04B);
                const __m128i T2_26A = _mm_sub_epi32(T15A, O05A);          // E5
                const __m128i T2_26B = _mm_sub_epi32(T15B, O05B);
                const __m128i T2_25A = _mm_sub_epi32(T16A, O06A);          // E6
                const __m128i T2_25B = _mm_sub_epi32(T16B, O06B);
                const __m128i T2_24A = _mm_sub_epi32(T17A, O07A);          // E7
                const __m128i T2_24B = _mm_sub_epi32(T17B, O07B);
                const __m128i T2_23A = _mm_sub_epi32(T18A, O08A);          //
                const __m128i T2_23B = _mm_sub_epi32(T18B, O08B);
                const __m128i T2_22A = _mm_sub_epi32(T19A, O09A);          //
                const __m128i T2_22B = _mm_sub_epi32(T19B, O09B);
                const __m128i T2_21A = _mm_sub_epi32(T1AA, O10A);          //
                const __m128i T2_21B = _mm_sub_epi32(T1AB, O10B);
                const __m128i T2_20A = _mm_sub_epi32(T1BA, O11A);          //
                const __m128i T2_20B = _mm_sub_epi32(T1BB, O11B);
                const __m128i T2_19A = _mm_sub_epi32(T1CA, O12A);          //
                const __m128i T2_19B = _mm_sub_epi32(T1CB, O12B);
                const __m128i T2_18A = _mm_sub_epi32(T1DA, O13A);          //
                const __m128i T2_18B = _mm_sub_epi32(T1DB, O13B);
                const __m128i T2_17A = _mm_sub_epi32(T1EA, O14A);          //
                const __m128i T2_17B = _mm_sub_epi32(T1EB, O14B);
                const __m128i T2_16A = _mm_sub_epi32(T1FA, O15A);          //
                const __m128i T2_16B = _mm_sub_epi32(T1FB, O15B);

                const __m128i T3_00A = _mm_srai_epi32(T2_00A, nShift);             // [30 20 10 00]
                const __m128i T3_00B = _mm_srai_epi32(T2_00B, nShift);             // [70 60 50 40]
                const __m128i T3_01A = _mm_srai_epi32(T2_01A, nShift);             // [31 21 11 01]
                const __m128i T3_01B = _mm_srai_epi32(T2_01B, nShift);             // [71 61 51 41]
                const __m128i T3_02A = _mm_srai_epi32(T2_02A, nShift);             // [32 22 12 02]
                const __m128i T3_02B = _mm_srai_epi32(T2_02B, nShift);             // [72 62 52 42]
                const __m128i T3_03A = _mm_srai_epi32(T2_03A, nShift);             // [33 23 13 03]
                const __m128i T3_03B = _mm_srai_epi32(T2_03B, nShift);             // [73 63 53 43]
                const __m128i T3_04A = _mm_srai_epi32(T2_04A, nShift);             // [33 24 14 04]
                const __m128i T3_04B = _mm_srai_epi32(T2_04B, nShift);             // [74 64 54 44]
                const __m128i T3_05A = _mm_srai_epi32(T2_05A, nShift);             // [35 25 15 05]
                const __m128i T3_05B = _mm_srai_epi32(T2_05B, nShift);             // [75 65 55 45]
                const __m128i T3_06A = _mm_srai_epi32(T2_06A, nShift);             // [36 26 16 06]
                const __m128i T3_06B = _mm_srai_epi32(T2_06B, nShift);             // [76 66 56 46]
                const __m128i T3_07A = _mm_srai_epi32(T2_07A, nShift);             // [37 27 17 07]
                const __m128i T3_07B = _mm_srai_epi32(T2_07B, nShift);             // [77 67 57 47]
                const __m128i T3_08A = _mm_srai_epi32(T2_08A, nShift);             // [30 20 10 00] x8
                const __m128i T3_08B = _mm_srai_epi32(T2_08B, nShift);             // [70 60 50 40]
                const __m128i T3_09A = _mm_srai_epi32(T2_09A, nShift);             // [31 21 11 01] x9
                const __m128i T3_09B = _mm_srai_epi32(T2_09B, nShift);             // [71 61 51 41]
                const __m128i T3_10A = _mm_srai_epi32(T2_10A, nShift);             // [32 22 12 02] xA
                const __m128i T3_10B = _mm_srai_epi32(T2_10B, nShift);             // [72 62 52 42]
                const __m128i T3_11A = _mm_srai_epi32(T2_11A, nShift);             // [33 23 13 03] xB
                const __m128i T3_11B = _mm_srai_epi32(T2_11B, nShift);             // [73 63 53 43]
                const __m128i T3_12A = _mm_srai_epi32(T2_12A, nShift);             // [33 24 14 04] xC
                const __m128i T3_12B = _mm_srai_epi32(T2_12B, nShift);             // [74 64 54 44]
                const __m128i T3_13A = _mm_srai_epi32(T2_13A, nShift);             // [35 25 15 05] xD
                const __m128i T3_13B = _mm_srai_epi32(T2_13B, nShift);             // [75 65 55 45]
                const __m128i T3_14A = _mm_srai_epi32(T2_14A, nShift);             // [36 26 16 06] xE
                const __m128i T3_14B = _mm_srai_epi32(T2_14B, nShift);             // [76 66 56 46]
                const __m128i T3_15A = _mm_srai_epi32(T2_15A, nShift);             // [37 27 17 07] xF
                const __m128i T3_15B = _mm_srai_epi32(T2_15B, nShift);             // [77 67 57 47]

                const __m128i T3_16A = _mm_srai_epi32(T2_16A, nShift);             // [30 20 10 00]
                const __m128i T3_16B = _mm_srai_epi32(T2_16B, nShift);             // [70 60 50 40]
                const __m128i T3_17A = _mm_srai_epi32(T2_17A, nShift);             // [31 21 11 01]
                const __m128i T3_17B = _mm_srai_epi32(T2_17B, nShift);             // [71 61 51 41]
                const __m128i T3_18A = _mm_srai_epi32(T2_18A, nShift);             // [32 22 12 02]
                const __m128i T3_18B = _mm_srai_epi32(T2_18B, nShift);             // [72 62 52 42]
                const __m128i T3_19A = _mm_srai_epi32(T2_19A, nShift);             // [33 23 13 03]
                const __m128i T3_19B = _mm_srai_epi32(T2_19B, nShift);             // [73 63 53 43]
                const __m128i T3_20A = _mm_srai_epi32(T2_20A, nShift);             // [33 24 14 04]
                const __m128i T3_20B = _mm_srai_epi32(T2_20B, nShift);             // [74 64 54 44]
                const __m128i T3_21A = _mm_srai_epi32(T2_21A, nShift);             // [35 25 15 05]
                const __m128i T3_21B = _mm_srai_epi32(T2_21B, nShift);             // [75 65 55 45]
                const __m128i T3_22A = _mm_srai_epi32(T2_22A, nShift);             // [36 26 16 06]
                const __m128i T3_22B = _mm_srai_epi32(T2_22B, nShift);             // [76 66 56 46]
                const __m128i T3_23A = _mm_srai_epi32(T2_23A, nShift);             // [37 27 17 07]
                const __m128i T3_23B = _mm_srai_epi32(T2_23B, nShift);             // [77 67 57 47]
                const __m128i T3_24A = _mm_srai_epi32(T2_24A, nShift);             // [30 20 10 00] x8
                const __m128i T3_24B = _mm_srai_epi32(T2_24B, nShift);             // [70 60 50 40]
                const __m128i T3_25A = _mm_srai_epi32(T2_25A, nShift);             // [31 21 11 01] x9
                const __m128i T3_25B = _mm_srai_epi32(T2_25B, nShift);             // [71 61 51 41]
                const __m128i T3_26A = _mm_srai_epi32(T2_26A, nShift);             // [32 22 12 02] xA
                const __m128i T3_26B = _mm_srai_epi32(T2_26B, nShift);             // [72 62 52 42]
                const __m128i T3_27A = _mm_srai_epi32(T2_27A, nShift);             // [33 23 13 03] xB
                const __m128i T3_27B = _mm_srai_epi32(T2_27B, nShift);             // [73 63 53 43]
                const __m128i T3_28A = _mm_srai_epi32(T2_28A, nShift);             // [33 24 14 04] xC
                const __m128i T3_28B = _mm_srai_epi32(T2_28B, nShift);             // [74 64 54 44]
                const __m128i T3_29A = _mm_srai_epi32(T2_29A, nShift);             // [35 25 15 05] xD
                const __m128i T3_29B = _mm_srai_epi32(T2_29B, nShift);             // [75 65 55 45]
                const __m128i T3_30A = _mm_srai_epi32(T2_30A, nShift);             // [36 26 16 06] xE
                const __m128i T3_30B = _mm_srai_epi32(T2_30B, nShift);             // [76 66 56 46]
                const __m128i T3_31A = _mm_srai_epi32(T2_31A, nShift);             // [37 27 17 07] xF
                const __m128i T3_31B = _mm_srai_epi32(T2_31B, nShift);             // [77 67 57 47]

                res00[part] = _mm_packs_epi32(T3_00A, T3_00B);        // [70 60 50 40 30 20 10 00]
                res01[part] = _mm_packs_epi32(T3_01A, T3_01B);        // [71 61 51 41 31 21 11 01]
                res02[part] = _mm_packs_epi32(T3_02A, T3_02B);        // [72 62 52 42 32 22 12 02]
                res03[part] = _mm_packs_epi32(T3_03A, T3_03B);        // [73 63 53 43 33 23 13 03]
                res04[part] = _mm_packs_epi32(T3_04A, T3_04B);        // [74 64 54 44 34 24 14 04]
                res05[part] = _mm_packs_epi32(T3_05A, T3_05B);        // [75 65 55 45 35 25 15 05]
                res06[part] = _mm_packs_epi32(T3_06A, T3_06B);        // [76 66 56 46 36 26 16 06]
                res07[part] = _mm_packs_epi32(T3_07A, T3_07B);        // [77 67 57 47 37 27 17 07]
                res08[part] = _mm_packs_epi32(T3_08A, T3_08B);        // [A0 ... 80]
                res09[part] = _mm_packs_epi32(T3_09A, T3_09B);        // [A1 ... 81]
                res10[part] = _mm_packs_epi32(T3_10A, T3_10B);        // [A2 ... 82]
                res11[part] = _mm_packs_epi32(T3_11A, T3_11B);        // [A3 ... 83]
                res12[part] = _mm_packs_epi32(T3_12A, T3_12B);        // [A4 ... 84]
                res13[part] = _mm_packs_epi32(T3_13A, T3_13B);        // [A5 ... 85]
                res14[part] = _mm_packs_epi32(T3_14A, T3_14B);        // [A6 ... 86]
                res15[part] = _mm_packs_epi32(T3_15A, T3_15B);        // [A7 ... 87]
                res16[part] = _mm_packs_epi32(T3_16A, T3_16B);
                res17[part] = _mm_packs_epi32(T3_17A, T3_17B);
                res18[part] = _mm_packs_epi32(T3_18A, T3_18B);
                res19[part] = _mm_packs_epi32(T3_19A, T3_19B);
                res20[part] = _mm_packs_epi32(T3_20A, T3_20B);
                res21[part] = _mm_packs_epi32(T3_21A, T3_21B);
                res22[part] = _mm_packs_epi32(T3_22A, T3_22B);
                res23[part] = _mm_packs_epi32(T3_23A, T3_23B);
                res24[part] = _mm_packs_epi32(T3_24A, T3_24B);
                res25[part] = _mm_packs_epi32(T3_25A, T3_25B);
                res26[part] = _mm_packs_epi32(T3_26A, T3_26B);
                res27[part] = _mm_packs_epi32(T3_27A, T3_27B);
                res28[part] = _mm_packs_epi32(T3_28A, T3_28B);
                res29[part] = _mm_packs_epi32(T3_29A, T3_29B);
                res30[part] = _mm_packs_epi32(T3_30A, T3_30B);
                res31[part] = _mm_packs_epi32(T3_31A, T3_31B);
            }
        }
        //transpose matrix 8x8 16bit.
        {
            __m128i tr0_0, tr0_1, tr0_2, tr0_3, tr0_4, tr0_5, tr0_6, tr0_7;
            __m128i tr1_0, tr1_1, tr1_2, tr1_3, tr1_4, tr1_5, tr1_6, tr1_7;
#define TRANSPOSE_8x8_16BIT(I0, I1, I2, I3, I4, I5, I6, I7, O0, O1, O2, O3, O4, O5, O6, O7) \
    tr0_0 = _mm_unpacklo_epi16(I0, I1); \
    tr0_1 = _mm_unpacklo_epi16(I2, I3); \
    tr0_2 = _mm_unpackhi_epi16(I0, I1); \
    tr0_3 = _mm_unpackhi_epi16(I2, I3); \
    tr0_4 = _mm_unpacklo_epi16(I4, I5); \
    tr0_5 = _mm_unpacklo_epi16(I6, I7); \
    tr0_6 = _mm_unpackhi_epi16(I4, I5); \
    tr0_7 = _mm_unpackhi_epi16(I6, I7); \
    tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1); \
    tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3); \
    tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1); \
    tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3); \
    tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5); \
    tr1_5 = _mm_unpacklo_epi32(tr0_6, tr0_7); \
    tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5); \
    tr1_7 = _mm_unpackhi_epi32(tr0_6, tr0_7); \
    O0 = _mm_unpacklo_epi64(tr1_0, tr1_4); \
    O1 = _mm_unpackhi_epi64(tr1_0, tr1_4); \
    O2 = _mm_unpacklo_epi64(tr1_2, tr1_6); \
    O3 = _mm_unpackhi_epi64(tr1_2, tr1_6); \
    O4 = _mm_unpacklo_epi64(tr1_1, tr1_5); \
    O5 = _mm_unpackhi_epi64(tr1_1, tr1_5); \
    O6 = _mm_unpacklo_epi64(tr1_3, tr1_7); \
    O7 = _mm_unpackhi_epi64(tr1_3, tr1_7); \

            TRANSPOSE_8x8_16BIT(res00[0], res01[0], res02[0], res03[0], res04[0], res05[0], res06[0], res07[0], in00[0], in01[0], in02[0], in03[0], in04[0], in05[0], in06[0], in07[0])
                TRANSPOSE_8x8_16BIT(res00[1], res01[1], res02[1], res03[1], res04[1], res05[1], res06[1], res07[1], in08[0], in09[0], in10[0], in11[0], in12[0], in13[0], in14[0], in15[0])
                TRANSPOSE_8x8_16BIT(res00[2], res01[2], res02[2], res03[2], res04[2], res05[2], res06[2], res07[2], in16[0], in17[0], in18[0], in19[0], in20[0], in21[0], in22[0], in23[0])
                TRANSPOSE_8x8_16BIT(res00[3], res01[3], res02[3], res03[3], res04[3], res05[3], res06[3], res07[3], in24[0], in25[0], in26[0], in27[0], in28[0], in29[0], in30[0], in31[0])

                TRANSPOSE_8x8_16BIT(res08[0], res09[0], res10[0], res11[0], res12[0], res13[0], res14[0], res15[0], in00[1], in01[1], in02[1], in03[1], in04[1], in05[1], in06[1], in07[1])
                TRANSPOSE_8x8_16BIT(res08[1], res09[1], res10[1], res11[1], res12[1], res13[1], res14[1], res15[1], in08[1], in09[1], in10[1], in11[1], in12[1], in13[1], in14[1], in15[1])
                TRANSPOSE_8x8_16BIT(res08[2], res09[2], res10[2], res11[2], res12[2], res13[2], res14[2], res15[2], in16[1], in17[1], in18[1], in19[1], in20[1], in21[1], in22[1], in23[1])
                TRANSPOSE_8x8_16BIT(res08[3], res09[3], res10[3], res11[3], res12[3], res13[3], res14[3], res15[3], in24[1], in25[1], in26[1], in27[1], in28[1], in29[1], in30[1], in31[1])

                TRANSPOSE_8x8_16BIT(res16[0], res17[0], res18[0], res19[0], res20[0], res21[0], res22[0], res23[0], in00[2], in01[2], in02[2], in03[2], in04[2], in05[2], in06[2], in07[2])
                TRANSPOSE_8x8_16BIT(res16[1], res17[1], res18[1], res19[1], res20[1], res21[1], res22[1], res23[1], in08[2], in09[2], in10[2], in11[2], in12[2], in13[2], in14[2], in15[2])
                TRANSPOSE_8x8_16BIT(res16[2], res17[2], res18[2], res19[2], res20[2], res21[2], res22[2], res23[2], in16[2], in17[2], in18[2], in19[2], in20[2], in21[2], in22[2], in23[2])
                TRANSPOSE_8x8_16BIT(res16[3], res17[3], res18[3], res19[3], res20[3], res21[3], res22[3], res23[3], in24[2], in25[2], in26[2], in27[2], in28[2], in29[2], in30[2], in31[2])

                TRANSPOSE_8x8_16BIT(res24[0], res25[0], res26[0], res27[0], res28[0], res29[0], res30[0], res31[0], in00[3], in01[3], in02[3], in03[3], in04[3], in05[3], in06[3], in07[3])
                TRANSPOSE_8x8_16BIT(res24[1], res25[1], res26[1], res27[1], res28[1], res29[1], res30[1], res31[1], in08[3], in09[3], in10[3], in11[3], in12[3], in13[3], in14[3], in15[3])
                TRANSPOSE_8x8_16BIT(res24[2], res25[2], res26[2], res27[2], res28[2], res29[2], res30[2], res31[2], in16[3], in17[3], in18[3], in19[3], in20[3], in21[3], in22[3], in23[3])
                TRANSPOSE_8x8_16BIT(res24[3], res25[3], res26[3], res27[3], res28[3], res29[3], res30[3], res31[3], in24[3], in25[3], in26[3], in27[3], in28[3], in29[3], in30[3], in31[3])



#undef TRANSPOSE_8x8_16BIT
        }
    }


    //clip
    {
        __m128i max_val = _mm_set1_epi16((1 << (clip - 1)) - 1);
        __m128i min_val = _mm_set1_epi16(-(1 << (clip - 1)));
        int k;

        for (k = 0; k < 4; k++)
        {
            in00[k] = _mm_min_epi16(in00[k], max_val);
            in00[k] = _mm_max_epi16(in00[k], min_val);
            in01[k] = _mm_min_epi16(in01[k], max_val);
            in01[k] = _mm_max_epi16(in01[k], min_val);
            in02[k] = _mm_min_epi16(in02[k], max_val);
            in02[k] = _mm_max_epi16(in02[k], min_val);
            in03[k] = _mm_min_epi16(in03[k], max_val);
            in03[k] = _mm_max_epi16(in03[k], min_val);
            in04[k] = _mm_min_epi16(in04[k], max_val);
            in04[k] = _mm_max_epi16(in04[k], min_val);
            in05[k] = _mm_min_epi16(in05[k], max_val);
            in05[k] = _mm_max_epi16(in05[k], min_val);
            in06[k] = _mm_min_epi16(in06[k], max_val);
            in06[k] = _mm_max_epi16(in06[k], min_val);
            in07[k] = _mm_min_epi16(in07[k], max_val);
            in07[k] = _mm_max_epi16(in07[k], min_val);
            in08[k] = _mm_min_epi16(in08[k], max_val);
            in08[k] = _mm_max_epi16(in08[k], min_val);
            in09[k] = _mm_min_epi16(in09[k], max_val);
            in09[k] = _mm_max_epi16(in09[k], min_val);
            in10[k] = _mm_min_epi16(in10[k], max_val);
            in10[k] = _mm_max_epi16(in10[k], min_val);
            in11[k] = _mm_min_epi16(in11[k], max_val);
            in11[k] = _mm_max_epi16(in11[k], min_val);
            in12[k] = _mm_min_epi16(in12[k], max_val);
            in12[k] = _mm_max_epi16(in12[k], min_val);
            in13[k] = _mm_min_epi16(in13[k], max_val);
            in13[k] = _mm_max_epi16(in13[k], min_val);
            in14[k] = _mm_min_epi16(in14[k], max_val);
            in14[k] = _mm_max_epi16(in14[k], min_val);
            in15[k] = _mm_min_epi16(in15[k], max_val);
            in15[k] = _mm_max_epi16(in15[k], min_val);
            in16[k] = _mm_min_epi16(in16[k], max_val);
            in16[k] = _mm_max_epi16(in16[k], min_val);
            in17[k] = _mm_min_epi16(in17[k], max_val);
            in17[k] = _mm_max_epi16(in17[k], min_val);
            in18[k] = _mm_min_epi16(in18[k], max_val);
            in18[k] = _mm_max_epi16(in18[k], min_val);
            in19[k] = _mm_min_epi16(in19[k], max_val);
            in19[k] = _mm_max_epi16(in19[k], min_val);
            in20[k] = _mm_min_epi16(in20[k], max_val);
            in20[k] = _mm_max_epi16(in20[k], min_val);
            in21[k] = _mm_min_epi16(in21[k], max_val);
            in21[k] = _mm_max_epi16(in21[k], min_val);
            in22[k] = _mm_min_epi16(in22[k], max_val);
            in22[k] = _mm_max_epi16(in22[k], min_val);
            in23[k] = _mm_min_epi16(in23[k], max_val);
            in23[k] = _mm_max_epi16(in23[k], min_val);
            in24[k] = _mm_min_epi16(in24[k], max_val);
            in24[k] = _mm_max_epi16(in24[k], min_val);
            in25[k] = _mm_min_epi16(in25[k], max_val);
            in25[k] = _mm_max_epi16(in25[k], min_val);
            in26[k] = _mm_min_epi16(in26[k], max_val);
            in26[k] = _mm_max_epi16(in26[k], min_val);
            in27[k] = _mm_min_epi16(in27[k], max_val);
            in27[k] = _mm_max_epi16(in27[k], min_val);
            in28[k] = _mm_min_epi16(in28[k], max_val);
            in28[k] = _mm_max_epi16(in28[k], min_val);
            in29[k] = _mm_min_epi16(in29[k], max_val);
            in29[k] = _mm_max_epi16(in29[k], min_val);
            in30[k] = _mm_min_epi16(in30[k], max_val);
            in30[k] = _mm_max_epi16(in30[k], min_val);
            in31[k] = _mm_min_epi16(in31[k], max_val);
            in31[k] = _mm_max_epi16(in31[k], min_val);
        }
    }

    // Add
    for (i = 0; i < 2; i++)
    {
#define STORE_LINE(L0, L1, L2, L3, L4, L5, L6, L7, H0, H1, H2, H3, H4, H5, H6, H7, offsetV, offsetH) \
    _mm_store_si128((__m128i*)&blk[(0 + (offsetV)) * 32 + (offsetH)+0], L0); \
    _mm_store_si128((__m128i*)&blk[(0 + (offsetV)) * 32 + (offsetH)+8], H0); \
    _mm_store_si128((__m128i*)&blk[(1 + (offsetV)) * 32 + (offsetH)+0], L1); \
    _mm_store_si128((__m128i*)&blk[(1 + (offsetV)) * 32 + (offsetH)+8], H1); \
    _mm_store_si128((__m128i*)&blk[(2 + (offsetV)) * 32 + (offsetH)+0], L2); \
    _mm_store_si128((__m128i*)&blk[(2 + (offsetV)) * 32 + (offsetH)+8], H2); \
    _mm_store_si128((__m128i*)&blk[(3 + (offsetV)) * 32 + (offsetH)+0], L3); \
    _mm_store_si128((__m128i*)&blk[(3 + (offsetV)) * 32 + (offsetH)+8], H3); \
    _mm_store_si128((__m128i*)&blk[(4 + (offsetV)) * 32 + (offsetH)+0], L4); \
    _mm_store_si128((__m128i*)&blk[(4 + (offsetV)) * 32 + (offsetH)+8], H4); \
    _mm_store_si128((__m128i*)&blk[(5 + (offsetV)) * 32 + (offsetH)+0], L5); \
    _mm_store_si128((__m128i*)&blk[(5 + (offsetV)) * 32 + (offsetH)+8], H5); \
    _mm_store_si128((__m128i*)&blk[(6 + (offsetV)) * 32 + (offsetH)+0], L6); \
    _mm_store_si128((__m128i*)&blk[(6 + (offsetV)) * 32 + (offsetH)+8], H6); \
    _mm_store_si128((__m128i*)&blk[(7 + (offsetV)) * 32 + (offsetH)+0], L7); \
    _mm_store_si128((__m128i*)&blk[(7 + (offsetV)) * 32 + (offsetH)+8], H7);

        const int k = i * 2;
        STORE_LINE(in00[k], in01[k], in02[k], in03[k], in04[k], in05[k], in06[k], in07[k], in00[k + 1], in01[k + 1], in02[k + 1], in03[k + 1], in04[k + 1], in05[k + 1], in06[k + 1], in07[k + 1], 0, i * 16)
            STORE_LINE(in08[k], in09[k], in10[k], in11[k], in12[k], in13[k], in14[k], in15[k], in08[k + 1], in09[k + 1], in10[k + 1], in11[k + 1], in12[k + 1], in13[k + 1], in14[k + 1], in15[k + 1], 8, i * 16)
            STORE_LINE(in16[k], in17[k], in18[k], in19[k], in20[k], in21[k], in22[k], in23[k], in16[k + 1], in17[k + 1], in18[k + 1], in19[k + 1], in20[k + 1], in21[k + 1], in22[k + 1], in23[k + 1], 16, i * 16)
            STORE_LINE(in24[k], in25[k], in26[k], in27[k], in28[k], in29[k], in30[k], in31[k], in24[k + 1], in25[k + 1], in26[k + 1], in27[k + 1], in28[k + 1], in29[k + 1], in30[k + 1], in31[k + 1], 24, i * 16)
#undef STORE_LINE
    }
}

/* wrapper: 适配 (coeff, w, h, bit_depth) 签名 */
static void idct_32x32_sse41(int16_t *coeff, int w, int h, int bit_depth)
{
    (void)w; (void)h;
    idct_32x32_sse41_core(coeff, 20 - bit_depth, bit_depth + 1);
}

/* ===========================================================================
 * 残差叠加: dst[i] = clip(dst[i] + coeff[i], 0, max_val)
 * SSE4 实现: 一次处理 8 个 int16 像素 (128-bit)
 * C fallback 在 itx.c 中实现并注册
 * ===========================================================================
 */

/* SSE4: 10-bit 路径 (uint16_t dst + int16_t coeff) */
static void recon_residual_sse4_10bit(uint8_t *dst, ptrdiff_t stride,
                                      const int16_t *coeff, int w, int h,
                                      int bit_depth)
{
    uint16_t *dst16 = (uint16_t *)(void *)dst;
    int stride16 = (int)(stride >> 1);
    int max_val = (1 << bit_depth) - 1;
    __m128i v_zero = _mm_setzero_si128();
    __m128i v_max = _mm_set1_epi16((short)max_val);
    /* 计算首行对齐偏移 (stride 64 字节对齐, 各行偏移相同) */
    int misalign = (int)((uintptr_t)&dst16[0] & 15) >> 1;  /* uint16_t 元素数 */
    int head = (8 - misalign) & 7;  /* 对齐到 16 字节边界需跳过的元素数 */
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
        /* SSE4: 一次处理 8 个像素 (dst 已对齐) */
        for (; x + 7 < w; x += 8) {
            __m128i d = _mm_load_si128((const __m128i *)&dst16[y * stride16 + x]);
            __m128i c = _mm_load_si128((const __m128i *)&coeff[y * w + x]);
            __m128i r = _mm_add_epi16(d, c);
            /* clip: 先 max(0) 去负值 (有符号比较), 再 min(max_val) (无符号比较) */
            r = _mm_max_epi16(r, v_zero);
            r = _mm_min_epu16(r, v_max);
            _mm_store_si128((__m128i *)&dst16[y * stride16 + x], r);
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

/* SSE4: 8-bit 路径 (uint8_t dst + int16_t coeff, 需扩展/打包) */
static void recon_residual_sse4_8bit(uint8_t *dst, ptrdiff_t stride,
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

/* SSE4 dispatcher */
static void recon_residual_sse4(uint8_t *dst, ptrdiff_t stride, const int16_t *coeff,
                                int w, int h, int bit_depth)
{
    if (bit_depth > 8) {
        recon_residual_sse4_10bit(dst, stride, coeff, w, h, bit_depth);
    } else {
        recon_residual_sse4_8bit(dst, stride, coeff, w, h, bit_depth);
    }
}

/* ===========================================================================
 * 注册函数
 * ===========================================================================
 */

/* SSE4: 注册 4x4/8x8/16x16/32x32 IDCT + 残差叠加 */
void avs2_itx_init_sse41(const avs2_cpu_flags *flags)
{
    (void)flags;
    avs2_dsp_table.itx[0] = idct_4x4_sse41;
    avs2_dsp_table.itx[1] = idct_8x8_sse41;
    avs2_dsp_table.itx[2] = idct_16x16_sse41;
    avs2_dsp_table.itx[3] = idct_32x32_sse41;
    avs2_dsp_table.recon_residual = recon_residual_sse4;
}

/* AVX2: 已统一到 SSE4 路径, 不再覆盖注册 */
void avs2_itx_init_avx2(const avs2_cpu_flags *flags) { (void)flags; }

/* 测试接口: 暴露内部 static 函数供 test_idct.c 调用 */
#ifdef AVS2_TEST_EXPOSE
void idct_4x4_sse41_test(int16_t *coeff, int w, int h, int bit_depth)
{
    idct_4x4_sse41(coeff, w, h, bit_depth);
}
void idct_8x8_sse41_test(int16_t *coeff, int w, int h, int bit_depth)
{
    idct_8x8_sse41(coeff, w, h, bit_depth);
}
void idct_16x16_sse41_test(int16_t *coeff, int w, int h, int bit_depth)
{
    idct_16x16_sse41(coeff, w, h, bit_depth);
}
void idct_32x32_sse41_test(int16_t *coeff, int w, int h, int bit_depth)
{
    idct_32x32_sse41(coeff, w, h, bit_depth);
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
