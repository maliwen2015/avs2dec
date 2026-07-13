/*
 * itx.c - 反变换实现
 *
 * 从 davs2 (source/common/transform.cc) 移植到 C。
 * 实现 AVS2 反 DCT (4x4/8x8/16x16/32x32/64x64)、NSST 二次变换、
 * 以及 64x64 块的 LOT (线性正交变换)。
 *
 * 系数类型: int16_t; 位深通过参数运行时传递, 同时支持 8-bit 与
 * 10-bit (等价于 davs2 的 HIGH_BIT_DEPTH 逻辑)。
 */

#include "internal.h"
#include "tables.h"
#include <string.h>

/* ---- 局部宏 ---- */
#ifndef AVS2_CLIP3
#define AVS2_CLIP3(min_val, max_val, val) \
    ((val) < (min_val) ? (min_val) : ((val) > (max_val) ? (max_val) : (val)))
#endif

/* LOT 小波变换抽头数 (5-3 小波) */
#define LOT_MAX_WLT_TAP 2

/* ---------------------------------------------------------------------------
 * DCT 变换矩阵 (用于 partial butterfly 反变换, 取自 davs2)
 * 注意: 这些矩阵的缩放与 avs2_t4/avs2_t8 等不同, 不能混用。
 */
static const int16_t g_t4[4][4] = {
    { 32,  32,  32,  32 },
    { 42,  17, -17, -42 },
    { 32, -32, -32,  32 },
    { 17, -42,  42, -17 }
};

static const int16_t g_t8[8][8] = {
    { 32,  32,  32,  32,  32,  32,  32,  32 },
    { 44,  38,  25,   9,  -9, -25, -38, -44 },
    { 42,  17, -17, -42, -42, -17,  17,  42 },
    { 38,  -9, -44, -25,  25,  44,   9, -38 },
    { 32, -32, -32,  32,  32, -32, -32,  32 },
    { 25, -44,   9,  38, -38,  -9,  44, -25 },
    { 17, -42,  42, -17, -17,  42, -42,  17 },
    {  9, -25,  38, -44,  44, -38,  25,  -9 }
};

static const int16_t g_t16[16][16] = {
    { 32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32,  32 },
    { 45,  43,  40,  35,  29,  21,  13,   4,  -4, -13, -21, -29, -35, -40, -43, -45 },
    { 44,  38,  25,   9,  -9, -25, -38, -44, -44, -38, -25,  -9,   9,  25,  38,  44 },
    { 43,  29,   4, -21, -40, -45, -35, -13,  13,  35,  45,  40,  21,  -4, -29, -43 },
    { 42,  17, -17, -42, -42, -17,  17,  42,  42,  17, -17, -42, -42, -17,  17,  42 },
    { 40,   4, -35, -43, -13,  29,  45,  21, -21, -45, -29,  13,  43,  35,  -4, -40 },
    { 38,  -9, -44, -25,  25,  44,   9, -38, -38,   9,  44,  25, -25, -44,  -9,  38 },
    { 35, -21, -43,   4,  45,  13, -40, -29,  29,  40, -13, -45,  -4,  43,  21, -35 },
    { 32, -32, -32,  32,  32, -32, -32,  32,  32, -32, -32,  32,  32, -32, -32,  32 },
    { 29, -40, -13,  45,  -4, -43,  21,  35, -35, -21,  43,   4, -45,  13,  40, -29 },
    { 25, -44,   9,  38, -38,  -9,  44, -25, -25,  44,  -9, -38,  38,   9, -44,  25 },
    { 21, -45,  29,  13, -43,  35,   4, -40,  40,  -4, -35,  43, -13, -29,  45, -21 },
    { 17, -42,  42, -17, -17,  42, -42,  17,  17, -42,  42, -17, -17,  42, -42,  17 },
    { 13, -35,  45, -40,  21,   4, -29,  43, -43,  29,  -4, -21,  40, -45,  35, -13 },
    {  9, -25,  38, -44,  44, -38,  25,  -9,  -9,  25, -38,  44, -44,  38, -25,   9 },
    {  4, -13,  21, -29,  35, -40,  43, -45,  45, -43,  40, -35,  29, -21,  13,  -4 }
};

static const int16_t g_t32[32][32] = {
    { 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32 },
    { 45, 45, 44, 43, 41, 39, 36, 34, 30, 27, 23, 19, 15, 11,  7,  2, -2, -7,-11,-15,-19,-23,-27,-30,-34,-36,-39,-41,-43,-44,-45,-45 },
    { 45, 43, 40, 35, 29, 21, 13,  4, -4,-13,-21,-29,-35,-40,-43,-45,-45,-43,-40,-35,-29,-21,-13, -4,  4, 13, 21, 29, 35, 40, 43, 45 },
    { 45, 41, 34, 23, 11, -2,-15,-27,-36,-43,-45,-44,-39,-30,-19, -7,  7, 19, 30, 39, 44, 45, 43, 36, 27, 15,  2,-11,-23,-34,-41,-45 },
    { 44, 38, 25,  9, -9,-25,-38,-44,-44,-38,-25, -9,  9, 25, 38, 44, 44, 38, 25,  9, -9,-25,-38,-44,-44,-38,-25, -9,  9, 25, 38, 44 },
    { 44, 34, 15, -7,-27,-41,-45,-39,-23, -2, 19, 36, 45, 43, 30, 11,-11,-30,-43,-45,-36,-19,  2, 23, 39, 45, 41, 27,  7,-15,-34,-44 },
    { 43, 29,  4,-21,-40,-45,-35,-13, 13, 35, 45, 40, 21, -4,-29,-43,-43,-29, -4, 21, 40, 45, 35, 13,-13,-35,-45,-40,-21,  4, 29, 43 },
    { 43, 23, -7,-34,-45,-36,-11, 19, 41, 44, 27, -2,-30,-45,-39,-15, 15, 39, 45, 30,  2,-27,-44,-41,-19, 11, 36, 45, 34,  7,-23,-43 },
    { 42, 17,-17,-42,-42,-17, 17, 42, 42, 17,-17,-42,-42,-17, 17, 42, 42, 17,-17,-42,-42,-17, 17, 42, 42, 17,-17,-42,-42,-17, 17, 42 },
    { 41, 11,-27,-45,-30,  7, 39, 43, 15,-23,-45,-34,  2, 36, 44, 19,-19,-44,-36, -2, 34, 45, 23,-15,-43,-39, -7, 30, 45, 27,-11,-41 },
    { 40,  4,-35,-43,-13, 29, 45, 21,-21,-45,-29, 13, 43, 35, -4,-40,-40, -4, 35, 43, 13,-29,-45,-21, 21, 45, 29,-13,-43,-35,  4, 40 },
    { 39, -2,-41,-36,  7, 43, 34,-11,-44,-30, 15, 45, 27,-19,-45,-23, 23, 45, 19,-27,-45,-15, 30, 44, 11,-34,-43, -7, 36, 41,  2,-39 },
    { 38, -9,-44,-25, 25, 44,  9,-38,-38,  9, 44, 25,-25,-44, -9, 38, 38, -9,-44,-25, 25, 44,  9,-38,-38,  9, 44, 25,-25,-44, -9, 38 },
    { 36,-15,-45,-11, 39, 34,-19,-45, -7, 41, 30,-23,-44, -2, 43, 27,-27,-43,  2, 44, 23,-30,-41,  7, 45, 19,-34,-39, 11, 45, 15,-36 },
    { 35,-21,-43,  4, 45, 13,-40,-29, 29, 40,-13,-45, -4, 43, 21,-35,-35, 21, 43, -4,-45,-13, 40, 29,-29,-40, 13, 45,  4,-43,-21, 35 },
    { 34,-27,-39, 19, 43,-11,-45,  2, 45,  7,-44,-15, 41, 23,-36,-30, 30, 36,-23,-41, 15, 44, -7,-45, -2, 45, 11,-43,-19, 39, 27,-34 },
    { 32,-32,-32, 32, 32,-32,-32, 32, 32,-32,-32, 32, 32,-32,-32, 32, 32,-32,-32, 32, 32,-32,-32, 32, 32,-32,-32, 32, 32,-32,-32, 32 },
    { 30,-36,-23, 41, 15,-44, -7, 45, -2,-45, 11, 43,-19,-39, 27, 34,-34,-27, 39, 19,-43,-11, 45,  2,-45,  7, 44,-15,-41, 23, 36,-30 },
    { 29,-40,-13, 45, -4,-43, 21, 35,-35,-21, 43,  4,-45, 13, 40,-29,-29, 40, 13,-45,  4, 43,-21,-35, 35, 21,-43, -4, 45,-13,-40, 29 },
    { 27,-43, -2, 44,-23,-30, 41,  7,-45, 19, 34,-39,-11, 45,-15,-36, 36, 15,-45, 11, 39,-34,-19, 45, -7,-41, 30, 23,-44,  2, 43,-27 },
    { 25,-44,  9, 38,-38, -9, 44,-25,-25, 44, -9,-38, 38,  9,-44, 25, 25,-44,  9, 38,-38, -9, 44,-25,-25, 44, -9,-38, 38,  9,-44, 25 },
    { 23,-45, 19, 27,-45, 15, 30,-44, 11, 34,-43,  7, 36,-41,  2, 39,-39, -2, 41,-36, -7, 43,-34,-11, 44,-30,-15, 45,-27,-19, 45,-23 },
    { 21,-45, 29, 13,-43, 35,  4,-40, 40, -4,-35, 43,-13,-29, 45,-21,-21, 45,-29,-13, 43,-35, -4, 40,-40,  4, 35,-43, 13, 29,-45, 21 },
    { 19,-44, 36, -2,-34, 45,-23,-15, 43,-39,  7, 30,-45, 27, 11,-41, 41,-11,-27, 45,-30, -7, 39,-43, 15, 23,-45, 34,  2,-36, 44,-19 },
    { 17,-42, 42,-17,-17, 42,-42, 17, 17,-42, 42,-17,-17, 42,-42, 17, 17,-42, 42,-17,-17, 42,-42, 17, 17,-42, 42,-17,-17, 42,-42, 17 },
    { 15,-39, 45,-30,  2, 27,-44, 41,-19,-11, 36,-45, 34, -7,-23, 43,-43, 23,  7,-34, 45,-36, 11, 19,-41, 44,-27, -2, 30,-45, 39,-15 },
    { 13,-35, 45,-40, 21,  4,-29, 43,-43, 29, -4,-21, 40,-45, 35,-13,-13, 35,-45, 40,-21, -4, 29,-43, 43,-29,  4, 21,-40, 45,-35, 13 },
    { 11,-30, 43,-45, 36,-19, -2, 23,-39, 45,-41, 27, -7,-15, 34,-44, 44,-34, 15,  7,-27, 41,-45, 39,-23,  2, 19,-36, 45,-43, 30,-11 },
    {  9,-25, 38,-44, 44,-38, 25, -9, -9, 25,-38, 44,-44, 38,-25,  9,  9,-25, 38,-44, 44,-38, 25, -9, -9, 25,-38, 44,-44, 38,-25,  9 },
    {  7,-19, 30,-39, 44,-45, 43,-36, 27,-15,  2, 11,-23, 34,-41, 45,-45, 41,-34, 23,-11, -2, 15,-27, 36,-43, 45,-44, 39,-30, 19, -7 },
    {  4,-13, 21,-29, 35,-40, 43,-45, 45,-43, 40,-35, 29,-21, 13, -4, -4, 13,-21, 29,-35, 40,-43, 45,-45, 43,-40, 35,-29, 21,-13,  4 },
    {  2, -7, 11,-15, 19,-23, 27,-30, 34,-36, 39,-41, 43,-44, 45,-45, 45,-45, 44,-43, 41,-39, 36,-34, 30,-27, 23,-19, 15,-11,  7, -2 }
};

/* ---------------------------------------------------------------------------
 * 4x4 二次变换 (NSST) 矩阵
 */
/* 亮度用 */
static const int16_t g_2t[SEC_TR_SIZE * SEC_TR_SIZE] = {
    123,  -35,   -8,   -3,
    -32, -120,   30,   10,
     14,   25,  123,  -22,
      8,   13,   19,  126
};
/* 色度用 */
static const int16_t g_2t_c[SEC_TR_SIZE * SEC_TR_SIZE] = {
     34,   58,   72,   81,
     77,   69,   -7,  -75,
     79,  -33,  -75,   58,
     55,  -84,   73,  -28
};


/**
 * ===========================================================================
 * partial butterfly 反变换
 * ===========================================================================
 */

/* 4 点 partial butterfly 反变换 */
static void partial_butterfly_inverse4(const int16_t *src, int16_t *dst,
                                       int shift, int line, int clip_depth)
{
    int e[2], o[2];
    int max_val = (1 << (clip_depth - 1)) - 1;
    int min_val = -max_val - 1;
    int add = 1 << (shift - 1);
    int j;

    for (j = 0; j < line; j++) {
        /* 利用对称性最大化减少乘法 */
        o[0] = g_t4[1][0] * src[line] + g_t4[3][0] * src[3 * line];
        o[1] = g_t4[1][1] * src[line] + g_t4[3][1] * src[3 * line];
        e[0] = g_t4[0][0] * src[0   ] + g_t4[2][0] * src[2 * line];
        e[1] = g_t4[0][1] * src[0   ] + g_t4[2][1] * src[2 * line];

        /* 合并偶数项与奇数项得到最终空域向量 */
        dst[0] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[0] + o[0] + add) >> shift));
        dst[1] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[1] + o[1] + add) >> shift));
        dst[2] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[1] - o[1] + add) >> shift));
        dst[3] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[0] - o[0] + add) >> shift));

        src++;
        dst += 4;
    }
}

/* 8 点 partial butterfly 反变换 */
static void partial_butterfly_inverse8(const int16_t *src, int16_t *dst,
                                       int shift, int line, int clip_depth)
{
    int e[4], o[4];
    int ee[2], eo[2];
    int max_val = (1 << (clip_depth - 1)) - 1;
    int min_val = -max_val - 1;
    int add = 1 << (shift - 1);
    int j, k;

    for (j = 0; j < line; j++) {
        /* 利用对称性最大化减少乘法 */
        for (k = 0; k < 4; k++) {
            o[k] = g_t8[1][k] * src[    line] +
                   g_t8[3][k] * src[3 * line] +
                   g_t8[5][k] * src[5 * line] +
                   g_t8[7][k] * src[7 * line];
        }

        eo[0] = g_t8[2][0] * src[2 * line] + g_t8[6][0] * src[6 * line];
        eo[1] = g_t8[2][1] * src[2 * line] + g_t8[6][1] * src[6 * line];
        ee[0] = g_t8[0][0] * src[0       ] + g_t8[4][0] * src[4 * line];
        ee[1] = g_t8[0][1] * src[0       ] + g_t8[4][1] * src[4 * line];

        /* 合并偶数项与奇数项得到最终空域向量 */
        e[0] = ee[0] + eo[0];
        e[3] = ee[0] - eo[0];
        e[1] = ee[1] + eo[1];
        e[2] = ee[1] - eo[1];

        for (k = 0; k < 4; k++) {
            dst[k]     = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[k] + o[k] + add) >> shift));
            dst[k + 4] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[3 - k] - o[3 - k] + add) >> shift));
        }

        src++;
        dst += 8;
    }
}

/* 16 点 partial butterfly 反变换 */
static void partial_butterfly_inverse16(const int16_t *src, int16_t *dst,
                                        int shift, int line, int clip_depth)
{
    int e[8], o[8];
    int ee[4], eo[4];
    int eee[2], eeo[2];
    int max_val = (1 << (clip_depth - 1)) - 1;
    int min_val = -max_val - 1;
    int add = 1 << (shift - 1);
    int j, k;

    for (j = 0; j < line; j++) {
        /* 利用对称性最大化减少乘法 */
        for (k = 0; k < 8; k++) {
            o[k] = g_t16[ 1][k] * src[     line] +
                   g_t16[ 3][k] * src[ 3 * line] +
                   g_t16[ 5][k] * src[ 5 * line] +
                   g_t16[ 7][k] * src[ 7 * line] +
                   g_t16[ 9][k] * src[ 9 * line] +
                   g_t16[11][k] * src[11 * line] +
                   g_t16[13][k] * src[13 * line] +
                   g_t16[15][k] * src[15 * line];
        }

        for (k = 0; k < 4; k++) {
            eo[k] = g_t16[ 2][k] * src[ 2 * line] +
                    g_t16[ 6][k] * src[ 6 * line] +
                    g_t16[10][k] * src[10 * line] +
                    g_t16[14][k] * src[14 * line];
        }

        eeo[0] = g_t16[4][0] * src[4 * line] + g_t16[12][0] * src[12 * line];
        eee[0] = g_t16[0][0] * src[0       ] + g_t16[ 8][0] * src[ 8 * line];
        eeo[1] = g_t16[4][1] * src[4 * line] + g_t16[12][1] * src[12 * line];
        eee[1] = g_t16[0][1] * src[0       ] + g_t16[ 8][1] * src[ 8 * line];

        /* 合并偶数项与奇数项得到最终空域向量 */
        for (k = 0; k < 2; k++) {
            ee[k    ] = eee[k    ] + eeo[k    ];
            ee[k + 2] = eee[1 - k] - eeo[1 - k];
        }

        for (k = 0; k < 4; k++) {
            e[k    ] = ee[k    ] + eo[k    ];
            e[k + 4] = ee[3 - k] - eo[3 - k];
        }

        for (k = 0; k < 8; k++) {
            dst[k]     = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[k] + o[k] + add) >> shift));
            dst[k + 8] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[7 - k] - o[7 - k] + add) >> shift));
        }

        src++;
        dst += 16;
    }
}

/* 32 点 partial butterfly 反变换 */
static void partial_butterfly_inverse32(const int16_t *src, int16_t *dst,
                                        int shift, int line, int clip_depth)
{
    int e[16], o[16];
    int ee[8], eo[8];
    int eee[4], eeo[4];
    int eeee[2], eeeo[2];
    int max_val = (1 << (clip_depth - 1)) - 1;
    int min_val = -max_val - 1;
    int add = 1 << (shift - 1);
    int j, k;

    for (j = 0; j < line; j++) {
        /* 利用对称性最大化减少乘法 */
        for (k = 0; k < 16; k++) {
            o[k] = g_t32[ 1][k] * src[     line] +
                   g_t32[ 3][k] * src[ 3 * line] +
                   g_t32[ 5][k] * src[ 5 * line] +
                   g_t32[ 7][k] * src[ 7 * line] +
                   g_t32[ 9][k] * src[ 9 * line] +
                   g_t32[11][k] * src[11 * line] +
                   g_t32[13][k] * src[13 * line] +
                   g_t32[15][k] * src[15 * line] +
                   g_t32[17][k] * src[17 * line] +
                   g_t32[19][k] * src[19 * line] +
                   g_t32[21][k] * src[21 * line] +
                   g_t32[23][k] * src[23 * line] +
                   g_t32[25][k] * src[25 * line] +
                   g_t32[27][k] * src[27 * line] +
                   g_t32[29][k] * src[29 * line] +
                   g_t32[31][k] * src[31 * line];
        }

        for (k = 0; k < 8; k++) {
            eo[k] = g_t32[ 2][k] * src[ 2 * line] +
                    g_t32[ 6][k] * src[ 6 * line] +
                    g_t32[10][k] * src[10 * line] +
                    g_t32[14][k] * src[14 * line] +
                    g_t32[18][k] * src[18 * line] +
                    g_t32[22][k] * src[22 * line] +
                    g_t32[26][k] * src[26 * line] +
                    g_t32[30][k] * src[30 * line];
        }

        for (k = 0; k < 4; k++) {
            eeo[k] = g_t32[ 4][k] * src[ 4 * line] +
                     g_t32[12][k] * src[12 * line] +
                     g_t32[20][k] * src[20 * line] +
                     g_t32[28][k] * src[28 * line];
        }

        eeeo[0] = g_t32[8][0] * src[8 * line] + g_t32[24][0] * src[24 * line];
        eeeo[1] = g_t32[8][1] * src[8 * line] + g_t32[24][1] * src[24 * line];
        eeee[0] = g_t32[0][0] * src[0       ] + g_t32[16][0] * src[16 * line];
        eeee[1] = g_t32[0][1] * src[0       ] + g_t32[16][1] * src[16 * line];

        /* 合并偶数项与奇数项得到最终空域向量 */
        eee[0] = eeee[0] + eeeo[0];
        eee[3] = eeee[0] - eeeo[0];
        eee[1] = eeee[1] + eeeo[1];
        eee[2] = eeee[1] - eeeo[1];
        for (k = 0; k < 4; k++) {
            ee[k    ] = eee[k    ] + eeo[k    ];
            ee[k + 4] = eee[3 - k] - eeo[3 - k];
        }

        for (k = 0; k < 8; k++) {
            e[k    ] = ee[k    ] + eo[k    ];
            e[k + 8] = ee[7 - k] - eo[7 - k];
        }

        for (k = 0; k < 16; k++) {
            dst[k]      = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[k] + o[k] + add) >> shift));
            dst[k + 16] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[15 - k] - o[15 - k] + add) >> shift));
        }

        src++;
        dst += 32;
    }
}


/**
 * ===========================================================================
 * 反 DCT 入口 (方形块)
 * ===========================================================================
 */

/* 4x4 反 DCT */
void idct_4x4_c(const int16_t *src, int16_t *dst, int i_dst, int bit_depth)
{
#define BSIZE 4
    int16_t coeff[BSIZE * BSIZE];
    int16_t block[BSIZE * BSIZE];
    int shift1 = 5;
    int shift2 = 20 - bit_depth;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = bit_depth + 1;
    int i;

    partial_butterfly_inverse4(src,   coeff, shift1, BSIZE, clip_depth1);
    partial_butterfly_inverse4(coeff, block, shift2, BSIZE, clip_depth2);

    for (i = 0; i < BSIZE; i++) {
        memcpy(dst, &block[i * BSIZE], BSIZE * sizeof(int16_t));
        dst += i_dst;
    }
#undef BSIZE
}

/* 8x8 反 DCT */
void idct_8x8_c(const int16_t *src, int16_t *dst, int i_dst, int bit_depth)
{
#define BSIZE 8
    int16_t coeff[BSIZE * BSIZE];
    int16_t block[BSIZE * BSIZE];
    int shift1 = 5;
    int shift2 = 20 - bit_depth;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = bit_depth + 1;
    int i;

    partial_butterfly_inverse8(src,   coeff, shift1, BSIZE, clip_depth1);
    partial_butterfly_inverse8(coeff, block, shift2, BSIZE, clip_depth2);

    for (i = 0; i < BSIZE; i++) {
        memcpy(dst, &block[i * BSIZE], BSIZE * sizeof(int16_t));
        dst += i_dst;
    }
#undef BSIZE
}

/* 16x16 反 DCT */
void idct_16x16_c(const int16_t *src, int16_t *dst, int i_dst, int bit_depth)
{
#define BSIZE 16
    int16_t coeff[BSIZE * BSIZE];
    int16_t block[BSIZE * BSIZE];
    int shift1 = 5;
    int shift2 = 20 - bit_depth;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = bit_depth + 1;
    int i;

    partial_butterfly_inverse16(src,   coeff, shift1, BSIZE, clip_depth1);
    partial_butterfly_inverse16(coeff, block, shift2, BSIZE, clip_depth2);

    for (i = 0; i < BSIZE; i++) {
        memcpy(dst, &block[i * BSIZE], BSIZE * sizeof(int16_t));
        dst += i_dst;
    }
#undef BSIZE
}

/* ---------------------------------------------------------------------------
 * 32x32 反 DCT
 * 注意: i_dst 最低位作为附加小波标志位 (LOT 路径使用)
 */
void idct_32x32_c(const int16_t *src, int16_t *dst, int i_dst, int bit_depth)
{
#define BSIZE 32
    int16_t coeff[BSIZE * BSIZE];
    int16_t block[BSIZE * BSIZE];
    int a_flag = i_dst & 0x01;
    int shift1 = 5;
    int shift2 = 20 - bit_depth - a_flag;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = bit_depth + 1 + a_flag;
    int i;

    i_dst &= 0xFE;    /* 去除标志位 */

    partial_butterfly_inverse32(src,   coeff, shift1, BSIZE, clip_depth1);
    partial_butterfly_inverse32(coeff, block, shift2, BSIZE, clip_depth2);

    for (i = 0; i < BSIZE; i++) {
        memcpy(dst, &block[i * BSIZE], BSIZE * sizeof(int16_t));
        dst += i_dst;
    }
#undef BSIZE
}

/* ---------------------------------------------------------------------------
 * 64x64 反 DCT (基于 LOT)
 * 注意: i_dst 最低位作为附加小波标志位
 */
static void idct_64x64_c(const int16_t *src, int16_t *dst, int i_dst, int bit_depth)
{
    int16_t row_buf[64 + LOT_MAX_WLT_TAP * 2];
    int16_t *p_ext = row_buf + LOT_MAX_WLT_TAP;
    const int n0 = 64;
    const int n1 = 64 >> 1;
    int x, y, offset;

    /* 步骤 0: 先做 32x32 反变换 */
    idct_32x32_c(src, dst, i_dst | 1, bit_depth);

    /* 步骤 1: 竖直方向 LOT 滤波 */
    for (x = 0; x < n0; x++) {
        /* 拷贝: offset 按 i_dst (=64) 步进, 读取 32x32 IDCT 输出的 32 行.
         * 原 offset += 32 是步长错误 (i_dst 的一半), 会读到列错位的数据. */
        for (y = 0, offset = 0; y < n1; y++, offset += n0) {
            p_ext[y << 1] = dst[x + offset];
        }

        /* 镜像反射 */
        p_ext[n0] = p_ext[n0 - 2];

        /* 滤波 (偶数像素) */
        for (y = 0; y <= n0; y += 2) {
            p_ext[y] >>= 1;
        }

        /* 滤波 (奇数像素) */
        for (y = 1; y < n0; y += 2) {
            p_ext[y] = (int16_t)((p_ext[y - 1] + p_ext[y + 1]) >> 1);
        }

        /* 回写 */
        for (y = 0, offset = 0; y < n0; y++, offset += n0) {
            dst[x + offset] = p_ext[y];
        }
    }

    /* 步骤 2: 水平方向 LOT 滤波 */
    for (y = 0, offset = 0; y < n0; y++, offset += n0) {
        /* 拷贝 */
        for (x = 0; x < n1; x++) {
            p_ext[x << 1] = dst[offset + x];
        }

        /* 镜像反射 */
        p_ext[n0] = p_ext[n0 - 2];

        /* 滤波 (奇数像素) */
        for (x = 1; x < n0; x += 2) {
            p_ext[x] = (int16_t)((p_ext[x - 1] + p_ext[x + 1]) >> 1);
        }

        /* 回写 */
        memcpy(dst + offset, p_ext, n0 * sizeof(int16_t));
    }
}


/**
 * ===========================================================================
 * NSST (非分离二次变换) 反变换
 * ===========================================================================
 */

/* 4x4 二次变换 1D 反变换 - 竖直方向 */
static void x_tr_2nd_4_1d_inv_ver(int16_t *coeff, int i_coeff, int i_shift,
                                  const int16_t *tc)
{
    int tmp_dct[SEC_TR_SIZE * SEC_TR_SIZE];
    const int add = 1 << (i_shift - 1);
    int i, j, k, sum;

    for (i = 0; i < SEC_TR_SIZE; i++) {
        for (j = 0; j < SEC_TR_SIZE; j++) {
            tmp_dct[i * SEC_TR_SIZE + j] = coeff[i * i_coeff + j];
        }
    }

    for (i = 0; i < SEC_TR_SIZE; i++) {
        for (j = 0; j < SEC_TR_SIZE; j++) {
            sum = add;
            for (k = 0; k < SEC_TR_SIZE; k++) {
                sum += tc[k * SEC_TR_SIZE + i] * tmp_dct[k * SEC_TR_SIZE + j];
            }
            coeff[i * i_coeff + j] = (int16_t)AVS2_CLIP3(-32768, 32767, sum >> i_shift);
        }
    }
}

/* 4x4 二次变换 1D 反变换 - 水平方向 */
static void x_tr_2nd_4_1d_inv_hor(int16_t *coeff, int i_coeff, int i_shift,
                                  int clip_depth, const int16_t *tc)
{
    int tmp_dct[SEC_TR_SIZE * SEC_TR_SIZE];
    const int max_val = (1 << (clip_depth - 1)) - 1;
    const int min_val = -max_val - 1;
    const int add = 1 << (i_shift - 1);
    int i, j, k, sum;

    for (i = 0; i < SEC_TR_SIZE; i++) {
        for (j = 0; j < SEC_TR_SIZE; j++) {
            tmp_dct[i * SEC_TR_SIZE + j] = coeff[i * i_coeff + j];
        }
    }

    for (i = 0; i < SEC_TR_SIZE; i++) {
        for (j = 0; j < SEC_TR_SIZE; j++) {
            sum = add;
            for (k = 0; k < SEC_TR_SIZE; k++) {
                sum += tc[k * SEC_TR_SIZE + i] * tmp_dct[j * SEC_TR_SIZE + k];
            }
            coeff[j * i_coeff + i] = (int16_t)AVS2_CLIP3(min_val, max_val, sum >> i_shift);
        }
    }
}

/* ---------------------------------------------------------------------------
 * 4x4 二次反变换 (色度/特殊路径), 使用色度矩阵 g_2t_c
 */
static void inv_transform_4x4_2nd_c(int16_t *coeff, int i_coeff, int bit_depth)
{
    const int shift1 = 5;
    const int shift2 = 20 - bit_depth + 2;
    const int clip_depth2 = bit_depth + 1;

    x_tr_2nd_4_1d_inv_ver(coeff, i_coeff, shift1, g_2t_c);
    x_tr_2nd_4_1d_inv_hor(coeff, i_coeff, shift2, clip_depth2, g_2t_c);
}

/* ---------------------------------------------------------------------------
 * 二次反变换 (亮度, 4x4 以上块)
 * i_mode     - 亮度真实帧内模式
 * b_top      - 上邻块是否可用
 * b_left     - 左邻块是否可用
 */
static void inv_transform_2nd_c(int16_t *coeff, int i_coeff, int i_mode,
                                int b_top, int b_left)
{
    int vt = (i_mode >=  0 && i_mode <= 23);
    int ht = (i_mode >= 13 && i_mode <= 32) || (i_mode >= 0 && i_mode <= 2);

    if (ht && b_left) {
        x_tr_2nd_4_1d_inv_hor(coeff, i_coeff, 7, 16, g_2t);
    }
    if (vt && b_top) {
        x_tr_2nd_4_1d_inv_ver(coeff, i_coeff, 7, g_2t);
    }
}


/**
 * ===========================================================================
 * 反变换入口
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * 反变换入口 (对应 davs2 inv_transform)
 * coeff              - 系数缓冲 (原地变换)
 * bsx/bsy            - 块宽/高 (4/8/16/32/64)
 * bit_depth          - 位深 (8 或 10)
 * b_sec_t            - 是否启用二次变换 (NSST)
 * i_luma_intra_mode  - 亮度帧内模式 (仅 b_sec_t 时有效)
 * b_top/b_left       - 上/左邻块可用性 (仅 b_sec_t 时有效)
 */
void avs2_inverse_transform(int16_t *coeff, int bsx, int bsy, int bit_depth,
                            int b_sec_t, int i_luma_intra_mode,
                            int b_top, int b_left)
{
    int idx;

    if (bsx == 4 && bsy == 4) {
        idx = 0;
    } else if (bsx == 8 && bsy == 8) {
        idx = 1;
    } else if (bsx == 16 && bsy == 16) {
        idx = 2;
    } else if (bsx == 32 && bsy == 32) {
        idx = 3;
    } else if (bsx == 64 && bsy == 64) {
        idx = 4;
    } else {
        /* 非方形或不支持尺寸: 不做变换 */
        return;
    }

    if (idx == 0) {
        /* 4x4: 二次变换与主变换二选一 */
        if (b_sec_t) {
            inv_transform_4x4_2nd_c(coeff, bsx, bit_depth);
        } else {
            /* 通过 DSP 函数指针调用 (可能使用 SIMD) */
            avs2_dsp_table.itx[0](coeff, bsx, bsy, bit_depth);
        }
    } else {
        /* 其它尺寸: 先做二次变换, 再做主变换 */
        if (b_sec_t) {
            inv_transform_2nd_c(coeff, bsx, i_luma_intra_mode, b_top, b_left);
        }
        avs2_dsp_table.itx[idx](coeff, bsx, bsy, bit_depth);
    }
}


/**
 * ===========================================================================
 * DSP 注册
 * ===========================================================================
 */

/* DSP 入口: 4x4 反 DCT */
static void itrans_dct4(int16_t *coeff, int w, int h, int bit_depth)
{
    (void)h;
    idct_4x4_c(coeff, coeff, w, bit_depth);
}

/* DSP 入口: 8x8 反 DCT */
static void itrans_dct8(int16_t *coeff, int w, int h, int bit_depth)
{
    (void)h;
    idct_8x8_c(coeff, coeff, w, bit_depth);
}

/* DSP 入口: 16x16 反 DCT */
static void itrans_dct16(int16_t *coeff, int w, int h, int bit_depth)
{
    (void)h;
    idct_16x16_c(coeff, coeff, w, bit_depth);
}

/* DSP 入口: 32x32 反 DCT */
static void itrans_dct32(int16_t *coeff, int w, int h, int bit_depth)
{
    (void)h;
    idct_32x32_c(coeff, coeff, w, bit_depth);
}

/* DSP 入口: 64x64 反 DCT (LOT) */
static void itrans_dct64(int16_t *coeff, int w, int h, int bit_depth)
{
    (void)h;
    idct_64x64_c(coeff, coeff, w, bit_depth);
}

/* ---------------------------------------------------------------------------
 * 残差叠加 C fallback: dst[i] = clip(dst[i] + coeff[i], 0, max_val)
 */
static void recon_residual_c(uint8_t *dst, ptrdiff_t stride, const int16_t *coeff,
                             int w, int h, int bit_depth)
{
    int max_val = (1 << bit_depth) - 1;
    int x, y;
    if (bit_depth > 8) {
        uint16_t *dst16 = (uint16_t *)(void *)dst;
        int stride16 = (int)(stride >> 1);
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                int v = dst16[y * stride16 + x] + coeff[y * w + x];
                if (v < 0) v = 0;
                if (v > max_val) v = max_val;
                dst16[y * stride16 + x] = (uint16_t)v;
            }
        }
    } else {
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                int v = dst[y * stride + x] + coeff[y * w + x];
                if (v < 0) v = 0;
                if (v > max_val) v = max_val;
                dst[y * stride + x] = (uint8_t)v;
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * 反变换 DSP 初始化
 */
void avs2_itx_init(void)
{
    avs2_dsp_table.itx[0] = itrans_dct4;
    avs2_dsp_table.itx[1] = itrans_dct8;
    avs2_dsp_table.itx[2] = itrans_dct16;
    avs2_dsp_table.itx[3] = itrans_dct32;
    avs2_dsp_table.itx[4] = itrans_dct64;
    avs2_dsp_table.recon_residual = recon_residual_c;
}
