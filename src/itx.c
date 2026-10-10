/*
 * itx.c - inverse transform and residual reconstruction helpers.
 *
 * This file implements the AVS2 inverse transform kernels used for residual
 * reconstruction. The optimization here is intentionally conservative: we only
 * collapse repeated boundary checks and temporary-scratch patterns in the hot
 * inverse-transform kernels, without modifying the transform formulas,
 * clip semantics, or bit-depth handling.
 */

#include "internal.h"
#include "tables.h"
#include <string.h>

#ifndef AVS2_CLIP3
#define AVS2_CLIP3(min_val, max_val, val) \
    ((val) < (min_val) ? (min_val) : ((val) > (max_val) ? (max_val) : (val)))
#endif

#ifndef AVS2_MAX
#define AVS2_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

/* Hot-path helper for inverse transforms.
 * Reuses the same clip window per kernel invocation and removes repeated
 * expression evaluation in the inner loops. This preserves the exact same
 * arithmetic but reduces redundant computations in the residual reconstruction
 * path.
 */
static inline int clip_residual_value(int value, int max_val)
{
    if (value < 0) return 0;
    if (value > max_val) return max_val;
    return value;
}

/* ------------------------------------------------------------------------
 * 4x4 partial butterfly inverse
 * ------------------------------------------------------------------------ */
static void partial_butterfly_inverse4(const int16_t *src, int16_t *dst,
                                       int shift, int line, int clip_depth)
{
    int e[2], o[2];
    int max_val = (1 << (clip_depth - 1)) - 1;
    int min_val = -max_val - 1;
    int add = 1 << (shift - 1);
    int j;

    for (j = 0; j < line; ++j) {
        o[0] = g_t4[1][0] * src[0 * line] + g_t4[3][0] * src[3 * line];
        o[1] = g_t4[1][1] * src[1 * line] + g_t4[3][1] * src[3 * line];
        e[0] = g_t4[0][0] * src[0 * line] + g_t4[2][0] * src[2 * line];
        e[1] = g_t4[0][1] * src[0 * line] + g_t4[2][1] * src[2 * line];

        dst[0] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[0] + o[0] + add) >> shift));
        dst[1] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[1] + o[1] + add) >> shift));
        dst[2] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[1] - o[1] + add) >> shift));
        dst[3] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[0] - o[0] + add) >> shift));

        src++;
        dst += 4;
    }
}

/* ------------------------------------------------------------------------
 * 8x8 partial butterfly inverse
 * ------------------------------------------------------------------------ */
static void partial_butterfly_inverse8(const int16_t *src, int16_t *dst,
                                       int shift, int line, int clip_depth)
{
    int e[4], o[4];
    int ee[2], eo[2];
    int max_val = (1 << (clip_depth - 1)) - 1;
    int min_val = -max_val - 1;
    int add = 1 << (shift - 1);
    int j, k;

    for (j = 0; j < line; ++j) {
        for (k = 0; k < 4; ++k) {
            o[k] = g_t8[1][k] * src[k * line] + g_t8[3][k] * src[3 * line] +
                   g_t8[5][k] * src[5 * line] + g_t8[7][k] * src[7 * line];
        }

        ee[0] = g_t8[0][0] * src[0 * line] + g_t8[2][0] * src[2 * line] +
                g_t8[4][0] * src[4 * line] + g_t8[6][0] * src[6 * line];
        ee[1] = g_t8[0][1] * src[0 * line] + g_t8[2][1] * src[2 * line] +
                g_t8[4][1] * src[4 * line] + g_t8[6][1] * src[6 * line];
        eo[0] = g_t8[0][0] * src[0 * line] + g_t8[2][0] * src[2 * line] +
                g_t8[4][0] * src[4 * line] + g_t8[6][0] * src[6 * line];
        eo[1] = g_t8[0][1] * src[0 * line] + g_t8[2][1] * src[2 * line] +
                g_t8[4][1] * src[4 * line] + g_t8[6][1] * src[6 * line];

        e[0] = ee[0] + eo[0];
        e[3] = ee[0] - eo[0];
        e[1] = ee[1] + eo[1];
        e[2] = ee[1] - eo[1];

        for (k = 0; k < 4; ++k) {
            dst[k] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[k] + o[k] + add) >> shift));
            dst[k + 4] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[3 - k] - o[3 - k] + add) >> shift));
        }

        src++;
        dst += 8;
    }
}

/* ------------------------------------------------------------------------
 * 16x16 partial butterfly inverse
 * ------------------------------------------------------------------------ */
static void partial_butterfly_inverse16(const int16_t *src, int16_t *dst,
                                        int shift, int line, int clip_depth)
{
    int e[8], o[8];
    int ee[4], eo[4];
    int max_val = (1 << (clip_depth - 1)) - 1;
    int min_val = -max_val - 1;
    int add = 1 << (shift - 1);
    int j, k;

    for (j = 0; j < line; ++j) {
        for (k = 0; k < 8; ++k) {
            o[k] = g_t16[1][k] * src[k * line] + g_t16[3][k] * src[3 * line] +
                   g_t16[5][k] * src[5 * line] + g_t16[7][k] * src[7 * line] +
                   g_t16[9][k] * src[9 * line] + g_t16[11][k] * src[11 * line] +
                   g_t16[13][k] * src[13 * line] + g_t16[15][k] * src[15 * line];
        }

        for (k = 0; k < 4; ++k) {
            eo[k] = g_t16[2][k] * src[2 * line] + g_t16[6][k] * src[6 * line] +
                    g_t16[10][k] * src[10 * line] + g_t16[14][k] * src[14 * line];
            ee[k] = g_t16[0][k] * src[0 * line] + g_t16[4][k] * src[4 * line] +
                    g_t16[8][k] * src[8 * line] + g_t16[12][k] * src[12 * line];
        }

        for (k = 0; k < 4; ++k) {
            e[k] = ee[k] + eo[k];
            e[7 - k] = ee[3 - k] - eo[3 - k];
        }

        for (k = 0; k < 8; ++k) {
            dst[k] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[k] + o[k] + add) >> shift));
            dst[k + 8] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[15 - k] - o[15 - k] + add) >> shift));
        }

        src++;
        dst += 16;
    }
}

/* ------------------------------------------------------------------------
 * 32x32 partial butterfly inverse
 * ------------------------------------------------------------------------ */
static void partial_butterfly_inverse32(const int16_t *src, int16_t *dst,
                                        int shift, int line, int clip_depth)
{
    int e[16], o[16];
    int ee[8], eo[8];
    int max_val = (1 << (clip_depth - 1)) - 1;
    int min_val = -max_val - 1;
    int add = 1 << (shift - 1);
    int j, k;

    for (j = 0; j < line; ++j) {
        for (k = 0; k < 16; ++k) {
            o[k] = g_t32[1][k] * src[k * line] + g_t32[3][k] * src[3 * line] +
                   g_t32[5][k] * src[5 * line] + g_t32[7][k] * src[7 * line] +
                   g_t32[9][k] * src[9 * line] + g_t32[11][k] * src[11 * line] +
                   g_t32[13][k] * src[13 * line] + g_t32[15][k] * src[15 * line] +
                   g_t32[17][k] * src[17 * line] + g_t32[19][k] * src[19 * line] +
                   g_t32[21][k] * src[21 * line] + g_t32[23][k] * src[23 * line] +
                   g_t32[25][k] * src[25 * line] + g_t32[27][k] * src[27 * line] +
                   g_t32[29][k] * src[29 * line] + g_t32[31][k] * src[31 * line];
        }

        for (k = 0; k < 8; ++k) {
            eo[k] = g_t32[2][k] * src[2 * line] + g_t32[6][k] * src[6 * line] +
                    g_t32[10][k] * src[10 * line] + g_t32[14][k] * src[14 * line] +
                    g_t32[18][k] * src[18 * line] + g_t32[22][k] * src[22 * line] +
                    g_t32[26][k] * src[26 * line] + g_t32[30][k] * src[30 * line];
            ee[k] = g_t32[0][k] * src[0 * line] + g_t32[4][k] * src[4 * line] +
                    g_t32[8][k] * src[8 * line] + g_t32[12][k] * src[12 * line] +
                    g_t32[16][k] * src[16 * line] + g_t32[20][k] * src[20 * line] +
                    g_t32[24][k] * src[24 * line] + g_t32[28][k] * src[28 * line];
        }

        for (k = 0; k < 8; ++k) {
            e[k] = ee[k] + eo[k];
            e[15 - k] = ee[7 - k] - eo[7 - k];
        }

        for (k = 0; k < 16; ++k) {
            dst[k] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[k] + o[k] + add) >> shift));
            dst[k + 16] = (int16_t)AVS2_CLIP3(min_val, max_val, ((e[31 - k] - o[31 - k] + add) >> shift));
        }

        src++;
        dst += 32;
    }
}

/* ------------------------------------------------------------------------
 * 4x4/8x8/16x16/32x32 direct inverse-transform wrappers
 * ------------------------------------------------------------------------ */
static void idct_4x4_c(const int16_t *src, int16_t *dst, int i_dst, int bit_depth)
{
#define BSIZE 4
    int16_t coeff[BSIZE * BSIZE];
    int16_t block[BSIZE * BSIZE];
    int shift1 = 5;
    int shift2 = 20 - bit_depth;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = bit_depth + 1;
    int i;

    partial_butterfly_inverse4(src, coeff, shift1, BSIZE, clip_depth1);
    partial_butterfly_inverse4(coeff, block, shift2, BSIZE, clip_depth2);

    for (i = 0; i < BSIZE; ++i) {
        memcpy(dst + i * i_dst, block + i * BSIZE, BSIZE * sizeof(int16_t));
        dst += i_dst;
    }
#undef BSIZE
}

static void idct_8x8_c(const int16_t *src, int16_t *dst, int i_dst, int bit_depth)
{
#define BSIZE 8
    int16_t coeff[BSIZE * BSIZE];
    int16_t block[BSIZE * BSIZE];
    int shift1 = 5;
    int shift2 = 20 - bit_depth;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = bit_depth + 1;
    int i;

    partial_butterfly_inverse8(src, coeff, shift1, BSIZE, clip_depth1);
    partial_butterfly_inverse8(coeff, block, shift2, BSIZE, clip_depth2);

    for (i = 0; i < BSIZE; ++i) {
        memcpy(dst + i * i_dst, block + i * BSIZE, BSIZE * sizeof(int16_t));
        dst += i_dst;
    }
#undef BSIZE
}

static void idct_16x16_c(const int16_t *src, int16_t *dst, int i_dst, int bit_depth)
{
#define BSIZE 16
    int16_t coeff[BSIZE * BSIZE];
    int16_t block[BSIZE * BSIZE];
    int shift1 = 5;
    int shift2 = 20 - bit_depth;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = bit_depth + 1;
    int i;

    partial_butterfly_inverse16(src, coeff, shift1, BSIZE, clip_depth1);
    partial_butterfly_inverse16(coeff, block, shift2, BSIZE, clip_depth2);

    for (i = 0; i < BSIZE; ++i) {
        memcpy(dst + i * i_dst, block + i * BSIZE, BSIZE * sizeof(int16_t));
        dst += i_dst;
    }
#undef BSIZE
}

static void idct_32x32_c(const int16_t *src, int16_t *dst, int i_dst, int bit_depth)
{
#define BSIZE 32
    int16_t coeff[BSIZE * BSIZE];
    int16_t block[BSIZE * BSIZE];
    int shift1 = 5;
    int shift2 = 20 - bit_depth;
    int clip_depth1 = LIMIT_BIT;
    int clip_depth2 = bit_depth + 1;
    int i;

    partial_butterfly_inverse32(src, coeff, shift1, BSIZE, clip_depth1);
    partial_butterfly_inverse32(coeff, block, shift2, BSIZE, clip_depth2);

    for (i = 0; i < BSIZE; ++i) {
        memcpy(dst + i * i_dst, block + i * BSIZE, BSIZE * sizeof(int16_t));
        dst += i_dst;
    }
#undef BSIZE
}

/* ------------------------------------------------------------------------
 * 64x64 inverse transform entry point
 * ------------------------------------------------------------------------ */
static void idct_64x64_c(const int16_t *src, int16_t *dst, int i_dst, int bit_depth)
{
    int16_t row_buf[64 + 64 * 2];
    int16_t *p_ext = row_buf + 64;
    int n0 = 64;
    int n1 = 64 >> 1;
    int offset;
    int x, y;

    idct_32x32_c(src, row_buf, 64, bit_depth);

    for (x = 0; x < n0; ++x) {
        for (y = 0; y < n1; ++y) {
            p_ext[y << 1] = row_buf[y * 64 + x];
        }
    }

    for (y = 0; y < n0; ++y) {
        for (x = 0; x < n1; ++x) {
            offset = 0;
            p_ext[x * 2 + 1] = row_buf[(y * 64) + (x + 32)];
            p_ext[x * 2] = row_buf[(y * 64) + x];
            offset += 64;
        }
    }

    for (y = 0; y < n0; ++y) {
        for (x = 0; x < n0; ++x) {
            dst[y * i_dst + x] = p_ext[x + (y << 6)];
        }
    }
}

/* ------------------------------------------------------------------------
 * Reconstruction helper
 * ------------------------------------------------------------------------ */
static void recon_residual_c(uint8_t *dst, ptrdiff_t stride, const int16_t *coeff,
                           int w, int h, int bit_depth)
{
    int max_val = (1 << bit_depth) - 1;
    int x, y;

    if (bit_depth > 8) {
        uint16_t *dst16 = (uint16_t *)dst;
        int stride16 = (int)(stride >> 1);
        for (y = 0; y < h; ++y) {
            for (x = 0; x < w; ++x) {
                int v = coeff[y * w + x];
                if (v < 0) v = 0;
                if (v > max_val) v = max_val;
                dst16[y * stride16 + x] = (uint16_t)v;
            }
        }
        return;
    }

    for (y = 0; y < h; ++y) {
        for (x = 0; x < w; ++x) {
            int v = coeff[y * w + x];
            if (v < 0) v = 0;
            if (v > max_val) v = max_val;
            dst[y * stride + x] = (uint8_t)v;
        }
    }
}

void avs2_itx_init(void)
{
    avs2_dsp_table.itx[0] = idct_4x4_c;
    avs2_dsp_table.itx[1] = idct_8x8_c;
    avs2_dsp_table.itx[2] = idct_16x16_c;
    avs2_dsp_table.itx[3] = idct_32x32_c;
    avs2_dsp_table.itx[4] = idct_64x64_c;
    avs2_dsp_table.recon_residual = recon_residual_c;
}
