/*
 * ALF (Adaptive Loop Filter) 环路滤波 (从 davs2 alf.cc 移植到 C)。
 *
 * 完整算法:
 *   - 9 抽头 7x7 对称滤波器, 16 个区域自适应 (region adaptive);
 *   - 两级滤波: block1 为主体 7x7 滤波, block2 为 4 角点边界修正;
 *   - 第 9 个 (中心) 系数由前 8 个系数重建:
 *     coeff[8] = (1<<ALF_NUM_BIT_SHIFT) - 2*sum(coeff[0..7]) + coeff[8].
 *
 * 命名: 小写加下划线。支持 8/10 位, 通过 bit_depth 分支。
 */

#include "internal.h"
#include <string.h>

/* 辅助宏 (本文件局部使用) */
#ifndef AVS2_CLIP3
#define AVS2_CLIP3(lo, hi, x) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif

/* --------------------------------------------------------------------------
 * 重建 ALF 第 9 个 (中心) 系数 (对应 davs2 alf_recon_coefficients 单组)
 * coeff[8] = (1<<shift) - 2*sum(coeff[0..7]) + coeff[8]
 * -------------------------------------------------------------------------- */
void avs2_alf_recon_last_coeff(int16_t *coeff)
{
    int sum = 0;
    int i;
    for (i = 0; i < ALF_MAX_NUM_COEF - 1; i++)
        sum += 2 * coeff[i];
    coeff[ALF_MAX_NUM_COEF - 1] = (int16_t)((1 << ALF_NUM_BIT_SHIFT) - sum
                                          + coeff[ALF_MAX_NUM_COEF - 1]);
}

/* ------------------------------------------------------------------
 * ALF 滤波函数模板: 通过宏为 8 位 (uint8_t) 和 10 位 (uint16_t) 各生成
 * 一份实现, 再由分发函数按 bit_depth 选择调用。
 * stride 均为元素步长 (byte_stride / bytes_per_sample)。
 * ------------------------------------------------------------------ */

/* --------------------------------------------------------------------------
 * block1: 主体 7x7 滤波 (对应 davs2 alf_filter_block1)
 * 对 LCU 区域 (含上下 4 行扩展) 应用 9 抽头对称滤波
 * dst/src 为平面基址, stride 为元素步长
 *
 * 系数加权布局:
 *   coeff[0]: ±3 行同列
 *   coeff[1]: ±2 行同列
 *   coeff[2]: +1 行右1列 + -1 行左1列 (交叉)
 *   coeff[3]: ±1 行同列
 *   coeff[4]: +1 行左1列 + -1 行右1列 (交叉)
 *   coeff[5]: 0 行 ±3 列
 *   coeff[6]: 0 行 ±2 列
 *   coeff[7]: 0 行 ±1 列
 *   coeff[8]: 0 行 0 列 (中心, 由重建得到)
 * -------------------------------------------------------------------------- */
#define DEFINE_ALF_FILTER_BLOCK1(func_name, pel_t) \
static void func_name(pel_t *dst, const pel_t *src, int stride, \
                      int lcu_pix_x, int lcu_pix_y, \
                      int lcu_width, int lcu_height, \
                      int *alf_coeff, int b_top_avail, int b_down_avail, \
                      int bit_depth) \
{ \
    const int pel_add = 1 << (ALF_NUM_BIT_SHIFT - 1); \
    const int pel_max = (1 << bit_depth) - 1; \
    const int min_x   = -3; \
    const int max_x   = lcu_width - 1 + 3; \
    int x, y; \
    const pel_t *imgPad1, *imgPad2, *imgPad3, *imgPad4, *imgPad5, *imgPad6; \
    { \
        /* 上下边界扩展 4 行 (b_top/b_down_avail 时) */ \
        int startPos = b_top_avail ? (lcu_pix_y - 4) : lcu_pix_y; \
        int endPos   = b_down_avail ? (lcu_pix_y + lcu_height - 4) : (lcu_pix_y + lcu_height); \
        src += (startPos * stride) + lcu_pix_x; \
        dst += (startPos * stride) + lcu_pix_x; \
        lcu_height = endPos - startPos; \
        lcu_height--;  /* 循环用 <= */ \
    } \
    for (y = 0; y <= lcu_height; y++) { \
        int yUp, yBottom; \
        /* ±1 行 (带行边界裁剪) */ \
        yUp     = AVS2_CLIP3(0, lcu_height, y - 1); \
        yBottom = AVS2_CLIP3(0, lcu_height, y + 1); \
        imgPad1 = src + (yBottom - y) * stride; \
        imgPad2 = src + (yUp     - y) * stride; \
        /* ±2 行 */ \
        yUp     = AVS2_CLIP3(0, lcu_height, y - 2); \
        yBottom = AVS2_CLIP3(0, lcu_height, y + 2); \
        imgPad3 = src + (yBottom - y) * stride; \
        imgPad4 = src + (yUp     - y) * stride; \
        /* ±3 行 */ \
        yUp     = AVS2_CLIP3(0, lcu_height, y - 3); \
        yBottom = AVS2_CLIP3(0, lcu_height, y + 3); \
        imgPad5 = src + (yBottom - y) * stride; \
        imgPad6 = src + (yUp     - y) * stride; \
        for (x = 0; x < lcu_width; x++) { \
            int xLeft, xRight; \
            int pel_val; \
            pel_val  = alf_coeff[0] * (imgPad5[x] + imgPad6[x]); \
            pel_val += alf_coeff[1] * (imgPad3[x] + imgPad4[x]); \
            xLeft    = AVS2_CLIP3(min_x, max_x, x - 1); \
            xRight   = AVS2_CLIP3(min_x, max_x, x + 1); \
            pel_val += alf_coeff[2] * (imgPad1[xRight] + imgPad2[xLeft ]); \
            pel_val += alf_coeff[3] * (imgPad1[x     ] + imgPad2[x     ]); \
            pel_val += alf_coeff[4] * (imgPad1[xLeft ] + imgPad2[xRight]); \
            pel_val += alf_coeff[7] * (src    [xRight] + src    [xLeft ]); \
            xLeft    = AVS2_CLIP3(min_x, max_x, x - 2); \
            xRight   = AVS2_CLIP3(min_x, max_x, x + 2); \
            pel_val += alf_coeff[6] * (src    [xRight] + src    [xLeft ]); \
            xLeft    = AVS2_CLIP3(min_x, max_x, x - 3); \
            xRight   = AVS2_CLIP3(min_x, max_x, x + 3); \
            pel_val += alf_coeff[5] * (src    [xRight] + src    [xLeft ]); \
            pel_val += alf_coeff[8] * (src    [x     ]); \
            pel_val = (pel_val + pel_add) >> ALF_NUM_BIT_SHIFT; \
            dst[x]  = (pel_t)AVS2_CLIP3(0, pel_max, pel_val); \
        } \
        src += stride; \
        dst += stride; \
    } \
}

DEFINE_ALF_FILTER_BLOCK1(alf_filter_block1_c_8,  uint8_t)
DEFINE_ALF_FILTER_BLOCK1(alf_filter_block1_c_16, uint16_t)

void alf_filter_block1_c(uint8_t *dst, const uint8_t *src, int stride,
                         int lcu_pix_x, int lcu_pix_y,
                         int lcu_width, int lcu_height,
                         int *alf_coeff, int b_top_avail, int b_down_avail,
                         int bit_depth)
{
    if (bit_depth > 8)
        alf_filter_block1_c_16((uint16_t *)dst, (const uint16_t *)src, stride,
                                lcu_pix_x, lcu_pix_y, lcu_width, lcu_height,
                                alf_coeff, b_top_avail, b_down_avail, bit_depth);
    else
        alf_filter_block1_c_8(dst, src, stride,
                               lcu_pix_x, lcu_pix_y, lcu_width, lcu_height,
                               alf_coeff, b_top_avail, b_down_avail, bit_depth);
}

/* --------------------------------------------------------------------------
 * block2: 边界修正滤波 (对应 davs2 alf_filter_block2)
 * 对 LCU 4 个角点 (左上/右上/左下/右下) 做修正,
 * 仅当角点像素与水平邻居不同时才滤波
 * -------------------------------------------------------------------------- */
#define DEFINE_ALF_FILTER_BLOCK2(func_name, pel_t) \
static void func_name(pel_t *dst, const pel_t *src, int stride, \
                      int lcu_pix_x, int lcu_pix_y, \
                      int lcu_width, int lcu_height, \
                      int *alf_coeff, int b_top_avail, int b_down_avail, \
                      int bit_depth) \
{ \
    const int pel_max = (1 << bit_depth) - 1; \
    const pel_t *p_src1, *p_src2, *p_src3, *p_src4, *p_src5, *p_src6; \
    int pixel_int; \
    int startPos = b_top_avail ? (lcu_pix_y - 4) : lcu_pix_y; \
    int endPos   = b_down_avail ? (lcu_pix_y + lcu_height - 4) : (lcu_pix_y + lcu_height); \
    /* 定位到起始行 */ \
    src += (startPos * stride) + lcu_pix_x; \
    dst += (startPos * stride) + lcu_pix_x; \
    /* --- 左上角 (与左侧邻居不同时滤波) --- */ \
    if (src[0] != src[-1]) { \
        p_src1 = src + 1 * stride; \
        p_src2 = src; \
        p_src3 = src + 2 * stride; \
        p_src4 = src; \
        p_src5 = src + 3 * stride; \
        p_src6 = src; \
        pixel_int  = alf_coeff[0] * (p_src5[ 0] + p_src6[ 0]); \
        pixel_int += alf_coeff[1] * (p_src3[ 0] + p_src4[ 0]); \
        pixel_int += alf_coeff[2] * (p_src1[ 1] + p_src2[ 0]); \
        pixel_int += alf_coeff[3] * (p_src1[ 0] + p_src2[ 0]); \
        pixel_int += alf_coeff[4] * (p_src1[-1] + p_src2[ 1]); \
        pixel_int += alf_coeff[7] * (src    [ 1] + src    [-1]); \
        pixel_int += alf_coeff[6] * (src    [ 2] + src    [-2]); \
        pixel_int += alf_coeff[5] * (src    [ 3] + src    [-3]); \
        pixel_int += alf_coeff[8] * (src    [ 0]); \
        pixel_int = (pixel_int + 32) >> 6; \
        dst[0] = (pel_t)AVS2_CLIP3(0, pel_max, pixel_int); \
    } \
    /* --- 右上角 (与右侧邻居不同时滤波) --- */ \
    src += lcu_width - 1; \
    dst += lcu_width - 1; \
    if (src[0] != src[1]) { \
        p_src1 = src + 1 * stride; \
        p_src2 = src; \
        p_src3 = src + 2 * stride; \
        p_src4 = src; \
        p_src5 = src + 3 * stride; \
        p_src6 = src; \
        pixel_int  = alf_coeff[0] * (p_src5[ 0] + p_src6[ 0]); \
        pixel_int += alf_coeff[1] * (p_src3[ 0] + p_src4[ 0]); \
        pixel_int += alf_coeff[2] * (p_src1[ 1] + p_src2[-1]); \
        pixel_int += alf_coeff[3] * (p_src1[ 0] + p_src2[ 0]); \
        pixel_int += alf_coeff[4] * (p_src1[-1] + p_src2[ 0]); \
        pixel_int += alf_coeff[7] * (src    [ 1] + src    [-1]); \
        pixel_int += alf_coeff[6] * (src    [ 2] + src    [-2]); \
        pixel_int += alf_coeff[5] * (src    [ 3] + src    [-3]); \
        pixel_int += alf_coeff[8] * (src    [ 0]); \
        pixel_int = (pixel_int + 32) >> 6; \
        dst[0] = (pel_t)AVS2_CLIP3(0, pel_max, pixel_int); \
    } \
    /* --- 左下角 (与左侧邻居不同时滤波) --- */ \
    src -= lcu_width - 1; \
    dst -= lcu_width - 1; \
    src += ((endPos - startPos - 1) * stride); \
    dst += ((endPos - startPos - 1) * stride); \
    if (src[0] != src[-1]) { \
        p_src1 = src; \
        p_src2 = src - 1 * stride; \
        p_src3 = src; \
        p_src4 = src - 2 * stride; \
        p_src5 = src; \
        p_src6 = src - 3 * stride; \
        pixel_int  = alf_coeff[0] * (p_src5[ 0] + p_src6[ 0]); \
        pixel_int += alf_coeff[1] * (p_src3[ 0] + p_src4[ 0]); \
        pixel_int += alf_coeff[2] * (p_src1[ 1] + p_src2[-1]); \
        pixel_int += alf_coeff[3] * (p_src1[ 0] + p_src2[ 0]); \
        pixel_int += alf_coeff[4] * (p_src1[ 0] + p_src2[ 1]); \
        pixel_int += alf_coeff[7] * (src    [ 1] + src    [-1]); \
        pixel_int += alf_coeff[6] * (src    [ 2] + src    [-2]); \
        pixel_int += alf_coeff[5] * (src    [ 3] + src    [-3]); \
        pixel_int += alf_coeff[8] * (src    [ 0]); \
        pixel_int = (pixel_int + 32) >> 6; \
        dst[0] = (pel_t)AVS2_CLIP3(0, pel_max, pixel_int); \
    } \
    /* --- 右下角 (与右侧邻居不同时滤波) --- */ \
    src += lcu_width - 1; \
    dst += lcu_width - 1; \
    if (src[0] != src[1]) { \
        p_src1 = src; \
        p_src2 = src - 1 * stride; \
        p_src3 = src; \
        p_src4 = src - 2 * stride; \
        p_src5 = src; \
        p_src6 = src - 3 * stride; \
        pixel_int  = alf_coeff[0] * (p_src5[ 0] + p_src6[ 0]); \
        pixel_int += alf_coeff[1] * (p_src3[ 0] + p_src4[ 0]); \
        pixel_int += alf_coeff[2] * (p_src1[ 0] + p_src2[-1]); \
        pixel_int += alf_coeff[3] * (p_src1[ 0] + p_src2[ 0]); \
        pixel_int += alf_coeff[4] * (p_src1[-1] + p_src2[ 1]); \
        pixel_int += alf_coeff[7] * (src    [ 1] + src    [-1]); \
        pixel_int += alf_coeff[6] * (src    [ 2] + src    [-2]); \
        pixel_int += alf_coeff[5] * (src    [ 3] + src    [-3]); \
        pixel_int += alf_coeff[8] * (src    [ 0]); \
        pixel_int = (pixel_int + 32) >> 6; \
        dst[0] = (pel_t)AVS2_CLIP3(0, pel_max, pixel_int); \
    } \
}

DEFINE_ALF_FILTER_BLOCK2(alf_filter_block2_c_8,  uint8_t)
DEFINE_ALF_FILTER_BLOCK2(alf_filter_block2_c_16, uint16_t)

void alf_filter_block2_c(uint8_t *dst, const uint8_t *src, int stride,
                         int lcu_pix_x, int lcu_pix_y,
                         int lcu_width, int lcu_height,
                         int *alf_coeff, int b_top_avail, int b_down_avail,
                         int bit_depth)
{
    if (bit_depth > 8)
        alf_filter_block2_c_16((uint16_t *)dst, (const uint16_t *)src, stride,
                               lcu_pix_x, lcu_pix_y, lcu_width, lcu_height,
                               alf_coeff, b_top_avail, b_down_avail, bit_depth);
    else
        alf_filter_block2_c_8(dst, src, stride,
                              lcu_pix_x, lcu_pix_y, lcu_width, lcu_height,
                              alf_coeff, b_top_avail, b_down_avail, bit_depth);
}

/* --------------------------------------------------------------------------
 * 对一个 LCU 块应用 ALF 滤波 (对应 davs2 alf_lcu_block)
 * 亮度 + 色度, 先 block1 (主体) 后 block2 (边界修正)
 * dst_frm 为解码帧 (输出), src_frm 为临时帧 (输入, 解码帧的副本)
 * -------------------------------------------------------------------------- */
void avs2_alf_on_block(avs2_frame *dst_frm, avs2_frame *src_frm,
                       const avs2_alf_param *ap, int lcu_x, int lcu_y,
                       int lcu_size, int img_w, int img_h, int bit_depth,
                       int b_top_avail, int b_down_avail)
{
    const int bps = dst_frm->bytes_per_sample;
    int lcu_pix_x  = lcu_x * lcu_size;
    int lcu_pix_y  = lcu_y * lcu_size;
    int lcu_width  = (lcu_pix_x + lcu_size > img_w) ? (img_w - lcu_pix_x) : lcu_size;
    int lcu_height = (lcu_pix_y + lcu_size > img_h) ? (img_h - lcu_pix_y) : lcu_size;

    /* 亮度 Y */
    if (ap->alf_enable[0]) {
        int alf_coeff[ALF_MAX_NUM_COEF];
        int stride = (int)(dst_frm->stride[0] / bps);
        int i, sum = 0;
        uint8_t *dst = dst_frm->data[0];
        const uint8_t *src = src_frm->data[0];

        /* 重建系数 (使用第 0 组, avs2dec 简化: 不做区域自适应) */
        for (i = 0; i < ALF_MAX_NUM_COEF - 1; i++) {
            alf_coeff[i] = ap->alf_coeff_y[0][i];
            sum += 2 * alf_coeff[i];
        }
        alf_coeff[ALF_MAX_NUM_COEF - 1] = (1 << ALF_NUM_BIT_SHIFT) - sum
                                        + ap->alf_coeff_y[0][ALF_MAX_NUM_COEF - 1];

        avs2_dsp_table.alf_block[0](dst, src, stride,
                                    lcu_pix_x, lcu_pix_y, lcu_width, lcu_height,
                                    alf_coeff, b_top_avail, b_down_avail, bit_depth);
        avs2_dsp_table.alf_block[1](dst, src, stride,
                                    lcu_pix_x, lcu_pix_y, lcu_width, lcu_height,
                                    alf_coeff, b_top_avail, b_down_avail, bit_depth);
    }

    /* 色度 U/V (坐标减半, 420) */
    lcu_pix_x  >>= 1;
    lcu_pix_y  >>= 1;
    lcu_width  >>= 1;
    lcu_height >>= 1;

    /* U 分量 */
    if (ap->alf_enable[1]) {
        int alf_coeff[ALF_MAX_NUM_COEF];
        int stride = (int)(dst_frm->stride[1] / bps);
        int i, sum = 0;
        uint8_t *dst = dst_frm->data[1];
        const uint8_t *src = src_frm->data[1];

        for (i = 0; i < ALF_MAX_NUM_COEF - 1; i++) {
            alf_coeff[i] = ap->alf_coeff_c[0][i];
            sum += 2 * alf_coeff[i];
        }
        alf_coeff[ALF_MAX_NUM_COEF - 1] = (1 << ALF_NUM_BIT_SHIFT) - sum
                                        + ap->alf_coeff_c[0][ALF_MAX_NUM_COEF - 1];

        avs2_dsp_table.alf_block[0](dst, src, stride,
                                    lcu_pix_x, lcu_pix_y, lcu_width, lcu_height,
                                    alf_coeff, b_top_avail, b_down_avail, bit_depth);
        avs2_dsp_table.alf_block[1](dst, src, stride,
                                    lcu_pix_x, lcu_pix_y, lcu_width, lcu_height,
                                    alf_coeff, b_top_avail, b_down_avail, bit_depth);
    }

    /* V 分量 */
    if (ap->alf_enable[2]) {
        int alf_coeff[ALF_MAX_NUM_COEF];
        int stride = (int)(dst_frm->stride[2] / bps);
        int i, sum = 0;
        uint8_t *dst = dst_frm->data[2];
        const uint8_t *src = src_frm->data[2];

        for (i = 0; i < ALF_MAX_NUM_COEF - 1; i++) {
            alf_coeff[i] = ap->alf_coeff_c[1][i];
            sum += 2 * alf_coeff[i];
        }
        alf_coeff[ALF_MAX_NUM_COEF - 1] = (1 << ALF_NUM_BIT_SHIFT) - sum
                                        + ap->alf_coeff_c[1][ALF_MAX_NUM_COEF - 1];

        avs2_dsp_table.alf_block[0](dst, src, stride,
                                    lcu_pix_x, lcu_pix_y, lcu_width, lcu_height,
                                    alf_coeff, b_top_avail, b_down_avail, bit_depth);
        avs2_dsp_table.alf_block[1](dst, src, stride,
                                    lcu_pix_x, lcu_pix_y, lcu_width, lcu_height,
                                    alf_coeff, b_top_avail, b_down_avail, bit_depth);
    }
}

/* --------------------------------------------------------------------------
 * ALF DSP 初始化 (对应 davs2 davs2_alf_init)
 * -------------------------------------------------------------------------- */
void avs2_alf_init(void)
{
    avs2_dsp_table.alf_block[0] = alf_filter_block1_c;
    avs2_dsp_table.alf_block[1] = alf_filter_block2_c;
}
