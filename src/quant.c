/*
 * quant.c - 反量化实现
 *
 * 从 davs2 (source/common/quant.cc) 移植到 C。
 * 实现 AVS2 反量化 (含加权量化 WQ 支持) 与色度 QP 映射。
 * 位深通过参数运行时传递, 同时支持 8-bit 与 10-bit。
 */

#include "internal.h"
#include "tables.h"
#include "quant.h"

/* ---- 局部宏 ---- */
#ifndef AVS2_CLIP3
#define AVS2_CLIP3(min_val, max_val, val) \
    ((val) < (min_val) ? (min_val) : ((val) > (max_val) ? (max_val) : (val)))
#endif

/**
 * ===========================================================================
 * 内部辅助函数
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * 计算反量化参数 (对应 davs2 cu_get_quant_params)
 * qp        - 量化参数
 * bit_size  - 变换尺寸的对数 (如 8x8 取 3)
 * bit_depth - 样本位深 (8 或 10)
 * shift/scale - 输出移位与缩放因子
 */
void avs2_get_quant_params(int qp, int bit_size, int bit_depth,
                           int *shift, int *scale)
{
    int q = qp;
    if (q < 0) {
        q = 0;
    } else if (q > 79) {
        q = 79;
    }
    *shift = avs2_iq_shift[q] + (bit_depth + 1) + bit_size - LIMIT_BIT;
    *scale = avs2_iq_tab[q];
}

/* ---------------------------------------------------------------------------
 * 简单反量化 (对应 davs2 dequant_c)
 * 对 n 个系数逐个乘以 scale 并右移 shift。
 */
void dequant_c(int16_t *coeff, int n, int scale, int shift)
{
    int add = (shift > 0) ? (1 << (shift - 1)) : 0;
    int i;

    for (i = 0; i < n; i++) {
        if (coeff[i]) {
            int c;
            if (shift > 0) {
                c = (coeff[i] * scale + add) >> shift;
            } else {
                c = coeff[i] * scale;
            }
            coeff[i] = (int16_t)AVS2_CLIP3(-32768, 32767, c);
        }
    }
}

/* ---------------------------------------------------------------------------
 * 加权反量化 (对应 davs2 dequant_weighted_c)
 * coeff      - 系数缓冲 (原地变换)
 * i_coeff    - 系数缓冲行步长
 * bsx/bsy    - 块宽/高
 * scale/shift- 量化缩放与移位
 * wq_matrix  - 加权量化矩阵
 * wqm_shift  - 加权矩阵移位 (pic_wq_data_index==1 时为 3, 否则 0)
 * wqm_size_id- 矩阵尺寸标识 (0:4x4, 1:8x8, 2/3:8x8 下采样)
 */
static void dequant_weighted_c(int16_t *coeff, int i_coeff, int bsx, int bsy,
                               int scale, int shift,
                               const int16_t *wq_matrix,
                               int wqm_shift, int wqm_size_id)
{
    int add = (shift > 0) ? (1 << (shift - 1)) : 0;
    int wqm_size = 1 << (wqm_size_id + 2);
    int stride_shift = AVS2_CLIP3(0, 2, wqm_size_id - 1);
    int stride = wqm_size >> stride_shift;
    int i, j;

    for (j = 0; j < bsy; j++) {
        for (i = 0; i < bsx; i++) {
            int wqm_coef = wq_matrix[((j >> stride_shift) & (stride - 1)) * stride +
                                     ((i >> stride_shift) & (stride - 1))];
            if (coeff[i]) {
                int t1 = (coeff[i] * wqm_coef) >> wqm_shift;
                int t2 = (t1 * scale) >> 4;
                int c;
                if (shift > 0) {
                    c = (t2 + add) >> shift;
                } else {
                    c = t2;
                }
                coeff[i] = (int16_t)AVS2_CLIP3(-32768, 32767, c);
            }
        }
        coeff += i_coeff;
    }
}


/**
 * ===========================================================================
 * 公开接口
 * ===========================================================================
 */

/* ---------------------------------------------------------------------------
 * 反量化 (简化接口, 供 cu.c 等使用)
 * 当 weighted 非零且 wqm 非空时, 使用平铺加权矩阵 (wqm_shift=0) 进行加权反量化;
 * 否则进行简单反量化。
 */
void avs2_dequant(int16_t *coeff, int n, int qp, int bit_size,
                  int bit_depth, int weighted, const uint8_t *wqm)
{
    int shift, scale;
    avs2_get_quant_params(qp, bit_size, bit_depth, &shift, &scale);

    if (weighted && wqm) {
        /* 平铺加权路径: 每个系数对应一个 wqm 系数, wqm_shift=0 */
        int add = (shift > 0) ? (1 << (shift - 1)) : 0;
        int i;
        for (i = 0; i < n; i++) {
            if (coeff[i]) {
                int t1 = coeff[i] * (int)wqm[i];
                int t2 = (t1 * scale) >> 4;
                int c;
                if (shift > 0) {
                    c = (t2 + add) >> shift;
                } else {
                    c = t2;
                }
                coeff[i] = (int16_t)AVS2_CLIP3(-32768, 32767, c);
            }
        }
    } else {
        dequant_c(coeff, n, scale, shift);
    }
}

/* ---------------------------------------------------------------------------
 * 加权反量化 (完整接口, 对应 davs2 dequant_coeffs + dequant_weighted_c)
 * 根据 qp 计算缩放/移位, 使用给定加权矩阵对 bsx*bsy 块进行反量化。
 */
void avs2_dequant_coeff(int16_t *coeff, int bsx, int bsy, int qp,
                        int bit_size, int bit_depth,
                        const int16_t *wq_matrix,
                        int wqm_shift, int wqm_size_id)
{
    int shift, scale;
    avs2_get_quant_params(qp, bit_size, bit_depth, &shift, &scale);
    dequant_weighted_c(coeff, bsx, bsx, bsy, scale, shift,
                       wq_matrix, wqm_shift, wqm_size_id);
}

/* ---------------------------------------------------------------------------
 * 色度 QP 映射 (对应 davs2 cu_get_chroma_qp)
 * luma_qp   - 亮度 QP
 * delta     - 色度 QP 偏移 (Cb 或 Cr)
 * bit_depth - 样本位深 (8 或 10); bit_depth>8 时启用 HIGH_BIT_DEPTH 路径
 */
int avs2_chroma_qp(int luma_qp, int delta, int bit_depth)
{
    int qp = luma_qp + delta;

    if (bit_depth > 8) {
        /* HIGH_BIT_DEPTH 路径: 先减去位深偏移再查表, 最后加回并裁剪 */
        int bit_depth_offset = ((bit_depth - 8) << 3);
        qp -= bit_depth_offset;
        if (qp >= 0) {
            int q_idx = qp > 63 ? 63 : qp;
            qp = avs2_qp_scale_cr[q_idx];
        }
        qp = AVS2_CLIP3(0, 63 + bit_depth_offset, qp + bit_depth_offset);
    } else {
        int q_idx = qp < 0 ? 0 : (qp > 63 ? 63 : qp);
        qp = avs2_qp_scale_cr[q_idx];
    }

    return qp;
}

/* ---------------------------------------------------------------------------
 * 反量化初始化: 注册 C 回退到 DSP 表
 */
void avs2_quant_init(void)
{
    avs2_dsp_table.dequant_block = dequant_c;
}
