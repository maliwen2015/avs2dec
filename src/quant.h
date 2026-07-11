#ifndef AVS2DEC_SRC_QUANT_H
#define AVS2DEC_SRC_QUANT_H

#include <stdint.h>

/* ===================================================================
 * 反量化接口 (从 davs2 quant.cc 移植)
 * =================================================================== */

/* 反量化 (简化接口): 对 n 个系数按 qp/bit_size 反量化。
 * 当 weighted 非零且 wqm 非空时, 使用平铺加权矩阵 (wqm_shift=0)。
 * - coeff     : 系数缓冲 (原地变换)
 * - n         : 系数个数
 * - qp        : 量化参数
 * - bit_size  : 变换尺寸的对数 (如 8x8 取 3)
 * - bit_depth : 样本位深 (8 或 10)
 * - weighted  : 是否启用加权量化
 * - wqm       : 加权量化矩阵 (n 个系数, 仅 weighted 时使用) */
void avs2_dequant(int16_t *coeff, int n, int qp, int bit_size,
                  int bit_depth, int weighted, const uint8_t *wqm);

/* 加权反量化 (完整接口, 对应 davs2 dequant_coeffs + dequant_weighted_c):
 * 使用给定加权矩阵对 bsx*bsy 块进行反量化。
 * - wq_matrix   : 加权量化矩阵
 * - wqm_shift   : 加权矩阵移位 (pic_wq_data_index==1 时为 3, 否则 0)
 * - wqm_size_id : 矩阵尺寸标识 (0:4x4, 1:8x8, 2/3:8x8 下采样) */
void avs2_dequant_coeff(int16_t *coeff, int bsx, int bsy, int qp,
                        int bit_size, int bit_depth,
                        const int16_t *wq_matrix,
                        int wqm_shift, int wqm_size_id);

/* 色度 QP 映射 (对应 davs2 cu_get_chroma_qp)。
 * - luma_qp   : 亮度 QP
 * - delta     : 色度 QP 偏移 (Cb 或 Cr)
 * - bit_depth : 样本位深 (8 或 10); bit_depth>8 时启用 HIGH_BIT_DEPTH 路径 */
int avs2_chroma_qp(int luma_qp, int delta, int bit_depth);

/* 反量化初始化 (对应 davs2 davs2_quant_init)。 */
void avs2_quant_init(void);

/* 计算反量化参数 (对应 davs2 cu_get_quant_params)。
 * - qp        : 量化参数
 * - bit_size  : 变换尺寸的对数 (如 8x8 取 3)
 * - bit_depth : 样本位深 (8 或 10)
 * - shift/scale: 输出移位与缩放因子 */
void avs2_get_quant_params(int qp, int bit_size, int bit_depth,
                           int *shift, int *scale);

/* 反量化 C 参考实现 (供 SIMD 文件引用) */
void dequant_c(int16_t *coeff, int n, int scale, int shift);

/* ===================================================================
 * 反变换入口 (从 davs2 transform.cc 移植, 实现见 itx.c)
 * =================================================================== */

/* 反变换入口 (对应 davs2 inv_transform): 按块尺寸分发反 DCT, 并可选地
 * 在主变换前应用 NSST 二次变换。
 * - coeff              : 系数缓冲 (原地变换)
 * - bsx/bsy            : 块宽/高 (4/8/16/32/64)
 * - bit_depth          : 位深 (8 或 10)
 * - b_sec_t            : 是否启用二次变换 (NSST)
 * - i_luma_intra_mode  : 亮度帧内模式 (仅 b_sec_t 时有效)
 * - b_top/b_left       : 上/左邻块可用性 (仅 b_sec_t 时有效) */
void avs2_inverse_transform(int16_t *coeff, int bsx, int bsy, int bit_depth,
                            int b_sec_t, int i_luma_intra_mode,
                            int b_top, int b_left);

#endif /* AVS2DEC_SRC_QUANT_H */
