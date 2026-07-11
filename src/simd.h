#ifndef AVS2DEC_SRC_SIMD_H
#define AVS2DEC_SRC_SIMD_H

/*
 * SIMD 抽象层
 *
 * 按位深 (8-bit / 16-bit) 分离 DSP 函数指针, 参考 dav1d 的 bitdepth 模板设计.
 * 每个模块 (itx, mc, ipred, lf, sao, alf) 提供独立初始化函数,
 * 在 avs2_dsp_init 中根据 cpu flags 选择最优实现.
 */

#include "cpu.h"
#include <stdint.h>
#include <stddef.h>

/* 反变换函数指针: 就地变换 coeff (int16_t), w/h 为块尺寸 */
typedef void (*avs2_itx_fn)(int16_t *coeff, int w, int h, int bit_depth);

/* 运动补偿函数指针 */
typedef void (*avs2_mc_fn)(const uint8_t *src, ptrdiff_t sstride,
                           uint8_t *dst, ptrdiff_t dstride,
                           int w, int h, int mx, int my, int bit_depth);

/* 帧内预测单模式函数指针 */
typedef void (*avs2_intra_pred_fn)(uint16_t *src, uint16_t *dst, int i_dst,
                                   int mode, int bsx, int bsy, int bit_depth);

/* 参考样本填充函数指针 */
typedef void (*avs2_fill_edge_fn)(const uint16_t *pTL, int i_TL,
                                  const uint16_t *pLcuEP, uint16_t *EP,
                                  uint32_t i_avai, int bsx, int bsy);

/* 去块滤波函数指针 (支持 8/10-bit, src 按 bit_depth 决定访问宽度) */
typedef void (*avs2_deblock_luma_fn)(void *src, int stride,
                                    int alpha, int beta, uint8_t *flt_flag, int bit_depth);
typedef void (*avs2_deblock_chroma_fn)(void *src_u, void *src_v, int stride,
                                       int alpha, int beta, uint8_t *flt_flag, int bit_depth);

/* SAO 函数指针 */
typedef void (*avs2_sao_eo_fn)(uint16_t *dst, int dst_stride,
                               const uint16_t *src, int src_stride,
                               int w, int h, int bit_depth,
                               const int *avail, const int *offset);
typedef void (*avs2_sao_bo_fn)(uint16_t *dst, int dst_stride,
                               const uint16_t *src, int src_stride,
                               int w, int h, int bit_depth, const int *offset);

/* ALF 函数指针 */
typedef void (*avs2_alf_fn)(uint16_t *dst, const uint16_t *src, int stride,
                            int lcu_pix_x, int lcu_pix_y, int lcu_width, int lcu_height,
                            int *alf_coeff, int b_top_avail, int b_down_avail, int bit_depth);

/* ---- x86 SIMD 初始化函数 (在 src/x86/ 下实现) ---- */
/* 每个 init 函数接收 cpu flags, 注册对应指令集的实现到全局 dsp_table */
void avs2_itx_init_sse41(const avs2_cpu_flags *flags);
void avs2_itx_init_avx2(const avs2_cpu_flags *flags);
void avs2_itx_init_avx512(const avs2_cpu_flags *flags);
void avs2_itx_init_neon(const avs2_cpu_flags *flags);

void avs2_mc_init_sse41(const avs2_cpu_flags *flags);
void avs2_mc_init_avx2(const avs2_cpu_flags *flags);
void avs2_mc_init_neon(const avs2_cpu_flags *flags);

void avs2_ipred_init_sse41(const avs2_cpu_flags *flags);
void avs2_ipred_init_avx2(const avs2_cpu_flags *flags);
void avs2_ipred_init_neon(const avs2_cpu_flags *flags);

void avs2_lf_init_sse41(const avs2_cpu_flags *flags);
void avs2_lf_init_avx2(const avs2_cpu_flags *flags);
void avs2_lf_init_neon(const avs2_cpu_flags *flags);

void avs2_sao_init_sse41(const avs2_cpu_flags *flags);
void avs2_sao_init_avx2(const avs2_cpu_flags *flags);
void avs2_sao_init_neon(const avs2_cpu_flags *flags);

void avs2_alf_init_sse41(const avs2_cpu_flags *flags);
void avs2_alf_init_avx2(const avs2_cpu_flags *flags);
void avs2_alf_init_neon(const avs2_cpu_flags *flags);

void avs2_quant_init_sse41(const avs2_cpu_flags *flags);
void avs2_quant_init_avx2(const avs2_cpu_flags *flags);
void avs2_quant_init_neon(const avs2_cpu_flags *flags);

#endif /* AVS2DEC_SRC_SIMD_H */
