/*
 * cu.c - CU (编码单元) 解码与重建
 *
 * 从 davs2 (source/common/cu.cc) 移植为 C。
 * 实现完整的 CU 解码流水线:
 *   - 四叉树递归分割 (split flag AEC 解码)
 *   - CU 类型解码 (skip/direct/intra/inter)
 *   - 帧内模式解码 (亮度 5-bit + DMH + 色度)
 *   - 帧间模式解码 (PU 类型、MV 解码、参考索引)
 *   - 残差解码 (CBP、TU 分割、系数 CG 遍历)
 *   - 重建 (预测 + 反量化 + 反变换 = 最终像素)
 *
 * 命名约定: 小写字母加下划线, 字段名与 davs2 cu_t 对应。
 */

#include "internal.h"
#include "aec_internal.h"
#include "quant.h"
#include "tables.h"
#include <string.h>
#include <stdio.h>

/* 辅助宏 (对应 davs2 DAVS2_ABS / DAVS2_MIN) */
#ifndef DAVS2_ABS
#define DAVS2_ABS(a)  ((a) >= 0 ? (a) : -(a))
#endif
#ifndef DAVS2_MIN
#define DAVS2_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

/* 非环路滤波热点：把重复的边界裁剪公用化，减少分支与重复计算。
 * 这不会改变解码结果，只减少运动补偿路径中的冗余判断。
 */
static inline int clamp_coord(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

/* ===================================================================
 * 行级并行: 线程本地系数 scratch 缓冲区
 *
 * 问题: coeff_scratch_y/u/v 是 fc 级别共享字段, Pass 2 中多个 worker
 *   并行调用 avs2_decode_lcu → load_cu_coeffs → reconstruct_residual,
 *   它们共享同一 scratch 缓冲区导致数据竞争.
 *
 * 解决: 每个 worker 在 avs2_row_parallel_pass2 入口分配栈上 scratch,
 *   通过 TLS (线程本地存储) 设置, avs2_decode_lcu 在 pass=2 时使用
 *   TLS scratch 而非 fc->coeff_scratch.
 *
 * __thread 是 GCC 扩展, MinGW (GCC) 亦支持.
 * =================================================================== */
static __thread int16_t *tls_coeff_scratch_y = NULL;
static __thread int16_t *tls_coeff_scratch_u = NULL;
static __thread int16_t *tls_coeff_scratch_v = NULL;

/* 行级并行 Pass 2: per-worker LCU 位置 (避免 fc->lcu_x/lcu_y 多 worker 竞争).
 * 在 avs2_decode_lcu 中设置, 供 load_cu_coeffs_y/c 读取. */
static __thread int tls_lcu_x = 0;
static __thread int tls_lcu_y = 0;

/* 设置当前线程的系数 scratch 缓冲区 (行级并行 Pass 2 使用).
 * 传入 NULL 清除 (Pass 2 结束时调用). */
void avs2_set_thread_scratch(int16_t *y, int16_t *u, int16_t *v)
{
    tls_coeff_scratch_y = y;
    tls_coeff_scratch_u = u;
    tls_coeff_scratch_v = v;
}

/* ======================= 其余文件内容保持不变 ======================= */

/* 楼上新增的 clamp_coord 先保留；后续直接复用于运动补偿的边界裁剪。 */

/* ... existing content ... */

/* 帧间预测: 对每个 PU 执行运动补偿 */
static int reconstruct_inter(avs2_frame_ctx *fc, avs2_cu *cu,
                              int x, int y, int bit_depth)
{
    avs2_frame *f = fc->fdec;
    int frame_type = fc->slice_type;
    int pu_idx;

    for (pu_idx = 0; pu_idx < cu->num_pu; pu_idx++) {
        int pix_x = x + cu->pu_x[pu_idx];
        int pix_y = y + cu->pu_y[pu_idx];
        int width = cu->pu_w[pu_idx];
        int height = cu->pu_h[pu_idx];
        int pred_dir = cu->b8pdir[pu_idx];
        int ref_1st = 0, ref_2nd = 0;
        avs2_mv mv_1st, mv_2nd;
        avs2_frame *p_fref1 = NULL, *p_fref2 = NULL;

        mv_1st.x = mv_1st.y = 0;
        mv_2nd.x = mv_2nd.y = 0;

        if (frame_type == AVS2_B_SLICE) {
            if (pred_dir == PDIR_BWD) {
                mv_1st = cu->mv[pu_idx][1];
                if (fc->n_refs > 0) p_fref1 = fc->fref[B_BWD];
            } else if (pred_dir == PDIR_SYM || pred_dir == PDIR_BID) {
                mv_1st = cu->mv[pu_idx][0];
                mv_2nd = cu->mv[pu_idx][1];
                if (fc->n_refs > 1) {
                    p_fref1 = fc->fref[B_FWD];
                    p_fref2 = fc->fref[B_BWD];
                }
            } else {
                mv_1st = cu->mv[pu_idx][0];
                if (fc->n_refs > 1) p_fref1 = fc->fref[B_FWD];
            }
        } else {
            int dmh_mode = cu->i_dmh_mode;
            ref_1st = cu->i_ref[pu_idx][0];
            if (ref_1st < 0 || ref_1st >= fc->n_refs) ref_1st = 0;
            mv_1st = cu->mv[pu_idx][0];

            if (pred_dir == PDIR_DUAL) {
                mv_2nd = cu->mv[pu_idx][1];
                ref_2nd = cu->i_ref[pu_idx][1];
                if (ref_2nd < 0 || ref_2nd >= fc->n_refs) ref_2nd = 0;
                if (ref_1st < fc->n_refs) p_fref1 = fc->fref[ref_1st];
                if (ref_2nd < fc->n_refs) p_fref2 = fc->fref[ref_2nd];
            } else if (dmh_mode) {
                mv_2nd.x = mv_1st.x + tab_dmh_pos[dmh_mode][1][0];
                mv_2nd.y = mv_1st.y + tab_dmh_pos[dmh_mode][1][1];
                mv_1st.x += tab_dmh_pos[dmh_mode][0][0];
                mv_1st.y += tab_dmh_pos[dmh_mode][0][1];
                ref_2nd = ref_1st;
                if (ref_1st < fc->n_refs) {
                    p_fref1 = fc->fref[ref_1st];
                    p_fref2 = p_fref1;
                }
            } else {
                if (ref_1st < fc->n_refs) p_fref1 = fc->fref[ref_1st];
            }
        }

        mv_1st.x = (int16_t)get_mc_pos(f->width, width, pix_x, mv_1st.x);
        mv_1st.y = (int16_t)get_mc_pos(f->height, height, pix_y, mv_1st.y);
        mv_2nd.x = (int16_t)get_mc_pos(f->width, width, pix_x, mv_2nd.x);
        mv_2nd.y = (int16_t)get_mc_pos(f->height, height, pix_y, mv_2nd.y);

        if (p_fref1 != NULL) {
            int bps = f->bytes_per_sample;
            int ref_y = pix_y + (mv_1st.y >> 2);
            int ref_x = pix_x + (mv_1st.x >> 2);
            int max_y = p_fref1->height + 60;
            int min_y = -60;
            int max_x = p_fref1->width + 60;
            int min_x = -60;

            ref_y = clamp_coord(ref_y, min_y, max_y);
            ref_x = clamp_coord(ref_x, min_x, max_x);

            uint8_t *p_pred = f->data[0] + pix_y * f->stride[0] + (ptrdiff_t)pix_x * bps;
            const uint8_t *p_ref = p_fref1->data[0] + (ptrdiff_t)ref_y * p_fref1->stride[0]
                                   + (ptrdiff_t)ref_x * bps;
            int mx = mv_1st.x & 3;
            int my = mv_1st.y & 3;

            avs2_dsp_table.mc_luma(p_ref, p_fref1->stride[0], p_pred, f->stride[0],
                                   width, height, mx, my, bit_depth);

            if (p_fref2 != NULL) {
                int ref2_y = pix_y + (mv_2nd.y >> 2);
                int ref2_x = pix_x + (mv_2nd.x >> 2);
                ref2_y = clamp_coord(ref2_y, min_y, max_y);
                ref2_x = clamp_coord(ref2_x, min_x, max_x);
                const uint8_t *p_ref2 = p_fref2->data[0] + (ptrdiff_t)ref2_y * p_fref2->stride[0]
                                        + (ptrdiff_t)ref2_x * bps;
                int mx2 = mv_2nd.x & 3;
                int my2 = mv_2nd.y & 3;

                avs2_dsp_table.mc_luma_avg(p_ref2, p_fref2->stride[0],
                                           p_pred, f->stride[0],
                                           width, height, mx2, my2, bit_depth);
            }
        } else {
            int bps = f->bytes_per_sample;
            uint8_t *p_pred = f->data[0] + pix_y * f->stride[0] + (ptrdiff_t)pix_x * bps;
            int fill_val = 1 << (bit_depth - 1);
            avs2_dsp_table.fill_block(p_pred, f->stride[0], width, height,
                                      fill_val, bit_depth);
        }

        if (f->chroma_format == AVS2_CHROMA_420) {
            int cx = pix_x >> 1, cy = pix_y >> 1;
            int cw = width >> 1, ch = height >> 1;
            int cmv1_x = mv_1st.x, cmv1_y = mv_1st.y;
            int cmv2_x = mv_2nd.x, cmv2_y = mv_2nd.y;
            int cbps = f->bytes_per_sample;
            int cfill_val = 1 << (bit_depth - 1);
            int pl;

            for (pl = 1; pl < 3; pl++) {
                uint8_t *p_pred = f->data[pl] + cy * f->stride[pl] + (ptrdiff_t)cx * cbps;
                if (p_fref1 != NULL) {
                    int cref_y = cy + (cmv1_y >> 3);
                    int cref_x = cx + (cmv1_x >> 3);
                    int cmax_y = (p_fref1->height >> 1) + 60;
                    int cmin_y = -60;
                    int cmax_x = (p_fref1->width >> 1) + 60;
                    int cmin_x = -60;

                    cref_y = clamp_coord(cref_y, cmin_y, cmax_y);
                    cref_x = clamp_coord(cref_x, cmin_x, cmax_x);

                    const uint8_t *p_ref = p_fref1->data[pl] +
                                           (ptrdiff_t)cref_y * p_fref1->stride[pl] +
                                           (ptrdiff_t)cref_x * cbps;
                    int cmx = cmv1_x & 7;
                    int cmy = cmv1_y & 7;
                    avs2_dsp_table.mc_chroma(p_ref, p_fref1->stride[pl],
                                             p_pred, f->stride[pl],
                                             cw, ch, cmx, cmy, bit_depth);

                    if (p_fref2 != NULL) {
                        int cref2_y = cy + (cmv2_y >> 3);
                        int cref2_x = cx + (cmv2_x >> 3);
                        cref2_y = clamp_coord(cref2_y, cmin_y, cmax_y);
                        cref2_x = clamp_coord(cref2_x, cmin_x, cmax_x);
                        const uint8_t *p_ref2 = p_fref2->data[pl] +
                                                (ptrdiff_t)cref2_y * p_fref2->stride[pl] +
                                                (ptrdiff_t)cref2_x * cbps;
                        int cmx2 = cmv2_x & 7;
                        int cmy2 = cmv2_y & 7;

                        avs2_dsp_table.mc_chroma_avg(p_ref2, p_fref2->stride[pl],
                                                     p_pred, f->stride[pl],
                                                     cw, ch, cmx2, cmy2, bit_depth);
                    }
                } else {
                    avs2_dsp_table.fill_block(p_pred, f->stride[pl], cw, ch,
                                              cfill_val, bit_depth);
                }
            }
        }
    }

    return 0;
}

/* ... existing remainder unchanged ... */
