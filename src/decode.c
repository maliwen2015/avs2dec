/*
 * Main decode orchestration. Parses Annex B start codes, drives header
 * parsing, allocates frames, and runs the LCU-row decode loop.
 */

#include "internal.h"
#include "aec_internal.h"
#include "quant.h"
#include <string.h>
#include <stdio.h>

static int process_start_code(struct avs2_internal *c, const uint8_t *data,
                               int sz, int sc_pos, int sc_id);

/* ------------------------------------------------------------------
 * 非滤波热点：行级依赖等待的公共化。
 * 作用：减少 Phase-2 里对多个参考帧的重复判断，避免拆散的分支和
 *       反复计算 required_row，保持行为一致但减少 spin-wait 热路径开销。
 * ------------------------------------------------------------------ */
static inline int is_lf_row_ready_for_refs(avs2_frame_ctx *fc, int required_row)
{
    if (fc->n_refs <= 0) {
        return 1;
    }

    for (int j = 0; j < fc->n_refs; ++j) {
        avs2_frame *ref = fc->fref[j];
        if (!ref) {
            continue;
        }

        int rr = required_row;
        if (rr >= ref->h_lcu) {
            rr = ref->h_lcu - 1;
        }
        if (rr < 0) {
            rr = 0;
        }

        if (!avs2_atomic_load(&ref->lf_row_done[rr])) {
            return 0;
        }
    }

    return 1;
}

static inline int required_mv_row_span(int mvy, int lcu_size)
{
    if (mvy > 0) {
        int pix_offset = (mvy + 3) / 4;
        int row_offset = (pix_offset + lcu_size - 1) / lcu_size;
        return row_offset + 1;
    }
    return 1;
}

/* ===================================================================
 * 行级并行 (块并行): 2-pass 模式
 *
 * Pass 1 (AEC, 串行): owning worker 逐行执行 AEC 解码, 存储系数到 per-LCU
 *   缓冲区, 设置 row_aec_done[i]. AEC 概率状态在 slice 内持续演进, 必须串行.
 *
 * Pass 2 (重建+LF, 并行): 多个 worker 并行处理 LCU 行.
 *   - 重建任务: row_recon_next 行, 依赖 row_aec_done[row] + row_recon_done[row-1]
 *     (上一行重建完成, 提供 intra 预测 top 参考)
 *   - LF 任务: row_lf_next 行, 依赖 row_recon_done[row] + row_lf_done[row-1]
 *     (SAO/ALF 跨行依赖, LF 必须按行序执行)
 *   重建与 LF 流水化: 重建 row i+1 可与 LF row i 并行.
 * =================================================================== */

/* 确保 per-LCU 系数缓冲区已分配 (尺寸变化时重新分配).
 * per-row 进度数组 (row_aec_done 等) 和 mv_row_range 为静态数组, 无需分配. */
static int ensure_row_parallel_buffers(avs2_frame_ctx *fc, avs2_frame *f, int lcu_log2)
{
    int w_lcu = f->w_lcu;
    int h_lcu = f->h_lcu;
    int n_lcu = w_lcu * h_lcu;
    int lcu_dim = 1 << lcu_log2;       /* 亮度 LCU 边长 (32 或 64) */
    int c_dim  = lcu_dim >> 1;         /* 色度 LCU 边长 */

    if (h_lcu > AVS2_MAX_H_LCU) {
        fprintf(stderr, "avs2dec [Warning]: h_lcu %d exceeds AVS2_MAX_H_LCU %d\n", h_lcu, AVS2_MAX_H_LCU);
        return AVS2_ERR_NOMEM;
    }

    /* per-LCU 系数缓冲区: 按实际 LCU 大小分配, 避免浪费 */
    if (fc->coeff_lcu_w_lcu != w_lcu || fc->coeff_lcu_h_lcu != h_lcu ||
        fc->coeff_lcu_log2 != lcu_log2) {
        avs2_mem_free(fc->coeff_lcu_y);  fc->coeff_lcu_y = NULL;
        avs2_mem_free(fc->coeff_lcu_u);  fc->coeff_lcu_u = NULL;
        avs2_mem_free(fc->coeff_lcu_v);  fc->coeff_lcu_v = NULL;
        fc->coeff_lcu_y = avs2_mem_allocz(sizeof(int16_t) * (size_t)n_lcu * lcu_dim * lcu_dim);
        fc->coeff_lcu_u = avs2_mem_allocz(sizeof(int16_t) * (size_t)n_lcu * c_dim * c_dim);
        fc->coeff_lcu_v = avs2_mem_allocz(sizeof(int16_t) * (size_t)n_lcu * c_dim * c_dim);
        if (!fc->coeff_lcu_y || !fc->coeff_lcu_u || !fc->coeff_lcu_v) {
            /* 失败时清理已分配的缓冲区, 避免悬空指针和内存泄漏.
             * 下次进入此函数会因尺寸不匹配而重新分配. */
            avs2_mem_free(fc->coeff_lcu_y); fc->coeff_lcu_y = NULL;
            avs2_mem_free(fc->coeff_lcu_u); fc->coeff_lcu_u = NULL;
            avs2_mem_free(fc->coeff_lcu_v); fc->coeff_lcu_v = NULL;
            fc->coeff_lcu_w_lcu = 0;
            fc->coeff_lcu_h_lcu = 0;
            return AVS2_ERR_NOMEM;
        }
        fc->coeff_lcu_w_lcu = w_lcu;
        fc->coeff_lcu_h_lcu = h_lcu;
        fc->coeff_lcu_log2  = (int8_t)lcu_log2;
    }

    fc->row_parallel_h_lcu = h_lcu;
    fc->mv_row_range_h_lcu = h_lcu;

    return AVS2_OK;
}

static void compute_mv_row_ranges(avs2_frame *f, int lcu_size, int *mv_row_range);

/* 行级并行: owning worker 的帧解码入口.
 * Pass 1 (AEC 串行) + Pass 2 (参与行级并行重建). */
static int avs2_decode_frame_fc_row(struct avs2_internal *c, avs2_frame_ctx *fc)
{
    avs2_frame *f = fc->fdec;
    int b_slice_checked = 0;

    if (!f) {
        avs2_warn(c, "no frame allocated\n");
        return AVS2_ERR_INVALID;
    }
    if (!fc->bs.buf) {
        avs2_warn(c, "no slice data (bs.buf is NULL)\n");
        return AVS2_ERR_INVALID;
    }

    /* 分配 per-LCU 系数缓冲区和 per-row 进度数组 */
    int r = ensure_row_parallel_buffers(fc, f, c->seq->log2_lcu_size);
    if (r) return r;

    /* 重置 per-row 进度 (fc 级 + 帧级).
     * 帧级 lf_row_done/aec_row_done 必须重置: DPB 帧缓冲复用时残留的 1 会导致
     * 依赖帧的 derive_skip_mv (读 col_pic->aec_row_done) 和 Pass 2 跨帧 LF 依赖
     * (读 ref->lf_row_done) 误判就绪. */
    {
        int h_lcu = f->h_lcu;
        int i;
        for (i = 0; i < h_lcu; i++) {
            fc->row_aec_done[i] = 0;
            fc->row_recon_done[i] = 0;
            fc->row_lf_done[i] = 0;
            avs2_atomic_store(&f->lf_row_done[i], 0);
            avs2_atomic_store(&f->aec_row_done[i], 0);
        }
        avs2_atomic_store(&f->lf_row_done_count, 0);
        f->done = 0;
        fc->row_aec_completed = 0;
        fc->row_recon_completed = 0;
        fc->row_recon_next = 0;
        fc->row_lf_next = 0;
        fc->row_lf_done_count = 0;
        fc->n_row_workers = 0;
    }

    /* AEC 初始化 */
    avs2_aec *aec = fc->aec_pool;
    avs2_aec_init_contexts(aec, fc->slice_type);
    fc->bs.pos = (fc->bs.pos + 7) & ~7;
    avs2_aec_start_decoding(aec, fc->bs.buf, fc->bs.sz, fc->bs.pos);
    fc->aec = aec;
    fc->i_last_dquant = 0;
    f->coi = (int)fc->pic_local.coding_order;

    /* 通知辅助 worker 可以参与 Pass 2 (广播唤醒, helper 扫描 recon_active) */
    avs2_mutex_lock(&c->task_lock);
    avs2_cond_broadcast(&c->task_cond, &c->task_lock, c->n_waiters_task);
    avs2_mutex_unlock(&c->task_lock);

    /* ---- Pass 1: AEC 串行, 逐行解码 ---- */
    for (int lcu_y = 0; lcu_y < f->h_lcu; lcu_y++) {
        for (int lcu_x = 0; lcu_x < f->w_lcu; lcu_x++) {
            if (b_slice_checked) {
                int byte_pos = (avs2_aec_get_bits_read(aec) + 7) >> 3;
                int found = 0;
                int k;
                for (k = 0; k < 4 && byte_pos + k + 3 < fc->bs.sz; k++) {
                    const uint8_t *d = fc->bs.buf + byte_pos + k;
                    if (d[0] == 0 && d[1] == 0 && d[2] == 1 && d[3] <= AVS2_SC_SLICE_MAX) {
                        found = 1;
                        break;
                    }
                }
                if (found) {
                    fc->bs.pos = (byte_pos + k) << 3;
                    avs2_parse_slice_header(c, &fc->bs,
                                            fc->bs.buf[byte_pos + k + 3], fc);
                    avs2_aec_init_contexts(aec, fc->slice_type);
                    fc->i_last_dquant = 0;
                    avs2_aec_start_decoding(aec, fc->bs.buf, fc->bs.sz, fc->bs.pos);
                    b_slice_checked = 0;
                }
            }

            int lr = avs2_decode_lcu(fc, c, lcu_x, lcu_y, 1);  /* pass=1: AEC only */
            if (lr < 0) {
                avs2_warn(c, "LCU (%d,%d) AEC error\n", lcu_x, lcu_y);
                break;
            }
            b_slice_checked = aec_startcode_follows(aec, 1);
            fc->bs.pos = avs2_aec_get_bits_read(aec);
        }

        /* 通知此行 AEC 完成.
         * fc->row_aec_done: Pass 2 重建任务按行等待 (本帧内).
         * f->aec_row_done:  依赖此帧 col_pic 的后续帧在 derive_skip_mv 中按行等待.
         *   ROW 模式必须设置此项, 否则 inter 帧的 derive_skip_mv 无限自旋.
         * 无需 lock+broadcast: Pass 2 worker 用 spin-wait 轮询, 不等待 cond. */
        fc->row_aec_done[lcu_y] = 1;
        avs2_atomic_store(&f->aec_row_done[lcu_y], 1);
    }

    /* 计算行级 MV 范围 (Pass 2 跨帧行级 LF 依赖使用).
     * cu_grid 已在 AEC 中填充, 可直接扫描 MV 计算. */
    compute_mv_row_ranges(f, 1 << c->seq->log2_lcu_size, fc->mv_row_range);

    /* ---- Pass 2: owning worker 也参与行级并行重建+LF ---- */
    avs2_row_parallel_pass2(c, fc, 0, 0);

    /* 等待辅助 worker 退出 */
    avs2_mutex_lock(&c->task_lock);
    while (fc->n_row_workers > 0) {
        c->n_waiters_done++;
        avs2_cond_wait(&c->done_cond, &c->task_lock);
        c->n_waiters_done--;
    }
    avs2_mutex_unlock(&c->task_lock);

    /* AEC 上下文由 aec_pool 管理, 仅清除指针 */
    fc->aec = NULL;

    /* 设置帧元数据 */
    f->poc = fc->pic_local.poc;
    f->coi = (int)fc->pic_local.coding_order;
    f->type = (int)fc->pic_local.picture_coding_type;
    f->pts = fc->saved_pts;
    f->dts = fc->saved_dts;

    return AVS2_OK;
}

/* 计算每行的 MV 行范围 (行级 LF 依赖).
 * mv_row_range[i] = 当前帧行 i 的 inter 块需要参考帧 LF 完成的额外行数.
 *   所需参考帧行 = min(i + mv_row_range[i], ref->h_lcu-1).
 * MV y 为正: 参考帧在下方, 需要额外行. 含 8-tap 插值滤波器余量 (+1 行).
 * MV y 为零或负: 参考帧在当前行或上方, 仅需当前行 + 滤波器余量 (1 行).
 * Phase 1 完成后 (cu_grid 已填充) 调用. */
static void compute_mv_row_ranges(avs2_frame *f, int lcu_size, int *mv_row_range)
{
    const int w8 = f->w8;
    const int h8 = f->h8;
    const int h_lcu = f->h_lcu;
    const int lcu_8 = lcu_size / 8;  /* LCU size in 8x8 blocks */

    for (int i = 0; i < h_lcu; i++) {
        mv_row_range[i] = 0;
    }

    for (int by = 0; by < h8; by++) {
        int lcu_row = by / lcu_8;
        if (lcu_row >= h_lcu) lcu_row = h_lcu - 1;
        for (int bx = 0; bx < w8; bx++) {
            avs2_cu *cu = &f->cu_grid[by * w8 + bx];
            if (!cu->b_intra) {
                for (int pu = 0; pu < cu->num_pu; pu++) {
                    for (int list = 0; list < 2; list++) {
                        if (cu->i_ref[pu][list] >= 0) {
                            int required_extra = required_mv_row_span(cu->mv[pu][list].y, lcu_size);
                            if (required_extra > mv_row_range[lcu_row]) {
                                mv_row_range[lcu_row] = required_extra;
                            }
                        }
                    }
                }
            }
        }
    }
}

/* 计算单个 LCU 行的 MV 行范围 (行级 pipeline 用).
 * 在 AEC 完成该行所有 LCU 后调用, 仅扫描该行的 8x8 块.
 * 确保 mv_row_range[lcu_y] 在 aec_row_done[lcu_y] 设置前就绪. */
static void compute_mv_row_range_for_row(avs2_frame *f, int lcu_size,
                                          int lcu_y, int *mv_row_range)
{
    const int w8 = f->w8;
    const int lcu_8 = lcu_size / 8;
    int by_start = lcu_y * lcu_8;
    int by_end = by_start + lcu_8;
    if (by_end > f->h8) by_end = f->h8;

    mv_row_range[lcu_y] = 0;
    for (int by = by_start; by < by_end; by++) {
        for (int bx = 0; bx < w8; bx++) {
            avs2_cu *cu = &f->cu_grid[by * w8 + bx];
            if (!cu->b_intra) {
                for (int pu = 0; pu < cu->num_pu; pu++) {
                    for (int list = 0; list < 2; list++) {
                        if (cu->i_ref[pu][list] >= 0) {
                            int required_extra = required_mv_row_span(cu->mv[pu][list].y, lcu_size);
                            if (required_extra > mv_row_range[lcu_y]) {
                                mv_row_range[lcu_y] = required_extra;
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ... rest of file unchanged ... */

/* 处理一行的完整 P2 流程: lf_wait → recon → save_border → deblock → pad → lf_row_done */
static void p2_do_row(avs2_frame_ctx *fc, struct avs2_internal *c, int lcu_y,
                      int is_inter, int need_pad)
{
    avs2_frame *f = fc->fdec;

    /* 跨帧 LF 依赖: inter 帧需等待参考帧 LF 完成 */
    if (is_inter) {
        int required_row = lcu_y + fc->mv_row_range[lcu_y];
        int spin_cnt = 0;
        while (!is_lf_row_ready_for_refs(fc, required_row)) {
            if ((++spin_cnt & 0x3ff) == 0 && avs2_atomic_load(&c->shutdown)) {
                break;
            }
            avs2_cpu_relax();
        }
    }

    /* 重建: pass=2 */
    for (int lcu_x = 0; lcu_x < f->w_lcu; lcu_x++) {
        avs2_decode_lcu(fc, c, lcu_x, lcu_y, 2);
    }

    /* inline deblock: 重建后立即滤波, 保持 L2 cache 局部性 */
    avs2_save_intra_border(fc, c, lcu_y);
    avs2_lf_apply_lcu_row_nosave(fc, c, lcu_y);

    if (need_pad) {
        avs2_pad_line_lcu(f, lcu_y, (int)c->seq->log2_lcu_size);
    }

    avs2_atomic_store(&f->lf_row_done[lcu_y], 1);
    avs2_atomic_inc(&f->lf_row_done_count);
}
