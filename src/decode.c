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
    avs2_row_parallel_pass2(c, fc, 0);

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
                            int mvy = cu->mv[pu][list].y;
                            int required_extra;
                            if (mvy > 0) {
                                /* 像素偏移 = ceil(mvy/4), LCU 行偏移 = ceil(像素/lcu) */
                                int pix_offset = (mvy + 3) / 4;
                                int row_offset = (pix_offset + lcu_size - 1) / lcu_size;
                                required_extra = row_offset + 1;  /* +1: 滤波器余量 */
                            } else {
                                required_extra = 1;  /* 当前行 + 滤波器余量 */
                            }
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
                            int mvy = cu->mv[pu][list].y;
                            int required_extra;
                            if (mvy > 0) {
                                int pix_offset = (mvy + 3) / 4;
                                int row_offset = (pix_offset + lcu_size - 1) / lcu_size;
                                required_extra = row_offset + 1;
                            } else {
                                required_extra = 1;
                            }
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

/* ===================================================================
 * 2-pass 帧并行模式
 *
 * 将帧解码分为两个阶段:
 *   Phase 1 (AEC, 无需等待参考帧): 逐 LCU 执行 pass=1, 解析熵编码并
 *     保存系数到 per-LCU 缓冲区. AEC 只读码流, 不访问参考帧像素.
 *   Phase 2 (重建+LF, 需参考帧就绪): 逐 LCU 行执行 pass=2 重建 + 环路滤波.
 *
 * 优势: Phase 1 可与前一帧的 Phase 2 并行执行, 大幅提升帧间并行度.
 *       AEC 占解码时间 ~80%, 将其从依赖链中解耦后, 串行瓶颈仅剩 ~20%.
 * =================================================================== */
/* 2-pass 帧并行: Phase 1 (AEC)
 * 等待 col_pic 的 AEC 完成, 执行 AEC 解码, 发送 aec_done 信号.
 * Phase 1 不等待参考帧重建, 与参考帧 Phase 2 并行. */
int avs2_decode_frame_fc_phase1(struct avs2_internal *c, avs2_frame_ctx *fc)
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

    /* 分配 per-LCU 系数缓冲区 */
    int r = ensure_row_parallel_buffers(fc, f, c->seq->log2_lcu_size);
    if (r) return r;

    /* 重置行级 LF 依赖状态 (行级 LF 依赖).
     * 此处安全: 帧刚被选取 (ref_cnt==1), 上一轮所有依赖帧已完成.
     * Phase 2 将按行设置 lf_row_done[row]=1, 依赖此帧的后续帧在 Phase 2 中按行轮询.
     * p2_started/done 也需重置, 防止帧复用时残留的旧值导致依赖帧误判就绪. */
    {
        int h_lcu = f->h_lcu;
        for (int i = 0; i < h_lcu; i++) {
            avs2_atomic_store(&f->lf_row_done[i], 0);
            avs2_atomic_store(&f->aec_row_done[i], 0);
        }
        avs2_atomic_store(&f->lf_row_done_count, 0);
        avs2_atomic_store(&f->p2_started, 0);
        f->done = 0;
    }

    /* ---- Phase 1: AEC (n_aec_deps 已由 worker 调度保证为 0) ----
     * worker 只在 n_aec_deps==0 时才取此任务, 无需在此等待.
     * col_pic 的 mvbuf/refbuf 在 col_pic 的 Phase 1 中由 store_mv_to_buf 逐行填充.
     * aec_started 在 AEC 初始化后立即设置, 允许依赖此帧的 B 帧 AEC 启动;
     * derive_skip_mv 中按行轮询 col_pic->aec_row_done 保证 mvbuf 可用. */

    if (avs2_atomic_load(&c->shutdown)) return AVS2_OK;
    {
        avs2_aec *aec = fc->aec_pool;
        avs2_aec_init_contexts(aec, fc->slice_type);
        fc->bs.pos = (fc->bs.pos + 7) & ~7;
        avs2_aec_start_decoding(aec, fc->bs.buf, fc->bs.sz, fc->bs.pos);
        fc->aec = aec;
        fc->i_last_dquant = 0;
        f->coi = (int)fc->pic_local.coding_order;

        /* 通知依赖此帧 AEC 的后续帧: Phase 1 已开始.
         * 通用模式 (多 worker): aec_started 信号立即递减 n_aec_deps, 允许 B 帧
         *   AEC 与此帧 AEC 在不同 worker 并行 (derive_skip_mv 按行 spin-wait).
         * 行级流水线模式 (单 AEC 线程): 不在此递减, 等 Phase 1 完成 (aec_done)
         *   时递减. 否则 AEC 线程在 B 帧 derive_skip_mv spin-wait col_pic AEC
         *   行, 但 col_pic AEC 尚未开始 (串行), 死锁. */
        avs2_mutex_lock(&c->task_lock);
        fc->aec_started = 1;
        if (c->n_aec_threads == 0) {
            for (int i = 0; i < c->n_fc; i++) {
                avs2_frame_ctx *other = &c->fc[i];
                if (other != fc && (other->task_state == 1 || other->task_state == 2 ||
                                    other->task_state == 5 || other->task_state == 6) &&
                    other->n_refs > 0 && other->fref[0] == fc->fdec &&
                    !IS_INTRA(other->slice_type)) {
                    other->n_aec_deps--;
                    if (other->n_aec_deps == 0 && other->task_state == 1) {
                        avs2_cond_signal(&c->task_cond);
                    }
                }
            }
        }
        avs2_mutex_unlock(&c->task_lock);

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
            /* 行级 AEC 完成信号: 此行所有 LCU 的 AEC (含 store_mv_to_buf) 已完成,
             * 依赖此帧 mvbuf 的后续帧在 derive_skip_mv 中按行轮询 aec_row_done.
             * 行级流水线模式: signal(recon_cond) 唤醒等待新行的重建线程.
             * 在设置 aec_row_done 前计算此行的 MV 行范围, 确保 Phase 2 重建此行时
             * mv_row_range[lcu_y] 已就绪 (行级 LF 跨帧依赖). */
            compute_mv_row_range_for_row(f, 1 << c->seq->log2_lcu_size,
                                          lcu_y, fc->mv_row_range);
            avs2_atomic_store(&f->aec_row_done[lcu_y], 1);
            if (c->n_aec_threads > 0) {
                avs2_mutex_lock(&c->task_lock);
                avs2_cond_broadcast(&c->recon_cond, &c->task_lock, c->n_waiters_recon);
                avs2_mutex_unlock(&c->task_lock);
            }
        }
        fc->aec = NULL;
    }

    /* ---- Phase 1 完成: 设置 aec_done ----
     * 通用模式: n_aec_deps 已在 Phase 1 开始时 (aec_started 信号) 递减.
     * 行级流水线模式: n_aec_deps 在此 (aec_done 信号) 递减, 确保 AEC 线程
     *   串行处理时 B 帧 AEC 不会在 col_pic AEC 完成前启动. */
    avs2_mutex_lock(&c->task_lock);
    fc->aec_done = 1;
    if (c->n_aec_threads > 0) {
        for (int i = 0; i < c->n_fc; i++) {
            avs2_frame_ctx *other = &c->fc[i];
            if (other != fc && (other->task_state == 1 || other->task_state == 2 ||
                                other->task_state == 5 || other->task_state == 6) &&
                other->n_refs > 0 && other->fref[0] == fc->fdec &&
                !IS_INTRA(other->slice_type)) {
                other->n_aec_deps--;
                if (other->n_aec_deps == 0 && other->task_state == 1) {
                    avs2_cond_signal(&c->task_cond);
                }
            }
        }
    }
    avs2_cond_broadcast(&c->done_cond, &c->task_lock, c->n_waiters_done);
    avs2_mutex_unlock(&c->task_lock);

    return AVS2_OK;
}

/* 2-pass 帧并行: Phase 2 (重建+LF, inline deblock)
 *
 * 逐行执行: 跨帧 LF 依赖等待 → 重建 (pass=2) → save_border → deblock → pad → lf_row_done
 * inline deblock 保持 cache 局部性: 重建后的 LCU 数据仍在 L2 时立即执行 deblock. */

/* 处理一行的完整 P2 流程: lf_wait → recon → save_border → deblock → pad → lf_row_done */
static void p2_do_row(avs2_frame_ctx *fc, struct avs2_internal *c, int lcu_y,
                      int is_inter, int need_pad)
{
    avs2_frame *f = fc->fdec;

    /* 跨帧 LF 依赖: inter 帧需等待参考帧 LF 完成 */
    if (is_inter) {
        int required_row = lcu_y + fc->mv_row_range[lcu_y];
        int spin_cnt = 0;
        for (;;) {
            int all_ready = 1;
            for (int j = 0; j < fc->n_refs; j++) {
                avs2_frame *ref = fc->fref[j];
                if (!ref) continue;
                int rr = required_row;
                if (rr >= ref->h_lcu) rr = ref->h_lcu - 1;
                if (!avs2_atomic_load(&ref->lf_row_done[rr])) {
                    all_ready = 0;
                    break;
                }
            }
            if (all_ready) break;
            if ((++spin_cnt & 0x3ff) == 0 && avs2_atomic_load(&c->shutdown)) break;
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

int avs2_decode_frame_fc_phase2(struct avs2_internal *c, avs2_frame_ctx *fc)
{
    avs2_frame *f = fc->fdec;

    if (!f) {
        return AVS2_ERR_INVALID;
    }

    avs2_atomic_store(&f->p2_started, 1);

    const int h_lcu = f->h_lcu;
    const int is_inter = (fc->n_refs > 0 && !IS_INTRA(fc->slice_type));
    const int need_pad = fc->pic_local.rps.refered_by_others;

    /* 设置 TLS 系数 scratch 缓冲区 (pass=2 重建所需, 32 字节对齐) */
#if defined(_MSC_VER)
    __declspec(align(32)) int16_t scratch_y[64 * 64];
    __declspec(align(32)) int16_t scratch_u[32 * 32];
    __declspec(align(32)) int16_t scratch_v[32 * 32];
#else
    int16_t scratch_y[64 * 64] __attribute__((aligned(32)));
    int16_t scratch_u[32 * 32] __attribute__((aligned(32)));
    int16_t scratch_v[32 * 32] __attribute__((aligned(32)));
#endif
    avs2_set_thread_scratch(scratch_y, scratch_u, scratch_v);

    /* inline deblock: 逐行 重建+deblock+pad, 保持 L2 cache 局部性 */
    for (int lcu_y = 0; lcu_y < h_lcu; lcu_y++) {
        p2_do_row(fc, c, lcu_y, is_inter, need_pad);
    }

    avs2_set_thread_scratch(NULL, NULL, NULL);

    /* 设置帧元数据 */
    f->poc = fc->pic_local.poc;
    f->coi = (int)fc->pic_local.coding_order;
    f->type = (int)fc->pic_local.picture_coding_type;
    f->pts = fc->saved_pts;
    f->dts = fc->saved_dts;

    return AVS2_OK;
}

/* 2-pass 行级并行: Phase 2 (重建+LF, 行级并行).
 *
 * 由 worker_thread 在 phase==2 时调用 (ROW 模式). Phase 1 (AEC) 已完成,
 * f->aec_row_done[row] 已由 Phase 1 逐行设置. 此函数:
 *   1. 重置 fc 级行状态 (row_recon_next, row_lf_next, row_recon_done[], ...)
 *   2. 设置 row_task_fc 唤醒辅助 worker 参与行级并行
 *   3. 调用 avs2_row_parallel_pass2 (owning worker 也参与)
 *   4. 清除 row_task_fc, 等待辅助 worker 退出
 *   5. 设置帧元数据
 *
 * 帧级状态 (f->aec_row_done, f->lf_row_done) 由 Phase 1 重置, 此处不重置.
 * fc->row_aec_done 在 2-pass 模式下不使用 (pass2 检查 f->aec_row_done). */
int avs2_decode_frame_fc_phase2_row(struct avs2_internal *c, avs2_frame_ctx *fc)
{
    avs2_frame *f = fc->fdec;

    if (!f) {
        return AVS2_OK;
    }

    avs2_atomic_store(&f->p2_started, 1);

    const int h_lcu = f->h_lcu;

    /* 重置 fc 级行状态 (Phase 2 行级并行使用).
     * f->aec_row_done 由 Phase 1 设置, 不在此重置.
     * f->lf_row_done 由 Phase 1 重置为 0, 由 LF 任务逐行设置为 1. */
    {
        int i;
        for (i = 0; i < h_lcu; i++) {
            fc->row_aec_done[i] = 0;
            fc->row_recon_done[i] = 0;
            fc->row_lf_done[i] = 0;
        }
        fc->row_recon_completed = 0;
        fc->row_recon_next = 0;
        fc->row_lf_next = 0;
        fc->row_lf_done_count = 0;
        fc->n_row_workers = 0;
    }

    /* 广播唤醒 helper 线程参与 Pass 2 行级并行.
     * Phase C: 不再使用 row_task_fc 单值, helper 通过扫描 recon_active 找到帧.
     * 行级流水线模式 (n_aec_threads>0): 重建线程等待 recon_cond.
     * 通用模式: worker 等待 task_cond. */
    avs2_mutex_lock(&c->task_lock);
    if (c->n_aec_threads > 0) {
        avs2_cond_broadcast(&c->recon_cond, &c->task_lock, c->n_waiters_recon);
    } else {
        avs2_cond_broadcast(&c->task_cond, &c->task_lock, c->n_waiters_task);
    }
    avs2_mutex_unlock(&c->task_lock);

    /* ---- Pass 2: owning worker 参与行级并行重建+LF ---- */
    avs2_row_parallel_pass2(c, fc, 0);

    /* 等待辅助 worker 退出 */
    avs2_mutex_lock(&c->task_lock);
    while (fc->n_row_workers > 0) {
        c->n_waiters_done++;
        avs2_cond_wait(&c->done_cond, &c->task_lock);
        c->n_waiters_done--;
    }
    avs2_mutex_unlock(&c->task_lock);

    /* 设置帧元数据 */
    f->poc = fc->pic_local.poc;
    f->coi = (int)fc->pic_local.coding_order;
    f->type = (int)fc->pic_local.picture_coding_type;
    f->pts = fc->saved_pts;
    f->dts = fc->saved_dts;

    return AVS2_OK;
}

/* 解码指定 fc 的帧 (LCU 解码循环). 由 worker 线程或主线程 (n_threads=1) 调用.
 * 使用 fc->pic_local 而非 c->pic, 避免与主线程的 header 解析竞争.
 * 包含原 avs2_decode_frame 中 got_picture 分支的逻辑 (LCU 循环 + 环路滤波 + padding). */
int avs2_decode_frame_fc(struct avs2_internal *c, avs2_frame_ctx *fc)
{
    avs2_frame *f = fc->fdec;

    if (!f) {
        avs2_warn(c, "no frame allocated\n");
        return AVS2_OK;
    }
    if (!fc->bs.buf) {
        avs2_warn(c, "no slice data (bs.buf is NULL)\n");
        return AVS2_OK;
    }

    /* 行级并行模式: 2-pass (AEC 串行 + 重建并行) */
    if (c->thread_mode == AVS2_THREAD_ROW && c->n_threads > 1) {
        return avs2_decode_frame_fc_row(c, fc);
    }

    /* 帧级并行模式: 2-pass 由 worker_thread 分别调用 phase1/phase2, 不在此处调用. */

    /* 单 pass 模式 (单线程): AEC+重建一气呵成, 无需中间等待 */
    {
        int b_slice_checked = 0;  /* 对应 davs2 h->b_slice_checked, 初始 0 */

        /* 重置行级进度标志.
         * aec_row_done: 后续帧的 derive_skip_mv 按行轮询 col_pic->aec_row_done,
         *   单 pass 必须设置此项, 否则 inter 帧无限自旋.
         * lf_row_done: 跨帧 LF 依赖 (多线程模式复用同一帧时需要).
         * avs2_frame_alloc 已用 allocz 清零, 此处显式重置以防帧未重新分配的情况. */
        {
            int h_lcu = f->h_lcu;
            int i;
            for (i = 0; i < h_lcu; i++) {
                avs2_atomic_store(&f->aec_row_done[i], 0);
                avs2_atomic_store(&f->lf_row_done[i], 0);
            }
            avs2_atomic_store(&f->lf_row_done_count, 0);
            f->done = 0;
        }

        /* AEC 上下文初始化 (使用预分配的 aec_pool, 避免每帧堆分配) */
        avs2_aec *aec = fc->aec_pool;
        avs2_aec_init_contexts(aec, fc->slice_type);
        fc->bs.pos = (fc->bs.pos + 7) & ~7;
        avs2_aec_start_decoding(aec, fc->bs.buf, fc->bs.sz, fc->bs.pos);
        fc->aec = aec;
        fc->i_last_dquant = 0;

        f->coi = (int)fc->pic_local.coding_order;

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

                int r = avs2_decode_lcu(fc, c, lcu_x, lcu_y, 0);
                if (r < 0) {
                    avs2_warn(c, "LCU (%d,%d) decode error\n", lcu_x, lcu_y);
                    break;
                }
                b_slice_checked = aec_startcode_follows(aec, 1);
                fc->bs.pos = avs2_aec_get_bits_read(aec);
            }
            /* 此行 AEC+重建完成 (pass=0 中 store_mv_to_buf 已填充 mvbuf),
             * 通知后续帧的 derive_skip_mv 可以读取此行 mvbuf. */
            avs2_atomic_store(&f->aec_row_done[lcu_y], 1);

            avs2_lf_apply_lcu_row(fc, c, lcu_y);
            if (fc->pic_local.rps.refered_by_others) {
                avs2_pad_line_lcu(f, lcu_y, (int)c->seq->log2_lcu_size);
            }
            /* 此行 LF 完成, 通知跨帧 LF 依赖. */
            avs2_atomic_store(&f->lf_row_done[lcu_y], 1);
            avs2_atomic_inc(&f->lf_row_done_count);
        }

        /* AEC 上下文由 aec_pool 管理, 仅清除指针 */
        fc->aec = NULL;

        f->poc = fc->pic_local.poc;
        f->coi = (int)fc->pic_local.coding_order;
        f->type = (int)fc->pic_local.picture_coding_type;
        f->pts = fc->saved_pts;
        f->dts = fc->saved_dts;
        f->done = 1;

        return AVS2_OK;
    }
}

int avs2_decode_frame(struct avs2_internal *c, const uint8_t *data, int sz)
{
    int pos = 0;
    int got_picture = 0;
    int sc_pos, sc_id;

    /* 设置 cur_fc 供 header 解析使用.
     * n_threads=1 时: cur_fc = &c->fc[0] (同步路径)
     * n_threads>1 时: cur_fc 由 lib.c 在提交新帧前设置 (从空闲 fc 中选取) */
    if (!c->cur_fc) {
        c->cur_fc = &c->fc[0];
    }

    while (pos < sz && avs2_find_start_code(data + pos, sz - pos, &sc_pos, &sc_id)) {
        sc_pos += pos;
        int r = process_start_code(c, data, sz, sc_pos, sc_id);
        if (r == AVS2_ERR_NOMEM) {
            /* DPB 满: 立即返回, 不消费后续数据.
             * avs2_send_data 检测到 NOMEM 后不 shift in_buf,
             * 让调用者先输出帧释放 DPB 空间再重试. */
            return AVS2_ERR_NOMEM;
        }
        if (r < 0) {
            avs2_warn(c, "error processing start code %02x: %d\n", sc_id, r);
            /* 不中断解码, 跳过错误继续处理下一个 start code */
        } else {
            if (sc_id == AVS2_SC_INTRA_PICTURE || sc_id == AVS2_SC_INTER_PICTURE)
                got_picture = 1;
        }
        pos = sc_pos + 4;
    }

    if (got_picture) {
        avs2_frame_ctx *fc = c->cur_fc;
        if (!fc) {
            avs2_warn(c, "no cur_fc set for decode\n");
            return AVS2_OK;
        }

        /* 保存 pts/dts 到 fc, 避免 worker 执行时 c->in_pts 已被下一帧覆盖 */
        fc->saved_pts = c->in_pts;
        fc->saved_dts = c->in_dts;

        if (c->n_threads_active > 0) {
            /* 多线程: 提交任务到队列, 由 worker 解码.
             * lib.c 中的 submit_frame_task 负责入队和唤醒 worker. */
            extern int avs2_submit_frame_task(struct avs2_internal *c, avs2_frame_ctx *fc);
            int r = avs2_submit_frame_task(c, fc);
            if (r < 0) return r;
            /* 主线程不等待, 立即返回以便解析下一帧 */
        } else {
            /* 单线程: 同步解码, 保持原有行为 */
            fc->task_state = 2;  /* decoding */
            int r = avs2_decode_frame_fc(c, fc);
            fc->task_state = 3;  /* done */
            if (r < 0) return r;
        }
    }
    return AVS2_OK;
}

static int process_start_code(struct avs2_internal *c, const uint8_t *data,
                              int sz, int sc_pos, int sc_id)
{
    /* find the extent of this unit (up to next start code) */
    int unit_end = sz;
    {
        int p2, id2;
        if (avs2_find_start_code(data + sc_pos + 4, sz - sc_pos - 4, &p2, &id2))
            unit_end = sc_pos + 4 + p2;
    }
    avs2_bs bs;
    avs2_bs_init(&bs, data + sc_pos, unit_end - sc_pos);

    switch (sc_id) {
        case AVS2_SC_SEQUENCE_HEADER:
            /* avs2_parse_sequence_header 内部会比较新旧 sequence header:
             * 内容相同则跳过写入 (无竞争), 内容不同则等待 worker 后写入.
             * 无需在此处额外等待. */
            return avs2_parse_sequence_header(c, &bs);
        case AVS2_SC_SEQUENCE_END:
            return AVS2_OK;
        case AVS2_SC_USER_DATA:
        case AVS2_SC_EXTENSION:
        case AVS2_SC_VIDEO_EDIT:
            /* skipped per davs2 behavior */
            return AVS2_OK;
        case AVS2_SC_INTRA_PICTURE:
        case AVS2_SC_INTER_PICTURE: {
            int r = avs2_parse_picture_header(c, &bs, sc_id);
            if (r) {
                avs2_warn(c, "parse_picture_header sc=%02x failed: %d\n", sc_id, r);
                return r;
            }
            /* 分配解码帧. 使用 c->cur_fc (由 avs2_decode_frame 设置). */
            avs2_frame_ctx *fc = c->cur_fc;
            if (!fc) {
                avs2_warn(c, "no cur_fc for picture header\n");
                return AVS2_ERR_INVALID;
            }
            /* 复制图像头到 fc->pic_local, 供 worker 安全读取 (避免与下一帧 header 解析竞争) */
            fc->pic_local = *c->pic;
            fc->slice_type = (int)c->pic->picture_coding_type;  /* 设置当前帧的 slice 类型 */
            /* 多线程: 在分配帧和构建参考列表时持有 task_lock,
             * 防止 worker 同时修改 DPB (减少 ref_cnt) 导致数据竞争.
             * worker 完成帧后减少 ref_cnt 也在 task_lock 保护下 (lib.c worker_thread),
             * 确保 avs2_dpb_get_free 和 avs2_build_reference_list 中
             * ref_cnt 检查的原子性, 避免 use-after-free. */
            int need_lock = c->n_threads_active > 0;
            if (need_lock) avs2_mutex_lock(&c->task_lock);
            fc->fdec = avs2_dpb_get_free(c);
            if (!fc->fdec) {
                if (need_lock) avs2_mutex_unlock(&c->task_lock);
                return AVS2_ERR_NOMEM;
            }
            r = avs2_frame_alloc(fc->fdec, c);
            if (r) {
                if (need_lock) avs2_mutex_unlock(&c->task_lock);
                avs2_warn(c, "frame_alloc failed: %d\n", r);
                return r;
            }
            /* 在 build_reference_list 前设置 coi, 供 DPB 查找 (虽然新帧 coi 不会被自身查找, 但保持一致性) */
            fc->fdec->coi = (int)c->pic->coding_order;
            fc->fdec->poc = c->pic->poc;
            fc->fdec->type = (int)c->pic->picture_coding_type;
            avs2_build_reference_list(c, fc);
            /* 在 task_lock 保护下设置 referenced, 供 worker 和 avs2_dpb_get_free 读取.
             * worker 不再写 f->referenced (避免与主线程 avs2_build_reference_list 竞争),
             * padding 判断改用 fc->pic_local.rps.refered_by_others (快照, 不变). */
            fc->fdec->referenced = fc->pic_local.rps.refered_by_others;
            /* 多线程: 在 task_lock 保护下递增参考帧 ref_cnt, 防止 avs2_picture_unref
             * 在 avs2_build_reference_list 和 avs2_submit_frame_task 之间
             * 释放帧 (referenced=0 的帧会被 avs2_picture_unref 判为可释放).
             * 单线程: 无并发, referenced 标志已足够, 无需 ref_cnt (complete_frame
             * 不被调用, 递增后无法递减). */
            if (need_lock) {
                for (int j = 0; j < fc->n_refs; j++) {
                    if (fc->fref[j]) {
                        fc->fref[j]->ref_cnt++;
                    }
                }
            }
            if (need_lock) avs2_mutex_unlock(&c->task_lock);
            return AVS2_OK;
        }
        default:
            /* slice start code (0x00..0x8F) */
            if (sc_id <= AVS2_SC_SLICE_MAX) {
                avs2_frame_ctx *fc = c->cur_fc;
                if (!fc) {
                    avs2_warn(c, "no cur_fc for slice header\n");
                    return AVS2_ERR_INVALID;
                }
                int r = avs2_parse_slice_header(c, &bs, sc_id, fc);
                if (r) return r;
                /* save slice data bitstream for LCU decoding */
                /* 对 slice 数据移除伪起始码防竞争比特 (对应 davs2
                 * bs_dispose_pseudo_code). 在原始数据中查找 unit_end
                 * (避免处理后 AEC 数据中的 00 00 01 模式被误识别),
                 * 然后对单个 slice 的数据应用位级重排.
                 * start code 和 slice header 不含伪起始码, 故 bs.pos
                 * (slice header 后的比特位置) 对处理后数据仍然有效. */
                int slice_len = unit_end - sc_pos;
                if (slice_len > fc->slice_buf_cap) {
                    uint8_t *nb = avs2_mem_realloc(fc->slice_buf, (size_t)slice_len);
                    if (!nb) return AVS2_ERR_NOMEM;
                    fc->slice_buf = nb;
                    fc->slice_buf_cap = slice_len;
                }
                int proc_sz = avs2_dispose_pseudo_code(fc->slice_buf,
                                                       data + sc_pos,
                                                       slice_len);
                fc->bs = bs;  /* 保留 pos (slice header 后位置) 和 error */
                fc->bs.buf = fc->slice_buf;
                fc->bs.sz  = proc_sz;
                return AVS2_OK;
            }
            return AVS2_OK;
    }
}
