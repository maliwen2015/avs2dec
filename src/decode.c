/*
 * Main decode orchestration. Parses Annex B start codes, drives header
 * parsing, allocates frames, and runs the LCU-row decode loop.
 */

#include "internal.h"
#include "aec_internal.h"
#include "quant.h"
#include <string.h>
#include <stdio.h>

/* 编译期性能统计开关: 1=启用每帧计时 (Release 下应保持 0 以避免
 * clock_gettime/QueryPerformanceCounter 系统调用进入热路径). */
#ifndef AVS2_PROFILE
#define AVS2_PROFILE 0
#endif

#if AVS2_PROFILE
#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
static double dbg_time_ms(void) {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart * 1000.0;
}
#else
#include <time.h>
static double dbg_time_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
#endif
/* 2-pass 性能统计 (调试用) */
static double g_p2_recon_total = 0;
static int g_2pass_count = 0;
static int g_frame_types[8] = {0};
/* profiling counters defined in lib.c */
extern volatile double g_p1_wait_total;
extern volatile double g_p1_aec_total;
extern volatile double g_p2_wait_total;
extern volatile double g_pick_fc_wait_total;
extern volatile int    g_pick_fc_block_count;
extern volatile double g_worker_idle_total;
#define AVS2_PROFILE_DECL double _prof_t0 = 0
#define AVS2_PROFILE_START() _prof_t0 = dbg_time_ms()
#define AVS2_PROFILE_END_ACCUM(counter) (counter) += dbg_time_ms() - _prof_t0
#else
#define AVS2_PROFILE_DECL ((void)0)
#define AVS2_PROFILE_START() ((void)0)
#define AVS2_PROFILE_END_ACCUM(counter) ((void)0)
#endif

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

/* 确保 per-LCU 系数缓冲区和 per-row 进度数组已分配 (尺寸变化时重新分配) */
static int ensure_row_parallel_buffers(avs2_frame_ctx *fc, avs2_frame *f, int lcu_log2)
{
    int w_lcu = f->w_lcu;
    int h_lcu = f->h_lcu;
    int n_lcu = w_lcu * h_lcu;
    int lcu_dim = 1 << lcu_log2;       /* 亮度 LCU 边长 (32 或 64) */
    int c_dim  = lcu_dim >> 1;         /* 色度 LCU 边长 */

    /* per-LCU 系数缓冲区: 按实际 LCU 大小分配, 避免浪费 */
    if (fc->coeff_lcu_w_lcu != w_lcu || fc->coeff_lcu_h_lcu != h_lcu ||
        fc->coeff_lcu_log2 != lcu_log2) {
        avs2_mem_free(fc->coeff_lcu_y);
        avs2_mem_free(fc->coeff_lcu_u);
        avs2_mem_free(fc->coeff_lcu_v);
        fc->coeff_lcu_y = avs2_mem_allocz(sizeof(int16_t) * (size_t)n_lcu * lcu_dim * lcu_dim);
        fc->coeff_lcu_u = avs2_mem_allocz(sizeof(int16_t) * (size_t)n_lcu * c_dim * c_dim);
        fc->coeff_lcu_v = avs2_mem_allocz(sizeof(int16_t) * (size_t)n_lcu * c_dim * c_dim);
        if (!fc->coeff_lcu_y || !fc->coeff_lcu_u || !fc->coeff_lcu_v) {
            return AVS2_ERR_NOMEM;
        }
        fc->coeff_lcu_w_lcu = w_lcu;
        fc->coeff_lcu_h_lcu = h_lcu;
        fc->coeff_lcu_log2  = (int8_t)lcu_log2;
    }

    /* per-row 进度数组 + mv_row_range (行级 LF 依赖) */
    if (fc->row_parallel_h_lcu != h_lcu) {
        avs2_mem_free((void *)fc->row_aec_done);
        avs2_mem_free((void *)fc->row_recon_done);
        avs2_mem_free((void *)fc->row_lf_done);
        avs2_mem_free(fc->mv_row_range);
        fc->row_aec_done   = (volatile int *)avs2_mem_allocz(sizeof(int) * (size_t)h_lcu);
        fc->row_recon_done = (volatile int *)avs2_mem_allocz(sizeof(int) * (size_t)h_lcu);
        fc->row_lf_done    = (volatile int *)avs2_mem_allocz(sizeof(int) * (size_t)h_lcu);
        fc->mv_row_range   = (int *)avs2_mem_allocz(sizeof(int) * (size_t)h_lcu);
        if (!fc->row_aec_done || !fc->row_recon_done || !fc->row_lf_done ||
            !fc->mv_row_range) {
            return AVS2_ERR_NOMEM;
        }
        fc->row_parallel_h_lcu = h_lcu;
        fc->mv_row_range_h_lcu = h_lcu;
    }

    return AVS2_OK;
}

/* 行级并行: owning worker 的帧解码入口.
 * Pass 1 (AEC 串行) + Pass 2 (参与行级并行重建). */
static int avs2_decode_frame_fc_row(struct avs2_internal *c, avs2_frame_ctx *fc)
{
    avs2_frame *f = fc->fdec;
    int b_slice_checked = 0;

    if (!f) {
        avs2_warn(c, "no frame allocated\n");
        return AVS2_OK;
    }
    if (!fc->bs.buf) {
        avs2_warn(c, "no slice data (bs.buf is NULL)\n");
        return AVS2_OK;
    }

    /* 分配 per-LCU 系数缓冲区和 per-row 进度数组 */
    int r = ensure_row_parallel_buffers(fc, f, c->seq->log2_lcu_size);
    if (r) return r;

    /* 重置 per-row 进度 */
    {
        int h_lcu = f->h_lcu;
        int i;
        for (i = 0; i < h_lcu; i++) {
            fc->row_aec_done[i] = 0;
            fc->row_recon_done[i] = 0;
            fc->row_lf_done[i] = 0;
        }
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

    /* 设置 row_task_fc, 通知辅助 worker 可以参与 Pass 2 */
    avs2_mutex_lock(&c->task_lock);
    c->row_task_fc = fc;
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

        /* 通知此行 AEC 完成, 唤醒等待的 Pass 2 worker */
        avs2_mutex_lock(&c->task_lock);
        fc->row_aec_done[lcu_y] = 1;
        avs2_cond_broadcast(&c->done_cond, &c->task_lock, c->n_waiters_done);
        avs2_mutex_unlock(&c->task_lock);
    }

    /* ---- Pass 2: owning worker 也参与行级并行重建+LF ---- */
    avs2_row_parallel_pass2(c, fc, 0);

    /* 清除 row_task_fc (仅当仍指向此 fc), 等待辅助 worker 退出 */
    avs2_mutex_lock(&c->task_lock);
    if (c->row_task_fc == fc) {
        c->row_task_fc = NULL;
    }
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
        return AVS2_OK;
    }
    if (!fc->bs.buf) {
        avs2_warn(c, "no slice data (bs.buf is NULL)\n");
        return AVS2_OK;
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

    /* 统计帧类型 */
#if AVS2_PROFILE
    if (fc->slice_type >= 0 && fc->slice_type < 8) g_frame_types[fc->slice_type]++;
#endif

    /* ---- Phase 1: AEC (n_aec_deps 已由 worker 调度保证为 0) ----
     * worker 只在 n_aec_deps==0 时才取此任务, 无需在此等待.
     * col_pic 的 mvbuf/refbuf 在 col_pic 的 Phase 1 中由 store_mv_to_buf 逐行填充.
     * aec_started 在 AEC 初始化后立即设置, 允许依赖此帧的 B 帧 AEC 启动;
     * derive_skip_mv 中按行轮询 col_pic->aec_row_done 保证 mvbuf 可用. */
    AVS2_PROFILE_DECL;
    AVS2_PROFILE_START();

    if (c->shutdown) return AVS2_OK;
    {
        avs2_aec *aec = fc->aec_pool;
        avs2_aec_init_contexts(aec, fc->slice_type);
        fc->bs.pos = (fc->bs.pos + 7) & ~7;
        avs2_aec_start_decoding(aec, fc->bs.buf, fc->bs.sz, fc->bs.pos);
        fc->aec = aec;
        fc->i_last_dquant = 0;
        f->coi = (int)fc->pic_local.coding_order;

        /* 通知依赖此帧 AEC 的后续帧: Phase 1 已开始, 可启动 B 帧 AEC.
         * 行级 aec_row_done 跟踪确保 derive_skip_mv 按行等待 col_pic AEC 完成. */
        avs2_mutex_lock(&c->task_lock);
        fc->aec_started = 1;
        for (int i = 0; i < c->n_fc; i++) {
            avs2_frame_ctx *other = &c->fc[i];
            if (other != fc && (other->task_state == 1 || other->task_state == 2 ||
                                other->task_state == 5) &&
                other->n_refs > 0 && other->fref[0] == fc->fdec &&
                !IS_INTRA(other->slice_type)) {
                other->n_aec_deps--;
                if (other->n_aec_deps == 0 && other->task_state == 1) {
                    avs2_cond_signal(&c->task_cond);
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
             * 依赖此帧 mvbuf 的后续帧在 derive_skip_mv 中按行轮询 aec_row_done. */
            avs2_atomic_store(&f->aec_row_done[lcu_y], 1);
        }
        fc->aec = NULL;
    }
    AVS2_PROFILE_END_ACCUM(g_p1_aec_total);

    /* 计算行级 MV 范围 (用于 Phase 2 行级 LF 依赖).
     * cu_grid 已在 AEC 中填充, 可直接扫描 MV 计算. */
    compute_mv_row_ranges(f, 1 << c->seq->log2_lcu_size, fc->mv_row_range);

    /* ---- Phase 1 完成: 设置 aec_done ----
     * n_aec_deps 递减已在 Phase 1 开始时 (aec_started 信号) 完成, 此处仅设置 aec_done.
     * aec_done 用于 complete_frame 的补偿递减判断 (非 2-pass 模式). */
    avs2_mutex_lock(&c->task_lock);
    fc->aec_done = 1;
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
    if (is_inter && fc->mv_row_range) {
        int required_row = lcu_y + fc->mv_row_range[lcu_y];
        int spin_cnt = 0;
        for (;;) {
            int all_ready = 1;
            for (int j = 0; j < fc->n_refs; j++) {
                avs2_frame *ref = fc->fref[j];
                if (!ref || !ref->lf_row_done) continue;
                int rr = required_row;
                if (rr >= ref->h_lcu) rr = ref->h_lcu - 1;
                if (!avs2_atomic_load(&ref->lf_row_done[rr])) {
                    all_ready = 0;
                    break;
                }
            }
            if (all_ready) break;
            if ((++spin_cnt & 0x3ff) == 0 && c->shutdown) break;
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
        return AVS2_OK;
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

    AVS2_PROFILE_DECL;
    AVS2_PROFILE_START();

    /* inline deblock: 逐行 重建+deblock+pad, 保持 L2 cache 局部性 */
    for (int lcu_y = 0; lcu_y < h_lcu; lcu_y++) {
        p2_do_row(fc, c, lcu_y, is_inter, need_pad);
    }

    avs2_set_thread_scratch(NULL, NULL, NULL);

    /* 累计统计 */
#if AVS2_PROFILE
    AVS2_PROFILE_END_ACCUM(g_p2_recon_total);
    g_2pass_count++;
    if (g_2pass_count % 100 == 0) {
        FILE *dbg = fopen("dbg_stats.txt", "a");
        if (dbg) {
            fprintf(dbg, "[2pass stats] frames=%d  P1_wait=%.1fms P1_aec=%.1fms P2_wait=%.1fms P2_recon=%.1fms fc_wait=%.1fms(%d) idle=%.1fms | types: I=%d P=%d B=%d F=%d S=%d G=%d GB=%d\n",
                g_2pass_count,
                g_p1_wait_total / g_2pass_count,
                g_p1_aec_total / g_2pass_count,
                g_p2_wait_total / g_2pass_count,
                g_p2_recon_total / g_2pass_count,
                g_pick_fc_wait_total / g_2pass_count,
                g_pick_fc_block_count,
                g_worker_idle_total / g_2pass_count,
                g_frame_types[0], g_frame_types[1], g_frame_types[2],
                g_frame_types[4], g_frame_types[5], g_frame_types[3], g_frame_types[6]);
            fclose(dbg);
        }
        fprintf(stderr, "[2pass stats] frames=%d  P1_wait=%.1fms P1_aec=%.1fms P2_wait=%.1fms P2_recon=%.1fms fc_wait=%.1fms(%d) idle=%.1fms | types: I=%d P=%d B=%d F=%d S=%d G=%d GB=%d\n",
                g_2pass_count,
                g_p1_wait_total / g_2pass_count,
                g_p1_aec_total / g_2pass_count,
                g_p2_wait_total / g_2pass_count,
                g_p2_recon_total / g_2pass_count,
                g_pick_fc_wait_total / g_2pass_count,
                g_pick_fc_block_count,
                g_worker_idle_total / g_2pass_count,
                g_frame_types[0], g_frame_types[1], g_frame_types[2],
                g_frame_types[4], g_frame_types[5], g_frame_types[3], g_frame_types[6]);
        fflush(stderr);
    }
#endif

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
            avs2_lf_apply_lcu_row(fc, c, lcu_y);
            if (fc->pic_local.rps.refered_by_others) {
                avs2_pad_line_lcu(f, lcu_y, (int)c->seq->log2_lcu_size);
            }
        }

        /* AEC 上下文由 aec_pool 管理, 仅清除指针 */
        fc->aec = NULL;

        f->poc = fc->pic_local.poc;
        f->coi = (int)fc->pic_local.coding_order;
        f->type = (int)fc->pic_local.picture_coding_type;
        f->pts = fc->saved_pts;
        f->dts = fc->saved_dts;

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
                avs2_warn(c, "dpb_get_free failed: n_dpb=%d\n", c->n_dpb);
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
