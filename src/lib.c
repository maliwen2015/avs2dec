#include "avs2dec/avs2dec.h"
#include "internal.h"
#include "aec_internal.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

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


/* 2-pass profiling counters (volatile double, approximate for tuning) */
volatile double g_p1_wait_total = 0;
volatile double g_p1_aec_total = 0;
volatile double g_p2_wait_total = 0;
volatile double g_pick_fc_wait_total = 0;
volatile int    g_pick_fc_block_count = 0;
volatile double g_worker_idle_total = 0;

void avs2_data_wrap(avs2_data *data, const uint8_t *buf, size_t sz,
                    int64_t pts, int64_t dts)
{
    memset(data, 0, sizeof(*data));
    data->data = buf;
    data->sz = sz;
    data->pts = pts;
    data->dts = dts;
}

void avs2_data_wrap_with_cb(avs2_data *data, const uint8_t *buf, size_t sz,
                            int64_t pts, int64_t dts, void *ref,
                            void (*free_cb)(const uint8_t *, void *))
{
    avs2_data_wrap(data, buf, sz, pts, dts);
    data->ref = ref;
    data->free_cb = free_cb;
}

const char *avs2_version(void) { return AVS2DEC_VERSION_STR; }

unsigned avs2_version_api(void) { return AVS2DEC_API_VERSION; }

void avs2_default_settings(avs2_settings *s)
{
    memset(s, 0, sizeof(*s));
    s->n_threads = 0;
    s->max_frame_delay = 0;
    s->log_level = AVS2_LOG_WARNING;
    s->frame_size_limit = 0;
    s->strict_std_compliance = 0;
    s->skip_loop_filter = 0;
    s->thread_mode = AVS2_THREAD_FRAME;
}

/* ===================================================================
 * 帧级并行: 线程池实现
 *
 * task_state 状态机:
 *   0 = idle      (空闲, 可被主线程选取)
 *   1 = queued    (已入队, 等待 worker 处理)
 *   2 = decoding  (worker 正在解码)
 *   3 = done      (解码完成, 帧在 DPB 中)
 *   4 = reserved  (主线程已选取, 正在解析 header, 尚未入队)
 *
 * 生命周期:
 *   idle(0) -> reserved(4) [主线程 pick_idle_fc]
 *            -> queued(1)  [主线程 submit_frame_task]
 *            -> decoding(2)[worker 出队]
 *            -> done(3)    [worker 完成]
 *            -> idle(0)    [主线程 pick_idle_fc 复用]
 *
 * n_deps: 未完成的参考帧数量. worker 等待 n_deps 归零后才开始解码.
 *   计算方法: 遍历 fc->fref[], 查找每个参考帧的 owning fc,
 *   若 owning fc 的 task_state 为 1(queued) 或 2(decoding), 则计为依赖.
 *   若找不到 owning fc (fc 已被复用), 说明参考帧已完成, 不计为依赖.
 * =================================================================== */

/* 行级并行 Pass 2: worker (owning 或辅助) 参与行级重建 + LF.
 *
 * 无锁设计: 使用原子 CAS 分配行任务, 原子 load/store 检查依赖和通知完成.
 * spin-wait 替代 cond_wait, 避免广播开销和 thundering herd.
 *
 * 任务类型:
 *   重建任务 (do_lf=0): 重建 LCU 行 (pass=2) + 保存 intra_border + 设置 row_recon_done
 *     依赖: row_aec_done[row] + row_recon_done[row-1] (intra top 参考)
 *     2-pass 模式: 还需等待参考帧 LF 行完成 (跨帧行级依赖)
 *   LF 任务 (do_lf=1): 环路滤波 + padding + 设置 row_lf_done + f->lf_row_done
 *     依赖: row_recon_done[row] + row_lf_done[row-1] (SAO/ALF 跨行依赖)
 *
 * is_helper=1: 辅助 worker, 在无可用任务且有 pending P1 任务时提前退出.
 * is_helper=0: owning worker, 必须等待帧完成.
 */
int avs2_row_parallel_pass2(struct avs2_internal *c, avs2_frame_ctx *fc, int is_helper)
{
    avs2_frame *f = fc->fdec;
    const int h_lcu = f->h_lcu;
    const int w_lcu = f->w_lcu;
    const int is_inter = (fc->n_refs > 0 && !IS_INTRA(fc->slice_type));

    /* 每个 worker 分配独立的栈上系数 scratch 缓冲区 (32 字节对齐). */
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

    for (;;) {
        /* 帧完成检查 */
        if (avs2_atomic_load(&fc->row_recon_completed)) {
            avs2_set_thread_scratch(NULL, NULL, NULL);
            return 1;
        }

        int got_task = 0;

        /* 尝试获取重建任务 (lock-free CAS) */
        {
            int row = avs2_atomic_load(&fc->row_recon_next);
            while (row < h_lcu) {
                /* 检查依赖: AEC 完成 + 上一行重建完成 */
                if (!avs2_atomic_load(&fc->row_aec_done[row]) ||
                    (row > 0 && !avs2_atomic_load(&fc->row_recon_done[row - 1]))) {
                    break;  /* 依赖未满足 */
                }
                /* CAS 领取行: row_recon_next 从 row 变为 row+1 */
                int expected = row;
                if (avs2_atomic_cas(&fc->row_recon_next, expected, row + 1)) {
                    got_task = 1;

                    /* 跨帧 LF 依赖: inter 帧需等待参考帧 LF 行完成 */
                    if (is_inter && fc->mv_row_range) {
                        int required_row = row + fc->mv_row_range[row];
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
                            if ((++spin_cnt & 0x3ff) == 0 &&
                                (c->shutdown ||
                                 avs2_atomic_load(&fc->row_recon_completed)))
                                break;
                            avs2_cpu_relax();
                        }
                    }

                    /* 重建 */
                    int lcu_x;
                    for (lcu_x = 0; lcu_x < w_lcu; lcu_x++) {
                        avs2_decode_lcu(fc, c, lcu_x, row, 2);
                    }
                    avs2_save_intra_border(fc, c, row);

                    /* 通知重建完成 (原子 store, 无需锁) */
                    avs2_atomic_store(&fc->row_recon_done[row], 1);
                    break;
                }
                /* CAS 失败: 重新加载 (GCC 已更新 expected, MSVC 需重新 load) */
                #if !defined(__GNUC__) && !defined(__clang__)
                row = avs2_atomic_load(&fc->row_recon_next);
                #else
                row = expected;
                #endif
            }
        }

        if (!got_task) {
            /* 尝试获取 LF 任务 (lock-free CAS) */
            int row = avs2_atomic_load(&fc->row_lf_next);
            while (row < h_lcu) {
                /* 检查依赖: 重建完成 + 上一行 LF 完成 */
                if (!avs2_atomic_load(&fc->row_recon_done[row]) ||
                    (row > 0 && !avs2_atomic_load(&fc->row_lf_done[row - 1]))) {
                    break;
                }
                /* CAS 领取行 */
                int expected = row;
                if (avs2_atomic_cas(&fc->row_lf_next, expected, row + 1)) {
                    got_task = 1;

                    /* LF: 去块 + SAO + ALF + padding */
                    avs2_lf_apply_lcu_row_nosave(fc, c, row);
                    if (fc->pic_local.rps.refered_by_others) {
                        avs2_pad_line_lcu(f, row, (int)c->seq->log2_lcu_size);
                    }

                    /* 跨帧 LF 依赖: 设置 f->lf_row_done */
                    avs2_atomic_store(&f->lf_row_done[row], 1);
                    avs2_atomic_inc(&f->lf_row_done_count);

                    /* 通知 LF 完成 + 检查帧完成 */
                    avs2_atomic_store(&fc->row_lf_done[row], 1);
                    int done_cnt = avs2_atomic_inc(&fc->row_lf_done_count);
                    if (done_cnt == h_lcu) {
                        avs2_atomic_store(&fc->row_recon_completed, 1);
                    }
                    break;
                }
                #if !defined(__GNUC__) && !defined(__clang__)
                row = avs2_atomic_load(&fc->row_lf_next);
                #else
                row = expected;
                #endif
            }
        }

        if (got_task) continue;

        /* 无可用任务 */
        if (c->shutdown) {
            avs2_set_thread_scratch(NULL, NULL, NULL);
            return 1;
        }

        /* 辅助 worker: 检查 pending P1 任务, 有则退出 */
        if (is_helper) {
            avs2_mutex_lock(&c->task_lock);
            for (int i = 0; i < c->n_fc; i++) {
                if (c->fc[i].task_state == 1 &&
                    c->fc[i].n_aec_deps == 0) {
                    avs2_mutex_unlock(&c->task_lock);
                    avs2_set_thread_scratch(NULL, NULL, NULL);
                    return 0;
                }
            }
            avs2_mutex_unlock(&c->task_lock);
        }

        /* spin-wait: 短暂等待新任务可用 */
        avs2_cpu_relax();
    }

    return 1;  /* unreachable */
}

/* LF helper: Phase 2 中并行执行环路滤波 (deblock + pad).
 *
 * 辅助 worker 或 owning worker (重建完成后) 调用此函数,
 * 通过 row_lf_next 顺序领取 LF 行. 每行等待:
 *   row_recon_done[row]  — owning worker 已完成此行重建
 *   lf_row_done[row-1]   — 上一行 LF 已完成 (水平边滤波跨行依赖)
 * 然后执行 avs2_lf_apply_lcu_row_nosave + avs2_pad_line_lcu,
 * 完成后设置 f->lf_row_done[row] + f->lf_row_done_count.
 *
 * LF 按行序执行: row_lf_next 顺序递增 + lf_row_done[row-1] 检查保证.
 * owning worker 重建阶段时, helper 可并行 deblock 已重建完成的行,
 * 实现 recon(row i+1) 与 deblock(row i) 的流水化.
 *
 * 使用 spin-wait (非 cond_wait) 等待行就绪: 避免每行 broadcast done_cond
 * 唤醒所有 worker (thundering herd). 仅在全部 LF 完成时 broadcast 一次. */
void avs2_lf_helper(struct avs2_internal *c, avs2_frame_ctx *fc)
{
    avs2_frame *f = fc->fdec;
    const int h_lcu = f->h_lcu;

    for (;;) {
        int row = -1;
        int spin_cnt = 0;

        /* spin-wait: 等待可用 LF 行 (lock-free 快速检查 + lock 领取) */
        for (;;) {
            if (fc->row_lf_done_count >= h_lcu) goto done;
            if (c->shutdown) goto done;

            /* lock-free 快速检查 (racy read 用于进度判断, 领取时再加锁) */
            int next = fc->row_lf_next;
            if (next < h_lcu &&
                avs2_atomic_load(&fc->row_recon_done[next]) &&
                (next == 0 ||
                 avs2_atomic_load(&f->lf_row_done[next - 1]))) {
                /* 加锁领取行 */
                avs2_mutex_lock(&c->task_lock);
                if (fc->row_lf_done_count >= h_lcu) {
                    avs2_mutex_unlock(&c->task_lock);
                    goto done;
                }
                if (fc->row_lf_next < h_lcu &&
                    avs2_atomic_load(&fc->row_recon_done[fc->row_lf_next]) &&
                    (fc->row_lf_next == 0 ||
                     avs2_atomic_load(&f->lf_row_done[fc->row_lf_next - 1]))) {
                    row = fc->row_lf_next++;
                    avs2_mutex_unlock(&c->task_lock);
                    break;
                }
                avs2_mutex_unlock(&c->task_lock);
            }

            if ((++spin_cnt & 0x3ff) == 0 && c->shutdown) goto done;
            avs2_cpu_relax();
        }

        /* 执行 LF: deblock + pad */
        avs2_lf_apply_lcu_row_nosave(fc, c, row);
        if (fc->pic_local.rps.refered_by_others) {
            avs2_pad_line_lcu(f, row, (int)c->seq->log2_lcu_size);
        }

        /* 通知 LF 完成 (跨帧依赖 + 帧内完成计数).
         * 不逐行 broadcast — 仅在全部完成时唤醒 owning worker. */
        avs2_atomic_store(&f->lf_row_done[row], 1);
        avs2_atomic_inc(&f->lf_row_done_count);

        avs2_mutex_lock(&c->task_lock);
        fc->row_lf_done_count++;
        if (fc->row_lf_done_count >= h_lcu) {
            /* 全部 LF 完成 — 唤醒等待的 owning worker */
            avs2_cond_broadcast(&c->done_cond, &c->task_lock, c->n_waiters_done);
        }
        avs2_mutex_unlock(&c->task_lock);
    }

done:
    /* 退出时唤醒 owning worker (可能在等待 LF 完成或 helper 退出) */
    avs2_mutex_lock(&c->task_lock);
    avs2_cond_broadcast(&c->done_cond, &c->task_lock, c->n_waiters_done);
    avs2_mutex_unlock(&c->task_lock);
}


static void complete_frame(struct avs2_internal *c, avs2_frame_ctx *fc)
{
    fc->task_state = 3;  /* done */
    fc->fdec->done = 1;
    c->n_pending--;

    /* 递减依赖此帧的 queued/decoding/phase1_done 任务的 n_deps.
     * state==1(queued): 尚未开始, n_deps 将在其 Phase 2 前被检查.
     * state==2(decoding): Phase 1 或 Phase 2 进行中.
     * state==5(phase1_done): 等待 Phase 2, n_deps 归零后可执行.
     * n_aec_deps: 若参考帧未通过 Phase 1 信号递减 (aec_started==0, 非 2-pass),
     *   在此补偿递减. 仅递减将此帧作为 col_pic (fref[0]) 的 inter 帧. */
    for (int i = 0; i < c->n_fc; i++) {
        avs2_frame_ctx *other = &c->fc[i];
        if (other != fc && (other->task_state == 1 || other->task_state == 2 ||
                            other->task_state == 5)) {
            for (int j = 0; j < other->n_refs; j++) {
                if (other->fref[j] == fc->fdec) {
                    other->n_deps--;
                    if (!fc->aec_started && j == 0 && !IS_INTRA(other->slice_type)) {
                        other->n_aec_deps--;
                    }
                    /* 行级 LF 依赖: Phase 2 在 Phase 1 完成后已推入 phase2_queue,
                     * 不再由 n_deps 归零触发推入, 避免重复入队. n_deps 仅用于
                     * 帧生命周期管理 (ref_cnt 防释放). */
                    break;
                }
            }
        }
    }

    /* 递减参考帧引用计数 */
    for (int j = 0; j < fc->n_refs; j++) {
        if (fc->fref[j]) {
            fc->fref[j]->ref_cnt--;
        }
    }

    /* 唤醒:
     * - done_cond: worker 等参考帧 (n_deps/n_aec_deps), get_picture 等输出
     * - task_cond: 2-pass 中 state==5 的 fc 可能因 n_deps 归零变为可执行 Phase 2 */
    avs2_cond_broadcast(&c->done_cond, &c->task_lock, c->n_waiters_done);
    avs2_cond_signal(&c->task_cond);
}

/* worker 线程主循环 */
static void *worker_thread(void *arg)
{
    struct avs2_internal *c = (struct avs2_internal *)arg;
    const int use_2pass = (c->thread_mode == AVS2_THREAD_FRAME && c->n_threads > 1);

    for (;;) {
        int fc_idx = -1;
        avs2_frame_ctx *row_fc = NULL;
        avs2_frame_ctx *fc = NULL;
        int phase = 0;  /* 0=单pass/行级, 1=Phase 1, 2=Phase 2 */

        avs2_mutex_lock(&c->task_lock);
        for (;;) {
            if (c->shutdown) break;

            if (use_2pass) {
                /* 1. Phase 2 任务: 扫描 fc 数组, 找 state==5 且所有参考帧
                 *    已开始 Phase 2 并完成至少 2 行 LF (lf_row_done_count >= 2)
                 *    或已完成 (done) 的帧.
                 *    要求 2 行 LF head start: 使参考帧领先依赖帧 ~2 行,
                 *    大幅减少依赖帧 Phase 2 的 lf_wait spin 时间.
                 *    P2 并发上限 (p2_cap): 防止 P2 优先调度导致 P1 饥饿.
                 *    当 P2 活跃 worker 数达到上限时, 跳过 P2 调度, 让 worker
                 *    执行 P1, 持续产生 state==5 帧保持 P2 不间断. */
                if (c->n_p2_active < c->p2_cap) {
                    for (int i = 0; i < c->n_fc; i++) {
                        avs2_frame_ctx *cand = &c->fc[i];
                        if (cand->task_state != 5) continue;
                        int refs_ready = 1;
                        for (int j = 0; j < cand->n_refs; j++) {
                            avs2_frame *ref = cand->fref[j];
                            if (ref && !ref->done &&
                                avs2_atomic_load(&ref->lf_row_done_count) < 2) {
                                refs_ready = 0;
                                break;
                            }
                        }
                        if (refs_ready) {
                            fc = cand;
                            fc->task_state = 2;  /* decoding */
                            phase = 2;
                            c->n_p2_active++;
                            break;
                        }
                    }
                }
                if (fc) break;

                /* 2. Phase 1 任务: 扫描 fc 数组, 只取 n_aec_deps==0 的任务. */
                for (int i = 0; i < c->n_fc; i++) {
                    if (c->fc[i].task_state == 1 && c->fc[i].n_aec_deps == 0) {
                        fc = &c->fc[i];
                        fc->task_state = 2;  /* decoding */
                        phase = 1;
                        break;
                    }
                }
                if (fc) break;
            } else if (c->task_q_head != c->task_q_tail) {
                /* 单 pass: 从 task_queue 取, 在 worker 循环内等待 n_deps. */
                fc_idx = c->task_queue[c->task_q_head];
                c->task_q_head = (c->task_q_head + 1) % AVS2_MAX_FRAME_DELAY;
                fc = &c->fc[fc_idx];
                fc->task_state = 2;  /* decoding */
                phase = 0;
                break;
            }

            /* 3. 单 pass row 辅助 (最低优先级, 仅在无帧级任务时参与). */
            if (c->row_task_fc) {
                avs2_frame_ctx *rfc = c->row_task_fc;
                if (rfc->task_state == 2 && !rfc->row_recon_completed) {
                    row_fc = rfc;
                    rfc->n_row_workers++;
                    break;
                }
            }

            /* 4. 无任务, 等待 */
            {
                double t_idle_start = dbg_time_ms();
                c->n_waiters_task++;
                avs2_cond_wait(&c->task_cond, &c->task_lock);
                c->n_waiters_task--;
                g_worker_idle_total += dbg_time_ms() - t_idle_start;
            }
        }
        avs2_mutex_unlock(&c->task_lock);

        if (c->shutdown) break;

        if (row_fc) {
            /* 行级并行辅助 (单 pass WPP 或 2-pass P2 helper) */
            avs2_row_parallel_pass2(c, row_fc, 1);
            avs2_mutex_lock(&c->task_lock);
            row_fc->n_row_workers--;
            if (row_fc->n_row_workers == 0) {
                avs2_cond_broadcast(&c->done_cond, &c->task_lock, c->n_waiters_done);
            }
            avs2_mutex_unlock(&c->task_lock);
            continue;
        }

        if (phase == 1) {
            /* ---- 2-pass Phase 1 (AEC) ---- */
            avs2_decode_frame_fc_phase1(c, fc);

            if (c->shutdown) break;

            /* Phase 1 完成: 设 state=5, 释放 worker. 不推入 phase2_queue
             * (已改用 fc 数组扫描 + p2_started 检查). 唤醒空闲 worker 让其
             * 扫描 state==5 帧并检查参考帧 p2_started/done 后启动 Phase 2. */
            avs2_mutex_lock(&c->task_lock);
            fc->task_state = 5;  /* phase1_done, waiting for Phase 2 */
            avs2_cond_signal(&c->task_cond);
            avs2_mutex_unlock(&c->task_lock);
        } else if (phase == 2) {
            /* ---- 2-pass Phase 2 (重建+LF, 行级 LF 依赖) ---- */
            avs2_decode_frame_fc_phase2(c, fc);
            avs2_mutex_lock(&c->task_lock);
            c->n_p2_active--;  /* 释放 P2 名额 */
            complete_frame(c, fc);
            avs2_mutex_unlock(&c->task_lock);
        } else {
            /* ---- 单 pass 模式 (行级并行或单线程) ---- */
            /* 等待参考帧就绪 (n_deps == 0) */
            avs2_mutex_lock(&c->task_lock);
            while (fc->n_deps > 0 && !c->shutdown) {
                c->n_waiters_done++;
                avs2_cond_wait(&c->done_cond, &c->task_lock);
                c->n_waiters_done--;
            }
            avs2_mutex_unlock(&c->task_lock);

            if (c->shutdown) break;

            avs2_decode_frame_fc(c, fc);

            avs2_mutex_lock(&c->task_lock);
            complete_frame(c, fc);
            avs2_mutex_unlock(&c->task_lock);
        }
    }

    return NULL;
}

/* 从空闲 fc 中选取一个用于新帧. n_threads>1 时由主线程调用.
 * 若无空闲 fc, 等待某个 worker 完成后释放. */
static avs2_frame_ctx *pick_idle_fc(struct avs2_internal *c)
{
    avs2_mutex_lock(&c->task_lock);
    avs2_frame_ctx *fc = NULL;
    while (!fc) {
        for (int i = 0; i < c->n_fc; i++) {
            /* idle(0) 或 done(3) 的 fc 都可以被复用 */
            if (c->fc[i].task_state == 0 || c->fc[i].task_state == 3) {
                fc = &c->fc[i];
                break;
            }
        }
        if (!fc) {
            /* 无空闲 fc, 等待 worker 完成 */
            double t_block_start = dbg_time_ms();
            c->n_waiters_done++;
            avs2_cond_wait(&c->done_cond, &c->task_lock);
            c->n_waiters_done--;
            g_pick_fc_wait_total += dbg_time_ms() - t_block_start;
            g_pick_fc_block_count++;
        }
    }
    /* 标记为 reserved, 防止其他调用选中同一个 fc */
    fc->task_state = 4;
    avs2_mutex_unlock(&c->task_lock);
    return fc;
}

/* 提交帧任务到队列. 由主线程在 header 解析完成后调用. */
int avs2_submit_frame_task(struct avs2_internal *c, avs2_frame_ctx *fc)
{
    int fc_idx = (int)(fc - c->fc);
    const int use_2pass = (c->thread_mode == AVS2_THREAD_FRAME && c->n_threads > 1);

    avs2_mutex_lock(&c->task_lock);

    /* 计算参考帧依赖数: 遍历 fref[], 检查每个参考帧的 owning fc 是否已完成.
     * 若找不到 owning fc (fc 已被复用), 说明参考帧已完成, 不计为依赖.
     * 去重: 同一参考帧 (相同指针) 只计数一次, 避免 n_deps 与递减不对称.
     *
     * n_deps: Phase 2 (重建) 依赖, 统计所有未完全完成的参考帧.
     * n_aec_deps: Phase 1 (AEC) 依赖, 仅统计 col_pic 且 AEC 未完成的参考帧.
     *   AEC 阶段 derive_skip_mv 时域直接模式唯一读取的参考帧是 col_pic:
     *     B 帧: col_pic = fref[B_BWD] = fref[0]
     *     P/F 帧: col_pic = fref[0]
     *     I/G 帧: 帧内, 无 AEC 依赖
     *   仅等待 col_pic 的 AEC 完成即可, 无需等待其他参考帧, 大幅缩短 P1 等待.
     *   若新帧在 col_pic 的 AEC 完成后才提交, 信号已错过, n_aec_deps 不应统计
     *   此参考帧, 否则 Phase 1 永远等待 n_aec_deps 归零导致死锁. */
    fc->n_deps = 0;
    fc->n_aec_deps = 0;
    {
        /* 确定 col_pic: AEC 阶段唯一需要读取的参考帧 */
        avs2_frame *col_pic = NULL;
        if (!IS_INTRA(fc->slice_type) && fc->n_refs > 0) {
            col_pic = fc->fref[0];
        }
        for (int j = 0; j < fc->n_refs; j++) {
            avs2_frame *ref = fc->fref[j];
            if (!ref) continue;
            /* 跳过已计数的重复参考帧 */
            int dup = 0;
            for (int k = 0; k < j; k++) {
                if (fc->fref[k] == ref) { dup = 1; break; }
            }
            if (dup) continue;
            for (int i = 0; i < c->n_fc; i++) {
                avs2_frame_ctx *fc2 = &c->fc[i];
                if (fc2 != fc && fc2->fdec == ref &&
                    (fc2->task_state == 1 || fc2->task_state == 2 ||
                     fc2->task_state == 5)) {
                    /* 参考帧的 owning fc 未完成 (queued/decoding/phase1_done).
                     * state==5: Phase 1 完成 but Phase 2 未完成, 像素不可用. */
                    fc->n_deps++;
                    /* n_aec_deps 仅统计 col_pic 且 AEC 未完成的参考帧.
                     * aec_started==1 表示参考帧 AEC 已开始, Phase 1 信号已发送,
                     * 新帧不会收到该信号, 不应计入 n_aec_deps. */
                    if (ref == col_pic && !fc2->aec_started) {
                        fc->n_aec_deps++;
                    }
                    break;
                }
            }
        }
    }
    fc->aec_done = 0;
    fc->aec_started = 0;

    /* 增加参考帧引用计数, 防止 DPB 在 worker 解码期间释放它们 */
    for (int j = 0; j < fc->n_refs; j++) {
        if (fc->fref[j]) {
            fc->fref[j]->ref_cnt++;
        }
    }

    /* 设置为 queued.
     * 2-pass: 不推入 task_queue, worker 扫描 fc 数组取 state==1 && n_aec_deps==0.
     *   若 n_aec_deps==0, 立即信号 task_cond 唤醒 worker.
     *   若 n_aec_deps>0, 等 col_pic Phase 1 完成时递减并信号.
     * 单 pass: 推入 task_queue, worker 取后在循环内等待 n_deps. */
    fc->task_state = 1;
    c->n_pending++;
    if (use_2pass) {
        if (fc->n_aec_deps == 0) {
            avs2_cond_signal(&c->task_cond);
        }
    } else {
        c->task_queue[c->task_q_tail] = fc_idx;
        c->task_q_tail = (c->task_q_tail + 1) % AVS2_MAX_FRAME_DELAY;
        avs2_cond_signal(&c->task_cond);
    }

    avs2_mutex_unlock(&c->task_lock);
    return AVS2_OK;
}

avs2_ctx *avs2_open(const avs2_settings *s)
{
    struct avs2_internal *c = avs2_mem_allocz(sizeof(*c));
    if (!c) return NULL;

    if (s) {
        c->n_threads = s->n_threads;
        c->max_frame_delay = s->max_frame_delay;
        c->log_level = s->log_level;
        c->frame_size_limit = s->frame_size_limit;
        c->strict_std_compliance = s->strict_std_compliance;
        c->skip_loop_filter = s->skip_loop_filter;
        c->thread_mode = s->thread_mode;
        c->allocator = s->allocator;
        c->logger = s->logger;
    } else {
        c->log_level = AVS2_LOG_WARNING;
    }

    avs2_cpu_detect(&c->cpu);

    aec_init_context_tab(c->aec_tab_ctx_mps, c->aec_tab_ctx_lps);

    int nthr = c->n_threads;
    if (nthr <= 0) nthr = avs2_cpu_count();
    if (nthr < 1) nthr = 1;
    if (nthr > AVS2_MAX_THREADS) nthr = AVS2_MAX_THREADS;
    c->n_threads = nthr;

    int nfc = c->max_frame_delay;
    if (nfc <= 0) {
        if (c->thread_mode == AVS2_THREAD_ROW) {
            /* 行级并行: 单帧内并行重建, 帧上下文数较少.
             * n_fc=2 允许一帧在重建时, 下一帧可以开始 AEC. */
            nfc = (nthr > 1) ? 2 : 1;
        } else {
            /* 帧级并行 (2-pass): Phase 1 (AEC) 完成后帧进入 state==5 等待 Phase 2.
             * 行级 LF 依赖 (spin-wait): Phase 2 可在参考帧 Phase 2 开始后即启动,
             * P2 重叠执行提高吞吐. n_fc = n_threads * 2: 足够覆盖 P1 并行 +
             * P2 pipeline 深度, 让更多帧同时进入 Phase 2 重叠执行. */
            nfc = nthr * 2;
        }
        if (nfc < 1) nfc = 1;
        if (nfc > AVS2_MAX_FRAME_DELAY) nfc = AVS2_MAX_FRAME_DELAY;
    }
    c->n_fc = nfc;

    c->fc = avs2_mem_allocz(sizeof(avs2_frame_ctx) * (size_t)c->n_fc);
    if (!c->fc) { avs2_mem_free(c); return NULL; }
    for (int i = 0; i < c->n_fc; i++) {
        avs2_mutex_init(&c->fc[i].lock);
        /* allocate per-frame-context coefficient scratch buffers.
         * These are reused for each CU during decoding, avoiding ~12KB
         * per CU entry in the cu_grid (saves ~400MB/frame at 1080p). */
        c->fc[i].coeff_scratch_y = avs2_mem_allocz(sizeof(int16_t) * 64 * 64);
        c->fc[i].coeff_scratch_u = avs2_mem_allocz(sizeof(int16_t) * 32 * 32);
        c->fc[i].coeff_scratch_v = avs2_mem_allocz(sizeof(int16_t) * 32 * 32);
        if (!c->fc[i].coeff_scratch_y || !c->fc[i].coeff_scratch_u ||
            !c->fc[i].coeff_scratch_v) {
            avs2_close((avs2_ctx **)&c);
            return NULL;
        }
        c->fc[i].task_state = 0;  /* idle */
        /* 预分配 AEC 上下文 (避免每帧 create/destroy 堆操作) */
        c->fc[i].aec_pool = avs2_aec_create(c->aec_tab_ctx_mps, c->aec_tab_ctx_lps);
        if (!c->fc[i].aec_pool) {
            avs2_close((avs2_ctx **)&c);
            return NULL;
        }
    }
    avs2_mutex_init(&c->out_lock);

    /* 初始化线程池同步原语 */
    avs2_mutex_init(&c->task_lock);
    avs2_cond_init(&c->task_cond);
    avs2_cond_init(&c->done_cond);
    c->task_q_head = c->task_q_tail = 0;
    c->phase2_q_head = c->phase2_q_tail = 0;
    c->n_pending = 0;
    c->shutdown = 0;
    c->n_waiters_task = 0;
    c->n_waiters_done = 0;
    c->n_p2_active = 0;
    /* P2 并发上限: 防止 P2 优先调度导致 P1 (AEC) 饥饿.
     * 设为 worker 数的一半, 留足够 worker 持续执行 P1 产生 state==5 帧.
     * P2 帧间并受依赖链限制 (~4 有效并发), p2_cap=7 已足够. */
    c->p2_cap = (c->n_threads > 1) ? (c->n_threads - 1) / 2 : 1;
    if (c->p2_cap < 1) c->p2_cap = 1;
    c->n_threads_active = 0;
    c->threads = NULL;
    c->row_task_fc = NULL;     /* 行级并行: 当前无 fc 在 WPP */

    /* 创建 worker 线程 (n_threads > 1 时).
     * n_threads=1: n_threads_active=0, 走同步路径.
     * n_threads>1: 创建 n_threads-1 个 worker (主线程负责 header 解析+提交). */
    if (c->n_threads > 1) {
        int n_workers = c->n_threads - 1;
        c->threads = avs2_mem_allocz(sizeof(avs2_thread_t) * (size_t)n_workers);
        if (!c->threads) {
            avs2_close((avs2_ctx **)&c);
            return NULL;
        }
        for (int i = 0; i < n_workers; i++) {
            if (avs2_thread_create(&c->threads[i], worker_thread, c) != 0) {
                /* 创建失败: 唤醒已创建的 worker 并 join, 回退到单线程模式 */
                avs2_mutex_lock(&c->task_lock);
                c->shutdown = 1;
                avs2_cond_broadcast(&c->task_cond, &c->task_lock, c->n_waiters_task);
                avs2_mutex_unlock(&c->task_lock);
                for (int j = 0; j < c->n_threads_active; j++) {
                    avs2_thread_join(&c->threads[j], NULL);
                }
                avs2_mem_free(c->threads);
                c->threads = NULL;
                c->n_threads_active = 0;
                c->shutdown = 0;
                break;
            }
            c->n_threads_active++;
        }
    }

    c->i_prev_coi = -1;  /* 对应 davs2 davs2.cc:504 mgr->i_prev_coi = -1 */

    c->seq = avs2_mem_allocz(sizeof(avs2_seq_header));
    c->pic = avs2_mem_allocz(sizeof(avs2_pic_header));
    if (!c->seq || !c->pic) { avs2_close((avs2_ctx **)&c); return NULL; }

    avs2_info(c, "avs2dec %s opened (%d threads, %d frame contexts, %d workers)\n",
              AVS2DEC_VERSION_STR, c->n_threads, c->n_fc, c->n_threads_active);
    return (avs2_ctx *)c;
}

void avs2_close(avs2_ctx **ctx)
{
    if (!ctx || !*ctx) return;
    struct avs2_internal *c = (struct avs2_internal *)*ctx;

    /* 关闭线程池: 设置 shutdown, 唤醒所有 worker, join */
    if (c->n_threads_active > 0) {
        avs2_mutex_lock(&c->task_lock);
        c->shutdown = 1;
        avs2_cond_broadcast(&c->task_cond, &c->task_lock, c->n_waiters_task);
        avs2_cond_broadcast(&c->done_cond, &c->task_lock, c->n_waiters_done);
        avs2_mutex_unlock(&c->task_lock);

        for (int i = 0; i < c->n_threads_active; i++) {
            if (c->threads[i]) {
                avs2_thread_join(&c->threads[i], NULL);
            }
        }
        avs2_mem_free(c->threads);
        c->threads = NULL;
        c->n_threads_active = 0;
    }

    /* 销毁线程池同步原语 */
    avs2_mutex_destroy(&c->task_lock);
    avs2_cond_destroy(&c->task_cond);
    avs2_cond_destroy(&c->done_cond);

    avs2_dpb_clear(c);
    for (int i = 0; i < c->n_dpb; i++)
        avs2_mem_free(c->dpb[i]);

    for (int i = 0; i < c->n_fc; i++) {
        avs2_mutex_destroy(&c->fc[i].lock);
        avs2_mem_free(c->fc[i].coeff_scratch_y);
        avs2_mem_free(c->fc[i].coeff_scratch_u);
        avs2_mem_free(c->fc[i].coeff_scratch_v);
        avs2_mem_free(c->fc[i].slice_buf);
        /* 行级并行: 释放 per-LCU 系数缓冲区和 per-row 进度数组 */
        avs2_mem_free(c->fc[i].coeff_lcu_y);
        avs2_mem_free(c->fc[i].coeff_lcu_u);
        avs2_mem_free(c->fc[i].coeff_lcu_v);
        avs2_mem_free((void *)c->fc[i].row_aec_done);
        avs2_mem_free((void *)c->fc[i].row_recon_done);
        avs2_mem_free((void *)c->fc[i].row_lf_done);
        /* LF 临时帧 (SAO/ALF 预分配复用) */
        avs2_lf_tmp_free(&c->fc[i]);
        /* 预分配的 AEC 上下文 */
        avs2_aec_destroy(c->fc[i].aec_pool);
    }
    avs2_mem_free(c->fc);
    avs2_mem_free(c->seq);
    avs2_mem_free(c->pic);
    avs2_mem_free(c->in_buf);
    avs2_mutex_destroy(&c->out_lock);
    avs2_mem_free(c);
    *ctx = NULL;
}

/*
 * Annex B frame accumulation. We buffer incoming bytes and scan for
 * picture boundaries: a new picture starts at an Intra (0xB3) or
 * Inter (0xB6) start code that is NOT preceded by the current picture's
 * slice data. We feed whole frames to avs2_decode_frame().
 */
static int append_input(struct avs2_internal *c, const avs2_data *data)
{
    if ((size_t)c->in_buf_sz + data->sz > (size_t)c->in_buf_cap) {
        int ncap = c->in_buf_cap ? c->in_buf_cap * 2 : 65536;
        while (ncap < c->in_buf_sz + (int)data->sz) ncap *= 2;
        uint8_t *nb = avs2_mem_realloc(c->in_buf, (size_t)ncap);
        if (!nb) return AVS2_ERR_NOMEM;
        c->in_buf = nb;
        c->in_buf_cap = ncap;
    }
    memcpy(c->in_buf + c->in_buf_sz, data->data, data->sz);
    c->in_buf_sz += (int)data->sz;
    c->in_pts = data->pts;
    c->in_dts = data->dts;
    return AVS2_OK;
}

/* Find the start of the next picture (B3/B6) after offset, return its
 * absolute byte position, or -1 if none. */
static int find_next_picture(struct avs2_internal *c, int offset)
{
    const uint8_t *buf = c->in_buf;
    int sz = c->in_buf_sz;
    for (int i = offset; i + 3 < sz; i++) {
        if (buf[i] == 0 && buf[i+1] == 0 && buf[i+2] == 1) {
            int id = buf[i+3];
            if (id == AVS2_SC_INTRA_PICTURE || id == AVS2_SC_INTER_PICTURE)
                return i;
        }
    }
    return -1;
}

int avs2_send_data(avs2_ctx *ctx, avs2_data *data)
{
    struct avs2_internal *c = (struct avs2_internal *)ctx;
    if (!data) {
        /* flush signal */
        c->flushing = 1;
        /* decode any remaining buffered frame */
        if (c->in_buf_sz > 0) {
            /* 多线程: 选取空闲 fc */
            if (c->n_threads_active > 0) {
                if (c->cur_fc && c->cur_fc->task_state == 4) {
                    c->cur_fc->task_state = 0;  /* 释放未提交的 reserved fc */
                }
                c->cur_fc = pick_idle_fc(c);
            }
            int r = avs2_decode_frame(c, c->in_buf, c->in_buf_sz);
            c->in_buf_sz = 0;
            if (r < 0) return r;
        }
        return AVS2_OK;
    }
    int r = append_input(c, data);
    if (r) return r;
    if (data->free_cb) data->free_cb(data->data, data->ref);

    /* Try to extract and decode complete frames. */
    int scan = 0;
    for (;;) {
        int p = find_next_picture(c, scan);
        if (p < 0) break;
        /* find the following picture to bound this one */
        int q = find_next_picture(c, p + 4);
        if (q < 0) {
            /* no end yet; unless flushing, wait for more data */
            if (!c->flushing) break;
            q = c->in_buf_sz;
        }
        /* 多线程: 在解码每帧前选取空闲 fc.
         * 单线程: cur_fc 由 avs2_decode_frame 内部设置 (默认 &c->fc[0]). */
        if (c->n_threads_active > 0) {
            /* 释放上一个未被提交的 reserved fc (没有 got_picture 的情况) */
            if (c->cur_fc && c->cur_fc->task_state == 4) {
                c->cur_fc->task_state = 0;
            }
            c->cur_fc = pick_idle_fc(c);
        }
        /* The frame bytes are [p, q). But the sequence header may precede p;
         * pass the whole prefix so headers are re-parsed. We decode from
         * the start of the buffer up to q. */
        int r2 = avs2_decode_frame(c, c->in_buf, q);
        if (r2 < 0) {
            avs2_warn(c, "decode error: %d\n", r2);
        }
        /* shift remaining bytes */
        int remain = c->in_buf_sz - q;
        if (remain > 0)
            memmove(c->in_buf, c->in_buf + q, (size_t)remain);
        c->in_buf_sz = remain;
        scan = 0;
    }
    return AVS2_OK;
}

int avs2_get_picture(avs2_ctx *ctx, avs2_picture *pic, avs2_seq_header *seq)
{
    struct avs2_internal *c = (struct avs2_internal *)ctx;
    int mt = c->n_threads_active > 0;
    avs2_frame *f = NULL;

    /* 多线程优化: 非 flushing 模式不等待所有任务完成, 而是按需检查 done 标志.
     * 这样主线程可以在 worker 解码时继续 send_data 提交新帧, 提升流水线深度.
     * flushing 模式仍需等待所有任务完成, 确保所有帧都已解码. */
    if (mt) {
        avs2_mutex_lock(&c->task_lock);
        if (c->flushing) {
            while (c->n_pending > 0) {
                c->n_waiters_done++;
                avs2_cond_wait(&c->done_cond, &c->task_lock);
                c->n_waiters_done--;
            }
        }
    } else {
        avs2_mutex_lock(&c->out_lock);
    }

    /* out_next_poc 在 header.c 中由第一个 I 帧的 POC 设定 (对应 davs2 outpics.output).
     * 输出排序逻辑对应 davs2 output_list_get_one_output_picture:
     * 1. 查找 POC == out_next_poc 的未输出帧 (精确匹配)
     * 2. flushing 模式下: 查找最小的 POC >= out_next_poc (跳过缺失的 POC),
     *    若不存在则输出最小的 POC (处理迟到/重复帧)
     * 3. 输出帧后检查是否有相同 POC 的重复帧, 若有则保持 out_next_poc 不变 */
    if (c->out_initialized) {
        /* 1. 精确匹配: 在所有 POC == out_next_poc 的帧中选 coi 最大的
         *    (对应 davs2 有序链表: 后解码的重复帧插入在前面, 先输出).
         *    多线程时只选 done 的帧 (已解码完成). */
        int best_coi = -1;
        for (int i = 0; i < c->n_dpb; i++) {
            avs2_frame *cand = c->dpb[i];
            if (cand && cand->used && !cand->output && cand->poc == c->out_next_poc) {
                if (!mt || cand->done) {
                    if (cand->coi > best_coi) {
                        best_coi = cand->coi;
                        f = cand;
                    }
                }
            }
        }

        /* 2. flushing 模式: 跳过缺失 POC 或处理迟到帧.
         *    flushing 时 n_pending==0, 所有帧都 done, 无需额外检查. */
        if (!f && c->flushing) {
            int min_poc_ge = 0x7fffffff;  /* >= out_next_poc 的最小 POC */
            int min_poc = 0x7fffffff;     /* 全局最小 POC (迟到帧) */
            for (int i = 0; i < c->n_dpb; i++) {
                avs2_frame *cand = c->dpb[i];
                if (!cand || !cand->used || cand->output) continue;
                if (cand->poc < min_poc) {
                    min_poc = cand->poc;
                }
                if (cand->poc >= c->out_next_poc && cand->poc < min_poc_ge) {
                    min_poc_ge = cand->poc;
                }
            }
            int target_poc = (min_poc_ge != 0x7fffffff) ? min_poc_ge : min_poc;
            if (target_poc != 0x7fffffff) {
                /* 在相同 POC 的帧中选 coi 最大的 */
                best_coi = -1;
                for (int i = 0; i < c->n_dpb; i++) {
                    avs2_frame *cand = c->dpb[i];
                    if (cand && cand->used && !cand->output && cand->poc == target_poc) {
                        if (cand->coi > best_coi) {
                            best_coi = cand->coi;
                            f = cand;
                        }
                    }
                }
            }
        }

        if (f) {
            f->output = 1;
            f->ref_cnt++;  /* 用户引用 */

            /* 检查是否有相同 POC 的重复帧 (对应 davs2 有序链表中重复 POC 相邻输出) */
            int has_dup = 0;
            for (int i = 0; i < c->n_dpb; i++) {
                avs2_frame *cand = c->dpb[i];
                if (cand && cand->used && !cand->output && cand->poc == f->poc) {
                    has_dup = 1;
                    break;
                }
            }
            /* 有重复帧时保持 out_next_poc 不变, 以便下次输出重复帧;
             * 无重复帧时前进到下一个 POC */
            c->out_next_poc = has_dup ? f->poc : f->poc + 1;
        }
    }
    if (mt) {
        avs2_mutex_unlock(&c->task_lock);
    } else {
        avs2_mutex_unlock(&c->out_lock);
    }
    if (!f) {
        return c->flushing ? AVS2_ERR_EOF : AVS2_ERR_AGAIN;
    }

    memset(pic, 0, sizeof(*pic));
    for (int i = 0; i < 3; i++) {
        pic->data[i] = f->data[i];
        pic->stride[i] = f->stride[i];
    }
    pic->p_w = f->width;
    pic->p_h = f->height;
    pic->width[0] = f->width;
    pic->height[0] = f->height;
    int cw = (f->width + 1) >> 1, ch = (f->height + 1) >> 1;
    pic->width[1] = pic->width[2] = cw;
    pic->height[1] = pic->height[2] = ch;
    pic->type = f->type;
    pic->poc = f->poc;
    pic->qp = f->qp;
    pic->bit_depth = f->bit_depth;
    pic->bytes_per_sample = f->bytes_per_sample;
    pic->chroma_format = f->chroma_format;
    pic->pts = f->pts;
    pic->dts = f->dts;
    pic->dec_frame = f;

    if (seq && c->seq) {
        memcpy(seq, c->seq, sizeof(*seq));
    }
    return AVS2_OK;
}

void avs2_picture_unref(avs2_ctx *ctx, avs2_picture *pic)
{
    struct avs2_internal *c = (struct avs2_internal *)ctx;
    if (!pic || !pic->dec_frame) return;
    avs2_frame *f = (avs2_frame *)pic->dec_frame;
    /* 多线程: ref_cnt 修改和帧释放需在 task_lock 保护下,
     * 防止与 worker 的 ref_cnt 修改 (worker_thread 完成时) 竞争,
     * 以及与主线程 avs2_dpb_get_free 的 ref_cnt 检查竞争. */
    int need_lock = c->n_threads_active > 0;
    if (need_lock) avs2_mutex_lock(&c->task_lock);
    /* 减少用户引用, 不重置 output 标志 (已输出的帧不应再次输出) */
    f->ref_cnt--;
    /* 如果只剩 DPB 引用且不再被参考且已输出, 则释放 */
    if (f->ref_cnt <= 1 && f->output && !f->referenced) {
        avs2_frame_free(f);
    }
    if (need_lock) avs2_mutex_unlock(&c->task_lock);
    pic->dec_frame = NULL;
}

void avs2_flush(avs2_ctx *ctx)
{
    struct avs2_internal *c = (struct avs2_internal *)ctx;
    /* 设置 flushing 标志, 解码剩余数据, 但不清 DPB —
     * 用户需要通过 avs2_get_picture 取走剩余帧.
     * DPB 和输出状态在 avs2_close 中清理. */
    c->flushing = 1;
    avs2_send_data(ctx, NULL);
    /* flushing 保持为 1, 让 avs2_get_picture 知道可以输出所有剩余帧 */
}
