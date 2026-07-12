#ifndef AVS2DEC_SRC_INTERNAL_H
#define AVS2DEC_SRC_INTERNAL_H

#include "avs2dec/avs2dec.h"
#include "avs2dec/headers.h"
#include "avs2dec/picture.h"

#include "levels.h"
#include "getbits.h"
#include "thread.h"
#include "cpu.h"
#include "mem.h"
#include "log.h"

#define AVS2_DEC_BLOCK_SIZE 8  /* min CU in pixels */
#define AVS2_LCU_MAX 64

/* 最大 LCU 行数, 覆盖 8K (4320/32=135) 并留余量 */
#define AVS2_MAX_H_LCU 256

/* 帧边界扩展 padding (像素数)
 * 对应 davs2 AVS2_PAD: 亮度 64, 色度 32 (420) */
#define AVS2_PAD_LUMA   64
#define AVS2_PAD_CHROMA 32

/* Motion vector (1/4 pixel for luma) */
typedef struct { int32_t x, y; } avs2_mv;

/* CU (coding unit) descriptor stored per min-CU in the LCU grid.
 * 从 davs2 cu_t 移植, 支持完整的多 PU/TU 解码。 */
typedef struct avs2_cu {
    /* ---- 邻域解码所需字段 ---- */
    int8_t   cu_level;          /* log2 尺寸 */
    int8_t   cu_type;           /* avs2_pu_mode */
    int8_t   i_slice_nr;        /* 条带编号 */
    int8_t   qp;                /* 量化参数 */
    int8_t   i_cbp;             /* 编码块模式 */
    int8_t   i_tu_split;        /* avs2_tu_split (TU 分割类型) */

    /* ---- 预测相关字段 ---- */
    int8_t   i_weighted_skipmode;  /* 加权跳过模式 */
    int8_t   i_skip_mode;          /* direct/skip 子模式 (DS_NONE 等) */
    int8_t   i_intra_mode_c;       /* 色度帧内预测模式 */
    int8_t   i_dmh_mode;           /* 方向多假设模式 */
    int8_t   num_pu;               /* 预测单元数量 */
    int8_t   b_intra;              /* 是否帧内 */

    /* ---- 每个块的预测数据 (4 个亮度块 + 2 个色度块) ---- */
    int8_t   b8pdir[4];            /* 每个 PU 的预测方向 (avs2_pdir) */
    int8_t   intra_pred_modes[4];  /* 每个亮度块的帧内模式 */
    int8_t   dct_pattern[6];       /* 每块的 DCT 模式 (4 亮度 + 2 色度) */

    /* PU 位置 (相对于 CU 原点, 像素单位) */
    int8_t   pu_x[4], pu_y[4], pu_w[4], pu_h[4];

    /* 运动矢量和参考索引: [PU 索引][前向/后向] */
    avs2_mv  mv[4][2];
    int8_t   i_ref[4][2];
} avs2_cu;

/* SAO parameters per LCU per component */
typedef struct {
    uint8_t sao_type;       /* avs2_sao_type (码流值) */
    uint8_t mode_idc;       /* SAO 模式: SAO_MODE_OFF=0, SAO_MODE_NEW=1 */
    uint8_t type_idc;       /* 滤波类型内部索引 (EO 方向 0..3, BO=4) */
    uint8_t start_band;     /* BO 起始带 */
    uint8_t start_band2;    /* BO 第二起始带 */
    uint8_t merge_left;     /* 左合并标志 */
    uint8_t merge_up;       /* 上合并标志 */
    int     offset[32];     /* BO: 32 带偏移; EO: offset[0..4] 为 5 类边界偏移 */
} avs2_sao_param;

/* ALF parameters per LCU */
typedef struct {
    uint8_t alf_enable[3];     /* Y, U, V */
    int16_t alf_coeff_y[ALF_NUM_VARS][ALF_MAX_NUM_COEF];
    int16_t alf_coeff_c[2][ALF_MAX_NUM_COEF]; /* U, V */
} avs2_alf_param;

/* A decoded frame in the DPB. */
typedef struct avs2_frame {
    uint8_t *data[3];
    ptrdiff_t stride[3];
    int width, height;       /* luma */
    int w8, h8;              /* in 8x8 units */
    int w_lcu, h_lcu;        /* in LCU units */

    int poc;
    int coi;
    int type;                /* avs2_slice_type */
    int structure;
    int qp;

    int bit_depth;
    int chroma_format;
    int bytes_per_sample;

    int referenced;          /* refered_by_others */
    int used;                /* in use (not free) */
    int output;              /* pending output */
    int64_t pts, dts;

    /* MV distance info (对应 davs2 dist_refs/dist_scale_refs)
     * 用于时域直接模式的 MV 缩放. */
    int dist_refs[AVS2_MAX_REFS];        /* 当前帧到各参考帧的距离 */
    int dist_scale_refs[AVS2_MAX_REFS];  /* = MULTI / dist_refs (缩放因子) */
    int8_t *refbuf;          /* 每个 4x4 块的参考帧索引 (对应 davs2 refbuf) */

    /* intra prediction mode buffer (4x4 granularity, with border padding) */
    int8_t *ipredmode;          /* offset to first real position */
    int ipredmode_stride;       /* stride in 4x4 blocks (incl. padding) */
    int8_t *ipredmode_base;     /* base pointer for free */

    /* per-CU metadata grid (w8 * h8 entries) */
    avs2_cu *cu_grid;
    /* MV buffer for temporal skip prediction */
    avs2_mv *mvbuf;
    /* loop filter flags */
    uint8_t *deblock_flags[2]; /* [edge_ver/hor] */
    /* SAO/ALF params per LCU */
    avs2_sao_param *sao_params;
    avs2_alf_param *alf_params;
    /* intra border cache: bottom row of previous LCU row (pre-deblock)
     * 用于下一 LCU 行顶部边界块的 intra 预测 TOP 参考 (对应 davs2 h->intra_border)
     * 按字节寻址 (8-bit=1 字节/像素, 10-bit=2 字节/像素), 与 f->data 一致 */
    uint8_t *intra_border[3];

    avs2_ref *ref;           /* refcounted buffer */
    void *pic_cookie;
    int ref_cnt;
    int done;                /* 帧解码完成标志 (多线程: worker 设置, get_picture 读取) */
    volatile int p2_started; /* Phase 2 (重建+LF) 已开始. 依赖帧在 Phase 2 调度时
                              * 检查此标志: 仅当所有参考帧 p2_started 或 done 时才
                              * 启动 Phase 2, 避免有限 worker 全部阻塞在行级 LF
                              * 等待导致死锁. Phase 1 重置为 0, Phase 2 开头置 1. */

    /* 行级 LF 完成跟踪 (2-pass 帧并行行级依赖).
     * lf_row_done[i]=1 当 LCU 行 i 的 LF+padding 完成.
     * 原子操作: 依赖此帧的后续帧在 Phase 2 中按行轮询, 无锁读取.
     * Phase 2 开始时重置 (此时无其他线程读取此帧的 lf_row_done). */
    volatile int  lf_row_done[AVS2_MAX_H_LCU]; /* [h_lcu] */
    volatile int  lf_row_done_count; /* 已完成 LF 的行数 */

    /* 行级 AEC 完成跟踪 (2-pass 帧并行 P1 依赖优化).
     * aec_row_done[i]=1 当 LCU 行 i 的 AEC 解码 (含 store_mv_to_buf) 完成.
     * 原子操作: 依赖此帧 mvbuf 的后续帧在 derive_skip_mv 中按行轮询.
     * Phase 1 开始时重置为 0, 每行 AEC 完成后原子存储. */
    volatile int  aec_row_done[AVS2_MAX_H_LCU]; /* [h_lcu] */

    /* 帧缓冲池化: 记录已分配缓冲区的尺寸, 用于复用判断.
     * alloc_*==0 表示尚未分配. 分辨率变化时重新分配. */
    int alloc_width;       /* 已分配的图像宽度 */
    int alloc_height;      /* 已分配的图像高度 */
    int alloc_bit_depth;   /* 已分配的位深 */
    int alloc_chroma;      /* 已分配的色度格式 */

    /* 辅助缓冲区: 将 refbuf/deblock_flags/intra_border/ipredmode_base
     * 合并为单次分配, 减少内存碎片和 alloc/free 次数.
     * 各子缓冲区按 32 字节对齐切分. */
    uint8_t *aux_buf;
} avs2_frame;

/* Frame decoding context (one per concurrent frame). */
typedef struct avs2_frame_ctx {
    avs2_frame *fdec;        /* frame being decoded */
    avs2_frame *fref[AVS2_MAX_REFS];
    int n_refs;

    avs2_bs bs;              /* bitstream reader for current slice */
    struct avs2_aec *aec;    /* AEC context (shared across LCUs in a slice) */
    struct avs2_aec *aec_pool;  /* 预分配的 AEC 上下文 (避免每帧堆分配) */

    /* 伪起始码处理后的 slice 数据缓冲区 (对应 davs2 bs_dispose_pseudo_code 输出).
     * bs.buf 在 slice header 解析后指向此缓冲区, 供 AEC 解码使用. */
    uint8_t *slice_buf;
    int      slice_buf_cap;

    int slice_type;
    int slice_qp;
    int b_dqp;              /* 是否启用 CU 级 delta QP (对应 davs2 h->b_DQP) */
    int i_last_dquant;      /* 上一个 CU 的 delta QP (对应 davs2 h->i_last_dquant) */
    int chroma_quant_param_delta_cb;  /* 色度 Cb QP 偏移 */
    int chroma_quant_param_delta_cr;  /* 色度 Cr QP 偏移 */

    /* SAO slice 级别开关 (Y, Cb, Cr) */
    uint8_t slice_sao_on[3];

    /* current LCU position */
    int lcu_x, lcu_y;
    int lcu_size;            /* log2 */

    /* neighbor availability caches */
    int left_avail, above_avail;

    /* scratch for intra reference samples */
    int16_t ref_samples_left[2 * AVS2_LCU_MAX + 1];
    int16_t ref_samples_top[2 * AVS2_LCU_MAX + 1];

    /* coefficient scratch buffers (reused per-CU, NOT stored in cu_grid).
     * 大小覆盖最大 TU: 亮度 64x64, 色度 32x32.
     * 帧级并行模式: 直接使用 coeff_scratch_y/u/v.
     * 行级并行模式: 使用 cur_lcu_coeff_y/u/v 指向 per-LCU 缓冲区.
     * 32 字节对齐: SIMD 量化/变换操作 (AVX2) 要求. */
    AVS2_ALIGNED_32(int16_t coeff_scratch_y[64 * 64]);  /* 亮度系数 scratch */
    AVS2_ALIGNED_32(int16_t coeff_scratch_u[32 * 32]);  /* U 色度系数 scratch */
    AVS2_ALIGNED_32(int16_t coeff_scratch_v[32 * 32]);  /* V 色度系数 scratch */
    /* 当前 LCU 的系数基址 (avs2_decode_lcu 中设置):
     * 帧级模式 = coeff_scratch_y/u/v; 行级模式 = per-LCU 缓冲区对应位置 */
    int16_t *cur_lcu_coeff_y;
    int16_t *cur_lcu_coeff_u;
    int16_t *cur_lcu_coeff_v;

    /* 行级并行/2-pass 帧并行: per-LCU 系数缓冲区 (Pass 1 存储, Pass 2 读取).
     * 大小 = w_lcu * h_lcu * lcu_dim * lcu_dim (按实际 LCU 大小分配). */
    int16_t *coeff_lcu_y;     /* 亮度系数 */
    int16_t *coeff_lcu_u;     /* U 色度系数 */
    int16_t *coeff_lcu_v;     /* V 色度系数 */
    int coeff_lcu_w_lcu;      /* 分配时的 w_lcu (检测是否需重新分配) */
    int coeff_lcu_h_lcu;      /* 分配时的 h_lcu */
    int8_t coeff_lcu_log2;    /* 分配时的 log2_lcu_size */

    avs2_mutex_t lock;
    int row_progress;        /* last finished LCU row (for threading) */

    /* 帧任务状态 (帧级并行).
     * task_state: 0=idle, 1=queued, 2=decoding, 3=done, 4=reserved, 5=phase1_done
     *   5=phase1_done: Phase 1 (AEC) 已完成, 等待 n_deps==0 后执行 Phase 2 (重建).
     *     worker 在 Phase 2 等待期间释放, 可处理其他帧的 Phase 1, 提高并行度.
     *   6=aec_running: AEC 线程正在执行 Phase 1, 重建线程可按行 pipeline 跟随.
     *     (行级流水线模式: aec_row_done[i] 逐行置位, 重建线程按行重建+LF)
     *   7=aec_done_state: AEC 整帧完成, 重建线程可能仍在收尾. 等重建完成 → done(3).
     * n_deps: 未完成 (非 done) 的参考帧数量, Phase 2 (重建) 等待其归零.
     *   worker 等待其归零后才开始解码 (单 pass 模式).
     * n_aec_deps: 未完成 AEC (非 aec_done) 的参考帧数量, Phase 1 (AEC) 等待其归零.
     *   2-pass 帧并行模式: AEC 需读参考帧 mvbuf (时域直接模式), 仅需参考帧 AEC 完成.
     * aec_done: Phase 1 (AEC) 完成标志. 非 2-pass 模式始终为 0 (由全完成补偿递减).
     * aec_started: Phase 1 (AEC) 已开始. 依赖此帧 mvbuf 的后续帧在提交时检查此标志:
     *   仅当 col_pic 的 aec_started=0 时才计入 n_aec_deps. aec_started 在 Phase 1
     *   开头设置, 允许 B 帧 AEC 与 col_pic AEC 行级重叠 (per-row aec_row_done).
     * pic_local: 当前帧的图像头副本 (worker 不应读 c->pic, 避免与主线程竞争) */
    int task_state;
    int n_deps;
    int n_aec_deps;
    int aec_done;
    int aec_started;
    double phase2_wait_start;  /* profiling: 进入 state==5 的时间戳 */
    avs2_pic_header pic_local;
    int64_t saved_pts, saved_dts;  /* 提交任务时保存的 pts/dts */

    /* 行级流水线: 重建线程在此帧上的工作状态.
     * recon_active: 1 = 有重建线程正在此帧上工作 (防止重复进入)
     * recon_started: 1 = 重建已启动 (AEC 完成首行后即可)
     * recon_cond: AEC 完成新行后唤醒等待的重建线程 */
    int recon_active;
    int recon_started;

    /* 行级 LF 依赖: 每行所需的额外参考帧 LF 行数 (基于 MV y 范围).
     * mv_row_range[i] = 当前帧行 i 的 inter 块需要参考帧 LF 完成的额外行数.
     *   所需参考帧行 = min(i + mv_row_range[i], h_lcu-1).
     * Phase 1 完成后 (cu_grid 已填充) 计算, Phase 2 中按行等待使用.
     * 含插值滤波器余量 (8-tap: +4 像素, 约 1 LCU 行). */
    int mv_row_range[AVS2_MAX_H_LCU]; /* [h_lcu] */
    int mv_row_range_h_lcu;        /* 分配时的 h_lcu */

    /* 行级并行: per-row 进度 (volatile, 在 task_lock 保护下访问).
     * row_aec_done[i]: 第 i 行的 AEC 解码完成 (Pass 1)
     * row_recon_done[i]: 第 i 行的重建完成 (Pass 2 重建, 不含 LF)
     * row_lf_done[i]: 第 i 行的环路滤波完成 (Pass 2 LF)
     * 仅 thread_mode==ROW 时使用, 在帧解码开始时重置 */
    volatile int row_aec_done[AVS2_MAX_H_LCU];    /* [h_lcu] */
    volatile int row_recon_done[AVS2_MAX_H_LCU];  /* [h_lcu] */
    volatile int row_lf_done[AVS2_MAX_H_LCU];     /* [h_lcu] */
    int row_parallel_h_lcu;        /* 分配时的 h_lcu */
    int row_aec_completed;         /* Pass 1 全部完成 (所有行 AEC done) */
    int row_recon_completed;       /* Pass 2 全部完成 (所有行 LF done) */
    int row_recon_next;            /* Pass 2: 下一个待重建的行号 */
    volatile int row_lf_next;      /* Pass 2: 下一个待 LF 的行号 (spin-wait 无锁读取, 需 volatile) */
    volatile int row_lf_done_count;/* Pass 2: 已完成 LF 的行数 (spin-wait 无锁读取, 需 volatile) */
    int n_row_workers;             /* 此帧当前在 avs2_row_parallel_pass2 中的辅助 worker 数 */
    int p2_split_row;              /* P2 行分割点: owning worker 做 [0,split), helper 做 [split,h_lcu).
                                     * 0=不分割 (intra 帧或无 helper). inter 帧设为 h_lcu/2. */
    volatile int p2_parallel;      /* 1 = 帧正在并行 P2 中, 接受辅助 worker (task_lock 保护) */

    /* Phase 2a (inter-parallel): 原子行计数器, 无行依赖.
     * inter_recon_next: 下一个待处理的行 (atomic CAS/INC)
     * inter_recon_done: 已完成的行数 (atomic INC) */
    volatile int inter_recon_next;
    volatile int inter_recon_done;

    /* 预分配的 LF 临时帧 (SAO/ALF 复用, 避免每 LCU 行堆分配).
     * 尺寸与 fdec 一致 (同 stride/padding), 按序列分辨率变化时重分配.
     * SAO 和 ALF 串行执行, 共用同一临时帧. */
    avs2_frame *lf_tmp;
    int lf_tmp_width;              /* 分配时的宽度 (检测重分配) */
    int lf_tmp_height;             /* 分配时的高度 */
} avs2_frame_ctx;

/* The global decoder context (cast from public avs2_ctx). */
struct avs2_internal {
    /* settings */
    int n_threads;
    int max_frame_delay;
    int log_level;
    unsigned frame_size_limit;
    int strict_std_compliance;
    int skip_loop_filter;
    int thread_mode;  /* avs2_thread_mode: 0=frame, 1=row */
    int force_8bit;   /* 1 = 强制 8-bit 解码 (有损) */
    avs2_picture_alloc allocator;
    avs2_logger logger;

    avs2_cpu_flags cpu;

    /* sequence/picture headers */
    avs2_seq_header *seq;
    avs2_pic_header *pic;
    avs2_rps sps_rps[AVS2_MAX_RPS]; /* RPS table from sequence header */

    uint16_t aec_tab_ctx_mps[4 * 2048];
    uint16_t aec_tab_ctx_lps[4 * 2048];

    /* frame contexts */
    avs2_frame_ctx *fc;
    int n_fc;
    avs2_frame_ctx *cur_fc;  /* 当前 header 解析所用的 fc (替代 &c->fc[0]) */

    /* DPB (容量需 >= n_fc + 参考帧数, 帧级并行时 n_fc=n_threads) */
    avs2_frame *dpb[AVS2_MAX_FRAME_DELAY];
    int n_dpb;

    /* pending output queue (ring) */
    avs2_frame *out_queue[AVS2_MAX_FRAME_DELAY];
    int out_head, out_tail;
    avs2_mutex_t out_lock;
    int out_next_poc;       /* 下一个期望输出的 POC (display order) */
    int out_initialized;    /* out_next_poc 是否已初始化 */
    int i_tr_wrap_cnt;      /* COI 回绕计数 (对应 davs2 i_tr_wrap_cnt) */
    int i_prev_coi;         /* 上一个 COI (对应 davs2 i_prev_coi), 初始 -1 */
    int seq_logged;         /* 序列头日志是否已打印 (去重, 避免重复序列头刷屏) */

    /* 线程池 (帧级并行). n_threads=1 时不创建 worker, 走同步路径.
     * 线程分组: AEC 线程 (专做 Phase 1) + 重建线程 (专做 Phase 2 行级 pipeline).
     * n_aec_threads: AEC 线程数 (建议 1-2). 0 = 不分组 (回退到通用 worker).
     * n_recon_threads: 重建线程数 = n_threads - 1 - n_aec_threads.
     * aec_threads[] / recon_threads[]: 分组线程句柄. */
    avs2_thread_t *threads;       /* 通用 worker (n_aec_threads==0 时使用) */
    avs2_thread_t *aec_threads;   /* AEC 专用线程 */
    avs2_thread_t *recon_threads; /* 重建专用线程 */
    int n_threads_active;   /* 实际创建的 worker 线程数 (n_threads-1 或 0) */
    int n_aec_threads;      /* AEC 线程数 (0=不分组) */
    int n_recon_threads;    /* 重建线程数 */
    avs2_mutex_t task_lock;  /* 保护任务队列/fc 状态/DPB 完成状态 */
    avs2_cond_t task_cond;   /* 任务就绪信号 (AEC 线程等待) */
    avs2_cond_t done_cond;   /* 帧完成信号 (get_picture 等待) */
    avs2_cond_t recon_cond;  /* 行级 pipeline 信号 (AEC→重建, 唤醒等待新行的重建线程) */
    int task_queue[AVS2_MAX_FRAME_DELAY]; /* 待解码 fc 索引队列 (Phase 1) */
    int task_q_head, task_q_tail;
    int phase2_queue[AVS2_MAX_FRAME_DELAY]; /* Phase 2 就绪 fc 索引队列 */
    int phase2_q_head, phase2_q_tail;
    int n_pending;           /* 队列中+解码中的任务总数 */
    int shutdown;            /* 关闭信号 */
    int n_waiters_task;      /* 等待 task_cond 的 worker 数 (Win32 广播用) */
    int n_waiters_done;      /* 等待 done_cond 的线程数 (Win32 广播用) */
    int n_waiters_recon;     /* 等待 recon_cond 的重建线程数 (Win32 广播用) */
    int n_p2_active;         /* 当前正在执行 Phase 2 的 worker 数 (task_lock 保护) */
    int p2_cap;              /* Phase 2 并发上限 (防止 P1 饥饿) */

    /* 行级并行: 当前正在行级并行解码的 fc (NULL = 无行级任务).
     * worker 在帧任务队列为空时检查此指针, 参与行级重建.
     * per-frame n_row_workers 在 avs2_frame_ctx 中, owning worker 等待
     * fc->n_row_workers 归零, 确保辅助 worker 不会操作已被复用的 fc. */
    avs2_frame_ctx *row_task_fc;

    /* input buffer (accumulate Annex B data to find frames) */
    uint8_t *in_buf;
    int in_buf_sz, in_buf_cap;
    int64_t in_pts, in_dts;
    int flushing;

    /* derived sequence params */
    int bit_depth;
    int max_cu_size;
    int min_cu_size;
};

/* picture.c */
int  avs2_frame_alloc(avs2_frame *f, struct avs2_internal *c);
void avs2_frame_free(avs2_frame *f);
void avs2_frame_free_buffers(avs2_frame *f);
void avs2_frame_ref(avs2_frame *dst, avs2_frame *src);
avs2_frame *avs2_dpb_get_free(struct avs2_internal *c);
void avs2_dpb_clear(struct avs2_internal *c);
void avs2_pad_line_lcu(avs2_frame *f, int lcu_y, int lcu_size_log2);

/* header.c */
int avs2_parse_sequence_header(struct avs2_internal *c, avs2_bs *bs);
int avs2_parse_picture_header(struct avs2_internal *c, avs2_bs *bs, int start_code);
int avs2_parse_slice_header(struct avs2_internal *c, avs2_bs *bs, int start_code,
                            avs2_frame_ctx *fc);
void avs2_build_reference_list(struct avs2_internal *c, avs2_frame_ctx *fc);

/* stream.c */
int avs2_find_start_code(const uint8_t *data, int sz, int *sc_pos, int *sc_id);

/* 移除 AVS2 码流中的伪起始码防竞争比特 (对应 davs2 bs_dispose_pseudo_code).
 * 在 AEC 解码前对整个 ES unit 调用, 处理 slice 数据中的 00 00 02 序列.
 * dst 和 src 可指向同一缓冲区 (原地处理), 返回处理后的字节数. */
int avs2_dispose_pseudo_code(uint8_t *dst, const uint8_t *src, int i_src);

/* decode.c */
/* 解析一帧的 start code 序列 (sequence/picture/slice header), 由主线程调用.
 * n_threads=1 时同步调用 avs2_decode_frame_fc; n_threads>1 时提交任务到队列. */
int avs2_decode_frame(struct avs2_internal *c, const uint8_t *data, int sz);
/* 解码指定 fc 的帧 (LCU 解码循环, worker 线程调用). 使用 fc->pic_local 而非 c->pic. */
int avs2_decode_frame_fc(struct avs2_internal *c, avs2_frame_ctx *fc);
/* 2-pass 帧并行: Phase 1 (AEC) — 等待 n_aec_deps, 执行 AEC, 发送 aec_done 信号.
 * Phase 1 完成后由 worker 检查 n_deps: 若为 0 直接执行 Phase 2, 否则设 state=5 释放 worker. */
int avs2_decode_frame_fc_phase1(struct avs2_internal *c, avs2_frame_ctx *fc);
/* 2-pass 帧并行: Phase 2 (重建+LF) — 执行重建和环路滤波, 设置帧元数据.
 * 调用前需确保 n_deps==0 (所有参考帧完全完成). */
int avs2_decode_frame_fc_phase2(struct avs2_internal *c, avs2_frame_ctx *fc);
/* 2-pass 行级并行: Phase 2 (重建+LF, 行级并行). 重置 fc 行状态, 设置 row_task_fc
 * 唤醒辅助 worker, 调用 avs2_row_parallel_pass2 行级并行重建+LF, 等待辅助 worker 退出.
 * 调用前需确保 n_deps==0 (所有参考帧完全完成). */
int avs2_decode_frame_fc_phase2_row(struct avs2_internal *c, avs2_frame_ctx *fc);
/* 行级并行: worker 参与 Pass 2 重建+LF. is_helper=1 时为辅助 worker, 可在
 * 无任务且有待处理 P1 任务时提前退出. 返回 1=帧完成, 0=helper 提前退出. */
int avs2_row_parallel_pass2(struct avs2_internal *c, avs2_frame_ctx *fc, int is_helper);
/* LF helper: Phase 2 中并行执行 deblock + pad (流水化重建与 LF). */
void avs2_lf_helper(struct avs2_internal *c, avs2_frame_ctx *fc);
/* Inter-parallel P2 helper: 并行执行 inter 重建 (pass=3, 无行依赖).
 * Workers 通过 CAS 领取行, 执行 inter MC+residual, 设置 row_recon_done.
 * owning worker 和辅助 worker 均可调用. */
void avs2_inter_recon_helper(struct avs2_internal *c, avs2_frame_ctx *fc);

/* cu.c */
/* pass=0: AEC+重建 (单pass), pass=1: 仅AEC (行级Pass 1), pass=2: 仅重建 (行级Pass 2) */
int avs2_decode_lcu(avs2_frame_ctx *fc, struct avs2_internal *c, int lcu_x, int lcu_y, int pass);
/* 设置当前线程的系数 scratch 缓冲区 (行级并行 Pass 2 使用 TLS 避免竞争).
 * 传入 NULL 清除. y=[64*64], u/v=[32*32] int16_t. */
void avs2_set_thread_scratch(int16_t *y, int16_t *u, int16_t *v);

/* AEC (aec.c) */
typedef struct avs2_aec avs2_aec;
avs2_aec *avs2_aec_create(const uint16_t *aec_tab_ctx_mps, const uint16_t *aec_tab_ctx_lps);
void avs2_aec_destroy(avs2_aec *aec);
void avs2_aec_init_contexts(avs2_aec *aec, int slice_type);
int  avs2_aec_start_decoding(avs2_aec *aec, const uint8_t *buf, int sz, int bit_pos);
int  avs2_aec_decode_bin(avs2_aec *aec, void *ctx);
int  avs2_aec_decode_bin_eq_prob(avs2_aec *aec);
int  avs2_aec_decode_final(avs2_aec *aec);
unsigned avs2_aec_decode_ue(avs2_aec *aec, void *ctx);
int avs2_aec_decode_se(avs2_aec *aec, void *ctx);
int avs2_aec_get_bits_read(avs2_aec *aec);

/* DSP interfaces */

/* 帧内预测单模式函数指针类型 (src 为参考样本数组, dst 为预测输出)
 * src/dst 按 bytes_per_sample 寻址 (8-bit=uint8, 10-bit=uint16),
 * 内部实现统一按 uint16_t 处理 (8-bit 时由调用方扩展/打包) */
typedef void (*avs2_intra_pred_fn)(uint8_t *src, uint8_t *dst, int i_dst,
                                   int mode, int bsx, int bsy, int bit_depth);
/* 参考样本填充函数指针类型 (pTL 为左上角像素, pLcuEP 为 LCU 边界缓存, EP 为输出参考数组)
 * pTL/pLcuEP 按 bytes_per_sample 寻址, EP 为 uint16_t 输出 (内部按 bps 读取帧数据) */
typedef void (*avs2_fill_edge_fn)(const uint8_t *pTL, int i_TL,
                                  const uint8_t *pLcuEP, uint8_t *EP,
                                  uint32_t i_avai, int bsx, int bsy, int bps);

/* 去块滤波函数指针类型 (支持 8/10-bit).
 * src 指向边界 Q 侧首像素 (R0), stride 为元素步长 (8-bit 按 uint8, 10-bit 按 uint16).
 * flt_flag[2] 为每 4 像素段的滤波标志 (0=不滤, 非0=滤).
 * bit_depth 决定像素访问宽度, 实现据此选择 uint8/uint16 路径. */
typedef void (*deblock_luma_fn)(void *src, int stride, int alpha, int beta,
                                uint8_t *flt_flag, int bit_depth);
typedef void (*deblock_chroma_fn)(void *src_u, void *src_v, int stride, int alpha,
                                  int beta, uint8_t *flt_flag, int bit_depth);

typedef struct {
    void (*ipred_dc)(uint8_t *dst, ptrdiff_t stride, const int16_t *left, const int16_t *top,
                     int w, int h, int bit_depth);
    void (*ipred_plane)(uint8_t *dst, ptrdiff_t stride, const int16_t *left, const int16_t *top,
                        int w, int h, int bit_depth);
    void (*ipred_angular)(uint8_t *dst, ptrdiff_t stride, const int16_t *ref, int w, int h,
                          int mode, int bit_depth, int above_avail, int left_avail);
    /* 帧内预测: 33 种亮度模式函数指针表 (模式 0..32) */
    avs2_intra_pred_fn ipred_mode[NUM_INTRA_MODE];
    /* 参考样本填充: 4 种边界情况 (0=左上角, 1=左边界, 2=上边界, 3=内部) */
    avs2_fill_edge_fn fill_edge[4];
    void (*mc_luma)(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst, ptrdiff_t dstride,
                    int w, int h, int mx, int my, int bit_depth);
    void (*mc_chroma)(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst, ptrdiff_t dstride,
                      int w, int h, int mx, int my, int bit_depth);
    /* MC + 双向平均: dst[i] = (dst[i] + mc(src,mv)[i] + 1) >> 1.
     * 省去 pred2 中间缓冲, 减少内存带宽.
     * 参数与 mc_luma 相同. */
    void (*mc_luma_avg)(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst, ptrdiff_t dstride,
                        int w, int h, int mx, int my, int bit_depth);
    void (*mc_chroma_avg)(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst, ptrdiff_t dstride,
                          int w, int h, int mx, int my, int bit_depth);
    void (*itx[5])(int16_t *coeff, int w, int h, int bit_depth); /* 4,8,16,32,64 */
    /* 反量化: 对 n 个系数批量执行 (coeff[i]*scale+add)>>shift, 饱和到 int16.
     * coeff 为连续缓冲区, 零系数保持零. */
    void (*dequant_block)(int16_t *coeff, int n, int scale, int shift);
    /* 去块滤波 (从 davs2 deblock.cc 移植). dir: EDGE_VER=0 垂直边, EDGE_HOR=1 水平边.
     * src 指向边界 Q 侧首像素 (R0), stride 为元素步长 (8-bit 按 uint8, 10-bit 按 uint16).
     * flt_flag[2] 为每 4 像素段的滤波标志 (0=不滤, 非0=滤). */
    deblock_luma_fn   deblock_luma[2];
    deblock_chroma_fn deblock_chroma[2];
    /* SAO 边界偏移 (从 davs2 sao.cc 移植). 4 个方向:
     *   [0]=EO_0(水平), [1]=EO_45, [2]=EO_90(垂直), [3]=EO_135.
     * avail[8]={上,下,左,右,左上,右上,左下,右下} 邻域可用性.
     * offset[5] 为 EO 类偏移 (edge_type 0..4). stride 为元素步长
     * (byte_stride / bytes_per_sample). dst/src 按 bytes_per_sample 寻址. */
    void (*sao_eo[4])(uint8_t *dst, int dst_stride, const uint8_t *src, int src_stride,
                      int w, int h, int bit_depth, const int *avail, const int *offset);
    /* SAO 带偏移. offset[32] 为 32 个带的偏移. */
    void (*sao_bo)(uint8_t *dst, int dst_stride, const uint8_t *src, int src_stride,
                   int w, int h, int bit_depth, const int *offset);
    /* ALF (从 davs2 alf.cc 移植). 9 抽头 7x7 对称滤波器.
     * block[0]=主体滤波 (block1), block[1]=边界修正 (block2).
     * stride 为元素步长 (byte_stride / bytes_per_sample),
     * bit_depth 为样本位深. dst/src 按 bytes_per_sample 寻址. */
    void (*alf_block[2])(uint8_t *dst, const uint8_t *src, int stride,
                         int lcu_pix_x, int lcu_pix_y, int lcu_width, int lcu_height,
                         int *alf_coeff, int b_top_avail, int b_down_avail, int bit_depth);
    /* 残差叠加: dst[i] = clip(dst[i] + coeff[i], 0, (1<<bit_depth)-1).
     * dst 按 bytes_per_sample 寻址 (8-bit=uint8, 10-bit=uint16),
     * stride 为字节步长, coeff 为 int16 连续缓冲区 (w*h 个元素). */
    void (*recon_residual)(uint8_t *dst, ptrdiff_t stride, const int16_t *coeff,
                           int w, int h, int bit_depth);
    /* 双向预测平均: dst[i] = (dst[i] + pred2[i] + 1) >> 1.
     * dst 按 bytes_per_sample 寻址, dst_stride 为字节步长.
     * pred2 为 int16 连续缓冲, pred2_stride 为 int16 元素步长. */
    void (*bi_avg)(uint8_t *dst, ptrdiff_t dst_stride, const int16_t *pred2,
                   int pred2_stride, int w, int h, int bit_depth);
    /* 双向预测平均 (双源): dst[i] = (pred1[i] + pred2[i] + 1) >> 1.
     * 两个预测均来自栈缓冲 (L1), 避免帧缓冲 cache 往返.
     * dst 按 bytes_per_sample 寻址, dst_stride 为字节步长.
     * pred1/pred2 为 int16 连续缓冲, stride 为 int16 元素步长. */
    void (*bi_avg_2src)(uint8_t *dst, ptrdiff_t dst_stride,
                        const int16_t *pred1, int pred1_stride,
                        const int16_t *pred2, int pred2_stride,
                        int w, int h, int bit_depth);
    /* 块填充: dst[i] = fill_val.
     * dst 按 bytes_per_sample 寻址, dst_stride 为字节步长. */
    void (*fill_block)(uint8_t *dst, ptrdiff_t dst_stride, int w, int h,
                       int fill_val, int bit_depth);
} avs2_dsp;

extern avs2_dsp avs2_dsp_table;

/* 全局: 是否禁用 SIMD (测试用, 由 avs2_dsp_init 检查) */
extern int g_disable_simd;

/* DSP init */
void avs2_dsp_init(void);
void avs2_ipred_init(void);
void avs2_mc_init(void);
void avs2_itx_init(void);
void avs2_loopfilter_init(void);
void avs2_sao_init(void);
void avs2_alf_init(void);
void avs2_quant_init(void);

/* 帧内预测入口 (从 davs2 intra.cc 移植) */
/* 亮度帧内预测: c 为解码上下文, f 为当前帧, cu 为编码单元, x/y 为像素坐标, bsx/bsy 为块尺寸, predmode 为预测模式 */
void avs2_get_intra_pred(struct avs2_internal *c, avs2_frame *f, avs2_cu *cu,
                         int x, int y, int bsx, int bsy, int predmode);
/* 色度帧内预测: x_c/y_c 为色度像素坐标 */
void avs2_get_intra_pred_chroma(struct avs2_internal *c, avs2_frame *f, avs2_cu *cu,
                                int x_c, int y_c);
/* 帧内预测分发器 (对应 davs2 intra_pred 内联函数) */
void avs2_intra_pred_dispatch(uint8_t *src, uint8_t *dst, int i_dst,
                              int dir_mode, int bsy, int bsx, int i_avail, int bit_depth);

/* 检查 TOPRIGHT 4x4 块在 LCU Z-scan 重建顺序下是否已重建。
 * 供帧间 MVP / 空间直接模式复用 (对应 davs2 p_tab_TR_avail 查表)。 */
int avs2_check_topright_avail(int pix_x, int pix_y, int bsx, int lcu_level);

/* Loop filter application */
void avs2_lf_apply_lcu_row(avs2_frame_ctx *fc, struct avs2_internal *c, int lcu_y);
void avs2_save_intra_border(avs2_frame_ctx *fc, struct avs2_internal *c, int lcu_y);
void avs2_lf_apply_lcu_row_nosave(avs2_frame_ctx *fc, struct avs2_internal *c, int lcu_y);
void avs2_lf_tmp_free(avs2_frame_ctx *fc);  /* 释放预分配的 LF 临时帧 */

/* ---- 去块滤波 (loopfilter.c, 从 davs2 deblock.cc 移植) ---- */
/* 对一个 LCU 执行完整去块滤波 (设置边界标志 + 垂直/水平边滤波) */
void avs2_loop_filter(avs2_frame_ctx *fc, struct avs2_internal *c, avs2_frame *f,
                      int lcu_x, int lcu_y);

/* ---- SAO (sao.c, 从 davs2 sao.cc 移植) ---- */
/* 对一个块应用 SAO 滤波 (按 type_idc 分发到 EO 或 BO) */
void avs2_sao_on_block(avs2_frame *dst_frm, avs2_frame *src_frm, int pl,
                       int pix_x, int pix_y, int blk_w, int blk_h,
                       int bit_depth, const int *avail, const avs2_sao_param *sp);
/* 从 AEC 码流读取一个 LCU 的 SAO 参数 (对应 davs2 sao_read_lcu) */
void avs2_sao_read_param(avs2_frame_ctx *fc, struct avs2_internal *c,
                         int lcu_xy, const uint8_t *slice_sao_on,
                         avs2_sao_param *sao_param);

/* ---- ALF (alf.c, 从 davs2 alf.cc 移植) ---- */
/* 对一个 LCU 块应用 ALF 滤波 (亮度 + 色度, 两级滤波) */
void avs2_alf_on_block(avs2_frame *dst_frm, avs2_frame *src_frm,
                       const avs2_alf_param *ap, int lcu_x, int lcu_y,
                       int lcu_size, int img_w, int img_h, int bit_depth,
                       int b_top_avail, int b_down_avail);
/* 重建 ALF 第 9 个 (中心) 系数: coeff[8] = (1<<shift) - 2*sum(coeff[0..7]) + coeff[8] */
void avs2_alf_recon_last_coeff(int16_t *coeff);

/* ---- SAO/ALF C 分发函数 (供 SIMD 回退调用) ---- */
/* 8 位时直接处理, 10 位时转发到 uint16_t 实现。
 * SIMD 实现 (sao_simd.c/alf_simd.c) 在 bit_depth<=8 时调用这些函数回退。 */
void sao_eo_0_c(uint8_t *dst, int dst_stride, const uint8_t *src, int src_stride,
                int w, int h, int bit_depth, const int *avail, const int *offset);
void sao_eo_90_c(uint8_t *dst, int dst_stride, const uint8_t *src, int src_stride,
                 int w, int h, int bit_depth, const int *avail, const int *offset);
void sao_eo_135_c(uint8_t *dst, int dst_stride, const uint8_t *src, int src_stride,
                  int w, int h, int bit_depth, const int *avail, const int *offset);
void sao_eo_45_c(uint8_t *dst, int dst_stride, const uint8_t *src, int src_stride,
                 int w, int h, int bit_depth, const int *avail, const int *offset);
void sao_bo_c(uint8_t *dst, int dst_stride, const uint8_t *src, int src_stride,
              int w, int h, int bit_depth, const int *offset);
void alf_filter_block1_c(uint8_t *dst, const uint8_t *src, int stride,
                         int lcu_pix_x, int lcu_pix_y,
                         int lcu_width, int lcu_height,
                         int *alf_coeff, int b_top_avail, int b_down_avail,
                         int bit_depth);
void alf_filter_block2_c(uint8_t *dst, const uint8_t *src, int stride,
                         int lcu_pix_x, int lcu_pix_y,
                         int lcu_width, int lcu_height,
                         int *alf_coeff, int b_top_avail, int b_down_avail,
                         int bit_depth);

#endif /* AVS2DEC_SRC_INTERNAL_H */
