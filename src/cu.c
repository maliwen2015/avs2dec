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
 * __thread 是 GCC 扩展, MinGW 支持. MSVC 等价物为 __declspec(thread).
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

/* ===================================================================
 * 常量表
 * =================================================================== */

/* DMH (方向多假设) 模式偏移位置表 */
static const int8_t tab_dmh_pos[DMH_MODE_NUM + DMH_MODE_NUM - 1][2][2] = {
    { {  0,  0 }, { 0,  0 } },
    { { -1,  0 }, { 1,  0 } },
    { {  0, -1 }, { 0,  1 } },
    { { -1,  1 }, { 1, -1 } },
    { { -1, -1 }, { 1,  1 } },
    { { -2,  0 }, { 2,  0 } },
    { {  0, -2 }, { 0,  2 } },
    { { -2,  2 }, { 2, -2 } },
    { { -2, -2 }, { 2,  2 } }
};

/* 帧内预测模式到扫描类型映射 (VER=0, HOR=1, DC_DIAG=2) */
static const int tab_intra_mode_scan_type[NUM_INTRA_MODE] = {
    INTRA_PRED_DC_DIAG, INTRA_PRED_DC_DIAG, INTRA_PRED_DC_DIAG, INTRA_PRED_HOR, INTRA_PRED_HOR,  /* 0-4 */
    INTRA_PRED_DC_DIAG, INTRA_PRED_DC_DIAG, INTRA_PRED_DC_DIAG, INTRA_PRED_VER, INTRA_PRED_VER,  /* 5-9 */
    INTRA_PRED_VER, INTRA_PRED_VER, INTRA_PRED_VER, INTRA_PRED_VER, INTRA_PRED_VER, INTRA_PRED_VER,  /* 10-15 */
    INTRA_PRED_VER, INTRA_PRED_DC_DIAG,  /* 16-17 */
    INTRA_PRED_DC_DIAG, INTRA_PRED_DC_DIAG, INTRA_PRED_HOR, INTRA_PRED_HOR, INTRA_PRED_HOR,  /* 18-22 */
    INTRA_PRED_HOR, INTRA_PRED_HOR, INTRA_PRED_HOR, INTRA_PRED_HOR, INTRA_PRED_HOR, INTRA_PRED_HOR,  /* 23-28 */
    INTRA_PRED_DC_DIAG, INTRA_PRED_DC_DIAG, INTRA_PRED_DC_DIAG, INTRA_PRED_VER  /* 29-32 */
};

/* DSP 初始化标志 */
static int g_dsp_inited = 0;

/* 前向声明 */
static void decode_cu_recursive(avs2_frame_ctx *fc, struct avs2_internal *c,
                                avs2_aec *aec, int x, int y, int level,
                                int qp, int *prev_qp, int pass);
static void get_mvp_default(avs2_frame *f, avs2_cu *cu, int pix_x, int pix_y,
                            avs2_mv *pmv, int bwd_2nd, int ref_frame,
                            int bsx, int pu_type_for_mvp, int frame_type,
                            int lcu_level, int b_bkgnd_picture,
                            int num_of_references);


/* ===================================================================
 * MV 缩放辅助 (对应 davs2 predict.h)
 * =================================================================== */

/* MV 缩放 (对应 davs2 scale_mv_skip) */
static int16_t scale_mv_skip(int mv, int dist_dst, int dist_src)
{
    mv = (int16_t)((mv * dist_dst * dist_src + HALF_MULTI) >> OFFSET);
    if (mv < -32768) mv = -32768;
    if (mv > 32767)  mv = 32767;
    return (int16_t)mv;
}

/* MV 缩放 (对称模式, 对应 davs2 scale_mv_biskip) */
static int16_t scale_mv_biskip(int mv, int dist_dst, int dist_src)
{
    int sign = (mv > 0) - (mv < 0);
    mv = sign * ((dist_src * (1 + (mv < 0 ? -mv : mv) * dist_dst) - 1) >> OFFSET);
    if (mv < -32768) mv = -32768;
    if (mv > 32767)  mv = 32767;
    return (int16_t)mv;
}

/* 将 CU 的 MV 和参考索引存储到帧的 mvbuf/refbuf (对应 davs2 save_mv_ref_info)
 * mvbuf/refbuf 以 4x4 (SPU) 为单位, stride = w8 * 2 */
static void store_mv_to_buf(avs2_frame *f, avs2_cu *cu, int x, int y)
{
    int w_spu = f->w8 * 2;
    int pu_idx;

    for (pu_idx = 0; pu_idx < cu->num_pu; pu_idx++) {
        int pu_x = x + cu->pu_x[pu_idx];
        int pu_y = y + cu->pu_y[pu_idx];
        int pu_w = cu->pu_w[pu_idx];
        int pu_h = cu->pu_h[pu_idx];
        int spu_w = pu_w >> MIN_PU_SIZE_IN_BIT;
        int spu_h = pu_h >> MIN_PU_SIZE_IN_BIT;
        int spu_x = pu_x >> MIN_PU_SIZE_IN_BIT;
        int spu_y = pu_y >> MIN_PU_SIZE_IN_BIT;
        avs2_mv mv0 = cu->mv[pu_idx][0];
        int8_t ref0 = cu->i_ref[pu_idx][0];
        int i, j;

        for (j = 0; j < spu_h; j++) {
            for (i = 0; i < spu_w; i++) {
                int idx = (spu_y + j) * w_spu + (spu_x + i);
                if (idx >= 0 && idx < f->w8 * f->h8 * 4) {
                    /* 存储前向 MV (用于时域直接模式) */
                    f->mvbuf[idx] = mv0;
                    f->refbuf[idx] = (ref0 >= 0) ? (int8_t)ref0 : (int8_t)-1;
                }
            }
        }
    }
}

/* ---- 空间直接模式所需的邻居信息结构 (对应 davs2 neighbor_inter_t) ---- */
typedef struct {
    avs2_mv mv[2];          /* [0]=前向, [1]=后向 */
    int8_t  is_available;   /* 是否可用 */
    int8_t  i_dir_pred;     /* 预测方向 (PDIR_FWD/BWD/SYM/BID) */
    int8_t  ref[2];         /* [0]=前向参考, [1]=后向参考 */
} neighbor_inter_t;

/* 前向声明 (find_pu_index 定义在后面) */
static int find_pu_index(avs2_cu *cu, int rel_x, int rel_y);

/* 邻居块位置索引 (对应 davs2 neighbor_block_pos_e) */
enum {
    BLK_TOPLEFT  = 0,
    BLK_TOP      = 1,
    BLK_LEFT     = 2,
    BLK_TOPRIGHT = 3,
    BLK_TOP2     = 4,
    BLK_LEFT2    = 5,
    NUM_INTER_NEIGHBOR = 6
};

/* 从 cu_grid 获取一个 4x4 位置处的完整邻居信息 (对应 davs2 cu_get_neighbor_spatial) */
static void get_neighbor_inter(avs2_frame *f, int nx_pix, int ny_pix,
                               neighbor_inter_t *p_nb, int cur_slice_nr)
{
    int bx = nx_pix >> 3, by = ny_pix >> 3;
    avs2_cu *cu;
    int cu_size, cu_origin_x, cu_origin_y, rel_x, rel_y, pu_idx;

    p_nb->is_available = 0;
    p_nb->i_dir_pred = PDIR_INVALID;
    p_nb->ref[0] = INVALID_REF;
    p_nb->ref[1] = INVALID_REF;
    p_nb->mv[0].x = p_nb->mv[0].y = 0;
    p_nb->mv[1].x = p_nb->mv[1].y = 0;

    if (bx < 0 || bx >= f->w8 || by < 0 || by >= f->h8) return;

    cu = &f->cu_grid[by * f->w8 + bx];
    if (cu->i_slice_nr != cur_slice_nr) return;

    /* 位置可用 (同条带且在图像内). 帧内块: is_available=1 但 ref=INVALID_REF
     * (对应 davs2 cu_get_neighbor_spatial 的行为) */
    p_nb->is_available = 1;
    if (cu->b_intra) return;

    /* 计算 CU 原点和相对位置, 找到覆盖该位置的 PU */
    cu_size = 1 << cu->cu_level;
    cu_origin_x = (bx * 8) & ~(cu_size - 1);
    cu_origin_y = (by * 8) & ~(cu_size - 1);
    rel_x = bx * 8 - cu_origin_x;
    rel_y = by * 8 - cu_origin_y;
    pu_idx = find_pu_index(cu, rel_x, rel_y);

    p_nb->i_dir_pred = cu->b8pdir[pu_idx];
    p_nb->mv[0] = cu->mv[pu_idx][0];
    p_nb->mv[1] = cu->mv[pu_idx][1];
    p_nb->ref[0] = cu->i_ref[pu_idx][0];
    p_nb->ref[1] = cu->i_ref[pu_idx][1];
}

/* 收集 CU 的 6 个空间邻居信息 (对应 davs2 cu_get_neighbors 的空间部分) */
static void gather_neighbors(avs2_frame *f, avs2_cu *cu, int pix_x, int pix_y,
                             int bsx, int bsy, neighbor_inter_t *neighbors,
                             int lcu_level)
{
    int cur_slice_nr = cu->i_slice_nr;
    int x0 = pix_x >> MIN_PU_SIZE_IN_BIT;  /* SPU 坐标 */
    int y0 = pix_y >> MIN_PU_SIZE_IN_BIT;
    int x1 = (bsx >> MIN_PU_SIZE_IN_BIT) + x0 - 1;
    int y1 = (bsy >> MIN_PU_SIZE_IN_BIT) + y0 - 1;
    /* TOPRIGHT 在 LCU Z-scan 重建顺序下是否可用 (对应 davs2 block_available_TR) */
    int avail_tr = avs2_check_topright_avail(pix_x, pix_y, bsx, lcu_level);

    /* 转换为像素坐标 (8x8 单元左上角) */
    get_neighbor_inter(f, (x0 - 1) * 4, (y0    ) * 4, &neighbors[BLK_LEFT    ], cur_slice_nr);
    get_neighbor_inter(f, (x0    ) * 4, (y0 - 1) * 4, &neighbors[BLK_TOP     ], cur_slice_nr);
    get_neighbor_inter(f, (x1    ) * 4, (y0 - 1) * 4, &neighbors[BLK_TOP2    ], cur_slice_nr);
    get_neighbor_inter(f, (x0 - 1) * 4, (y0 - 1) * 4, &neighbors[BLK_TOPLEFT ], cur_slice_nr);
    get_neighbor_inter(f, (x0 - 1) * 4, (y1    ) * 4, &neighbors[BLK_LEFT2   ], cur_slice_nr);
    /* TOPRIGHT 不可用时传 -1 使其标记为不可用 (对应 davs2 avail_TR ? x1+1 : -1) */
    get_neighbor_inter(f, avail_tr ? (x1 + 1) * 4 : -4, (y0 - 1) * 4,
                       &neighbors[BLK_TOPRIGHT], cur_slice_nr);
}

/* B 帧空间直接模式 MV 推导 (对应 davs2 get_mv_bskip_spatial) */
static void get_mv_bskip_spatial(neighbor_inter_t *p_neighbors,
                                 avs2_mv *fw_pmv, avs2_mv *bw_pmv, int ds_mode)
{
    avs2_mv mv_1st[DS_MAX_NUM], mv_2nd[DS_MAX_NUM];
    int j;
    int bid_flag = 0, bw_flag = 0, fw_flag = 0, sym_flag = 0, bid2 = 0;

    for (j = 0; j < DS_MAX_NUM; j++) {
        mv_1st[j].x = mv_1st[j].y = 0;
        mv_2nd[j].x = mv_2nd[j].y = 0;
    }

    for (j = 0; j < 6; j++) {
        if (!p_neighbors[j].is_available) continue;

        if (p_neighbors[j].i_dir_pred == PDIR_BID) {
            mv_2nd[DS_B_BID] = p_neighbors[j].mv[1];
            mv_1st[DS_B_BID] = p_neighbors[j].mv[0];
            bid_flag++;
            if (bid_flag == 1) bid2 = j;
        } else if (p_neighbors[j].i_dir_pred == PDIR_SYM) {
            mv_2nd[DS_B_SYM] = p_neighbors[j].mv[1];
            mv_1st[DS_B_SYM] = p_neighbors[j].mv[0];
            sym_flag++;
        } else if (p_neighbors[j].i_dir_pred == PDIR_BWD) {
            mv_2nd[DS_B_BWD] = p_neighbors[j].mv[1];
            bw_flag++;
        } else if (p_neighbors[j].i_dir_pred == PDIR_FWD) {
            mv_1st[DS_B_FWD] = p_neighbors[j].mv[0];
            fw_flag++;
        }
    }

    /* 无 BID 邻居但有 FWD 和 BWD: 合成 BID */
    if (bid_flag == 0 && fw_flag != 0 && bw_flag != 0) {
        mv_2nd[DS_B_BID] = mv_2nd[DS_B_BWD];
        mv_1st[DS_B_BID] = mv_1st[DS_B_FWD];
    }

    /* 无 SYM 邻居: 从 BID 或 BWD/FWD 合成 */
    if (sym_flag == 0 && bid_flag > 1) {
        mv_2nd[DS_B_SYM] = p_neighbors[bid2].mv[1];
        mv_1st[DS_B_SYM] = p_neighbors[bid2].mv[0];
    } else if (sym_flag == 0 && bw_flag != 0) {
        mv_2nd[DS_B_SYM] = mv_2nd[DS_B_BWD];
        mv_1st[DS_B_SYM].x = (int16_t)(-mv_2nd[DS_B_BWD].x);
        mv_1st[DS_B_SYM].y = (int16_t)(-mv_2nd[DS_B_BWD].y);
    } else if (sym_flag == 0 && fw_flag != 0) {
        mv_2nd[DS_B_SYM].x = (int16_t)(-mv_1st[DS_B_FWD].x);
        mv_2nd[DS_B_SYM].y = (int16_t)(-mv_1st[DS_B_FWD].y);
        mv_1st[DS_B_SYM] = mv_1st[DS_B_FWD];
    }

    /* 无 BWD 邻居: 从 BID 合成 */
    if (bw_flag == 0 && bid_flag > 1) {
        mv_2nd[DS_B_BWD] = p_neighbors[bid2].mv[1];
    } else if (bw_flag == 0 && bid_flag != 0) {
        mv_2nd[DS_B_BWD] = mv_2nd[DS_B_BID];
    }

    /* 无 FWD 邻居: 从 BID 合成 */
    if (fw_flag == 0 && bid_flag > 1) {
        mv_1st[DS_B_FWD] = p_neighbors[bid2].mv[0];
    } else if (fw_flag == 0 && bid_flag != 0) {
        mv_1st[DS_B_FWD] = mv_1st[DS_B_BID];
    }

    *fw_pmv = mv_1st[ds_mode];
    *bw_pmv = mv_2nd[ds_mode];
}

/* P/F 帧空间直接模式 MV 推导 (对应 davs2 get_mv_fskip_spatial) */
static void get_mv_fskip_spatial(neighbor_inter_t *p_neighbors,
                                 avs2_mv *mv_1st_out, avs2_mv *mv_2nd_out,
                                 int8_t *ref_1st_out, int8_t *ref_2nd_out,
                                 int ds_mode)
{
    avs2_mv mv_tskip_1st[DS_MAX_NUM], mv_tskip_2nd[DS_MAX_NUM];
    int8_t  ref_skip_1st[DS_MAX_NUM], ref_skip_2nd[DS_MAX_NUM];
    int j;
    int bid_flag = 0, fw_flag = 0, bid2 = 0, fw2 = 0;

    for (j = 0; j < DS_MAX_NUM; j++) {
        mv_tskip_1st[j].x = mv_tskip_1st[j].y = 0;
        mv_tskip_2nd[j].x = mv_tskip_2nd[j].y = 0;
        ref_skip_1st[j] = 0;
        ref_skip_2nd[j] = INVALID_REF;
    }

    for (j = 0; j < 6; j++) {
        if (!p_neighbors[j].is_available) continue;

        if (p_neighbors[j].ref[0] != INVALID_REF && p_neighbors[j].ref[1] != INVALID_REF) {
            /* bid: 双向参考 */
            ref_skip_1st[DS_DUAL_1ST] = p_neighbors[j].ref[0];
            ref_skip_2nd[DS_DUAL_1ST] = p_neighbors[j].ref[1];
            mv_tskip_1st[DS_DUAL_1ST] = p_neighbors[j].mv[0];
            mv_tskip_2nd[DS_DUAL_1ST] = p_neighbors[j].mv[1];
            bid_flag++;
            if (bid_flag == 1) bid2 = j;
        } else if (p_neighbors[j].ref[0] != INVALID_REF && p_neighbors[j].ref[1] == INVALID_REF) {
            /* fw: 仅前向参考 */
            ref_skip_1st[DS_SINGLE_1ST] = p_neighbors[j].ref[0];
            mv_tskip_1st[DS_SINGLE_1ST] = p_neighbors[j].mv[0];
            fw_flag++;
            if (fw_flag == 1) fw2 = j;
        }
    }

    /* 无 BID 且有多个 FW: 用两个 FW 合成 DUAL_1ST */
    if (bid_flag == 0 && fw_flag > 1) {
        ref_skip_1st[DS_DUAL_1ST] = ref_skip_1st[DS_SINGLE_1ST];
        ref_skip_2nd[DS_DUAL_1ST] = p_neighbors[fw2].ref[0];
        mv_tskip_1st[DS_DUAL_1ST] = mv_tskip_1st[DS_SINGLE_1ST];
        mv_tskip_2nd[DS_DUAL_1ST] = p_neighbors[fw2].mv[0];
    }

    /* DUAL_2ND */
    if (bid_flag > 1) {
        ref_skip_1st[DS_DUAL_2ND] = p_neighbors[bid2].ref[0];
        ref_skip_2nd[DS_DUAL_2ND] = p_neighbors[bid2].ref[1];
        mv_tskip_1st[DS_DUAL_2ND] = p_neighbors[bid2].mv[0];
        mv_tskip_2nd[DS_DUAL_2ND] = p_neighbors[bid2].mv[1];
    } else if (bid_flag == 1 && fw_flag > 1) {
        ref_skip_1st[DS_DUAL_2ND] = ref_skip_1st[DS_SINGLE_1ST];
        ref_skip_2nd[DS_DUAL_2ND] = p_neighbors[fw2].ref[0];
        mv_tskip_1st[DS_DUAL_2ND] = mv_tskip_1st[DS_SINGLE_1ST];
        mv_tskip_2nd[DS_DUAL_2ND] = p_neighbors[fw2].mv[0];
    }

    /* SINGLE_1ST (前向) */
    ref_skip_2nd[DS_SINGLE_1ST] = INVALID_REF;
    mv_tskip_2nd[DS_SINGLE_1ST].x = mv_tskip_2nd[DS_SINGLE_1ST].y = 0;
    if (fw_flag == 0 && bid_flag > 1) {
        ref_skip_1st[DS_SINGLE_1ST] = p_neighbors[bid2].ref[0];
        mv_tskip_1st[DS_SINGLE_1ST] = p_neighbors[bid2].mv[0];
    } else if (fw_flag == 0 && bid_flag == 1) {
        ref_skip_1st[DS_SINGLE_1ST] = ref_skip_1st[DS_DUAL_1ST];
        mv_tskip_1st[DS_SINGLE_1ST] = mv_tskip_1st[DS_DUAL_1ST];
    }

    /* SINGLE_2ND (前向第二选择) */
    ref_skip_2nd[DS_SINGLE_2ND] = INVALID_REF;
    mv_tskip_2nd[DS_SINGLE_2ND].x = mv_tskip_2nd[DS_SINGLE_2ND].y = 0;
    if (fw_flag > 1) {
        ref_skip_1st[DS_SINGLE_2ND] = p_neighbors[fw2].ref[0];
        mv_tskip_1st[DS_SINGLE_2ND] = p_neighbors[fw2].mv[0];
    } else if (bid_flag > 1) {
        ref_skip_1st[DS_SINGLE_2ND] = p_neighbors[bid2].ref[1];
        mv_tskip_1st[DS_SINGLE_2ND] = p_neighbors[bid2].mv[1];
    } else if (bid_flag == 1) {
        ref_skip_1st[DS_SINGLE_2ND] = ref_skip_2nd[DS_DUAL_1ST];
        mv_tskip_1st[DS_SINGLE_2ND] = mv_tskip_2nd[DS_DUAL_1ST];
    }

    *mv_1st_out  = mv_tskip_1st[ds_mode];
    *mv_2nd_out  = mv_tskip_2nd[ds_mode];
    *ref_1st_out = ref_skip_1st[ds_mode];
    *ref_2nd_out = ref_skip_2nd[ds_mode];
}

/* 将 SPU 坐标映射到 mvbuf 子采样位置 (对应 davs2 save_mv_ref_info 的子采样逻辑)
 * davs2 在 save_mv_ref_info 中对 mvbuf 做 16x16 子采样:
 * 每个 16x16 像素区域 (4x4 SPU) 取中心位置 (offset+2) 的 MV,
 * 边界处取 (group_start + boundary) / 2. */
static inline int mvbuf_subsample_idx(int spu_x, int spu_y, int w_spu, int h_spu)
{
    int sub_x, sub_y;
    /* x: round down to 16x16 group, then offset +2 (center) */
    sub_x = ((spu_x >> MV_FACTOR_IN_BIT) << MV_FACTOR_IN_BIT) + 2;
    if (sub_x >= w_spu) {
        sub_x = (((spu_x >> MV_FACTOR_IN_BIT) << MV_FACTOR_IN_BIT) + w_spu) >> 1;
    }
    sub_y = ((spu_y >> MV_FACTOR_IN_BIT) << MV_FACTOR_IN_BIT) + 2;
    if (sub_y >= h_spu) {
        sub_y = (((spu_y >> MV_FACTOR_IN_BIT) << MV_FACTOR_IN_BIT) + h_spu) >> 1;
    }
    return sub_y * w_spu + sub_x;
}

/* Skip 模式 MV 推导 (对应 davs2 fill_mv_and_ref_for_skip) */
static void derive_skip_mv(avs2_frame_ctx *fc, avs2_cu *cu, int x, int y,
                           int b_bkgnd_picture)
{
    avs2_frame *f = fc->fdec;
    int frame_type = fc->slice_type;
    int ds_mode = cu->i_skip_mode;
    int w_spu = f->w8 * 2;
    int h_spu = f->h8 * 2;
    int num_refs = fc->n_refs;
    int i;

    if (frame_type == AVS2_B_SLICE && ds_mode == DS_NONE) {
        /* B 帧时域直接 (Symmetric): 从后向参考帧的同位置块获取 MV */
        avs2_frame *col_pic = (fc->n_refs > 0) ? fc->fref[B_BWD] : NULL;

        for (i = 0; i < cu->num_pu; i++) {
            /* davs2 用 PU 左上角 SPU 坐标访问 mvbuf (fill_mv_bskip 中 i8/j8) */
            int pu_x = x + cu->pu_x[i];
            int pu_y = y + cu->pu_y[i];
            int spu_x = pu_x >> MIN_PU_SIZE_IN_BIT;
            int spu_y = pu_y >> MIN_PU_SIZE_IN_BIT;
            avs2_mv tmv = {0, 0};
            int refframe = -1;

            if (spu_x < 0) spu_x = 0;
            if (spu_y < 0) spu_y = 0;
            if (spu_x >= w_spu) spu_x = w_spu - 1;
            if (spu_y >= h_spu) spu_y = h_spu - 1;

            if (col_pic && col_pic->mvbuf) {
                /* davs2 mvbuf 做 16x16 子采样, 取中心位置 MV.
                 * 注意: 使用 col_pic 自身的尺寸计算索引, 防止序列内
                 * 分辨率切换时当前帧尺寸大于参考帧导致越界读. */
                int col_w_spu = col_pic->w8 * 2;
                int col_h_spu = col_pic->h8 * 2;
                int cspu_x = spu_x, cspu_y = spu_y;
                if (cspu_x >= col_w_spu) cspu_x = col_w_spu - 1;
                if (cspu_y >= col_h_spu) cspu_y = col_h_spu - 1;
                if (cspu_x < 0) cspu_x = 0;
                if (cspu_y < 0) cspu_y = 0;
                int col_idx = mvbuf_subsample_idx(cspu_x, cspu_y, col_w_spu, col_h_spu);
                /* 行级 AEC 依赖: 等待 col_pic 对应行的 AEC (含 store_mv_to_buf) 完成.
                 * col_idx 对应 SPU y = col_idx / col_w_spu, LCU 行 = SPU y >> (lcu_log2 - spu_log2) */
                int accessed_spu_y = col_idx / col_w_spu;
                int required_row = accessed_spu_y >> (fc->lcu_size - MIN_PU_SIZE_IN_BIT);
                if (required_row < 0) required_row = 0;
                if (required_row >= col_pic->h_lcu) required_row = col_pic->h_lcu - 1;
                while (!avs2_atomic_load(&col_pic->aec_row_done[required_row])) {
                    avs2_cpu_relax();
                }
                tmv = col_pic->mvbuf[col_idx];
                refframe = col_pic->refbuf[col_idx];
            }

            if (refframe >= 0 && col_pic) {
                /* 时域直接: 缩放 colocated MV */
                int iTRp_src = col_pic->dist_scale_refs[refframe];
                int iTRb = f->dist_refs[B_FWD];  /* 当前帧到前向参考的距离 */
                int iTRd = f->dist_refs[B_BWD];  /* 当前帧到后向参考的距离 */

                if (iTRp_src > 0 && iTRb > 0 && iTRd > 0) {
                    cu->mv[i][0].x = scale_mv_biskip(tmv.x, iTRb, iTRp_src);
                    cu->mv[i][0].y = scale_mv_biskip(tmv.y, iTRb, iTRp_src);
                    cu->mv[i][1].x = (int16_t)(-scale_mv_biskip(tmv.x, iTRd, iTRp_src));
                    cu->mv[i][1].y = (int16_t)(-scale_mv_biskip(tmv.y, iTRd, iTRp_src));
                } else {
                    cu->mv[i][0].x = cu->mv[i][0].y = 0;
                    cu->mv[i][1].x = cu->mv[i][1].y = 0;
                }
            } else {
                /* 同位置块不可用: 使用空间 MVP (对应 davs2 fill_mv_bskip
                 * refframe==-1 时回退到 get_mvp_default) */
                avs2_mv mvp_fwd, mvp_bwd;
                int cu_size = 1 << cu->cu_level;
                get_mvp_default(f, cu, x, y, &mvp_fwd, 0, 0, cu_size, 0,
                                frame_type, fc->lcu_size, b_bkgnd_picture, num_refs);
                get_mvp_default(f, cu, x, y, &mvp_bwd, 1, 0, cu_size, 0,
                                frame_type, fc->lcu_size, b_bkgnd_picture, num_refs);
                cu->mv[i][0] = mvp_fwd;
                cu->mv[i][1] = mvp_bwd;
            }

            /* 参考帧: B 帧 DS_NONE = Symmetric, 前向=fref[1], 后向=fref[0] */
            cu->i_ref[i][0] = B_FWD;
            cu->i_ref[i][1] = B_BWD;
            cu->b8pdir[i] = PDIR_SYM;
        }
    } else if ((frame_type == AVS2_P_SLICE || frame_type == AVS2_F_SLICE) && ds_mode == 0) {
        /* P/F 帧时域直接: 从前向参考帧的同位置块获取 MV */
        avs2_frame *col_pic = (fc->n_refs > 0) ? fc->fref[0] : NULL;

        for (i = 0; i < cu->num_pu; i++) {
            /* davs2 用 PU 左上角 SPU 坐标访问 mvbuf (fill_mv_pf_skip_temporal 中 block_x/block_y) */
            int pu_x = x + cu->pu_x[i];
            int pu_y = y + cu->pu_y[i];
            int spu_x = pu_x >> MIN_PU_SIZE_IN_BIT;
            int spu_y = pu_y >> MIN_PU_SIZE_IN_BIT;
            avs2_mv tmv = {0, 0};
            int refframe = -1;

            if (spu_x < 0) spu_x = 0;
            if (spu_y < 0) spu_y = 0;
            if (spu_x >= w_spu) spu_x = w_spu - 1;
            if (spu_y >= h_spu) spu_y = h_spu - 1;

            if (col_pic && col_pic->mvbuf) {
                /* davs2 mvbuf 做 16x16 子采样, 取中心位置 MV.
                 * 注意: 使用 col_pic 自身的尺寸计算索引, 防止序列内
                 * 分辨率切换时当前帧尺寸大于参考帧导致越界读. */
                int col_w_spu = col_pic->w8 * 2;
                int col_h_spu = col_pic->h8 * 2;
                int cspu_x = spu_x, cspu_y = spu_y;
                if (cspu_x >= col_w_spu) cspu_x = col_w_spu - 1;
                if (cspu_y >= col_h_spu) cspu_y = col_h_spu - 1;
                if (cspu_x < 0) cspu_x = 0;
                if (cspu_y < 0) cspu_y = 0;
                int col_idx = mvbuf_subsample_idx(cspu_x, cspu_y, col_w_spu, col_h_spu);
                /* 行级 AEC 依赖: 等待 col_pic 对应行的 AEC (含 store_mv_to_buf) 完成.
                 * col_idx 对应 SPU y = col_idx / col_w_spu, LCU 行 = SPU y >> (lcu_log2 - spu_log2) */
                int accessed_spu_y = col_idx / col_w_spu;
                int required_row = accessed_spu_y >> (fc->lcu_size - MIN_PU_SIZE_IN_BIT);
                if (required_row < 0) required_row = 0;
                if (required_row >= col_pic->h_lcu) required_row = col_pic->h_lcu - 1;
                while (!avs2_atomic_load(&col_pic->aec_row_done[required_row])) {
                    avs2_cpu_relax();
                }
                tmv = col_pic->mvbuf[col_idx];
                refframe = col_pic->refbuf[col_idx];
            }

            if (refframe >= 0 && col_pic) {
                /* 时域直接: 缩放 colocated MV (davs2 get_mv_pf_skip_temporal)
                 * davs2: col_dist = dist_scale_refs[refframe] (= MULTI/dist_refs)
                 *        scale_mv_skip(tmv, cur_dist, col_dist) */
                int cur_dist = f->dist_refs[0];
                int col_dist = col_pic->dist_scale_refs[refframe];

                if (col_dist > 0 && cur_dist > 0) {
                    cu->mv[i][0].x = scale_mv_skip(tmv.x, cur_dist, col_dist);
                    cu->mv[i][0].y = scale_mv_skip(tmv.y, cur_dist, col_dist);
                } else {
                    cu->mv[i][0].x = cu->mv[i][0].y = 0;
                }
            } else {
                cu->mv[i][0].x = cu->mv[i][0].y = 0;
            }

            cu->mv[i][1].x = cu->mv[i][1].y = 0;
            cu->i_ref[i][0] = 0;
            cu->i_ref[i][1] = INVALID_REF;
            cu->b8pdir[i] = PDIR_FWD;
        }
    } else {
        /* 空间直接模式: 从 6 个空间邻居推导 MV */
        int cu_size = 1 << cu->cu_level;
        neighbor_inter_t neighbors[NUM_INTER_NEIGHBOR];
        avs2_mv mv_1st, mv_2nd;
        int8_t  ref_1st, ref_2nd;
        int8_t  pdir;
        int i;

        gather_neighbors(f, cu, x, y, cu_size, cu_size, neighbors, fc->lcu_size);

        if (frame_type == AVS2_B_SLICE) {
            /* B 帧空间直接: DS_B_BID / DS_B_BWD / DS_B_SYM / DS_B_FWD */
            get_mv_bskip_spatial(neighbors, &mv_1st, &mv_2nd, ds_mode);

            /* 参考帧和预测方向由 ds_mode 决定 (对应 davs2 fill_mv_bskip + tab_pdir_bskip) */
            switch (ds_mode) {
            case DS_B_BID:
                ref_1st = B_FWD;
                ref_2nd = B_BWD;
                pdir = PDIR_BID;
                break;
            case DS_B_BWD:
                ref_1st = INVALID_REF;
                ref_2nd = B_BWD;
                pdir = PDIR_BWD;
                break;
            case DS_B_SYM:
                ref_1st = B_FWD;
                ref_2nd = B_BWD;
                pdir = PDIR_SYM;
                break;
            case DS_B_FWD:
            default:
                ref_1st = B_FWD;
                ref_2nd = INVALID_REF;
                pdir = PDIR_FWD;
                break;
            }
        } else {
            /* P/F 帧空间直接: DS_DUAL_1ST / DS_DUAL_2ND / DS_SINGLE_1ST / DS_SINGLE_2ND */
            get_mv_fskip_spatial(neighbors, &mv_1st, &mv_2nd, &ref_1st, &ref_2nd, ds_mode);
            if (ref_2nd != INVALID_REF) {
                pdir = PDIR_DUAL;
            } else {
                pdir = PDIR_FWD;
            }
        }

        /* 填充所有 PU (skip 模式所有 PU 使用相同 MV) */
        for (i = 0; i < cu->num_pu; i++) {
            cu->mv[i][0] = mv_1st;
            cu->mv[i][1] = mv_2nd;
            cu->i_ref[i][0] = ref_1st;
            cu->i_ref[i][1] = ref_2nd;
            cu->b8pdir[i] = pdir;
        }
    }
}


/* ===================================================================
 * 辅助函数: PU/TU 初始化 (对应 davs2 cu_init_prediction_units / cu_init_transform_units)
 * =================================================================== */

/* 获取预测单元的位置和尺寸 (对应 davs2 CODING_BLOCK_INFO 表) */
static void init_prediction_units(avs2_cu *cu, int frame_type, int slice_type)
{
    /* 每种预测模式的 PU 数量 */
    static const int num_prediction_unit[MAX_PRED_MODES] = {
        1, 1, 2, 2, 2, 2, 2, 2, 1, 4, 4, 4
    };

    /* PU 位置和尺寸表 [模式][块] = {x, y, w, h} (以 8x8 为基准, 后续移位) */
    static const int8_t coding_block_info[MAX_PRED_MODES + 1][4][4] = {
        /* 0: PRED_SKIP    */ { {0,0,8,8}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0} },
        /* 1: PRED_2Nx2N   */ { {0,0,8,8}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0} },
        /* 2: PRED_2NxN    */ { {0,0,8,4}, {0,4,8,4}, {0,0,0,0}, {0,0,0,0} },
        /* 3: PRED_Nx2N    */ { {0,0,4,8}, {4,0,4,8}, {0,0,0,0}, {0,0,0,0} },
        /* 4: PRED_2NxnU   */ { {0,0,8,2}, {0,2,8,6}, {0,0,0,0}, {0,0,0,0} },
        /* 5: PRED_2NxnD   */ { {0,0,8,6}, {0,6,8,2}, {0,0,0,0}, {0,0,0,0} },
        /* 6: PRED_nLx2N   */ { {0,0,2,8}, {2,0,6,8}, {0,0,0,0}, {0,0,0,0} },
        /* 7: PRED_nRx2N   */ { {0,0,6,8}, {6,0,2,8}, {0,0,0,0}, {0,0,0,0} },
        /* 8: PRED_I_2Nx2N */ { {0,0,8,8}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0} },
        /* 9: PRED_I_NxN   */ { {0,0,4,4}, {4,0,4,4}, {0,4,4,4}, {4,4,4,4} },
        /*10: PRED_I_2Nxn  */ { {0,0,8,2}, {0,2,8,2}, {0,4,8,2}, {0,6,8,2} },
        /*11: PRED_I_nx2N  */ { {0,0,2,8}, {2,0,2,8}, {4,0,2,8}, {6,0,2,8} },
        /* X: 默认 4x4     */ { {0,0,4,4}, {4,0,4,4}, {0,4,4,4}, {4,4,4,4} }
    };

    int level = cu->cu_level;
    int mode = cu->cu_type;
    int shift_bits = level - MIN_CU_SIZE_IN_BIT;
    int block_num;
    int ds_mode = cu->i_skip_mode;
    int i;

    if (mode == PRED_SKIP) {
        /* 特殊 Skip/Direct 模式: CU 超过 8x8 时划分成 4 个 PU */
        if (level > 3 &&
            (frame_type == AVS2_P_SLICE ||
             (frame_type == AVS2_F_SLICE && ds_mode == DS_NONE) ||
             (frame_type == AVS2_B_SLICE && ds_mode == DS_NONE))) {
            cu->num_pu = 4;
            for (i = 0; i < 4; i++) {
                const int8_t *info = coding_block_info[PRED_I_nx2N + 1][i];
                cu->pu_x[i] = info[0] << shift_bits;
                cu->pu_y[i] = info[1] << shift_bits;
                cu->pu_w[i] = info[2] << shift_bits;
                cu->pu_h[i] = info[3] << shift_bits;
            }
        } else {
            cu->num_pu = 1;
            const int8_t *info = coding_block_info[PRED_SKIP][0];
            cu->pu_x[0] = info[0] << shift_bits;
            cu->pu_y[0] = info[1] << shift_bits;
            cu->pu_w[0] = info[2] << shift_bits;
            cu->pu_h[0] = info[3] << shift_bits;
        }
    } else {
        block_num = num_prediction_unit[mode];
        cu->num_pu = (int8_t)block_num;
        for (i = 0; i < block_num; i++) {
            const int8_t *info = coding_block_info[mode][i];
            cu->pu_x[i] = info[0] << shift_bits;
            cu->pu_y[i] = info[1] << shift_bits;
            cu->pu_w[i] = info[2] << shift_bits;
            cu->pu_h[i] = info[3] << shift_bits;
        }
    }
}

/* 获取变换单元的位置和尺寸 (对应 davs2 cu_init_transform_units) */
static void init_transform_units(avs2_cu *cu, int8_t tu_x[4], int8_t tu_y[4],
                                 int8_t tu_w[4], int8_t tu_h[4])
{
    /* TU 分割信息表 [模式][块] = {x, y, w, h} (以 8x8 为基准) */
    static const int8_t tu_split_info[4][4][4] = {
        /* TU_SPLIT_NON   */ { {0,0,8,8}, {0,0,0,0}, {0,0,0,0}, {0,0,0,0} },
        /* TU_SPLIT_HOR   */ { {0,0,8,2}, {0,2,8,2}, {0,4,8,2}, {0,6,8,2} },
        /* TU_SPLIT_VER   */ { {0,0,2,8}, {2,0,2,8}, {4,0,2,8}, {6,0,2,8} },
        /* TU_SPLIT_CROSS */ { {0,0,4,4}, {4,0,4,4}, {0,4,4,4}, {4,4,4,4} }
    };

    int shift_bits = cu->cu_level - MIN_CU_SIZE_IN_BIT;
    int tu_type = cu->i_tu_split;
    int i;

    for (i = 0; i < 4; i++) {
        const int8_t *info = tu_split_info[tu_type][i];
        tu_x[i] = info[0] << shift_bits;
        tu_y[i] = info[1] << shift_bits;
        tu_w[i] = info[2] << shift_bits;
        tu_h[i] = info[3] << shift_bits;
    }
}


/* ===================================================================
 * AEC 描述符同步 (avs2_cu <-> aec_cu_t)
 * =================================================================== */

/* 将 avs2_cu 的字段填充到 aec_cu_t (供 AEC 函数使用) */
static void fill_aec_cu_desc(aec_cu_t *desc, avs2_cu *cu)
{
    int i;
    desc->i_cu_type = cu->cu_type;
    desc->i_cu_level = cu->cu_level;
    desc->i_trans_size = cu->i_tu_split;
    for (i = 0; i < 4; i++) {
        desc->b8pdir[i] = cu->b8pdir[i];
        desc->ref_idx[i].r[0] = cu->i_ref[i][0];
        desc->ref_idx[i].r[1] = cu->i_ref[i][1];
    }
    desc->i_cbp = cu->i_cbp;
    desc->i_qp = cu->qp;
    desc->i_weighted_skipmode = cu->i_weighted_skipmode;
    desc->i_md_directskip_mode = cu->i_skip_mode;
}

/* 将 aec_cu_t 的字段同步回 avs2_cu */
static void sync_aec_cu_desc(avs2_cu *cu, aec_cu_t *desc)
{
    int i;
    cu->cu_type = desc->i_cu_type;
    cu->i_tu_split = desc->i_trans_size;
    for (i = 0; i < 4; i++) {
        cu->b8pdir[i] = desc->b8pdir[i];
        cu->i_ref[i][0] = desc->ref_idx[i].r[0];
        cu->i_ref[i][1] = desc->ref_idx[i].r[1];
    }
    cu->i_cbp = desc->i_cbp;
    cu->i_weighted_skipmode = desc->i_weighted_skipmode;
    cu->i_skip_mode = desc->i_md_directskip_mode;
}


/* ===================================================================
 * 帧内预测模式解码 (对应 davs2 cu_read_intrapred_mode_luma)
 * =================================================================== */

/* 读取亮度帧内预测模式 (5-bit FL + MPM 推导)
 * 从帧的 ipredmode 数组获取邻域模式, 解码后存储回去供后续块使用。
 * \param aec     AEC 解码器
 * \param cu      当前 CU
 * \param b8      块索引 (0..3)
 * \param f       当前解码帧 (含 ipredmode 数组)
 * \param x_4x4   当前块在 4x4 单位下的 x 坐标
 * \param y_4x4   当前块在 4x4 单位下的 y 坐标
 * \return 0 成功, -1 失败
 */
static int read_intra_pred_mode_luma(avs2_aec *aec, avs2_cu *cu, int b8,
                                     avs2_frame *f, int x_4x4, int y_4x4)
{
    int8_t *p_ipredmode = f->ipredmode;
    int stride = f->ipredmode_stride;
    int8_t *p_intramode = p_ipredmode + y_4x4 * stride + x_4x4;
    int intra_mode_top  = p_intramode[-stride];
    int intra_mode_left = p_intramode[-1];
    int luma_mode = aec_read_intra_pmode(aec);
    int mpm[2];
    int8_t real_luma_mode;

    /* MPM (Most Probable Mode) 推导 */
    mpm[0] = (intra_mode_top < intra_mode_left) ? intra_mode_top : intra_mode_left;
    mpm[1] = (intra_mode_top > intra_mode_left) ? intra_mode_top : intra_mode_left;

    if (mpm[0] == mpm[1]) {
        mpm[0] = DC_PRED;
        mpm[1] = (mpm[1] == DC_PRED) ? BI_PRED : mpm[1];
    }

    /* 将码流中的模式值映射到实际模式 */
    if (luma_mode < 0) {
        real_luma_mode = (int8_t)mpm[luma_mode + 2];
    } else {
        real_luma_mode = (int8_t)(luma_mode + (luma_mode >= mpm[0]) + (luma_mode + 1 >= mpm[1]));
    }

    /* 范围检查 */
    if (real_luma_mode < 0 || real_luma_mode >= NUM_INTRA_MODE) {
        real_luma_mode = DC_PRED;
    }

    cu->intra_pred_modes[b8] = real_luma_mode;

    /* 存储模式到 ipredmode 数组, 供后续块的 MPM 推导使用。
     * 仅存储右列和末行 (对应 davs2 的存储优化)。 */
    {
        int size_in_scu = 1 << (cu->cu_level - MIN_CU_SIZE_IN_BIT);
        int w_4x4 = size_in_scu << 1;
        int h_4x4 = size_in_scu << 1;
        int j;

        switch (cu->i_tu_split) {
        case TU_SPLIT_HOR:
            h_4x4 >>= 2;
            break;
        case TU_SPLIT_VER:
            w_4x4 >>= 2;
            break;
        case TU_SPLIT_CROSS:
            w_4x4 >>= 1;
            h_4x4 >>= 1;
            break;
        default:
            break;
        }

        for (j = 0; j < h_4x4; j++) {
            int i = (j == h_4x4 - 1) ? 0 : w_4x4 - 1;
            for (; i < w_4x4; i++) {
                p_intramode[i] = real_luma_mode;
            }
            p_intramode += stride;
        }
    }

    return 0;
}


/* ===================================================================
 * CU 头部解码 (对应 davs2 cu_read_header)
 * =================================================================== */

/* 读取 CU 头部: CU 类型、帧内/帧间模式
 * \return 实际 CU 类型 (skip 为 -1), 或 -1 表示错误 (通过 real_cu_type 指针返回)
 */
static int read_cu_header(avs2_aec *aec, avs2_cu *cu, int frame_type,
                          avs2_seq_header *seq, int num_refs,
                          int x, int y, int *real_cu_type_out,
                          avs2_frame *f)
{
    aec_cu_t aec_desc;
    int real_cu_type;

    fill_aec_cu_desc(&aec_desc, cu);
    aec_desc.i_md_directskip_mode = 0;

    /* 1. 读取 CU 类型 */
    if (frame_type == AVS2_S_SLICE) {
        real_cu_type = aec_read_cu_type_sframe(aec);
    } else {
        real_cu_type = aec_read_cu_type(aec, &aec_desc, frame_type,
                                        seq->enable_amp, seq->enable_mhpskip,
                                        seq->enable_wsm, num_refs);
    }

    *real_cu_type_out = real_cu_type;
    real_cu_type = (real_cu_type < 0) ? 0 : real_cu_type;
    aec_desc.i_cu_type = (int8_t)real_cu_type;

    /* 2. 帧间预测方向解析 */
    if (frame_type != AVS2_I_SLICE && !IS_INTRA_MODE(real_cu_type)) {
        aec_read_inter_pred_dir(aec, &aec_desc, frame_type,
                                seq->enable_dhp, num_refs);
    }

    /* 3. 帧内模式解析 */
    if (IS_INTRA_MODE(real_cu_type)) {
        int size_8x8   = 1 << (cu->cu_level - B8X8_IN_BIT);
        int size_16x16 = 1 << (cu->cu_level - B16X16_IN_BIT);
        int y_4x4      = y >> MIN_PU_SIZE_IN_BIT;
        int x_4x4      = x >> MIN_PU_SIZE_IN_BIT;

        /* 读取帧内 CU 子类型 (2Nx2N / NxN / SDIP) */
        real_cu_type = aec_read_intra_cu_type(aec, &aec_desc,
                                               seq->enable_sdip, seq->enable_nsqt);
        aec_desc.i_cu_type = (int8_t)real_cu_type;
        /* 提前同步 i_tu_split, 供 read_intra_pred_mode_luma 使用 */
        cu->i_tu_split = aec_desc.i_trans_size;

        /* 读取亮度预测模式 (块 0) */
        read_intra_pred_mode_luma(aec, cu, 0, f, x_4x4, y_4x4);

        /* 根据子类型读取剩余块的预测模式 */
        switch (real_cu_type) {
        case PRED_I_2Nxn:
            read_intra_pred_mode_luma(aec, cu, 1, f, x_4x4, y_4x4 + 1 * size_16x16);
            read_intra_pred_mode_luma(aec, cu, 2, f, x_4x4, y_4x4 + 2 * size_16x16);
            read_intra_pred_mode_luma(aec, cu, 3, f, x_4x4, y_4x4 + 3 * size_16x16);
            break;
        case PRED_I_nx2N:
            read_intra_pred_mode_luma(aec, cu, 1, f, x_4x4 + 1 * size_16x16, y_4x4);
            read_intra_pred_mode_luma(aec, cu, 2, f, x_4x4 + 2 * size_16x16, y_4x4);
            read_intra_pred_mode_luma(aec, cu, 3, f, x_4x4 + 3 * size_16x16, y_4x4);
            break;
        case PRED_I_NxN:
            read_intra_pred_mode_luma(aec, cu, 1, f, x_4x4 + size_8x8, y_4x4);
            read_intra_pred_mode_luma(aec, cu, 2, f, x_4x4, y_4x4 + size_8x8);
            read_intra_pred_mode_luma(aec, cu, 3, f, x_4x4 + size_8x8, y_4x4 + size_8x8);
            break;
        default:
            break;
        }

        /* 读取色度预测模式 */
        if (seq->chroma_format != AVS2_CHROMA_400) {
            /* 计算色度模式上下文 (对应 davs2 cu_init: c_ipred_mode_ctx)
             * 根据左邻 CU 的 c_ipred_mode 是否非 0 设置 */
            int c_ipred_mode_ctx = 0;
            if (x > 0) {
                int left_bx = (x >> 3) - 1;
                int left_by = y >> 3;
                if (left_bx >= 0 && left_by >= 0 && left_bx < f->w8 && left_by < f->h8) {
                    avs2_cu *left_cu = &f->cu_grid[left_by * f->w8 + left_bx];
                    if (left_cu->i_slice_nr == cu->i_slice_nr) {
                        c_ipred_mode_ctx = (left_cu->i_intra_mode_c != 0) ? 1 : 0;
                    }
                }
            }
            cu->i_intra_mode_c = (int8_t)aec_read_intra_pmode_c(aec,
                                                                 cu->intra_pred_modes[0],
                                                                 c_ipred_mode_ctx);
        } else {
            cu->i_intra_mode_c = 0;
        }
    }

    /* 同步回 avs2_cu */
    sync_aec_cu_desc(cu, &aec_desc);
    cu->cu_type = (int8_t)real_cu_type;

    return 0;
}


/* ===================================================================
 * MV 解码 (对应 davs2 cu_read_mv)
 * =================================================================== */

/* MVPRED 类型常量 (对应 davs2 defines.h) */
#define MVPRED_xy_MIN  0
#define MVPRED_L       1
#define MVPRED_U       2
#define MVPRED_UR      3

/* 符号函数 (对应 davs2 davs2_sign2 / davs2_sign3) */
static int davs2_sign2(int val) { return ((val >> 31) << 1) + 1; }
static int davs2_sign3(int val) { return (val >> 31) | (int)(((uint32_t)-val) >> 31u); }

/* MV 缩放: 普通帧间模式 (对应 davs2 scale_mv_default)
 * mv = sign3(mv) * ((|mv| * dist_dst * dist_src + HALF_MULTI) >> OFFSET) */
static int16_t scale_mv_default(int mv, int dist_dst, int dist_src)
{
    mv = davs2_sign3(mv) * ((DAVS2_ABS(mv) * dist_dst * dist_src + HALF_MULTI) >> OFFSET);
    if (mv < -32768) mv = -32768;
    if (mv > 32767)  mv = 32767;
    return (int16_t)mv;
}

/* PMVR MV 推导 (对应 davs2 pmvr_mv_derivation) */
static void pmvr_mv_derivation(int enable_pmvr, avs2_mv *mv,
                               const avs2_mv *mvd, const avs2_mv *mvp)
{
    int mvx, mvy;

    if (enable_pmvr) {
        int ctr_x = ((mvp->x >> 1) << 1) - mvp->x;
        int ctr_y = ((mvp->y >> 1) << 1) - mvp->y;

        if (DAVS2_ABS(mvd->x - ctr_x) > THRESHOLD_PMVR) {
            mvx = mvp->x + (mvd->x << 1) - ctr_x - davs2_sign2(mvd->x - ctr_x) * THRESHOLD_PMVR;
            mvy = mvp->y + (mvd->y << 1) + ctr_y;
        } else if (DAVS2_ABS(mvd->y - ctr_y) > THRESHOLD_PMVR) {
            mvx = mvp->x + (mvd->x << 1) + ctr_x;
            mvy = mvp->y + (mvd->y << 1) - ctr_y - davs2_sign2(mvd->y - ctr_y) * THRESHOLD_PMVR;
        } else {
            mvx = mvd->x + mvp->x;
            mvy = mvd->y + mvp->y;
        }
    } else {
        mvx = mvd->x + mvp->x;
        mvy = mvd->y + mvp->y;
    }

    mvx = (mvx < -32768) ? -32768 : (mvx > 32767 ? 32767 : mvx);
    mvy = (mvy < -32768) ? -32768 : (mvy > 32767 ? 32767 : mvy);
    mv->x = (int16_t)mvx;
    mv->y = (int16_t)mvy;
}

/* 中位数 MV 推导 (对应 davs2 derive_median_mv)
 * 符号聚类 + 最小差异对均值 */
static int16_t derive_median_mv(int mva, int mvb, int mvc)
{
    int mvp;

    if (((mva < 0) && (mvb > 0) && (mvc > 0)) || ((mva > 0) && (mvb < 0) && (mvc < 0))) {
        mvp = (mvb + mvc) / 2;
    } else if (((mvb < 0) && (mva > 0) && (mvc > 0)) || ((mvb > 0) && (mva < 0) && (mvc < 0))) {
        mvp = (mvc + mva) / 2;
    } else if (((mvc < 0) && (mva > 0) && (mvb > 0)) || ((mvc > 0) && (mva < 0) && (mvb < 0))) {
        mvp = (mva + mvb) / 2;
    } else {
        int dAB = DAVS2_ABS(mva - mvb);
        int dBC = DAVS2_ABS(mvb - mvc);
        int dCA = DAVS2_ABS(mvc - mva);
        int min_diff = DAVS2_MIN(dAB, DAVS2_MIN(dBC, dCA));

        if (min_diff == dAB) {
            mvp = (mva + mvb) / 2;
        } else if (min_diff == dBC) {
            mvp = (mvb + mvc) / 2;
        } else {
            mvp = (mvc + mva) / 2;
        }
    }

    return (int16_t)mvp;
}

/* MV 预测类型推导 (对应 davs2 derive_mv_pred_type) */
static int derive_mv_pred_type(int ref_frame, int rFrameL, int rFrameU,
                               int rFrameUR, int pu_type_for_mvp)
{
    int mvp_type = MVPRED_xy_MIN;

    if ((rFrameL != INVALID_REF) && (rFrameU == INVALID_REF) && (rFrameUR == INVALID_REF)) {
        mvp_type = MVPRED_L;
    } else if ((rFrameL == INVALID_REF) && (rFrameU != INVALID_REF) && (rFrameUR == INVALID_REF)) {
        mvp_type = MVPRED_U;
    } else if ((rFrameL == INVALID_REF) && (rFrameU == INVALID_REF) && (rFrameUR != INVALID_REF)) {
        mvp_type = MVPRED_UR;
    } else {
        switch (pu_type_for_mvp) {
        case 1:
        case 4:
            if (rFrameL == ref_frame) mvp_type = MVPRED_L;
            break;
        case 2:
            if (rFrameUR == ref_frame) mvp_type = MVPRED_UR;
            break;
        case 3:
            if (rFrameU == ref_frame) mvp_type = MVPRED_U;
            break;
        default:
            break;
        }
    }

    return mvp_type;
}

/* 获取 PU 类型用于 MVP (对应 davs2 get_pu_type_for_mvp) */
static int get_pu_type_for_mvp(int bsx, int bsy, int cu_pix_x, int cu_pix_y)
{
    if (bsx < bsy) {
        if (cu_pix_x == 0) return 1;
        else return 2;
    } else if (bsx > bsy) {
        if (cu_pix_y == 0) return 3;
        else return 4;
    }
    return 0;  /* 默认 */
}

/* 查找覆盖相对位置 (rel_x, rel_y) 的 PU 索引 */
static int find_pu_index(avs2_cu *cu, int rel_x, int rel_y)
{
    int i;
    for (i = 0; i < cu->num_pu; i++) {
        if (rel_x >= cu->pu_x[i] && rel_x < cu->pu_x[i] + cu->pu_w[i] &&
            rel_y >= cu->pu_y[i] && rel_y < cu->pu_y[i] + cu->pu_h[i]) {
            return i;
        }
    }
    return 0;
}

/* 从 cu_grid 获取邻居 MV 和参考索引 (对应 davs2 cu_get_neighbor_spatial)
 * \param is_available_out 非NULL时输出可用性: 1=同条带且在图像内, 0=不可用 */
static void get_neighbor_mv_ref(avs2_frame *f, int nx_pix, int ny_pix,
                                avs2_mv *mv, int8_t *ref, int list,
                                int cur_slice_nr, int *is_available_out)
{
    int bx = nx_pix >> 3, by = ny_pix >> 3;
    avs2_cu *cu;
    int cu_size, cu_origin_x, cu_origin_y, rel_x, rel_y, pu_idx;

    mv->x = mv->y = 0;
    *ref = INVALID_REF;
    if (is_available_out) *is_available_out = 0;

    if (bx < 0 || bx >= f->w8 || by < 0 || by >= f->h8) return;

    cu = &f->cu_grid[by * f->w8 + bx];
    if (cu->i_slice_nr != cur_slice_nr) return;

    /* 帧内块: is_available=1 (位置可用), 但 ref=INVALID_REF (对应 davs2 行为) */
    if (is_available_out) *is_available_out = 1;
    if (cu->b_intra) return;

    /* 计算 CU 原点和相对位置, 找到覆盖该位置的 PU */
    cu_size = 1 << cu->cu_level;
    cu_origin_x = (bx * 8) & ~(cu_size - 1);
    cu_origin_y = (by * 8) & ~(cu_size - 1);
    rel_x = bx * 8 - cu_origin_x;
    rel_y = by * 8 - cu_origin_y;
    pu_idx = find_pu_index(cu, rel_x, rel_y);

    *mv = cu->mv[pu_idx][list];
    *ref = cu->i_ref[pu_idx][list];
}

/* 检查邻居参考帧可用性 (对应 davs2 recheck_neighbor_ref_avail)
 * 处理背景图像和 S 帧的特殊情况: 背景参考帧与普通参考帧不能互为邻居 */
static int recheck_neighbor_ref_avail(int ref_frame, int neighbor_ref,
                                      int frame_type, int b_bkgnd_picture,
                                      int num_of_references)
{
    if (neighbor_ref != -1) {
        if (((ref_frame == num_of_references - 1 && neighbor_ref != num_of_references - 1) ||
             (ref_frame != num_of_references - 1 && neighbor_ref == num_of_references - 1)) &&
             (frame_type == AVS2_P_SLICE || frame_type == AVS2_F_SLICE) && b_bkgnd_picture) {
            neighbor_ref = -1;
        }
        if (frame_type == AVS2_S_SLICE) {
            neighbor_ref = -1;
        }
    }
    return neighbor_ref;
}

/* 4 邻居中位数 MVP (对应 davs2 get_mvp_default) */
static void get_mvp_default(avs2_frame *f, avs2_cu *cu, int pix_x, int pix_y,
                            avs2_mv *pmv, int bwd_2nd, int ref_frame,
                            int bsx, int pu_type_for_mvp, int frame_type,
                            int lcu_level, int b_bkgnd_picture,
                            int num_of_references)
{
    avs2_mv mva, mvb, mvc, mvd;
    int8_t rFrameL, rFrameU, rFrameUL, rFrameUR;
    int is_available_UR;
    int mvPredType;
    int cur_slice_nr = cu->i_slice_nr;
    int x0 = pix_x, y0 = pix_y;
    int x1 = pix_x + bsx - 1;
    /* TOPRIGHT 在 LCU Z-scan 重建顺序下是否可用 (对应 davs2 avail_TR) */
    int avail_tr = avs2_check_topright_avail(pix_x, pix_y, bsx, lcu_level);

    /* 获取 4 个邻居: L (左), U (上), UL (左上), UR (右上)
     * is_available 表示位置可用 (同条带且在图像内), 与 ref 是否有效无关 */
    get_neighbor_mv_ref(f, x0 - 1, y0, &mva, &rFrameL, bwd_2nd, cur_slice_nr, NULL);
    get_neighbor_mv_ref(f, x0, y0 - 1, &mvb, &rFrameU, bwd_2nd, cur_slice_nr, NULL);
    get_neighbor_mv_ref(f, x0 - 1, y0 - 1, &mvd, &rFrameUL, bwd_2nd, cur_slice_nr, NULL);
    get_neighbor_mv_ref(f, avail_tr ? (x1 + 1) : -1, y0 - 1,
                        &mvc, &rFrameUR, bwd_2nd, cur_slice_nr, &is_available_UR);

    /* UR 位置不可用时退化为 UL (对应 davs2 is_available_UR ? ... : rFrameUL) */
    if (!is_available_UR) {
        rFrameUR = rFrameUL;
        mvc = mvd;
    }

    /* 检查邻居参考帧可用性 (对应 davs2 recheck_neighbor_ref_avail) */
    rFrameL  = (int8_t)recheck_neighbor_ref_avail(ref_frame, rFrameL,  frame_type, b_bkgnd_picture, num_of_references);
    rFrameU  = (int8_t)recheck_neighbor_ref_avail(ref_frame, rFrameU,  frame_type, b_bkgnd_picture, num_of_references);
    rFrameUR = (int8_t)recheck_neighbor_ref_avail(ref_frame, rFrameUR, frame_type, b_bkgnd_picture, num_of_references);

    /* 邻居 MV 距离缩放 */
    if (frame_type == AVS2_B_SLICE) {
        /* B 帧: mult_distance * mult_distance_src = MULTI, MV 基本不变 */
        int mult_distance = f->dist_refs[bwd_2nd ? B_BWD : B_FWD];
        int mult_distance_src = f->dist_scale_refs[bwd_2nd ? B_BWD : B_FWD];

        if (rFrameL >= 0) {
            mva.x = scale_mv_default(mva.x, mult_distance, mult_distance_src);
            mva.y = scale_mv_default(mva.y, mult_distance, mult_distance_src);
        } else { mva.x = mva.y = 0; }
        if (rFrameU >= 0) {
            mvb.x = scale_mv_default(mvb.x, mult_distance, mult_distance_src);
            mvb.y = scale_mv_default(mvb.y, mult_distance, mult_distance_src);
        } else { mvb.x = mvb.y = 0; }
        if (rFrameUR >= 0) {
            mvc.x = scale_mv_default(mvc.x, mult_distance, mult_distance_src);
            mvc.y = scale_mv_default(mvc.y, mult_distance, mult_distance_src);
        } else { mvc.x = mvc.y = 0; }
    } else {
        /* P/F 帧: 从邻居参考距离缩放到当前参考距离 */
        int mult_distance = f->dist_refs[ref_frame];

        if (rFrameL >= 0) {
            int devide_distance_src = f->dist_scale_refs[rFrameL];
            mva.x = scale_mv_default(mva.x, mult_distance, devide_distance_src);
            mva.y = scale_mv_default(mva.y, mult_distance, devide_distance_src);
        } else { mva.x = mva.y = 0; }
        if (rFrameU >= 0) {
            int devide_distance_src = f->dist_scale_refs[rFrameU];
            mvb.x = scale_mv_default(mvb.x, mult_distance, devide_distance_src);
            mvb.y = scale_mv_default(mvb.y, mult_distance, devide_distance_src);
        } else { mvb.x = mvb.y = 0; }
        if (rFrameUR >= 0) {
            int devide_distance_src = f->dist_scale_refs[rFrameUR];
            mvc.x = scale_mv_default(mvc.x, mult_distance, devide_distance_src);
            mvc.y = scale_mv_default(mvc.y, mult_distance, devide_distance_src);
        } else { mvc.x = mvc.y = 0; }
    }

    mvPredType = derive_mv_pred_type(ref_frame, rFrameL, rFrameU, rFrameUR, pu_type_for_mvp);

    switch (mvPredType) {
    case MVPRED_xy_MIN:
        pmv->x = derive_median_mv(mva.x, mvb.x, mvc.x);
        pmv->y = derive_median_mv(mva.y, mvb.y, mvc.y);
        break;
    case MVPRED_L:
        *pmv = mva;
        break;
    case MVPRED_U:
        *pmv = mvb;
        break;
    default:  /* MVPRED_UR */
        *pmv = mvc;
        break;
    }
}

/* 读取运动矢量 (对应 davs2 cu_read_mv)
 * \param fc        帧上下文 (含距离信息)
 * \param aec       AEC 解码器
 * \param cu        当前 CU
 * \param frame_type 帧类型
 * \param x, y      CU 像素坐标
 * \param num_refs  参考帧数量
 * \param f         当前帧
 * \param seq       序列头 (含 enable_pmvr)
 * \return 0 成功, -1 失败
 */
static int read_cu_mv(avs2_frame_ctx *fc, avs2_aec *aec, avs2_cu *cu,
                      int frame_type, int x, int y, int num_refs,
                      avs2_frame *f, avs2_seq_header *seq,
                      int b_bkgnd_picture)
{
    int is_bframe = (frame_type == AVS2_B_SLICE);
    int enable_pmvr = seq->enable_pmvr;
    int pu_idx;

    /* F 帧 DMH 模式读取 */
    if (frame_type == AVS2_F_SLICE &&
        cu->b8pdir[0] == PDIR_FWD && cu->b8pdir[1] == PDIR_FWD &&
        cu->b8pdir[2] == PDIR_FWD && cu->b8pdir[3] == PDIR_FWD) {
        if (!(cu->cu_level == B8X8_IN_BIT && cu->cu_type >= PRED_2NxN &&
              cu->cu_type <= PRED_nRx2N)) {
            cu->i_dmh_mode = (int8_t)aec_read_dmh_mode(aec, cu->cu_level);
        } else {
            cu->i_dmh_mode = 0;
        }
    }

    /* 读取前向 MV */
    for (pu_idx = 0; pu_idx < cu->num_pu; pu_idx++) {
        if (cu->b8pdir[pu_idx] != PDIR_BWD) {
            int pu_x = cu->pu_x[pu_idx];
            int pu_y = cu->pu_y[pu_idx];
            int bsx = cu->pu_w[pu_idx];
            int bsy = cu->pu_h[pu_idx];
            int refframe = cu->i_ref[pu_idx][0];
            avs2_mv mvp, mvd, mv;
            int pu_mvp_type = get_pu_type_for_mvp(bsx, bsy, pu_x, pu_y);

            /* MV 预测 (4 邻居中位数) */
            get_mvp_default(f, cu, x + pu_x, y + pu_y, &mvp, 0,
                            refframe, bsx, pu_mvp_type, frame_type, fc->lcu_size,
                            b_bkgnd_picture, num_refs);

            if (frame_type != AVS2_S_SLICE) {
                aec_read_mvds(aec, &mvd);
                /* PMVR MV 推导 */
                pmvr_mv_derivation(enable_pmvr, &mv, &mvd, &mvp);
            } else {
                /* S 帧: 无 MVD, 直接使用 MVP */
                mv = mvp;
            }

            cu->mv[pu_idx][0] = mv;

            /* F 帧 DUAL 模式: 后向 MV = scale_mv_skip(mv_fwd, dist_2nd, dist_1st_src) */
            if (!is_bframe && cu->b8pdir[pu_idx] == PDIR_DUAL) {
                int dist_1st = f->dist_refs[refframe];
                int dist_1st_src = f->dist_scale_refs[refframe];
                int dist_2nd = f->dist_refs[!refframe];
                avs2_mv mv_2nd;
                mv_2nd.x = scale_mv_skip(mv.x, dist_2nd, dist_1st_src);
                mv_2nd.y = scale_mv_skip(mv.y, dist_2nd, dist_1st_src);
                cu->mv[pu_idx][1] = mv_2nd;
                (void)dist_1st;  /* 非场编时未使用 */
            } else if (!is_bframe) {
                cu->mv[pu_idx][1].x = 0;
                cu->mv[pu_idx][1].y = 0;
            }
        }
    }

    /* B 帧: 读取后向 MV */
    if (!is_bframe) {
        return 0;
    }

    {
        int distance_fwd_src = f->dist_scale_refs[B_FWD];
        int distance_bwd = f->dist_refs[B_BWD];

        for (pu_idx = 0; pu_idx < cu->num_pu; pu_idx++) {
            if (cu->b8pdir[pu_idx] != PDIR_FWD) {
                int pu_x = cu->pu_x[pu_idx];
                int pu_y = cu->pu_y[pu_idx];
                int bsx = cu->pu_w[pu_idx];
                int bsy = cu->pu_h[pu_idx];
                int refframe = cu->i_ref[pu_idx][1];
                avs2_mv mvp, mvd, mv;
                int pu_mvp_type = get_pu_type_for_mvp(bsx, bsy, pu_x, pu_y);

                get_mvp_default(f, cu, x + pu_x, y + pu_y, &mvp, 1,
                                refframe, bsx, pu_mvp_type, frame_type, fc->lcu_size,
                                b_bkgnd_picture, num_refs);

                if (cu->b8pdir[pu_idx] == PDIR_SYM) {
                    /* 对称 MV: 后向 = -scale_mv_skip(前向, dist_bwd, dist_fwd_src) */
                    avs2_mv mv_1st = cu->mv[pu_idx][0];
                    mv.x = (int16_t)(-scale_mv_skip(mv_1st.x, distance_bwd, distance_fwd_src));
                    mv.y = (int16_t)(-scale_mv_skip(mv_1st.y, distance_bwd, distance_fwd_src));
                } else {
                    aec_read_mvds(aec, &mvd);
                    pmvr_mv_derivation(enable_pmvr, &mv, &mvd, &mvp);
                }

                cu->mv[pu_idx][1] = mv;
            }
        }
    }

    return 0;
}


/* ===================================================================
 * 系数解码 (对应 davs2 cu_read_all_coeffs)
 * =================================================================== */

/* 注: get_quant_params 和 get_chroma_qp 已移至 quant.c (avs2_get_quant_params / avs2_chroma_qp) */

/* 读取一个块的所有系数 (对应 davs2 cu_get_block_coeffs 调用) */
static int8_t read_block_coeffs(avs2_aec *aec, avs2_cu *cu, aec_cu_t *aec_desc,
                                coeff_t *coeff, int w_tr, int h_tr,
                                int tu_level, int tu_split_type, int is_luma,
                                int intra_pred_class, int b_swap_xy,
                                int scale, int shift, int wq_size_id)
{
    runlevel_t runlevel;
    runlevel_pair_t pairs[16];
    const uint8_t (*cg_scan)[2];

    /* 通过分派表选择 CG 扫描表 (对应 davs2 tab_scan_cg[tu_level][split_type]) */
    cg_scan = avs2_tab_scan_cg[tu_level][tu_split_type];
    if (cg_scan == NULL) {
        cg_scan = avs2_tab_scan_cg[tu_level][TU_SPLIT_NON];
    }

    /* 设置 runlevel 结构 */
    memset(&runlevel, 0, sizeof(runlevel));
    memset(pairs, 0, sizeof(pairs));

    runlevel.cg_scan = cg_scan;
    if (is_luma) {
        runlevel.p_ctx_run = aec->syn_ctx.coeff_run[0];
        runlevel.p_ctx_level = aec->syn_ctx.coeff_level;
        runlevel.p_ctx_sig_cg = aec->syn_ctx.sig_cg_contexts;
        runlevel.p_ctx_last_cg = aec->syn_ctx.last_cg_contexts;
        runlevel.p_ctx_last_pos_in_cg = aec->syn_ctx.last_coeff_pos;
    } else {
        runlevel.p_ctx_run = aec->syn_ctx.coeff_run[1];
        runlevel.p_ctx_level = aec->syn_ctx.coeff_level + 20;
        runlevel.p_ctx_sig_cg = aec->syn_ctx.sig_cg_contexts + NUM_SIGCG_CTX_LUMA;
        runlevel.p_ctx_last_cg = aec->syn_ctx.last_cg_contexts + NUM_LAST_CG_CTX_LUMA;
        runlevel.p_ctx_last_pos_in_cg = aec->syn_ctx.last_coeff_pos + NUM_LAST_POS_CTX_LUMA;
    }
    runlevel.run_level = pairs;
    runlevel.p_res = coeff;
    runlevel.i_res = w_tr;
    runlevel.b_swap_xy = b_swap_xy;
    runlevel.i_tu_level = tu_level;
    runlevel.w_tr = w_tr;
    runlevel.h_tr = h_tr;

    return cu_get_block_coeffs(aec, &runlevel, aec_desc, coeff,
                               w_tr, h_tr, tu_level, is_luma,
                               intra_pred_class, b_swap_xy,
                               scale, shift, wq_size_id);
}

/* 读取 CU 的所有系数 (对应 davs2 cu_read_all_coeffs) */
static int read_cu_coeffs(avs2_frame_ctx *fc, avs2_aec *aec, avs2_cu *cu,
                          int bit_depth, int chroma_format, int x, int y)
{
    aec_cu_t aec_desc;
    int bit_size = cu->cu_level;
    int tu_level = cu->cu_level;
    int b8, uv;

    fill_aec_cu_desc(&aec_desc, cu);

    /* 亮度系数解码 */
    if (cu->i_tu_split == TU_SPLIT_NON) {
        /* 无 TU 分割: 整块解码 */
        tu_level = tu_level - B4X4_IN_BIT;
        if (tu_level > 3) tu_level = 3;
        if (tu_level < 0) tu_level = 0;

        if (cu->i_cbp & 0x0F) {
            int intra_pred_class = IS_INTRA_MODE(cu->cu_type)
                                   ? tab_intra_mode_scan_type[cu->intra_pred_modes[0]]
                                   : INTRA_PRED_DC_DIAG;
            int b_swap_xy = (IS_INTRA_MODE(cu->cu_type) &&
                             intra_pred_class == INTRA_PRED_HOR);
            int blocksize = 1 << (tu_level + B4X4_IN_BIT);
            int shift, scale;
            int wq_size_id = (bit_size > 5) ? 3 : (bit_size - B4X4_IN_BIT);

            /* bit_size - (i_trans_size != TU_SPLIT_NON) = bit_size - 0 = bit_size */
            avs2_get_quant_params(cu->qp, bit_size, bit_depth, &shift, &scale);
            memset(fc->coeff_scratch_y, 0, sizeof(int16_t) * 64 * 64);

            cu->dct_pattern[0] = read_block_coeffs(aec, cu, &aec_desc,
                                                    fc->coeff_scratch_y,
                                                    blocksize, blocksize,
                                                    tu_level, TU_SPLIT_NON, 1,
                                                    intra_pred_class, b_swap_xy,
                                                    scale, shift, wq_size_id);
        }
    } else {
        /* 有 TU 分割: 4 个子块分别解码 */
        int8_t tu_x[4], tu_y[4], tu_w[4], tu_h[4];
        int b_wavelet = (bit_size == B64X64_IN_BIT && cu->i_tu_split != TU_SPLIT_CROSS);
        int shift, scale;
        int wq_size_id;

        init_transform_units(cu, tu_x, tu_y, tu_w, tu_h);

        tu_level = cu->cu_level - B8X8_IN_BIT - b_wavelet;
        if (tu_level < 0) tu_level = 0;

        /* 对应 davs2: bit_size - (i_trans_size != TU_SPLIT_NON) = cu_level - 1 */
        avs2_get_quant_params(cu->qp, cu->cu_level - 1, bit_depth, &shift, &scale);

        if (cu->i_tu_split == TU_SPLIT_CROSS) {
            wq_size_id = (bit_size > 5) ? 3 : (bit_size - B8X8_IN_BIT);
        } else {
            wq_size_id = bit_size - B8X8_IN_BIT;
            if (cu->cu_level == B64X64_IN_BIT) wq_size_id--;
        }

        for (b8 = 0; b8 < 4; b8++) {
            if (cu->i_cbp & (1 << b8)) {
                int bsx = tu_w[b8];
                int bsy = tu_h[b8];
                int intra_pred_class = IS_INTRA_MODE(cu->cu_type)
                                       ? tab_intra_mode_scan_type[cu->intra_pred_modes[b8]]
                                       : INTRA_PRED_DC_DIAG;
                int b_swap_xy = (IS_INTRA_MODE(cu->cu_type) &&
                                 intra_pred_class == INTRA_PRED_HOR &&
                                 cu->cu_type != PRED_I_2Nxn &&
                                 cu->cu_type != PRED_I_nx2N);
                coeff_t *p_res = fc->coeff_scratch_y + (b8 << ((bit_size - 1) << 1));

                memset(p_res, 0, sizeof(coeff_t) * bsx * bsy);
                cu->dct_pattern[b8] = read_block_coeffs(aec, cu, &aec_desc,
                                                         p_res, bsx, bsy,
                                                         tu_level, cu->i_tu_split, 1,
                                                         intra_pred_class, b_swap_xy,
                                                         scale, shift, wq_size_id);
                if (cu->dct_pattern[b8] < 0) {
                    return -1;
                }
            }
        }
    }

    /* 色度系数解码 */
    if (chroma_format != AVS2_CHROMA_400) {
        int wq_size_id = cu->cu_level - 1;
        int c_tu_level = cu->cu_level - B8X8_IN_BIT;
        if (c_tu_level < 0) c_tu_level = 0;
        if (c_tu_level > 3) c_tu_level = 3;

        for (uv = 0; uv < 2; uv++) {
            if ((cu->i_cbp >> (uv + 4)) & 0x1) {
                int blocksize = 1 << wq_size_id;
                coeff_t *p_res = (uv == 0) ? fc->coeff_scratch_u : fc->coeff_scratch_v;
                int shift, scale;
                int chroma_qp = avs2_chroma_qp(cu->qp,
                                                uv == 0 ? fc->chroma_quant_param_delta_cb
                                                        : fc->chroma_quant_param_delta_cr,
                                                bit_depth);

                avs2_get_quant_params(chroma_qp, wq_size_id, bit_depth, &shift, &scale);
                memset(p_res, 0, sizeof(coeff_t) * blocksize * blocksize);

                cu->dct_pattern[4 + uv] = read_block_coeffs(aec, cu, &aec_desc,
                                                             p_res, blocksize, blocksize,
                                                             c_tu_level, TU_SPLIT_NON, 0,
                                                             INTRA_PRED_DC_DIAG, 0,
                                                             scale, shift, wq_size_id);
                if (cu->dct_pattern[4 + uv] < 0) {
                    return -1;
                }
            }
        }
    }

    return 0;
}


/* ===================================================================
 * CBP 解码 (对应 davs2 cu_read_cbp)
 * =================================================================== */

/* 读取 CBP 和 delta QP (对应 davs2 cu_read_cbp + cu_read_delta_qp) */
static int read_cu_cbp(avs2_frame_ctx *fc, avs2_aec *aec, avs2_cu *cu,
                       int chroma_format, int *prev_qp, int x, int y)
{
    aec_cu_t aec_desc;
    int cbp;

    fill_aec_cu_desc(&aec_desc, cu);
    /* 设置邻块查找所需信息 */
    aec_desc.scu_x = x >> MIN_CU_SIZE_IN_BIT;
    aec_desc.scu_y = y >> MIN_CU_SIZE_IN_BIT;
    aec_desc.p_frame = fc->fdec;

    /* 读取 CBP (对应 davs2 cu_read_cbp: p_cu->i_cbp = aec_read_cbp(...)) */
    cbp = aec_read_cbp(aec, &aec_desc, chroma_format);
    aec_desc.i_cbp = (int8_t)cbp;

    /* 读取 delta QP (对应 davs2 cu_read_cbp + cu_read_delta_qp)
     * davs2 使用左 CU 的 QP (h->lcu.i_left_cu_qp) 作为预测基准,
     * 而非 Z 扫描前一个 CU 的 QP。左 CU 从 cu_grid 获取。 */
    if (fc->b_dqp) {
        int i_delta_qp = 0;
        int left_qp;  /* 左 CU 的 QP (对应 davs2 h->lcu.i_left_cu_qp) */

        if (aec_desc.i_cbp != 0) {
            i_delta_qp = aec_read_cu_delta_qp(aec, fc->i_last_dquant);
        }
        fc->i_last_dquant = i_delta_qp;

        /* 获取左 CU 的 QP (对应 davs2 cu_init 中 i_left_cu_qp 的设置)
         * 默认使用 slice QP; 若左侧有同条带 CU 则使用其 QP */
        left_qp = fc->slice_qp;
        if (x > 0) {
            avs2_frame *f = fc->fdec;
            int bx = x >> 3, by = y >> 3;
            avs2_cu *left_cu = &f->cu_grid[by * f->w8 + bx - 1];
            if (left_cu->i_slice_nr == cu->i_slice_nr) {
                left_qp = left_cu->qp;
            }
        }

        aec_desc.i_qp = (int8_t)(i_delta_qp + left_qp);
    } else {
        aec_desc.i_qp = (int8_t)fc->slice_qp;
    }

    *prev_qp = aec_desc.i_qp;

    sync_aec_cu_desc(cu, &aec_desc);
    cu->qp = aec_desc.i_qp;

    return 0;
}


/* ===================================================================
 * CU 信息读取 (对应 davs2 cu_read_info)
 * =================================================================== */

/* 读取 CU 全部信息: 头部 + MV + 残差
 * \return 0 成功, -1 失败
 */
static int read_cu_info(avs2_frame_ctx *fc, avs2_aec *aec, avs2_cu *cu, int level,
                        int x, int y, int frame_type,
                        avs2_seq_header *seq, int num_refs,
                        int bit_depth, int chroma_format, int *prev_qp,
                        avs2_frame *f)
{
    int real_cu_type = 0;

    /* 初始化 CU (对应 davs2 cu_init: p_cu->i_qp = h->i_qp) */
    cu->cu_level = (int8_t)level;
    cu->qp = (int8_t)fc->slice_qp;
    cu->cu_type = PRED_SKIP;
    cu->i_cbp = 0;
    cu->i_intra_mode_c = DC_PRED_C;
    cu->i_dmh_mode = 0;
    cu->i_weighted_skipmode = 0;
    cu->i_skip_mode = DS_NONE;
    cu->b_intra = 0;
    cu->i_slice_nr = 0;  /* 单条带帧: 所有 CU 同属条带 0 */
    memset(cu->dct_pattern, 0, sizeof(cu->dct_pattern));
    memset(cu->intra_pred_modes, 0, sizeof(cu->intra_pred_modes));
    memset(cu->b8pdir, 0, sizeof(cu->b8pdir));
    memset(cu->mv, 0, sizeof(cu->mv));
    memset(cu->i_ref, 0xFF, sizeof(cu->i_ref));  /* INVALID_REF = -1 */

    /* 1. 读取 CU 头部 */
    {
        if (read_cu_header(aec, cu, frame_type, seq, num_refs, x, y, &real_cu_type, f) < 0) {
            return -1;
        }
    }

    /* 初始化 PU */
    init_prediction_units(cu, frame_type, frame_type);

    /* 2. 读取 MV 和参考索引 */
    if (IS_INTRA_MODE(cu->cu_type)) {
        /* 帧内: 无 MV */
        int i;
        cu->b_intra = 1;
        for (i = 0; i < 4; i++) {
            cu->i_ref[i][0] = INVALID_REF;
            cu->i_ref[i][1] = INVALID_REF;
            cu->b8pdir[i] = PDIR_INVALID;
        }
    } else if (cu->cu_type == PRED_SKIP) {
        /* Skip 模式: MV 通过时域直接推导 (对应 davs2 fill_mv_and_ref_for_skip) */
        cu->b_intra = 0;
        derive_skip_mv(fc, cu, x, y, !seq->background_picture_disable);
    } else {
        /* 普通帧间: 读取 MV */
        cu->b_intra = 0;
        if (read_cu_mv(fc, aec, cu, frame_type, x, y, num_refs, f, seq,
                       !seq->background_picture_disable) < 0) {
            return -1;
        }
    }

    /* 3. 读取 CBP 和系数 */
    if (real_cu_type < 0) {
        /* true skip 模式: 无残差。
         * QP 继承自左侧 CU (对应 davs2: p_cu->i_qp = h->lcu.i_left_cu_qp) */
        int left_qp_skip = fc->slice_qp;
        if (x > 0) {
            int bx_skip = x >> 3, by_skip = y >> 3;
            avs2_cu *left_cu_skip = &f->cu_grid[by_skip * f->w8 + bx_skip - 1];
            if (left_cu_skip->i_slice_nr == cu->i_slice_nr) {
                left_qp_skip = left_cu_skip->qp;
            }
        }
        cu->qp = (int8_t)left_qp_skip;
        *prev_qp = left_qp_skip;
        cu->i_tu_split = TU_SPLIT_NON;
        cu->i_cbp = 0;
    } else {
        /* 非 Skip: 读取 CBP */
        if (read_cu_cbp(fc, aec, cu, chroma_format, prev_qp, x, y) < 0) {
            return -1;
        }
        if (cu->i_cbp != 0) {
            if (read_cu_coeffs(fc, aec, cu, bit_depth, chroma_format, x, y) < 0) {
                return -1;
            }
        }
    }

    return 0;
}


/* ===================================================================
 * 帧间预测 (对应 davs2 davs2_get_inter_pred)
 * =================================================================== */

/* MV 位置裁剪 (对应 davs2 cu_get_mc_pos)
 * 裁剪 blk_pos + (mv >> 2) 到 [-blk_size - 8, img_size + 4] 范围内,
 * 返回裁剪后的 MV 值 (使得 blk_pos + (ret >> 2) == 裁剪后的位置)。 */
static int get_mc_pos(int img_size, int blk_size, int blk_pos, int mv)
{
    int imv = mv >> 2;  /* 整像素精度 */
    int fmv = mv & 7;   /* 分像素精度 */
    int pos = blk_pos + imv;  /* 绝对整像素位置 */

    if (pos < -blk_size - 8) {
        /* 裁剪到下界: blk_pos + (ret >> 2) == -blk_size - 8 */
        return ((-blk_size - 8 - blk_pos) << 2) + fmv;
    } else if (pos > img_size + 4) {
        /* 裁剪到上界: blk_pos + (ret >> 2) == img_size + 4 */
        return ((img_size + 4 - blk_pos) << 2) + fmv;
    } else {
        return mv;
    }
}

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

        /* 确定参考帧和 MV
         * B 帧: fref[0]=后向(高POC, B_BWD=0), fref[1]=前向(低POC, B_FWD=1)
         * P/F 帧: fref[i]=第 i 个前向参考 */
        if (frame_type == AVS2_B_SLICE) {
            /* B 帧: 前向 MV 用 fref[1], 后向 MV 用 fref[0] */
            if (pred_dir == PDIR_BWD) {
                mv_1st = cu->mv[pu_idx][1];
                if (fc->n_refs > 0) p_fref1 = fc->fref[B_BWD];  /* 后向参考 */
            } else if (pred_dir == PDIR_SYM || pred_dir == PDIR_BID) {
                mv_1st = cu->mv[pu_idx][0];  /* 前向 MV */
                mv_2nd = cu->mv[pu_idx][1];  /* 后向 MV */
                if (fc->n_refs > 1) {
                    p_fref1 = fc->fref[B_FWD];  /* 前向参考 */
                    p_fref2 = fc->fref[B_BWD];  /* 后向参考 */
                }
            } else {
                /* PDIR_FWD */
                mv_1st = cu->mv[pu_idx][0];
                if (fc->n_refs > 1) p_fref1 = fc->fref[B_FWD];  /* 前向参考 */
            }
        } else {
            /* P/F 帧: fref[i] 为前向参考 */
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
                /* DMH 模式: 两个假设 */
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

        /* 裁剪 MV 到图像边界 */
        mv_1st.x = (int16_t)get_mc_pos(f->width, width, pix_x, mv_1st.x);
        mv_1st.y = (int16_t)get_mc_pos(f->height, height, pix_y, mv_1st.y);
        mv_2nd.x = (int16_t)get_mc_pos(f->width, width, pix_x, mv_2nd.x);
        mv_2nd.y = (int16_t)get_mc_pos(f->height, height, pix_y, mv_2nd.y);

        /* 亮度运动补偿 */
        if (p_fref1 != NULL) {
            int bps = f->bytes_per_sample;
            int ref_y = pix_y + (mv_1st.y >> 2);
            int ref_x = pix_x + (mv_1st.x >> 2);
            /* 边界裁剪: 确保参考位置在 padding 范围内 [-64, ref_h+64) */
            int max_y = p_fref1->height + 60;
            int min_y = -60;
            int max_x = p_fref1->width + 60;
            int min_x = -60;
            if (ref_y < min_y) ref_y = min_y;
            if (ref_y > max_y) ref_y = max_y;
            if (ref_x < min_x) ref_x = min_x;
            if (ref_x > max_x) ref_x = max_x;

            uint8_t *p_pred = f->data[0] + pix_y * f->stride[0] + (ptrdiff_t)pix_x * bps;
            const uint8_t *p_ref = p_fref1->data[0] + (ptrdiff_t)ref_y * p_fref1->stride[0]
                                   + (ptrdiff_t)ref_x * bps;
            int mx = mv_1st.x & 3;
            int my = mv_1st.y & 3;

            avs2_dsp_table.mc_luma(p_ref, p_fref1->stride[0], p_pred, f->stride[0],
                                   width, height, mx, my, bit_depth);

            /* 双向预测: 第二参考帧 (融合 MC+avg, 省去 pred2 中间缓冲) */
            if (p_fref2 != NULL) {
                int ref2_y = pix_y + (mv_2nd.y >> 2);
                int ref2_x = pix_x + (mv_2nd.x >> 2);
                if (ref2_y < min_y) ref2_y = min_y;
                if (ref2_y > max_y) ref2_y = max_y;
                if (ref2_x < min_x) ref2_x = min_x;
                if (ref2_x > max_x) ref2_x = max_x;
                const uint8_t *p_ref2 = p_fref2->data[0] + (ptrdiff_t)ref2_y * p_fref2->stride[0]
                                        + (ptrdiff_t)ref2_x * bps;
                int mx2 = mv_2nd.x & 3;
                int my2 = mv_2nd.y & 3;

                /* 融合: dst[i] = (dst[i] + mc(src2,mv2)[i] + 1) >> 1 */
                avs2_dsp_table.mc_luma_avg(p_ref2, p_fref2->stride[0],
                                           p_pred, f->stride[0],
                                           width, height, mx2, my2, bit_depth);
            }
        } else {
            /* 参考帧不可用: 用 1<<(bit_depth-1) 填充 */
            int bps = f->bytes_per_sample;
            uint8_t *p_pred = f->data[0] + pix_y * f->stride[0] + (ptrdiff_t)pix_x * bps;
            int fill_val = 1 << (bit_depth - 1);
            avs2_dsp_table.fill_block(p_pred, f->stride[0], width, height,
                                      fill_val, bit_depth);
        }

        /* 色度运动补偿 (4:2:0) */
        if (f->chroma_format == AVS2_CHROMA_420) {
            int cx = pix_x >> 1, cy = pix_y >> 1;
            int cw = width >> 1, ch = height >> 1;
            /* 4:2:0 下色度 MV (1/8 像素) 数值上等于亮度 MV (1/4 像素),
             * 无需缩放 (对应 davs2 vec1_x = mv_1st.x 直接传给 mc_chroma) */
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
                    if (cref_y < cmin_y) cref_y = cmin_y;
                    if (cref_y > cmax_y) cref_y = cmax_y;
                    if (cref_x < cmin_x) cref_x = cmin_x;
                    if (cref_x > cmax_x) cref_x = cmax_x;
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
                        if (cref2_y < cmin_y) cref2_y = cmin_y;
                        if (cref2_y > cmax_y) cref2_y = cmax_y;
                        if (cref2_x < cmin_x) cref2_x = cmin_x;
                        if (cref2_x > cmax_x) cref2_x = cmax_x;
                        const uint8_t *p_ref2 = p_fref2->data[pl] +
                                                (ptrdiff_t)cref2_y * p_fref2->stride[pl] +
                                                (ptrdiff_t)cref2_x * cbps;
                        int cmx2 = cmv2_x & 7;
                        int cmy2 = cmv2_y & 7;

                        /* 融合: dst[i] = (dst[i] + mc(src2,mv2)[i] + 1) >> 1 */
                        avs2_dsp_table.mc_chroma_avg(p_ref2, p_fref2->stride[pl],
                                                     p_pred, f->stride[pl],
                                                     cw, ch, cmx2, cmy2, bit_depth);
                    }
                } else {
                    /* 参考帧不可用: 用 1<<(bit_depth-1) 填充 */
                    avs2_dsp_table.fill_block(p_pred, f->stride[pl], cw, ch,
                                              cfill_val, bit_depth);
                }
            }
        }
    }

    return 0;
}


/* ===================================================================
 * 残差重建 (对应 davs2 davs2_get_recons)
 * =================================================================== */

/* 对一个块执行反量化 + 反变换 + 加到预测上 */
static void reconstruct_residual(avs2_frame_ctx *fc, struct avs2_internal *c,
                                 avs2_cu *cu, int block_idx,
                                 int blk_x, int blk_y, int bsx, int bsy,
                                 int bit_depth, int is_luma)
{
    avs2_frame *f = fc->fdec;
    coeff_t *coeff;
    int pl;

    /* 获取系数缓冲区 (系数已在 aec_read_run_level 中完成反量化).
     * Pass 2 (行级并行): 使用 TLS scratch (per-worker 独立, 避免 fc->cur_lcu_coeff 竞争).
     * Pass 0/1: 使用 fc->cur_lcu_coeff (单线程或串行, 无竞争). */
    int16_t *scratch_y = tls_coeff_scratch_y ? tls_coeff_scratch_y : fc->cur_lcu_coeff_y;
    int16_t *scratch_u = tls_coeff_scratch_u ? tls_coeff_scratch_u : fc->cur_lcu_coeff_u;
    int16_t *scratch_v = tls_coeff_scratch_v ? tls_coeff_scratch_v : fc->cur_lcu_coeff_v;

    if (is_luma) {
        coeff = scratch_y;
        if (cu->i_tu_split != TU_SPLIT_NON) {
            int bit_size = cu->cu_level;
            coeff = scratch_y + (block_idx << ((bit_size - 1) << 1));
        }
    } else {
        coeff = (block_idx == 4) ? scratch_u : scratch_v;
    }

    /* 反变换 (系数已在 AEC 解码时反量化, 无需再次反量化)
     * 二次变换 (NSST) 仅对帧内亮度块生效 (对应 davs2 inv_transform 中
     * b_secT = b_secT && IS_INTRA(p_cu) && blockidx < 4) */
    {
        int b_sec_t = 0;
        int i_luma_intra_mode = 0;
        int b_top = 0;
        int b_left = 0;

        if (is_luma && IS_INTRA_MODE(cu->cu_type) &&
            c->seq->enable_2nd_transform) {
            b_sec_t = 1;
            i_luma_intra_mode = cu->intra_pred_modes[block_idx];
            b_top  = (blk_y > 0);
            b_left = (blk_x > 0);
        }

        avs2_inverse_transform(coeff, bsx, bsy, bit_depth,
                               b_sec_t, i_luma_intra_mode, b_top, b_left);
    }

    /* 将残差加到预测上 (SIMD 优化: 通过 dsp_table->recon_residual 函数指针) */
    pl = is_luma ? 0 : (block_idx == 4 ? 1 : 2);
    {
        int bps = f->bytes_per_sample;
        ptrdiff_t stride = f->stride[pl];
        uint8_t *dst = f->data[pl] + (ptrdiff_t)blk_y * stride + (ptrdiff_t)blk_x * bps;
        avs2_dsp_table.recon_residual(dst, stride, coeff, bsx, bsy, bit_depth);
    }
}


/* ===================================================================
 * CU 重建 (对应 davs2 cu_recon)
 * =================================================================== */

/* 重建一个 CU: 预测 + 残差 */
static int reconstruct_cu(avs2_frame_ctx *fc, struct avs2_internal *c,
                          avs2_cu *cu, int x, int y)
{
    avs2_frame *f = fc->fdec;
    int bit_depth = c->bit_depth;
    int chroma_format = f->chroma_format;
    int8_t tu_x[4], tu_y[4], tu_w[4], tu_h[4];
    int blockidx;

    init_transform_units(cu, tu_x, tu_y, tu_w, tu_h);

    if (IS_INTRA_MODE(cu->cu_type)) {
        /* ---- 帧内 CU ---- */
        /* 1. 亮度预测 */
        if (cu->i_tu_split == TU_SPLIT_NON) {
            /* 无 TU 分割: 整块预测 */
            avs2_get_intra_pred(c, f, cu, x, y, tu_w[0], tu_h[0],
                                cu->intra_pred_modes[0]);
            if (cu->i_cbp & 0x0F) {
                reconstruct_residual(fc, c, cu, 0, x, y, tu_w[0], tu_h[0],
                                     bit_depth, 1);
            }
        } else {
            /* 有 TU 分割: 4 个子块分别预测 */
            for (blockidx = 0; blockidx < 4; blockidx++) {
                int bx = x + tu_x[blockidx];
                int by = y + tu_y[blockidx];
                avs2_get_intra_pred(c, f, cu, bx, by,
                                    tu_w[blockidx], tu_h[blockidx],
                                    cu->intra_pred_modes[blockidx]);
                if (cu->i_cbp & (1 << blockidx)) {
                    reconstruct_residual(fc, c, cu, blockidx, bx, by,
                                         tu_w[blockidx], tu_h[blockidx],
                                         bit_depth, 1);
                }
            }
        }

        /* 2. 色度预测 */
        if (chroma_format == AVS2_CHROMA_420) {
            int cx = x >> 1, cy = y >> 1;
            avs2_get_intra_pred_chroma(c, f, cu, cx, cy);
        }
    } else {
        /* ---- 帧间 CU ---- */
        /* 1. 运动补偿预测 */
        reconstruct_inter(fc, cu, x, y, bit_depth);

        /* 2. 亮度残差 */
        if (cu->i_tu_split == TU_SPLIT_NON) {
            if (cu->i_cbp & 0x0F) {
                reconstruct_residual(fc, c, cu, 0, x, y, tu_w[0], tu_h[0],
                                     bit_depth, 1);
            }
        } else {
            for (blockidx = 0; blockidx < 4; blockidx++) {
                if (cu->i_cbp & (1 << blockidx)) {
                    int bx = x + tu_x[blockidx];
                    int by = y + tu_y[blockidx];
                    reconstruct_residual(fc, c, cu, blockidx, bx, by,
                                         tu_w[blockidx], tu_h[blockidx],
                                         bit_depth, 1);
                }
            }
        }
    }

    /* 3. 色度残差 */
    if (chroma_format == AVS2_CHROMA_420) {
        int cx = x >> 1, cy = y >> 1;
        int cw = 1 << (cu->cu_level - 1);
        int ch = cw;

        if (cu->i_cbp & (1 << 4)) {
            reconstruct_residual(fc, c, cu, 4, cx, cy, cw, ch, bit_depth, 0);
        }
        if (cu->i_cbp & (1 << 5)) {
            reconstruct_residual(fc, c, cu, 5, cx, cy, cw, ch, bit_depth, 0);
        }
    }

    return 0;
}


/* ===================================================================
 * 行级并行: per-LCU 系数保存/恢复 (按 LCU 行布局)
 *
 * pass=0 中 cur_lcu_coeff 是 per-CU 复用的 scratch, 每个 CU AEC 后立即重建,
 * 覆盖没问题. 但 2-pass 的 Pass 1 中所有 CU 的系数都写入 scratch[0..blocksize²-1],
 * 后面的 CU 覆盖前面的. 因此 Pass 1 需要将每个 CU 的系数保存到 per-LCU 缓冲区,
 * Pass 2 重建前恢复到 scratch.
 *
 * per-LCU 缓冲区按 LCU 行布局存储 (stride = lcu_size), CU 的系数偏移:
 *   y_offset = cu_y_in_lcu * lcu_size + cu_x_in_lcu
 * 色度 (420): stride = lcu_size/2, 偏移折半.
 * =================================================================== */

/* 保存 CU 亮度系数从 scratch 到 per-LCU 缓冲区 (按 LCU 行布局) */
static void save_cu_coeffs_y(avs2_frame_ctx *fc, avs2_cu *cu, int x, int y)
{
    avs2_frame *f = fc->fdec;
    int lcu_size = 1 << fc->lcu_size;
    int lcu_sq = lcu_size * lcu_size;  /* per-LCU 系数块大小 */
    /* 使用 TLS lcu_x/lcu_y: Pass 1 (AEC 串行) 与 Pass 2 (重建并行) 重叠执行时,
     * Pass 2 的 avs2_decode_lcu 会覆盖 fc->lcu_x/lcu_y, 导致 save 读取错误位置.
     * TLS 保证每个线程读取自己设置的 LCU 位置. */
    int lcu_idx = tls_lcu_y * f->w_lcu + tls_lcu_x;
    int cu_size = 1 << cu->cu_level;
    int cu_x_in_lcu = x - tls_lcu_x * lcu_size;
    int cu_y_in_lcu = y - tls_lcu_y * lcu_size;
    int bit_size = cu->cu_level;
    coeff_t *lcu_base = fc->coeff_lcu_y + lcu_idx * lcu_sq;

    if (cu->i_tu_split == TU_SPLIT_NON) {
        /* 无 TU 分割: 整块连续存储在 scratch[0..cu_size²-1] */
        if (cu->i_cbp & 0x0F) {
            coeff_t *src = fc->coeff_scratch_y;
            coeff_t *dst = lcu_base + cu_y_in_lcu * lcu_size + cu_x_in_lcu;
            int yy;
            for (yy = 0; yy < cu_size; yy++) {
                memcpy(dst + yy * lcu_size, src + yy * cu_size,
                       cu_size * sizeof(coeff_t));
            }
        }
    } else {
        /* 有 TU 分割: 4 个 TU 块分别存储 */
        int8_t tu_x[4], tu_y[4], tu_w[4], tu_h[4];
        int b8;
        init_transform_units(cu, tu_x, tu_y, tu_w, tu_h);
        for (b8 = 0; b8 < 4; b8++) {
            if (cu->i_cbp & (1 << b8)) {
                coeff_t *src = fc->coeff_scratch_y + (b8 << ((bit_size - 1) << 1));
                coeff_t *dst = lcu_base + (cu_y_in_lcu + tu_y[b8]) * lcu_size
                               + (cu_x_in_lcu + tu_x[b8]);
                int yy;
                for (yy = 0; yy < tu_h[b8]; yy++) {
                    memcpy(dst + yy * lcu_size, src + yy * tu_w[b8],
                           tu_w[b8] * sizeof(coeff_t));
                }
            }
        }
    }
}

/* 保存 CU 色度系数从 scratch 到 per-LCU 缓冲区 (按 LCU/2 行布局) */
static void save_cu_coeffs_c(avs2_frame_ctx *fc, avs2_cu *cu, int x, int y)
{
    avs2_frame *f = fc->fdec;
    int lcu_size = 1 << fc->lcu_size;
    int c_lcu_size = lcu_size >> 1;
    int c_lcu_sq = c_lcu_size * c_lcu_size;
    /* 使用 TLS lcu_x/lcu_y: 避免 Pass 1/Pass 2 重叠时 fc->lcu_x/lcu_y 竞争 */
    int lcu_idx = tls_lcu_y * f->w_lcu + tls_lcu_x;
    int cu_size = 1 << cu->cu_level;
    int c_cu_size = cu_size >> 1;
    int c_cu_x_in_lcu = (x - tls_lcu_x * lcu_size) >> 1;
    int c_cu_y_in_lcu = (y - tls_lcu_y * lcu_size) >> 1;
    int uv;

    for (uv = 0; uv < 2; uv++) {
        if ((cu->i_cbp >> (uv + 4)) & 0x1) {
            coeff_t *src = (uv == 0) ? fc->coeff_scratch_u : fc->coeff_scratch_v;
            coeff_t *lcu_base = ((uv == 0) ? fc->coeff_lcu_u : fc->coeff_lcu_v)
                                + lcu_idx * c_lcu_sq;
            coeff_t *dst = lcu_base + c_cu_y_in_lcu * c_lcu_size + c_cu_x_in_lcu;
            int yy;
            for (yy = 0; yy < c_cu_size; yy++) {
                memcpy(dst + yy * c_lcu_size, src + yy * c_cu_size,
                       c_cu_size * sizeof(coeff_t));
            }
        }
    }
}

/* 从 per-LCU 缓冲区恢复 CU 亮度系数到 scratch (按 LCU 行布局) */
static void load_cu_coeffs_y(avs2_frame_ctx *fc, avs2_cu *cu, int x, int y)
{
    avs2_frame *f = fc->fdec;
    int lcu_size = 1 << fc->lcu_size;
    int lcu_sq = lcu_size * lcu_size;
    /* 使用 TLS lcu_x/lcu_y: Pass 2 中多 worker 并行, 避免 fc->lcu_x/lcu_y 竞争 */
    int lcu_idx = tls_lcu_y * f->w_lcu + tls_lcu_x;
    int cu_size = 1 << cu->cu_level;
    int cu_x_in_lcu = x - tls_lcu_x * lcu_size;
    int cu_y_in_lcu = y - tls_lcu_y * lcu_size;
    int bit_size = cu->cu_level;
    coeff_t *lcu_base = fc->coeff_lcu_y + lcu_idx * lcu_sq;

    /* 仅清零 CU 大小的区域 (reconstruct_residual 只访问 cu_size^2 元素),
     * 避免对小 CU 清零整个 LCU 大小的 scratch (lcu=32 时 2048B vs 128B) */
    memset(tls_coeff_scratch_y, 0, sizeof(coeff_t) * cu_size * cu_size);

    if (cu->i_tu_split == TU_SPLIT_NON) {
        if (cu->i_cbp & 0x0F) {
            coeff_t *dst = tls_coeff_scratch_y;
            coeff_t *src = lcu_base + cu_y_in_lcu * lcu_size + cu_x_in_lcu;
            int yy;
            for (yy = 0; yy < cu_size; yy++) {
                memcpy(dst + yy * cu_size, src + yy * lcu_size,
                       cu_size * sizeof(coeff_t));
            }
        }
    } else {
        int8_t tu_x[4], tu_y[4], tu_w[4], tu_h[4];
        int b8;
        init_transform_units(cu, tu_x, tu_y, tu_w, tu_h);
        for (b8 = 0; b8 < 4; b8++) {
            if (cu->i_cbp & (1 << b8)) {
                coeff_t *dst = tls_coeff_scratch_y + (b8 << ((bit_size - 1) << 1));
                coeff_t *src = lcu_base + (cu_y_in_lcu + tu_y[b8]) * lcu_size
                               + (cu_x_in_lcu + tu_x[b8]);
                int yy;
                for (yy = 0; yy < tu_h[b8]; yy++) {
                    memcpy(dst + yy * tu_w[b8], src + yy * lcu_size,
                           tu_w[b8] * sizeof(coeff_t));
                }
            }
        }
    }
}

/* 从 per-LCU 缓冲区恢复 CU 色度系数到 scratch (按 LCU/2 行布局) */
static void load_cu_coeffs_c(avs2_frame_ctx *fc, avs2_cu *cu, int x, int y)
{
    avs2_frame *f = fc->fdec;
    int lcu_size = 1 << fc->lcu_size;
    int c_lcu_size = lcu_size >> 1;
    int c_lcu_sq = c_lcu_size * c_lcu_size;
    /* 使用 TLS lcu_x/lcu_y: 避免 Pass 2 中 fc->lcu_x/lcu_y 竞争 */
    int lcu_idx = tls_lcu_y * f->w_lcu + tls_lcu_x;
    int cu_size = 1 << cu->cu_level;
    int c_cu_size = cu_size >> 1;
    int c_cu_x_in_lcu = (x - tls_lcu_x * lcu_size) >> 1;
    int c_cu_y_in_lcu = (y - tls_lcu_y * lcu_size) >> 1;
    int uv;

    for (uv = 0; uv < 2; uv++) {
        /* 仅清零 CU 色度大小的区域, 避免清零整个 LCU/2 大小 */
        memset((uv == 0) ? tls_coeff_scratch_u : tls_coeff_scratch_v,
               0, sizeof(coeff_t) * c_cu_size * c_cu_size);
        if ((cu->i_cbp >> (uv + 4)) & 0x1) {
            coeff_t *dst = (uv == 0) ? tls_coeff_scratch_u : tls_coeff_scratch_v;
            coeff_t *lcu_base = ((uv == 0) ? fc->coeff_lcu_u : fc->coeff_lcu_v)
                                + lcu_idx * c_lcu_sq;
            coeff_t *src = lcu_base + c_cu_y_in_lcu * c_lcu_size + c_cu_x_in_lcu;
            int yy;
            for (yy = 0; yy < c_cu_size; yy++) {
                memcpy(dst + yy * c_cu_size, src + yy * c_lcu_size,
                       c_cu_size * sizeof(coeff_t));
            }
        }
    }
}


/* ===================================================================
 * 四叉树递归解码 (对应 davs2 decode_lcu_parse + decode_lcu_recon)
 * =================================================================== */

/* 递归解码 CU: 读取 split flag, 若分割则递归, 否则解码叶节点 CU.
 * pass=0: AEC 解码 + 重建 (单 pass 模式, 帧级并行)
 * pass=1: 仅 AEC 解码 (行级并行 Pass 1)
 * pass=2: 仅重建 (行级并行 Pass 2) */
static void decode_cu_recursive(avs2_frame_ctx *fc, struct avs2_internal *c,
                                avs2_aec *aec, int x, int y, int level,
                                int qp, int *prev_qp, int pass)
{
    avs2_frame *f = fc->fdec;
    avs2_seq_header *seq = c->seq;
    int bit_depth = c->bit_depth;
    int chroma_format = seq->chroma_format;
    int frame_type = fc->slice_type;
    int num_refs = fc->n_refs;
    const int size = 1 << level;

    /* 边界检查 */
    if (x >= f->width || y >= f->height) return;

    /* split flag 解码 (非最小 CU 时). pass>=2 (仅重建) 时从 cu_grid 读取已有 split 信息. */
    int split = 0;
    if (level > MIN_CU_SIZE_IN_BIT) {
        int pix_x_end = x + size;
        int pix_y_end = y + size;
        int b_cu_inside_pic = (pix_x_end <= f->width) && (pix_y_end <= f->height);
        if (b_cu_inside_pic) {
            if (pass < 2) {
                split = aec_read_split_flag(aec, level);
            } else {
                /* pass>=2: 从 8x8 网格推断 split (叶节点 CU 的 cu_level == level) */
                int bx = x >> 3, by = y >> 3;
                avs2_cu *cu = &f->cu_grid[by * f->w8 + bx];
                split = (cu->cu_level < level);
            }
        } else {
            split = 1;  /* 超出图像边界: 强制分割 */
        }
    }

    if (split) {
        /* 递归处理 4 个子块 */
        int half = size >> 1;
        decode_cu_recursive(fc, c, aec, x,       y,       level - 1, qp, prev_qp, pass);
        decode_cu_recursive(fc, c, aec, x + half, y,       level - 1, qp, prev_qp, pass);
        decode_cu_recursive(fc, c, aec, x,       y + half, level - 1, qp, prev_qp, pass);
        decode_cu_recursive(fc, c, aec, x + half, y + half, level - 1, qp, prev_qp, pass);
        return;
    }

    /* ---- 叶节点 CU ---- */
    {
        int bx = x >> 3, by = y >> 3;
        avs2_cu *cu;
        int w = size, h = size;

        if (bx >= f->w8 || by >= f->h8) return;
        cu = &f->cu_grid[by * f->w8 + bx];

        if (pass < 2) {
            /* pass=0 或 pass=1: AEC 解码 CU 信息 */
            if (read_cu_info(fc, aec, cu, level, x, y, frame_type, seq, num_refs,
                             bit_depth, chroma_format, prev_qp, f) < 0) {
                /* 解码错误: 标记 CU 为无效 */
                cu->i_slice_nr = -1;
                return;
            }

            /* pass=1: 保存系数到 per-LCU 缓冲区 (行级并行 Pass 2 需要恢复) */
            if (pass == 1 && fc->coeff_lcu_y) {
                save_cu_coeffs_y(fc, cu, x, y);
                if (f->chroma_format == AVS2_CHROMA_420) {
                    save_cu_coeffs_c(fc, cu, x, y);
                }
            }
        }

        if (pass == 0 || pass == 2 || pass == 3 || pass == 4) {
            /* pass>=2: 从 per-LCU 缓冲区恢复系数到 scratch (行级并行) */
            if (pass >= 2 && fc->coeff_lcu_y) {
                load_cu_coeffs_y(fc, cu, x, y);
                if (f->chroma_format == AVS2_CHROMA_420)
                    load_cu_coeffs_c(fc, cu, x, y);
            }

            /* pass=0/pass=2: 完整重建 (inter+intra)
             * pass=3: 仅 inter 重建 (无行依赖, 可并行)
             * pass=4: 仅 intra 重建 (有行依赖, 串行) */
            if (pass == 3) {
                if (!IS_INTRA_MODE(cu->cu_type)) {
                    reconstruct_cu(fc, c, cu, x, y);
                }
            } else if (pass == 4) {
                if (IS_INTRA_MODE(cu->cu_type)) {
                    reconstruct_cu(fc, c, cu, x, y);
                }
            } else {
                reconstruct_cu(fc, c, cu, x, y);
            }
        }

        /* 存储 MV 到 mvbuf/refbuf (供后续帧的时域直接模式使用).
         * pass=0 (单 pass) 和 pass=1 (AEC 阶段) 需要: MV 在 read_cu_info 中已解码,
         *   立即存入 mvbuf 供后续帧 Phase 1 (时域直接模式) 读取.
         * pass>=2 不需要: pass=1 已存储, 重建阶段不重复. */
        if (pass < 2 && !cu->b_intra) {
            store_mv_to_buf(f, cu, x, y);
        }

        /* 将 CU 信息复制到所有覆盖的 8x8 网格 (pass=0 和 pass=1 需要) */
        if (pass < 2) {
            if (x + w > f->width)  w = f->width - x;
            if (y + h > f->height) h = f->height - y;
            {
                int sx = size >> 3, sy = size >> 3;
                int gx, gy;
                if (sx < 1) sx = 1;
                if (sy < 1) sy = 1;
                for (gy = by; gy < by + sy && gy < f->h8; gy++) {
                    for (gx = bx; gx < bx + sx && gx < f->w8; gx++) {
                        if (gx != bx || gy != by) {
                            f->cu_grid[gy * f->w8 + gx] = *cu;
                        }
                    }
                }
            }
        }
    }
}


/* ===================================================================
 * LCU 解码入口 (对应 davs2 decode_lcu)
 * =================================================================== */

int avs2_decode_lcu(avs2_frame_ctx *fc, struct avs2_internal *c,
                    int lcu_x, int lcu_y, int pass)
{
    avs2_seq_header *seq = c->seq;
    int lcu_size = 1 << seq->log2_lcu_size;
    int x0 = lcu_x * lcu_size;
    int y0 = lcu_y * lcu_size;

    int qp = fc->slice_qp;  /* 对应 davs2 h->i_qp (slice QP) */
    int prev_qp = qp;

    /* 初始化 DSP (首次调用) */
    if (!g_dsp_inited) {
        avs2_dsp_init();
        g_dsp_inited = 1;
    }

    /* 记录 LCU 位置 */
    fc->lcu_x = lcu_x;
    fc->lcu_y = lcu_y;
    tls_lcu_x = lcu_x;  /* TLS: Pass 2 中 load_cu_coeffs 读取 (避免 fc 竞争) */
    tls_lcu_y = lcu_y;
    fc->lcu_size = (int)seq->log2_lcu_size;

    /* 邻域可用性 */
    fc->left_avail  = (lcu_x > 0);
    fc->above_avail = (lcu_y > 0);

    /* 设置活跃系数缓冲区 (AEC 写入, 重建读取).
     * pass>=2 (行级并行 Pass 2/3/4): 使用 TLS scratch (每个 worker 独立, 避免竞争).
     * 其他 pass: 使用 fc->coeff_scratch (Pass 1 串行, pass=0 单线程, 无竞争). */
    if (pass >= 2 && tls_coeff_scratch_y) {
        fc->cur_lcu_coeff_y = tls_coeff_scratch_y;
        fc->cur_lcu_coeff_u = tls_coeff_scratch_u;
        fc->cur_lcu_coeff_v = tls_coeff_scratch_v;
    } else {
        fc->cur_lcu_coeff_y = fc->coeff_scratch_y;
        fc->cur_lcu_coeff_u = fc->coeff_scratch_u;
        fc->cur_lcu_coeff_v = fc->coeff_scratch_v;
    }

    /* Pass 1 (AEC): 读取 SAO/ALF 参数, AEC 解码 CU 信息和系数.
     * pass<2 (单pass/Pass 1) 需要读取, pass>=2 (仅重建) 不需要. */
    if (pass < 2) {
        avs2_aec *aec = fc->aec;
        if (!aec) return AVS2_ERR_INVALID;

        avs2_frame *f = fc->fdec;
        int lcu_xy = lcu_y * f->w_lcu + lcu_x;

        /* LCU 级 SAO 参数读取 (对应 davs2 decode_lcu_row 中 sao_read_lcu_param) */
        if (seq->enable_sao &&
            (fc->slice_sao_on[0] || fc->slice_sao_on[1] || fc->slice_sao_on[2])) {
            avs2_sao_read_param(fc, c, lcu_xy, fc->slice_sao_on,
                                &f->sao_params[lcu_xy * 3]);
        }

        /* LCU 级 ALF 使能标志读取 (对应 davs2 aec_read_alf_lcu_ctrl).
         * 使用 fc->pic_local 而非 c->pic (worker 安全) */
        if (seq->enable_alf) {
            avs2_alf_param *ap = &f->alf_params[lcu_xy];
            ap->alf_enable[0] = (uint8_t)(fc->pic_local.alf_pic_flag_y  ? aec_read_alf_lcu_ctrl(aec) : 0);
            ap->alf_enable[1] = (uint8_t)(fc->pic_local.alf_pic_flag_cb ? aec_read_alf_lcu_ctrl(aec) : 0);
            ap->alf_enable[2] = (uint8_t)(fc->pic_local.alf_pic_flag_cr ? aec_read_alf_lcu_ctrl(aec) : 0);
        }
    }

    /* 递归解码 LCU (pass=0: AEC+重建, pass=1: 仅AEC, pass=2: 仅重建) */
    decode_cu_recursive(fc, c, fc->aec, x0, y0, (int)seq->log2_lcu_size,
                        qp, &prev_qp, pass);

    return AVS2_OK;
}
