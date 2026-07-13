/*
 * 去块滤波 (从 davs2 deblock.cc 移植到 C)。
 *
 * 完整算法:
 *   - 4 级边界强度 (fs = 0..4), 由边界两侧像素平坦度 (FlatnessL/FlatnessR) 决定;
 *   - alpha/beta 阈值表 (按 QP 索引), 位深移位支持 8/10-bit;
 *   - 亮度 8x8 边界 + 色度 4x4 边界, 分别滤波;
 *   - P/F 帧跳过滤波优化 (CBP=0 且 MV 接近 且参考帧相同时跳过)。
 *
 * 命名: 小写加下划线。pel_t -> uint16_t。
 */

#include "internal.h"
#include "aec_internal.h"
#include "tables.h"
#include "quant.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>  /* memset */

/* 辅助宏 (本文件局部使用) */
#ifndef AVS2_MIN
#define AVS2_MIN(a, b)     ((a) < (b) ? (a) : (b))
#endif
#ifndef AVS2_CLIP3
#define AVS2_CLIP3(lo, hi, x) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif

/* --------------------------------------------------------------------------
 * alpha/beta 表 (与 davs2 deblock.cc 完全一致, 按 QP 索引, 值域 0..64)
 * -------------------------------------------------------------------------- */
static const uint8_t alpha_table[64] = {
     0,  0,  0,  0,  0,  0,  1,  1,
     1,  1,  1,  2,  2,  2,  3,  3,
     4,  4,  5,  5,  6,  7,  8,  9,
    10, 11, 12, 13, 15, 16, 18, 20,
    22, 24, 26, 28, 30, 33, 33, 35,
    35, 36, 37, 37, 39, 39, 42, 44,
    46, 48, 50, 52, 53, 54, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64
};

static const uint8_t beta_table[64] = {
     0,  0,  0,  0,  0,  0,  1,  1,
     1,  1,  1,  1,  1,  2,  2,  2,
     2,  2,  3,  3,  3,  3,  4,  4,
     4,  4,  5,  5,  5,  5,  6,  6,
     6,  7,  7,  7,  8,  8,  8,  9,
     9, 10, 10, 11, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22,
    23, 23, 24, 24, 25, 25, 26, 27
};

/* --------------------------------------------------------------------------
 * 边界类型 (滤波控制)
 * -------------------------------------------------------------------------- */
enum edge_type_e {
    EDGE_TYPE_NOFILTER  = 0,  /* 无去块滤波 */
    EDGE_TYPE_ONLY_LUMA = 1,  /* TU 边界 (仅亮度) */
    EDGE_TYPE_BOTH      = 2   /* CU/PU 边界 (亮度+色度) */
};

/* PU 分区方向判定宏 (与 davs2 一致) */
#define IS_HOR_PU_PART(m) ((m) == PRED_2NxN || (m) == PRED_2NxnU || \
                           (m) == PRED_2NxnD || (m) == PRED_I_2Nxn)
#define IS_VER_PU_PART(m) ((m) == PRED_Nx2N || (m) == PRED_nLx2N || \
                           (m) == PRED_nRx2N || (m) == PRED_I_nx2N)

#define MAX_QP_DEBLOCK 63

/* --------------------------------------------------------------------------
 * 设置一条边的滤波标志 (对应 davs2 lf_set_edge_filter_param)
 * dir: EDGE_VER=0 垂直边, EDGE_HOR=1 水平边
 * -------------------------------------------------------------------------- */
static void set_deblock_flag(avs2_frame *f, int i_level, int scu_x, int scu_y,
                             int dir, int edge_type)
{
    const int w_in_scu = f->w8;
    int scu_num = 1 << (i_level - MIN_CU_SIZE_IN_BIT);
    int i;

    if (dir == EDGE_VER) {
        /* 垂直边: 位于 scu_x 左侧, 不能是帧左边界 */
        if (scu_x == 0) return;
        /* 注意: avs2dec 未记录逐 CU 的 slice 号, 跨 slice 边界检查省略
         * (等价于 cross_loop_filter_flag 为真)。 */
        for (i = 0; i < scu_num; i++) {
            int idx = (scu_y + i) * w_in_scu + scu_x;
            if (f->deblock_flags[EDGE_VER][idx] != EDGE_TYPE_NOFILTER)
                break;
            f->deblock_flags[EDGE_VER][idx] = (uint8_t)edge_type;
        }
    } else {
        /* 水平边: 位于 scu_y 上方, 不能是帧上边界 */
        if (scu_y == 0) return;
        for (i = 0; i < scu_num; i++) {
            int idx = scu_y * w_in_scu + scu_x + i;
            if (f->deblock_flags[EDGE_HOR][idx] != EDGE_TYPE_NOFILTER)
                break;
            f->deblock_flags[EDGE_HOR][idx] = (uint8_t)edge_type;
        }
    }
}

/* --------------------------------------------------------------------------
 * 递归设置一个 LCU 内所有边的滤波标志 (对应 davs2 lf_lcu_set_edge_filter)
 * -------------------------------------------------------------------------- */
static void lcu_set_deblock_flag(avs2_frame *f, int i_level, int scu_x, int scu_y)
{
    const int w_in_scu = f->w8;
    const int h_in_scu = f->h8;
    avs2_cu *p_cu = &f->cu_grid[scu_y * w_in_scu + scu_x];
    int i;

    if (p_cu->cu_level < i_level) {
        /* 当前 CU 小于该层级, 递归 4 个子块 */
        for (i = 0; i < 4; i++) {
            int sub_cu_x = scu_x + ((i & 1) << (i_level - MIN_CU_SIZE_IN_BIT - 1));
            int sub_cu_y = scu_y + ((i >> 1) << (i_level - MIN_CU_SIZE_IN_BIT - 1));
            if (sub_cu_x >= w_in_scu || sub_cu_y >= h_in_scu)
                continue;  /* 超出帧范围 */
            lcu_set_deblock_flag(f, i_level - 1, sub_cu_x, sub_cu_y);
        }
    } else {
        /* 设置左/上边界为 CU 边界 (亮度+色度) */
        set_deblock_flag(f, i_level, scu_x, scu_y, EDGE_VER, EDGE_TYPE_BOTH);
        set_deblock_flag(f, i_level, scu_x, scu_y, EDGE_HOR, EDGE_TYPE_BOTH);

        /* 大于 8x8 的 CU 需设置预测/变换内部边界 */
        if (p_cu->cu_level > B8X8_IN_BIT) {
            int shift = i_level - MIN_CU_SIZE_IN_BIT - 1;

            switch (p_cu->cu_type) {
            case PRED_2NxN:
                set_deblock_flag(f, i_level, scu_x, scu_y + (1 << shift), EDGE_HOR, EDGE_TYPE_BOTH);
                break;
            case PRED_Nx2N:
                set_deblock_flag(f, i_level, scu_x + (1 << shift), scu_y, EDGE_VER, EDGE_TYPE_BOTH);
                break;
            case PRED_I_NxN:
                set_deblock_flag(f, i_level, scu_x + (1 << shift), scu_y, EDGE_VER, EDGE_TYPE_BOTH);
                set_deblock_flag(f, i_level, scu_x, scu_y + (1 << shift), EDGE_HOR, EDGE_TYPE_BOTH);
                break;
            case PRED_I_2Nxn:
                if (shift > 0) {
                    set_deblock_flag(f, i_level, scu_x, scu_y + (1 << (shift - 1)),       EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
                    set_deblock_flag(f, i_level, scu_x, scu_y + (1 << (shift - 1)) * 2,   EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
                    set_deblock_flag(f, i_level, scu_x, scu_y + (1 << (shift - 1)) * 3,   EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
                } else {
                    set_deblock_flag(f, i_level, scu_x, scu_y + (1 << shift),             EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
                }
                break;
            case PRED_I_nx2N:
                if (shift > 0) {
                    set_deblock_flag(f, i_level, scu_x + (1 << (shift - 1)),     scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                    set_deblock_flag(f, i_level, scu_x + (1 << (shift - 1)) * 2, scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                    set_deblock_flag(f, i_level, scu_x + (1 << (shift - 1)) * 3, scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                } else {
                    set_deblock_flag(f, i_level, scu_x + (1 << shift),           scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                }
                break;
            case PRED_2NxnU:
                if (shift > 0)
                    set_deblock_flag(f, i_level, scu_x, scu_y + (1 << (shift - 1)), EDGE_HOR, EDGE_TYPE_BOTH);
                break;
            case PRED_2NxnD:
                if (shift > 0)
                    set_deblock_flag(f, i_level, scu_x, scu_y + (1 << (shift - 1)) * 3, EDGE_HOR, EDGE_TYPE_BOTH);
                break;
            case PRED_nLx2N:
                if (shift > 0)
                    set_deblock_flag(f, i_level, scu_x + (1 << (shift - 1)), scu_y, EDGE_VER, EDGE_TYPE_BOTH);
                break;
            case PRED_nRx2N:
                if (shift > 0)
                    set_deblock_flag(f, i_level, scu_x + (1 << (shift - 1)) * 3, scu_y, EDGE_VER, EDGE_TYPE_BOTH);
                break;
            default:
                /* skip/2Nx2N 等模式无需额外内部边界 */
                break;
            }

            /* 变换块边界 */
            if (p_cu->cu_type != PRED_I_NxN &&
                p_cu->i_tu_split != TU_SPLIT_NON && p_cu->i_cbp != 0) {
                /* nsqt 非对称变换分区 (简化: 仅处理通用十字分割) */
                set_deblock_flag(f, i_level, scu_x + (1 << shift), scu_y, EDGE_VER, EDGE_TYPE_ONLY_LUMA);
                set_deblock_flag(f, i_level, scu_x, scu_y + (1 << shift), EDGE_HOR, EDGE_TYPE_ONLY_LUMA);
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * 查找指定像素位置所属 PU 的前向 MV 和参考索引
 * scu    - 8x8 网格中的 CU 副本 (含全部 PU 信息)
 * pix_x  - 像素绝对 x 坐标
 * pix_y  - 像素绝对 y 坐标
 * scu_x  - 8x8 SCU 的 x 索引 (用于推算 CU 原点)
 * scu_y  - 8x8 SCU 的 y 索引
 * -------------------------------------------------------------------------- */
static inline void get_pu_mv_ref(avs2_cu *scu, int pix_x, int pix_y,
                          int scu_x, int scu_y,
                          avs2_mv *mv_out, int8_t *ref_out)
{
    /* 快速路径: 单 PU (最常见) 直接返回, 跳过坐标计算 */
    if (scu->num_pu <= 1) {
        *mv_out = scu->mv[0][0];
        *ref_out = scu->i_ref[0][0];
        return;
    }

    int cu_size = 1 << scu->cu_level;
    int cu_origin_x = (scu_x << MIN_CU_SIZE_IN_BIT) & ~(cu_size - 1);
    int cu_origin_y = (scu_y << MIN_CU_SIZE_IN_BIT) & ~(cu_size - 1);
    int rel_x = pix_x - cu_origin_x;
    int rel_y = pix_y - cu_origin_y;
    int i;

    for (i = 0; i < scu->num_pu; i++) {
        if (rel_x >= scu->pu_x[i] && rel_x < scu->pu_x[i] + scu->pu_w[i] &&
            rel_y >= scu->pu_y[i] && rel_y < scu->pu_y[i] + scu->pu_h[i]) {
            *mv_out = scu->mv[i][0];
            *ref_out = scu->i_ref[i][0];
            return;
        }
    }
    /* 未找到匹配 PU, 回退到 PU 0 */
    *mv_out = scu->mv[0][0];
    *ref_out = scu->i_ref[0][0];
}

/* --------------------------------------------------------------------------
 * 判定是否跳过滤波 (对应 davs2 lf_skip_filter)
 * 返回 0 表示跳过, 1 表示需要滤波
 * dir: EDGE_VER=0 垂直边, EDGE_HOR=1 水平边
 * half: 0=前 4 像素段, 1=后 4 像素段
 * -------------------------------------------------------------------------- */
static uint8_t skip_filter(avs2_frame *f, int frame_type,
                           avs2_cu *scu_p, avs2_cu *scu_q,
                           int scu_x, int scu_y, int dir, int half)
{
    if (frame_type == AVS2_PIC_P || frame_type == AVS2_PIC_F) {
        /* 计算 Q 侧和 L 侧的 SPU 像素位置 */
        int q_pix_x, q_pix_y, l_pix_x, l_pix_y;
        int l_scu_x, l_scu_y;
        avs2_mv mv_q, mv_p;
        int8_t ref_q, ref_p;

        if (dir == EDGE_VER) {
            /* 垂直边: Q 在右, L 在左 */
            q_pix_x = (scu_x << MIN_CU_SIZE_IN_BIT);
            q_pix_y = (scu_y << MIN_CU_SIZE_IN_BIT) + (half << 2);
            l_pix_x = q_pix_x - 4;
            l_pix_y = q_pix_y;
            l_scu_x = scu_x - 1;
            l_scu_y = scu_y;
        } else {
            /* 水平边: Q 在下, L 在上 */
            q_pix_x = (scu_x << MIN_CU_SIZE_IN_BIT) + (half << 2);
            q_pix_y = (scu_y << MIN_CU_SIZE_IN_BIT);
            l_pix_x = q_pix_x;
            l_pix_y = q_pix_y - 4;
            l_scu_x = scu_x;
            l_scu_y = scu_y - 1;
        }

        get_pu_mv_ref(scu_q, q_pix_x, q_pix_y, scu_x, scu_y, &mv_q, &ref_q);
        get_pu_mv_ref(scu_p, l_pix_x, l_pix_y, l_scu_x, l_scu_y, &mv_p, &ref_p);

        if ((scu_p->i_cbp == 0) && (scu_q->i_cbp == 0) &&
            (abs(mv_p.x - mv_q.x) < 4) &&
            (abs(mv_p.y - mv_q.y) < 4) &&
            (ref_p != INVALID_REF && ref_p == ref_q)) {
            return 0;
        }
    }
    return 1;
}

/* --------------------------------------------------------------------------
 * 去块滤波核心 (对应 davs2 lf_edge_core)
 *   src        - 指向边界 Q 侧首像素 R0
 *   b_chroma   - 是否色度 (色度仅迭代 4 次, 且 fs 减 1)
 *   ptr_inc    - 沿边界方向的步长 (垂直边=stride, 水平边=1)
 *   inc1       - 跨边界方向的步长 (垂直边=1, 水平边=stride)
 *   flt_flag[2]- 每 4 像素段的滤波使能标志
 *
 * 4 级边界强度 fs 由 FlatnessL+FlatnessR 决定:
 *   fs=4: 强滤波 (修改 L2..R2 共 6 像素)
 *   fs=3: 次强滤波 (修改 L2,L0,R0,R1)
 *   fs=2: 中等滤波 (修改 L0,R0)
 *   fs=1: 弱滤波 (修改 L0,R0, 3:1 加权)
 *
 * 按 type 实例化 8-bit (uint8_t) 与 10-bit (uint16_t) 两份实现,
 * 避免在 8-bit 帧上按 uint16_t 读写 (每次访问 2 字节) 的错误.
 * -------------------------------------------------------------------------- */
#define DEFINE_DEBLOCK_EDGE(name, type)                                     \
static void name(type *src, int b_chroma, int ptr_inc, int inc1,            \
                 int alpha, int beta, uint8_t *flt_flag)                    \
{                                                                           \
    int inc2 = inc1 << 1;                                                   \
    int inc3 = inc1 + inc2;                                                 \
    int abs_delta;                                                          \
    int L2, L1, L0, R0, R1, R2;                                             \
    int fs;                                                                 \
    int flat_l, flat_r;                                                     \
    int flag;                                                               \
    int pel;                                                                \
                                                                            \
    for (pel = 0; pel < MIN_CU_SIZE; pel++) {                               \
        L2 = src[-inc3];                                                    \
        L1 = src[-inc2];                                                    \
        L0 = src[-inc1];                                                    \
        R0 = src[    0];                                                    \
        R1 = src[ inc1];                                                    \
        R2 = src[ inc2];                                                    \
                                                                            \
        abs_delta = abs(R0 - L0);                                           \
        flag = (pel < 4) ? flt_flag[0] : flt_flag[1];                       \
                                                                            \
        if (flag && (abs_delta < alpha) && (abs_delta > 1)) {               \
            flat_l  = (abs(L1 - L0) < beta) ? 2 : 0;                        \
            flat_l += (abs(L2 - L0) < beta);                                \
                                                                            \
            flat_r  = (abs(R0 - R1) < beta) ? 2 : 0;                        \
            flat_r += (abs(R0 - R2) < beta);                                \
                                                                            \
            switch (flat_l + flat_r) {                                      \
            case 6:                                                         \
                fs = 3 + ((R1 == R0) && (L0 == L1));                        \
                break;                                                      \
            case 5:                                                         \
                fs = 2 + ((R1 == R0) && (L0 == L1));                        \
                break;                                                      \
            case 4:                                                         \
                fs = 1 + (flat_l == 2);                                     \
                break;                                                      \
            case 3:                                                         \
                fs = (abs(L1 - R1) < beta);                                 \
                break;                                                      \
            default:                                                        \
                fs = 0;                                                     \
                break;                                                      \
            }                                                               \
                                                                            \
            fs -= (b_chroma && fs > 0);                                     \
                                                                            \
            switch (fs) {                                                  \
            case 4:                                                         \
                src[-inc1] = (type)((L0 + ((L0 + L2) << 3) + L2 + (R0 << 3) + (R2 << 2) + (R2 << 1) + 16) >> 5); \
                src[-inc2] = (type)(((L0 << 3) - L0 + (L2 << 2) + (L2 << 1) + R0 + (R0 << 1) + 8) >> 4); \
                src[-inc3] = (type)(((L0 << 2) + L2 + (L2 << 1) + R0 + 4) >> 3); \
                src[    0] = (type)((R0 + ((R0 + R2) << 3) + R2 + (L0 << 3) + (L2 << 2) + (L2 << 1) + 16) >> 5); \
                src[ inc1] = (type)(((R0 << 3) - R0 + (R2 << 2) + (R2 << 1) + L0 + (L0 << 1) + 8) >> 4); \
                src[ inc2] = (type)(((R0 << 2) + R2 + (R2 << 1) + L0 + 4) >> 3); \
                break;                                                      \
            case 3:                                                         \
                src[-inc1] = (type)((L2 + (L1 << 2) + (L0 << 2) + (L0 << 1) + (R0 << 2) + R1 + 8) >> 4); \
                src[    0] = (type)((L1 + (L0 << 2) + (R0 << 2) + (R0 << 1) + (R1 << 2) + R2 + 8) >> 4); \
                src[-inc2] = (type)((L2 * 3 + L1 * 8 + L0 * 4 + R0 + 8) >> 4); \
                src[ inc1] = (type)((R2 * 3 + R1 * 8 + R0 * 4 + L0 + 8) >> 4); \
                break;                                                      \
            case 2:                                                         \
                src[-inc1] = (type)(((L1 << 1) + L1 + (L0 << 3) + (L0 << 1) + (R0 << 1) + R0 + 8) >> 4); \
                src[    0] = (type)(((L0 << 1) + L0 + (R0 << 3) + (R0 << 1) + (R1 << 1) + R1 + 8) >> 4); \
                break;                                                      \
            case 1:                                                         \
                src[-inc1] = (type)((L0 * 3 + R0 + 2) >> 2);                \
                src[    0] = (type)((R0 * 3 + L0 + 2) >> 2);                \
                break;                                                      \
            default:                                                        \
                break;                                                      \
            }                                                               \
        }                                                                   \
                                                                            \
        src += ptr_inc;                                                     \
        pel += b_chroma;                                                    \
    }                                                                       \
}

DEFINE_DEBLOCK_EDGE(deblock_edge_8,  uint8_t)
DEFINE_DEBLOCK_EDGE(deblock_edge_16, uint16_t)

#undef DEFINE_DEBLOCK_EDGE

/* --------------------------------------------------------------------------
 * 亮度去块滤波 (对应 davs2 deblock_edge_ver / deblock_edge_hor)
 * dir=EDGE_VER: 垂直边, 沿边向下遍历 (ptr_inc=stride), 跨边左右 (inc1=1)
 * dir=EDGE_HOR: 水平边, 沿边向右遍历 (ptr_inc=1), 跨边上下 (inc1=stride)
 * 按 bit_depth 选择 8/10-bit 实现 (C 回退, SIMD 可覆盖).
 * src 为 void*, stride 为元素步长 (与 bit_depth 匹配). */
void deblock_luma_ver(void *src, int stride, int alpha, int beta,
                      uint8_t *flt_flag, int bit_depth)
{
    if (bit_depth > 8)
        deblock_edge_16((uint16_t *)src, 0, stride, 1, alpha, beta, flt_flag);
    else
        deblock_edge_8((uint8_t *)src, 0, stride, 1, alpha, beta, flt_flag);
}

void deblock_luma_hor(void *src, int stride, int alpha, int beta,
                      uint8_t *flt_flag, int bit_depth)
{
    if (bit_depth > 8)
        deblock_edge_16((uint16_t *)src, 0, 1, stride, alpha, beta, flt_flag);
    else
        deblock_edge_8((uint8_t *)src, 0, 1, stride, alpha, beta, flt_flag);
}

/* --------------------------------------------------------------------------
 * 色度去块滤波 (对应 davs2 deblock_edge_ver_c / deblock_edge_hor_c)
 * 同时滤波 U/V 两个分量 */
void deblock_chroma_ver(void *src_u, void *src_v, int stride,
                        int alpha, int beta, uint8_t *flt_flag, int bit_depth)
{
    if (bit_depth > 8) {
        deblock_edge_16((uint16_t *)src_u, 1, stride, 1, alpha, beta, flt_flag);
        deblock_edge_16((uint16_t *)src_v, 1, stride, 1, alpha, beta, flt_flag);
    } else {
        deblock_edge_8((uint8_t *)src_u, 1, stride, 1, alpha, beta, flt_flag);
        deblock_edge_8((uint8_t *)src_v, 1, stride, 1, alpha, beta, flt_flag);
    }
}

void deblock_chroma_hor(void *src_u, void *src_v, int stride,
                        int alpha, int beta, uint8_t *flt_flag, int bit_depth)
{
    if (bit_depth > 8) {
        deblock_edge_16((uint16_t *)src_u, 1, 1, stride, alpha, beta, flt_flag);
        deblock_edge_16((uint16_t *)src_v, 1, 1, stride, alpha, beta, flt_flag);
    } else {
        deblock_edge_8((uint8_t *)src_u, 1, 1, stride, alpha, beta, flt_flag);
        deblock_edge_8((uint8_t *)src_v, 1, 1, stride, alpha, beta, flt_flag);
    }
}

/* --------------------------------------------------------------------------
 * 对一个 8x8 SCU 的指定方向边执行去块滤波 (对应 davs2 lf_scu_deblock)
 * -------------------------------------------------------------------------- */
static void scu_deblock(avs2_frame_ctx *fc, struct avs2_internal *c,
                        avs2_frame *f, int scu_x, int scu_y, int dir,
                        int frame_type)
{
    const int w_in_scu = f->w8;
    const int scu_xy   = scu_y * w_in_scu + scu_x;
    avs2_cu *scu_q     = &f->cu_grid[scu_xy];
    int edge_condition = f->deblock_flags[dir][scu_xy];

    if (edge_condition == EDGE_TYPE_NOFILTER)
        return;

    {
        const int bps = f->bytes_per_sample;
        const int byte_stride  = (int)f->stride[0];   /* 字节步长 */
        const int byte_stride_c = (int)f->stride[1];
        const int stride    = byte_stride / bps;     /* 元素步长 */
        const int stride_c  = byte_stride_c / bps;
        /* qp_shift: 码流中 QP 包含 8*(bit_depth-8) 的偏移, 需减回后查表.
         * val_shift: alpha/beta 表是 8-bit, 需左移到当前位深. */
        const int qp_shift  = c->bit_depth - 8;
        const int val_shift = c->bit_depth - 8;
        avs2_cu *scu_p = (dir) ? (scu_q - w_in_scu) : (scu_q - 1);
        uint8_t b_filter_flag[2];
        int qp;

        /* skip_filter 仅对 P/F 帧有意义 (检查 MV/ref 差异决定是否跳过).
         * B/I/G/S 帧: 始终滤波, 直接置 1 跳过函数调用开销.
         * (88% B 帧场景下省去 ~467K 次/frame 的函数调用) */
        if (frame_type == AVS2_PIC_P || frame_type == AVS2_PIC_F) {
            b_filter_flag[0] = skip_filter(f, frame_type,
                                           scu_p, scu_q, scu_x, scu_y, dir, 0);
            b_filter_flag[1] = skip_filter(f, frame_type,
                                           scu_p, scu_q, scu_x, scu_y, dir, 1);
            if (!b_filter_flag[0] && !b_filter_flag[1])
                return;  /* 两段都跳过 */
        } else {
            b_filter_flag[0] = 1;
            b_filter_flag[1] = 1;
        }

        /* 亮度边滤波 */
        {
            uint8_t *src_y = f->data[0]
                          + (scu_y << MIN_CU_SIZE_IN_BIT) * byte_stride
                          + (scu_x << MIN_CU_SIZE_IN_BIT) * bps;
            int alpha, beta;
            qp = (scu_p->qp + scu_q->qp + 1) >> 1;  /* 两块 QP 均值 */

            /* 10/12-bit 时 QP 已在配置中加 8*(bit_depth-8), 此处减回 */
            alpha = alpha_table[AVS2_CLIP3(0, MAX_QP_DEBLOCK,
                                          qp - (qp_shift << 3) + fc->pic_local.alpha_offset)] << val_shift;
            beta  = beta_table [AVS2_CLIP3(0, MAX_QP_DEBLOCK,
                                          qp - (qp_shift << 3) + fc->pic_local.beta_offset)] << val_shift;

            avs2_dsp_table.deblock_luma[dir](src_y, stride, alpha, beta,
                                             b_filter_flag, c->bit_depth);
        }

        /* 色度边滤波 (仅 CU/PU 边界, 且在偶数 scu 位置) */
        if (edge_condition == EDGE_TYPE_BOTH &&
            c->seq->chroma_format != AVS2_CHROMA_400) {
            if (((scu_y & 1) == 0 && dir) || (((scu_x & 1) == 0) && (!dir))) {
                int uv_pix_x = scu_x << (MIN_CU_SIZE_IN_BIT - 1);
                int uv_pix_y = scu_y << (MIN_CU_SIZE_IN_BIT - 1);
                uint8_t *src_u = f->data[1] + uv_pix_y * byte_stride_c + uv_pix_x * bps;
                uint8_t *src_v = f->data[2] + uv_pix_y * byte_stride_c + uv_pix_x * bps;
                int delta_cb = fc->pic_local.chroma_quant_param_disable ? 0
                              : fc->pic_local.chroma_quant_param_delta_cb;
                int alpha, beta;

                qp = avs2_chroma_qp(qp, delta_cb, c->bit_depth) - (qp_shift << 3);
                alpha = alpha_table[AVS2_CLIP3(0, MAX_QP_DEBLOCK,
                                              qp + fc->pic_local.alpha_offset)] << val_shift;
                beta  = beta_table [AVS2_CLIP3(0, MAX_QP_DEBLOCK,
                                              qp + fc->pic_local.beta_offset)] << val_shift;

                avs2_dsp_table.deblock_chroma[dir](src_u, src_v, stride_c,
                                                   alpha, beta, b_filter_flag,
                                                   c->bit_depth);
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * 对一个 LCU 执行完整去块滤波 (对应 davs2 davs2_lcu_deblock)
 * 先设置边界标志, 再依次滤波所有垂直边和水平边
 * -------------------------------------------------------------------------- */
void avs2_loop_filter(avs2_frame_ctx *fc, struct avs2_internal *c,
                      avs2_frame *f, int lcu_x, int lcu_y)
{
    const int lcu_level = (int)c->seq->log2_lcu_size;
    const int w_in_scu  = f->w8;
    const int h_in_scu  = f->h8;
    const int num_in_scu = 1 << (lcu_level - MIN_CU_SIZE_IN_BIT);
    int scu_x = lcu_x << (lcu_level - MIN_CU_SIZE_IN_BIT);
    int scu_y = lcu_y << (lcu_level - MIN_CU_SIZE_IN_BIT);
    int num_hor = AVS2_MIN(w_in_scu - scu_x, num_in_scu);
    int num_ver = AVS2_MIN(h_in_scu - scu_y, num_in_scu);
    const int frame_type = (int)fc->pic_local.picture_coding_type;
    int i, j;

    /* 清零当前 LCU 区域的滤波标志 (避免重复解码时的残留) */
    for (j = 0; j < num_ver; j++) {
        memset(&f->deblock_flags[EDGE_VER][(scu_y + j) * w_in_scu + scu_x], 0, num_hor);
        memset(&f->deblock_flags[EDGE_HOR][(scu_y + j) * w_in_scu + scu_x], 0, num_hor);
    }

    /* 设置 LCU 内所有边的滤波标志 */
    lcu_set_deblock_flag(f, lcu_level, scu_x, scu_y);

    /* 垂直边滤波 */
    for (j = 0; j < num_ver; j++)
        for (i = 0; i < num_hor; i++)
            scu_deblock(fc, c, f, scu_x + i, scu_y + j, EDGE_VER, frame_type);

    /* 水平边滤波: 调整起始位置以覆盖上一 LCU 的最后一条水平边 */
    if (scu_x == 0) {
        num_hor--;  /* 首 LCU: 留最后一条边 */
    } else {
        if (scu_x + num_hor == w_in_scu)
            num_hor++;  /* 末 LCU: 多滤一条边 */
        scu_x--;        /* 从上一 LCU 最后一条边开始 */
    }

    for (j = 0; j < num_ver; j++)
        for (i = 0; i < num_hor; i++)
            scu_deblock(fc, c, f, scu_x + i, scu_y + j, EDGE_HOR, frame_type);
}

/* --------------------------------------------------------------------------
 * 去块滤波 DSP 初始化 (对应 davs2 davs2_deblock_init)
 * -------------------------------------------------------------------------- */
void avs2_loopfilter_init(void)
{
    avs2_dsp_table.deblock_luma[EDGE_VER]   = deblock_luma_ver;
    avs2_dsp_table.deblock_luma[EDGE_HOR]   = deblock_luma_hor;
    avs2_dsp_table.deblock_chroma[EDGE_VER] = deblock_chroma_ver;
    avs2_dsp_table.deblock_chroma[EDGE_HOR] = deblock_chroma_hor;
}
