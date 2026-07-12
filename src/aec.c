/*
 * AVS2 算术熵编码 (AEC) 解码器。
 *
 * 从 davs2 的 aec.cc 移植到 C。实现 AVS2 双域 (R 域 / LG 域) 算术解码，
 * 对应 GB/T 33475.2。保留全部上下文模型、概率转移表、MPS/LPS 更新逻辑
 * 以及完整的系数解码 (run-level, TU, EGK, CG 遍历)。
 *
 * 命名约定：小写字母加下划线，字段名与 davs2 原始实现一致。
 */

#include "internal.h"
#include "aec_internal.h"
#include "tables.h"
#include <string.h>
#include <stdio.h>

/**
 * ===========================================================================
 * 辅助宏
 * ===========================================================================
 */
#define AEC_RETURN_ON_ERROR(x) do { if (unlikely(p_aec->b_bit_error)) return (x); } while (0)

#define DAVS2_MAX(a, b)     ((a) > (b) ? (a) : (b))
#define DAVS2_MIN(a, b)     ((a) < (b) ? (a) : (b))
#define DAVS2_ABS(a)        ((a) >= 0 ? (a) : -(a))
#define DAVS2_CLIP3(l, h, v) ((v) < (l) ? (l) : ((v) > (h) ? (h) : (v)))
#define DAVS2_SWAP(a, b)    do { int _t = (a); (a) = (b); (b) = _t; } while (0)

#define CTRL_OPT_AEC 1   /* 是否启用查表方式的 AEC 上下文状态更新 */

/* 分支预测提示: 热路径用 likely, 错误/罕见路径用 unlikely */
#if defined(__GNUC__) || defined(__clang__)
#define likely(x)   __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x)   (x)
#define unlikely(x) (x)
#endif

static const int EO_OFFSET_INV__MAP[] = { 1, 0, 2, -1, 3, 4, 5, 6 };
static const int T_Chr[5] = { 0, 1, 2, 4, 3000 };
static const int8_t tab_rank[6] = { 0, 1, 2, 3, 3, 4 };

static const uint8_t raster2ZZ_4x4[] = {
    0,  1,  5,  6,
    2,  4,  7, 12,
    3,  8, 11, 13,
    9, 10, 14, 15
};

static const uint8_t raster2ZZ_8x8[] = {
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63
};

static const uint8_t raster2ZZ_2x8[] = {
    0, 1, 4, 5,  8,  9, 12, 13,
    2, 3, 6, 7, 10, 11, 14, 15
};

static const uint8_t raster2ZZ_8x2[] = {
    0,  1,
    2,  4,
    3,  5,
    6,  8,
    7,  9,
    10, 12,
    11, 13,
    14, 15
};

static const uint8_t tab_scan_coeff_pos_in_cg[4][4] = {
    { 0,  1,  5,  6 },
    { 2,  4,  7, 12 },
    { 3,  8, 11, 13 },
    { 9, 10, 14, 15 }
};

static const uint8_t tab_cwr[] = { 3, 3, 4, 5, 5, 5, 5 };
static const uint16_t tab_lg_pmps_offset[] = { 0, 0, 0, 197, 95, 46 };

static const int tab_pdir_bskip[DS_MAX_NUM] = {
    PDIR_SYM, PDIR_BID, PDIR_BWD, PDIR_SYM, PDIR_FWD
};

static const int8_t tab_intra_mode_luma2chroma[NUM_INTRA_MODE] = {
    DC_PRED_C,   -1, BI_PRED_C, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    VERT_PRED_C, -1,        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    HOR_PRED_C,  -1,        -1, -1, -1, -1, -1, -1, -1
};

static const int saoclip[NUM_SAO_OFFSET][3] = {
    /* EO: 下界, 上界, 阈值 */
    { -1, 6, 7 },
    {  0, 1, 1 },
    {  0, 0, 0 },
    { -1, 0, 1 },
    { -6, 1, 7 },
    { -7, 7, 7 }  /* BO */
};

/**
 * ===========================================================================
 * 位读取
 * ===========================================================================
 */

/* 读取下一个比特，写入 i_value_t 低位.
 * 热函数: 63/64 次走快路径 (无需重新填充), 用 likely 标注快路径. */
static inline int aec_get_next_bit(avs2_aec *p_aec)
{
    uint32_t next_bit;

    if (unlikely(--p_aec->i_bits_to_go < 0)) {
        int diff = p_aec->i_bytes - p_aec->i_byte_pos;
        uint8_t *p_buffer = p_aec->p_buffer + p_aec->i_byte_pos;

        if (likely(diff > 7)) {
            p_aec->i_byte_buf =
                ((uint64_t)p_buffer[0] << 56) | ((uint64_t)p_buffer[1] << 48) |
                ((uint64_t)p_buffer[2] << 40) | ((uint64_t)p_buffer[3] << 32) |
                ((uint64_t)p_buffer[4] << 24) | ((uint64_t)p_buffer[5] << 16) |
                ((uint64_t)p_buffer[6] <<  8) |  (uint64_t)p_buffer[7];
            p_aec->i_bits_to_go = 63;
            p_aec->i_byte_pos += 8;
        } else if (diff > 0) {
            /* 一帧剩余字节数小于 8，只读一次 */
            int i;
            p_aec->i_bits_to_go += (int8_t)(diff << 3);
            p_aec->i_byte_pos += (p_aec->i_bits_to_go + 1) >> 3;

            p_aec->i_byte_buf = 0;
            for (i = 0; i < diff; i++) {
                p_aec->i_byte_buf = (p_aec->i_byte_buf << 8) | p_buffer[i];
            }
        } else {
            p_aec->b_bit_error = 1;
            return 1;
        }
    }

    /* 取出下一位 (快路径) */
    next_bit = (uint32_t)((p_aec->i_byte_buf >> p_aec->i_bits_to_go) & 0x01);
    p_aec->i_value_t = (p_aec->i_value_t << 1) | next_bit;

    return 0;
}

/* 读取 num_bits 位，写入 i_value_t 低位.
 * 快路径: 缓冲区有足够位, 一次读取. */
static inline int aec_get_next_n_bit(avs2_aec *p_aec, int num_bits)
{
    if (likely(p_aec->i_bits_to_go >= num_bits)) {
        uint32_t next_bits;
        p_aec->i_bits_to_go -= (int8_t)num_bits;
        next_bits = (uint32_t)((p_aec->i_byte_buf >> p_aec->i_bits_to_go) & ((1u << num_bits) - 1));
        p_aec->i_value_t = (p_aec->i_value_t << num_bits) | next_bits;
        return 0;
    } else {
        for (; num_bits != 0; num_bits--) {
            if (unlikely(aec_get_next_bit(p_aec))) {
                return 1;
            }
        }
        return p_aec->b_bit_error;
    }
}

/**
 * ===========================================================================
 * 上下文概率更新
 * ===========================================================================
 */

static inline void update_ctx_mps(aec_ctx *ctx, const uint16_t *tab_ctx_mps)
{
#if CTRL_OPT_AEC
    /* 压缩表查找: 移除 mps 位构建索引, 查表后 OR 回原 mps 位 */
    uint16_t v = ctx->v;
    int idx = (v & 3) | ((v & 0xFFF8) >> 1);
    ctx->v = tab_ctx_mps[idx] | (v & 4);
#else
    uint32_t lg_pmps = ctx->b.lg_pmps;
    uint8_t  cycno   = (uint8_t)ctx->b.cycno;
    uint32_t cwr     = tab_cwr[cycno];

    if (cycno == 0) {
        ctx->b.cycno = 1;
    }
    lg_pmps -= (lg_pmps >> cwr) + (lg_pmps >> (cwr + 2));
    ctx->b.lg_pmps = (uint16_t)lg_pmps;
#endif
}

static inline void update_ctx_lps(aec_ctx *ctx, const uint16_t *tab_ctx_lps)
{
#if CTRL_OPT_AEC
    /* 压缩表查找: 移除 mps 位构建索引, 查表后 XOR 原 mps 位
     * (LPS 中 mps 翻转与输入 mps 无关, 存储的是 mps=0 的结果) */
    uint16_t v = ctx->v;
    int idx = (v & 3) | ((v & 0xFFF8) >> 1);
    ctx->v = tab_ctx_lps[idx] ^ (v & 4);
#else
    uint32_t cycno   = ctx->b.cycno;
    uint32_t cwr     = tab_cwr[cycno];
    uint32_t lg_pmps = ctx->b.lg_pmps + tab_lg_pmps_offset[cwr];
    uint32_t mps     = ctx->b.mps;

    if (cycno != 3) {
        ++cycno;
    }

    if (lg_pmps >= (256 << LG_PMPS_SHIFTNO)) {
        lg_pmps = (512 << LG_PMPS_SHIFTNO) - 1 - lg_pmps;
        mps = !mps;
    }

    ctx->v = MAKE_CONTEXT(lg_pmps, mps, cycno);
#endif
}

/* 初始化概率转移表 (查表优化用).
 * 使用 g_tab_initialized 做线程安全的懒初始化.
 * 初始化完成后表内容只读, 多线程读取安全. */
void aec_init_context_tab(uint16_t aec_tab_ctx_mps[4 * 2048], uint16_t aec_tab_ctx_lps[4 * 2048])
{
#if CTRL_OPT_AEC
    int cycno;

    memset(aec_tab_ctx_mps, 0, sizeof(uint16_t) * 4 * 2048);
    memset(aec_tab_ctx_lps, 0, sizeof(uint16_t) * 4 * 2048);

    /* MPS 转移: 仅填充 mps=0 的条目, 查找时 OR 回实际 mps 位 */
    for (cycno = 0; cycno < 4; cycno++) {
        uint32_t cwr = tab_cwr[cycno];
        uint32_t new_cycno = DAVS2_MAX(cycno, 1);
        int lg_pmps;
        for (lg_pmps = 0; lg_pmps <= 1024; lg_pmps++) {
            uint32_t new_lg = (uint32_t)lg_pmps - ((uint32_t)lg_pmps >> cwr) - ((uint32_t)lg_pmps >> (cwr + 2));
            int idx = cycno | (lg_pmps << 2);
            aec_tab_ctx_mps[idx] = MAKE_CONTEXT(new_lg, 0, new_cycno);
        }
    }

    /* LPS 转移: 仅填充 mps=0 的条目, 存储翻转后的 mps, 查找时 XOR 原 mps 位 */
    for (cycno = 0; cycno < 4; cycno++) {
        uint32_t cwr = tab_cwr[cycno];
        uint32_t new_cycno = DAVS2_MIN(cycno + 1, 3);
        int lg_pmps;
        for (lg_pmps = 0; lg_pmps <= 1024; lg_pmps++) {
            uint32_t new_lg = (uint32_t)lg_pmps + tab_lg_pmps_offset[cwr];
            uint32_t new_mps = 0;
            if (new_lg >= (256 << LG_PMPS_SHIFTNO)) {
                new_lg = (512 << LG_PMPS_SHIFTNO) - 1 - new_lg;
                new_mps = 1;
            }
            int idx = cycno | (lg_pmps << 2);
            aec_tab_ctx_lps[idx] = MAKE_CONTEXT(new_lg, new_mps, new_cycno);
        }
    }
#else
    (void)0;
#endif
}

/**
 * ===========================================================================
 * AEC 初始化与状态查询
 * ===========================================================================
 */

/* 初始化算术解码器 (表已在 avs2_aec_create 中初始化, 无需再调) */
int aec_start_decoding(avs2_aec *p_aec, uint8_t *p_start, int i_byte_pos, int i_bytes)
{
    p_aec->p_buffer     = p_start;
    p_aec->i_byte_pos   = i_byte_pos;
    p_aec->i_bytes      = i_bytes;
    p_aec->i_bits_to_go = 0;
    p_aec->b_bit_error  = 0;
    p_aec->b_val_domain = 1;
    p_aec->i_s1         = 0;
    p_aec->i_t1         = QUARTER - 1; /* 0xff */
    p_aec->i_value_s    = 0;
    p_aec->i_value_t    = 0;

    if (p_aec->i_bits_to_go < B_BITS - 1) {
        if (aec_get_next_n_bit(p_aec, B_BITS - 1)) {
            return 0;
        }
    }

    return 0;
}

int aec_bits_read(avs2_aec *p_aec)
{
    return (p_aec->i_byte_pos << 3) - p_aec->i_bits_to_go;
}

/**
 * ===========================================================================
 * 核心算术解码原语
 * ===========================================================================
 */

/* AEC 归一化快速检查: 返回 1 表示需要归一化.
 * 内联到各 biari_decode_* 调用方, 避免常见路径 (无需归一化) 的函数调用开销.
 *   需要归一化: 处于 R 域 (b_val_domain!=0) 或 LG 域区间耗尽
 *               (i_s1==AEC_VALUE_BOUND && b_val_bound) */
static inline int aec_renormalize_needed(const avs2_aec *p_aec)
{
    return p_aec->b_val_domain != 0 ||
           (p_aec->i_s1 == AEC_VALUE_BOUND && p_aec->b_val_bound != 0);
}

/* AEC 归一化慢速路径: 实际执行归一化工作.
 * 调用方应先用 aec_renormalize_needed() 做快速检查, 仅在需要时调用本函数.
 * 用 clz 批量读取比特替代逐比特循环.
 * \param p_aec       AEC 状态
 * \param pi_value_s  指向 value_s (可能是局部或 p_aec->i_value_s)
 * \return 0=成功, 1=错误 */
static inline int aec_renormalize_slow(avs2_aec *p_aec, uint32_t *pi_value_s)
{
    *pi_value_s = 0;
    p_aec->i_s1 = 0;

    /* clz 快速归一化: i_value_t != 0 时精确计算移位位数 */
    if (likely(p_aec->i_value_t != 0)) {
        int n_bits = clz32(p_aec->i_value_t) - 23;
        if (n_bits < 0) n_bits = 0;
        if (n_bits > 0) {
            if (unlikely(aec_get_next_n_bit(p_aec, n_bits))) {
                return 1;
            }
            *pi_value_s += n_bits;
        }
    } else {
        /* i_value_t == 0: 回退到逐比特循环 */
        while (p_aec->i_value_t < QUARTER && *pi_value_s < AEC_VALUE_BOUND) {
            if (unlikely(aec_get_next_bit(p_aec))) {
                return 1;
            }
            (*pi_value_s)++;
        }
    }

    p_aec->b_val_bound = (p_aec->i_value_t < QUARTER);
    p_aec->i_value_t   = p_aec->i_value_t & 0xff;
    return 0;
}

/* 带上下文的双域算术解码.
 * 优化: (A) MPS 快速路径 + 显式位移替代位域访问; (C) 内联归一化检查.
 * 位域映射: lg_pmps = v >> 5 (字段 lg_pmps(11位, 起始位3) >> LG_PMPS_SHIFTNO=2),
 *           mps = (v >> 2) & 1. 显式位移避免位域实现的编译器依赖开销. */
static inline int biari_decode_symbol(avs2_aec *p_aec, aec_ctx *ctx)
{
    uint32_t v = ctx->v;
    uint32_t lg_pmps = v >> 5;            /* ctx->b.lg_pmps >> LG_PMPS_SHIFTNO */
    int bit = (int)((v >> 2) & 1);        /* ctx->b.mps */
    uint32_t i_value_s = p_aec->i_value_s;
    uint32_t s_flag;
    uint32_t s2;
    uint32_t t2;
    int is_LPS;

    /* 内联归一化快速检查 (方案 C): 常见路径无需归一化, 跳过函数调用 */
    if (unlikely(aec_renormalize_needed(p_aec))) {
        if (unlikely(aec_renormalize_slow(p_aec, &i_value_s))) {
            return 0;
        }
    }

    if (unlikely(i_value_s > AEC_VALUE_BOUND)) {
        p_aec->b_bit_error = 1;
        p_aec->i_value_s   = i_value_s;
        return 0;
    }

    s_flag = p_aec->i_t1 < lg_pmps;
    s2     = p_aec->i_s1 + s_flag;
    t2     = p_aec->i_t1 - lg_pmps + (s_flag << 8); /* 8 位 */
    is_LPS = (s2 > i_value_s || (s2 == i_value_s && p_aec->i_value_t >= t2)) && p_aec->b_val_bound == 0;

    p_aec->b_val_domain = is_LPS;

    if (likely(!is_LPS)) {     /* MPS 快速路径 (方案 A): 80%+ 的 bin 走此路径 */
        p_aec->i_s1 = s2;
        p_aec->i_t1 = t2;
        p_aec->i_value_s = i_value_s;
        update_ctx_mps(ctx, p_aec->tab_ctx_mps);
        return bit;
    }

    /* LPS 慢速路径 */
    {
        uint32_t t_rlps = (s_flag == 0) ? (lg_pmps) : (p_aec->i_t1 + lg_pmps);
        int n_bits;
        bit = !bit;

        if (s2 == i_value_s) {
            p_aec->i_value_t -= t2;
        } else {
            if (unlikely(aec_get_next_bit(p_aec))) {
                return 0;
            }
            p_aec->i_value_t += 256 - t2;
        }

        /* 恢复区间: clz 一次计算移位位数 (t_rlps 是确定性值) */
        n_bits = (t_rlps != 0) ? (clz32(t_rlps) - 23) : 0;
        if (n_bits < 0) n_bits = 0;
        t_rlps <<= n_bits;
        if (n_bits) {
            if (unlikely(aec_get_next_n_bit(p_aec, n_bits))) {
                return 0;
            }
        }

        p_aec->i_s1 = 0;
        p_aec->i_t1 = t_rlps & 0xff;
        p_aec->i_value_s = i_value_s;
        update_ctx_lps(ctx, p_aec->tab_ctx_lps);
    }

    return bit;
}

/* 等概率解码 (无上下文，无更新) */
static inline int biari_decode_symbol_eq_prob(avs2_aec *p_aec)
{
    if (unlikely(p_aec->b_val_domain != 0 || (p_aec->i_s1 == AEC_VALUE_BOUND && p_aec->b_val_bound != 0))) {
        p_aec->i_s1 = 0;

        if (unlikely(aec_get_next_bit(p_aec))) {
            return 0;
        }

        if (p_aec->i_value_t >= (256 + p_aec->i_t1)) {  /* LPS */
            p_aec->i_value_t -= (256 + p_aec->i_t1);
            return 1;
        } else {
            return 0;
        }
    } else {
        uint32_t s2 = p_aec->i_s1 + 1;
        uint32_t t2 = p_aec->i_t1;
        int is_LPS = s2 > p_aec->i_value_s || ((s2 == p_aec->i_value_s && p_aec->i_value_t >= t2) && p_aec->b_val_bound == 0);

        p_aec->b_val_domain = is_LPS;

        if (unlikely(is_LPS)) {    /* LPS */
            if (s2 == p_aec->i_value_s) {
                p_aec->i_value_t -= t2;
            } else {
                if (unlikely(aec_get_next_bit(p_aec))) {
                    return 0;
                }
                p_aec->i_value_t += 256 - t2;
            }
            return 1;
        } else {
            p_aec->i_s1 = s2;
            p_aec->i_t1 = t2;
            return 0;
        }
    }
}

/* 终止符号解码 (固定 lg_pmps=1).
 * 优化: (C) 内联归一化检查. */
static inline int biari_decode_final(avs2_aec *p_aec)
{
    const uint32_t lg_pmps = 1;
    uint32_t t2;
    uint32_t s2;
    uint32_t s_flag;
    int is_LPS;

    /* 内联归一化快速检查 (方案 C) */
    if (unlikely(aec_renormalize_needed(p_aec))) {
        if (unlikely(aec_renormalize_slow(p_aec, &p_aec->i_value_s))) {
            return 0;
        }
    }

    s_flag = p_aec->i_t1 < lg_pmps;
    s2 = p_aec->i_s1 + s_flag;
    t2 = p_aec->i_t1 - lg_pmps + (s_flag << 8); /* 8 位 */

    /* 比较值 */
    is_LPS = (s2 > p_aec->i_value_s || (s2 == p_aec->i_value_s && p_aec->i_value_t >= t2)) && p_aec->b_val_bound == 0;
    p_aec->b_val_domain = is_LPS;

    if (unlikely(is_LPS)) {     /* LPS */
        /* t_rlps=1: clz(1)=31, n_bits=31-23=8 */
        int n_bits = 8;

        if (s2 == p_aec->i_value_s) {
            p_aec->i_value_t -= t2;
        } else {
            if (unlikely(aec_get_next_bit(p_aec))) {
                return 0;
            }
            p_aec->i_value_t += 256 - t2;
        }

        if (n_bits) {
            if (unlikely(aec_get_next_n_bit(p_aec, n_bits))) {
                return 0;
            }
        }

        p_aec->i_s1 = 0;
        p_aec->i_t1 = 0;
    } else {        /* MPS */
        p_aec->i_s1 = s2;
        p_aec->i_t1 = t2;
    }

    return is_LPS;
}

/* 使用同一上下文连续解码，直到得到 0 或达到 max_num.
 * 优化: (A) MPS 快速路径 + 显式位移; (C) 内联归一化检查. */
static inline int biari_decode_symbol_continue0(avs2_aec *p_aec, aec_ctx *ctx, int max_num)
{
    uint32_t i_value_s = p_aec->i_value_s;
    int bit = 0;
    int i;

    for (i = 0; i < max_num && !bit; i++) {
        uint32_t v = ctx->v;
        uint32_t lg_pmps = v >> 5;          /* ctx->b.lg_pmps >> LG_PMPS_SHIFTNO */
        uint32_t t2;
        uint32_t s2;
        uint32_t s_flag;
        int is_LPS;

        bit = (int)((v >> 2) & 1);          /* ctx->b.mps */

        /* 内联归一化快速检查 (方案 C) */
        if (unlikely(aec_renormalize_needed(p_aec))) {
            if (unlikely(aec_renormalize_slow(p_aec, &i_value_s))) {
                return 0;
            }
        }

        s_flag = p_aec->i_t1 < lg_pmps;
        s2 = p_aec->i_s1 + s_flag;
        t2 = p_aec->i_t1 - lg_pmps + (s_flag << 8); /* 8 位 */

        if (unlikely(i_value_s > AEC_VALUE_BOUND)) {
            p_aec->b_bit_error = 1;
            return 0;
        }

        is_LPS = (s2 > i_value_s || (s2 == i_value_s && p_aec->i_value_t >= t2)) && p_aec->b_val_bound == 0;
        p_aec->b_val_domain = is_LPS;

        if (likely(!is_LPS)) {     /* MPS 快速路径 (方案 A) */
            p_aec->i_s1 = s2;
            p_aec->i_t1 = t2;
            update_ctx_mps(ctx, p_aec->tab_ctx_mps);
        } else {     /* LPS */
            uint32_t t_rlps = (s_flag == 0) ? (lg_pmps) : (p_aec->i_t1 + lg_pmps);
            int n_bits;
            bit = !bit;

            if (s2 == i_value_s) {
                p_aec->i_value_t -= t2;
            } else {
                if (unlikely(aec_get_next_bit(p_aec))) {
                    return 0;
                }
                p_aec->i_value_t += 256 - t2;
            }

            /* clz 计算 LPS 移位位数 */
            n_bits = (t_rlps != 0) ? (clz32(t_rlps) - 23) : 0;
            if (n_bits < 0) n_bits = 0;
            t_rlps <<= n_bits;
            if (n_bits) {
                if (unlikely(aec_get_next_n_bit(p_aec, n_bits))) {
                    return 0;
                }
            }

            p_aec->i_s1 = 0;
            p_aec->i_t1 = t_rlps & 0xff;
            update_ctx_lps(ctx, p_aec->tab_ctx_lps);
        }
    }

    p_aec->i_value_s = i_value_s;
    return i - bit;
}

/* 连续解码，上下文按 ctx_add 递增 (最多 max_ctx_inc).
 * 优化: (A) MPS 快速路径 + 显式位移; (C) 内联归一化检查. */
static inline int biari_decode_symbol_continu0_ext(avs2_aec *p_aec, aec_ctx *ctx, int max_ctx_inc, int max_num)
{
    int bit = 0;
    int i;

    for (i = 0; i < max_num && !bit; i++) {
        int ctx_add = DAVS2_MIN(i, max_ctx_inc);
        aec_ctx *p_ctx = ctx + ctx_add;
        uint32_t v = p_ctx->v;
        uint32_t lg_pmps = v >> 5;          /* p_ctx->b.lg_pmps >> LG_PMPS_SHIFTNO */
        uint32_t t2;
        uint32_t s2;
        int is_LPS;
        int s_flag;

        bit = (int)((v >> 2) & 1);          /* p_ctx->b.mps */

        /* 内联归一化快速检查 (方案 C) */
        if (unlikely(aec_renormalize_needed(p_aec))) {
            if (unlikely(aec_renormalize_slow(p_aec, &p_aec->i_value_s))) {
                return 0;
            }
        }

        s_flag = p_aec->i_t1 < lg_pmps;
        s2 = p_aec->i_s1 + s_flag;
        t2 = p_aec->i_t1 - lg_pmps + (s_flag << 8); /* 8 位 */

        if (unlikely(p_aec->i_value_s > AEC_VALUE_BOUND)) {
            p_aec->b_bit_error = 1;
            return 0;
        }

        is_LPS = (s2 > p_aec->i_value_s || (s2 == p_aec->i_value_s && p_aec->i_value_t >= t2)) && p_aec->b_val_bound == 0;
        p_aec->b_val_domain = is_LPS;

        if (likely(!is_LPS)) {     /* MPS 快速路径 (方案 A) */
            p_aec->i_s1 = s2;
            p_aec->i_t1 = t2;
            update_ctx_mps(p_ctx, p_aec->tab_ctx_mps);
        } else {     /* LPS */
            uint32_t t_rlps = (s_flag == 0) ? (lg_pmps) : (p_aec->i_t1 + lg_pmps);
            int n_bits;
            bit = !bit;

            if (s2 == p_aec->i_value_s) {
                p_aec->i_value_t -= t2;
            } else {
                if (unlikely(aec_get_next_bit(p_aec))) {
                    return 0;
                }
                p_aec->i_value_t += 256 - t2;
            }

            /* clz 计算移位位数, 批量读取替换逐比特循环 */
            n_bits = (t_rlps != 0) ? (clz32(t_rlps) - 23) : 0;
            if (n_bits < 0) n_bits = 0;
            t_rlps <<= n_bits;
            if (n_bits) {
                if (unlikely(aec_get_next_n_bit(p_aec, n_bits))) {
                    return 0;
                }
            }

            p_aec->i_s1 = 0;
            p_aec->i_t1 = t_rlps & 0xff;
            update_ctx_lps(p_ctx, p_aec->tab_ctx_lps);
        }
    }

    return i - bit;
}

/* 一元二值化解码: 首位用 ctx，后续用 ctx+ctx_offset，max_symbol 不带终止 0 */
static int unary_bin_max_decode(avs2_aec *p_aec, aec_ctx *ctx, int ctx_offset, int max_symbol)
{
    int symbol = biari_decode_symbol(p_aec, ctx);

    if (symbol == 1) {
        return 0;
    } else {
        if (max_symbol == 1) {
            return symbol;
        } else {
            aec_ctx *p_ctx = ctx + ctx_offset;
            symbol = 1 + biari_decode_symbol_continue0(p_aec, p_ctx, max_symbol - 1);
            return symbol;
        }
    }
}

/**
 * ===========================================================================
 * 上下文初始化
 * ===========================================================================
 */

void aec_init_contexts(avs2_aec *p_aec)
{
    const uint16_t lg_pmps = (uint16_t)((QUARTER << LG_PMPS_SHIFTNO) - 1);
    uint16_t v = (uint16_t)MAKE_CONTEXT(lg_pmps, 0, 0);
    uint16_t *d = (uint16_t *)&p_aec->syn_ctx;
    int ctx_cnt = (int)(sizeof(aec_context_set) / sizeof(uint16_t));

    while (ctx_cnt-- != 0) {
        *d++ = v;
    }
}

void aec_new_slice(struct avs2_internal *c)
{
    (void)c;
    /* davs2 中重置 h->i_last_dquant = 0；本实现由调用方跟踪 delta qp */
}

/**
 * ===========================================================================
 * 公共 API
 * ===========================================================================
 */

avs2_aec *avs2_aec_create(const uint16_t *aec_tab_ctx_mps, const uint16_t *aec_tab_ctx_lps)
{
    avs2_aec *a = (avs2_aec *)avs2_mem_allocz(sizeof(*a));
    if (!a) return NULL;

#if CTRL_OPT_AEC
    /* 存储表指针, 供 update_ctx_mps/lps 通过 AEC 状态访问 */
    a->tab_ctx_mps = aec_tab_ctx_mps;
    a->tab_ctx_lps = aec_tab_ctx_lps;
#endif

    return a;
}

void avs2_aec_destroy(avs2_aec *aec)
{
    avs2_mem_free(aec);
}

void avs2_aec_init_contexts(avs2_aec *aec, int slice_type)
{
    (void)slice_type;
    aec_init_contexts(aec);
}

int avs2_aec_start_decoding(avs2_aec *aec, const uint8_t *buf, int sz, int bit_pos)
{
    aec_start_decoding(aec, (uint8_t *)buf, bit_pos >> 3, sz);
    return aec->b_bit_error ? AVS2_ERR_INVALID : AVS2_OK;
}

int avs2_aec_decode_bin(avs2_aec *aec, void *ctx)
{
    return biari_decode_symbol(aec, (aec_ctx *)ctx);
}

int avs2_aec_decode_bin_eq_prob(avs2_aec *aec)
{
    return biari_decode_symbol_eq_prob(aec);
}

int avs2_aec_decode_final(avs2_aec *aec)
{
    return biari_decode_final(aec);
}

unsigned avs2_aec_decode_ue(avs2_aec *aec, void *ctx)
{
    return (unsigned)avs2_aec_decode_bin(aec, ctx);
}

int avs2_aec_decode_se(avs2_aec *aec, void *ctx)
{
    unsigned v = avs2_aec_decode_ue(aec, ctx);
    int r = (int)((v + 1) >> 1);
    if ((v & 1) == 0) r = -r;
    return r;
}

int avs2_aec_get_bits_read(avs2_aec *aec)
{
    return aec_bits_read(aec);
}

/**
 * ===========================================================================
 * 高层 AEC 解码函数 (自包含)
 * ===========================================================================
 */

/* DMH 模式解码 */
int aec_read_dmh_mode(avs2_aec *p_aec, int i_cu_level)
{
    aec_ctx *p_ctx = p_aec->syn_ctx.pu_type_index + (i_cu_level - 3) * 3 + NUM_INTER_DIR_DHP_CTX;

    if (biari_decode_symbol(p_aec, p_ctx) == 0) {
        return 0;
    } else {
        if (biari_decode_symbol(p_aec, p_ctx + 1) == 0) {
            return 3 + biari_decode_symbol_eq_prob(p_aec);    /* 3,4: 一元码 10x */
        } else {
            if (biari_decode_symbol(p_aec, p_ctx + 2) == 0) {
                return 7 + biari_decode_symbol_eq_prob(p_aec);    /* 7,8: 一元码 110x */
            } else {
                /* 1,2 对应一元码 1110x；5,6 对应一元码 1111x */
                int b3 = biari_decode_symbol_eq_prob(p_aec);
                int b4 = biari_decode_symbol_eq_prob(p_aec);
                return 1 + (b3 << 2) + b4;
            }
        }
    }
}

/* 运动矢量差解码 (单分量) */
static int aec_read_mvd(avs2_aec *p_aec, aec_ctx *p_ctx)
{
    int binary_symbol = 0;
    int golomb_order = 0;
    int act_sym;

    if (!biari_decode_symbol(p_aec, p_ctx + 0)) {
        act_sym = 0;
    } else if (!biari_decode_symbol(p_aec, p_ctx + 1)) {
        act_sym = 1;
    } else if (!biari_decode_symbol(p_aec, p_ctx + 2)) {
        act_sym = 2;
    } else {   /* 1110 */
        int add_sym = biari_decode_symbol_eq_prob(p_aec);
        act_sym = 0;

        for (;;) {
            int l = biari_decode_symbol_eq_prob(p_aec);
            AEC_RETURN_ON_ERROR(0);
            if (l == 0) {
                act_sym += (1 << golomb_order);
                golomb_order++;
            } else {
                break;
            }
        }

        while (golomb_order--) {
            if (biari_decode_symbol_eq_prob(p_aec)) {
                binary_symbol |= (1 << golomb_order);
            }
        }

        act_sym += binary_symbol;
        act_sym = (act_sym << 1) + 3 + add_sym;
    }

    if (act_sym != 0) {
        if (biari_decode_symbol_eq_prob(p_aec)) {
            act_sym = -act_sym;
        }
    }

    return act_sym;
}

/* 解码两个分量的运动矢量差 */
void aec_read_mvds(avs2_aec *p_aec, avs2_mv *p_mvd)
{
    p_mvd->x = aec_read_mvd(p_aec, p_aec->syn_ctx.mvd_contexts[0]);
    p_mvd->y = aec_read_mvd(p_aec, p_aec->syn_ctx.mvd_contexts[1]);
}

/* 加权跳过模式解码 */
static int aec_read_wpm(avs2_aec *p_aec, int num_of_references)
{
    aec_ctx *p_ctx = p_aec->syn_ctx.weighted_skip_mode;
    return biari_decode_symbol_continu0_ext(p_aec, p_ctx, 2, num_of_references - 1);
}

/* 直接跳过模式方向解码 */
static int aec_read_dir_skip_mode(avs2_aec *p_aec)
{
    aec_ctx *p_ctx = p_aec->syn_ctx.cu_subtype_index;
    int act_sym = biari_decode_symbol_continu0_ext(p_aec, p_ctx, 32768, 3);
    if (act_sym == 3) {
        act_sym += (!biari_decode_symbol(p_aec, p_ctx + 3));
    }
    return act_sym;
}

/* intra 亮度预测模式解码 */
int aec_read_intra_pmode(avs2_aec *p_aec)
{
    aec_ctx *p_ctx = p_aec->syn_ctx.intra_luma_pred_mode;
    int symbol;

    if (biari_decode_symbol(p_aec, p_ctx) == 1) {
        symbol = biari_decode_symbol(p_aec, p_ctx + 6) - 2;
    } else {
        symbol  = biari_decode_symbol(p_aec, p_ctx + 1) << 4;
        symbol += biari_decode_symbol(p_aec, p_ctx + 2) << 3;
        symbol += biari_decode_symbol(p_aec, p_ctx + 3) << 2;
        symbol += biari_decode_symbol(p_aec, p_ctx + 4) << 1;
        symbol += biari_decode_symbol(p_aec, p_ctx + 5);
    }

    return symbol;
}

/* delta QP 解码 */
int aec_read_cu_delta_qp(avs2_aec *p_aec, int i_last_dequant)
{
    aec_ctx *p_ctx = p_aec->syn_ctx.delta_qp_contexts;
    int act_sym;
    int dquant;

    act_sym = 1 - biari_decode_symbol(p_aec, p_ctx + (!!i_last_dequant));
    if (act_sym != 0) {
        act_sym = unary_bin_max_decode(p_aec, p_ctx + 2, 1, 256) + 1;
    }

    /* cu_qp_delta 范围: (-32-4*(BitDepth-8)) ~ (32+4*(BitDepth-8)) */
    dquant = (act_sym + 1) >> 1;
    if ((act_sym & 0x01) == 0) {    /* LSB 为符号位 */
        dquant = -dquant;
    }

    return dquant;
}

/* slice 结束判定 */
int aec_startcode_follows(avs2_aec *p_aec, int eos_bit)
{
    int bit = 0;

    if (eos_bit) {
        bit = biari_decode_final(p_aec);
    }

    return bit;
}

/* ALF LCU 控制标志解码 */
int aec_read_alf_lcu_ctrl(avs2_aec *p_aec)
{
    aec_ctx *ctx = p_aec->syn_ctx.alf_lcu_enable_scmodel;
    return biari_decode_symbol(p_aec, ctx);
}

/* split flag 解码 */
int aec_read_split_flag(avs2_aec *p_aec, int i_level)
{
    aec_ctx *p_ctx = p_aec->syn_ctx.cu_split_flag + (i_level - MIN_CU_SIZE_IN_BIT - 1);
    int split_flag = biari_decode_symbol(p_aec, p_ctx);
    return split_flag;
}

/* SAO merge flag 解码 */
static int read_sao_mergeflag(avs2_aec *p_aec, int act_ctx)
{
    int act_sym = 0;

    if (act_ctx == 1) {
        act_sym = biari_decode_symbol(p_aec, &p_aec->syn_ctx.sao_mergeflag_context[0]);
    } else if (act_ctx == 2) {
        act_sym = biari_decode_symbol(p_aec, &p_aec->syn_ctx.sao_mergeflag_context[1]);
        if (act_sym != 1) {
            act_sym += (biari_decode_symbol(p_aec, &p_aec->syn_ctx.sao_mergeflag_context[2]) << 1);
        }
    }

    return act_sym;
}

int aec_read_sao_mergeflag(avs2_aec *p_aec, int mergeleft_avail, int mergeup_avail)
{
    int merge_left  = 0;
    int merge_top   = 0;
    int merge_index = read_sao_mergeflag(p_aec, mergeleft_avail + mergeup_avail);

    if (mergeleft_avail) {
        merge_left  = merge_index & 0x01;
        merge_index = merge_index >> 1;
    }
    if (mergeup_avail && !merge_left) {
        merge_top = merge_index & 0x01;
    }

    return (merge_left << 1) + merge_top;
}

/* SAO 模式解码 */
int aec_read_sao_mode(avs2_aec *p_aec)
{
    int t2 = !biari_decode_symbol(p_aec, p_aec->syn_ctx.sao_mode_context);
    int act_sym;

    if (t2) {
        int t1 = !biari_decode_symbol_eq_prob(p_aec);
        act_sym = t2 + (t1 << 1);
    } else {
        act_sym = 0;
    }

    return act_sym;
}

/* SAO 偏移解码 */
static int read_sao_offset(avs2_aec *p_aec, int offset_type)
{
    int maxvalue = saoclip[offset_type][2];
    int cnt = 0;
    int act_sym, sym;

    if (offset_type == SAO_CLASS_BO) {
        sym = !biari_decode_symbol(p_aec, &p_aec->syn_ctx.sao_offset_context[0]);
    } else {
        sym = !biari_decode_symbol_eq_prob(p_aec);
    }

    while (sym) {
        cnt++;
        if (cnt == maxvalue) {
            break;
        }
        sym = !biari_decode_symbol_eq_prob(p_aec);
    }

    if (offset_type == SAO_CLASS_EO_FULL_VALLEY) {
        act_sym = EO_OFFSET_INV__MAP[cnt];
    } else if (offset_type == SAO_CLASS_EO_FULL_PEAK) {
        act_sym = -EO_OFFSET_INV__MAP[cnt];
    } else if (offset_type == SAO_CLASS_EO_HALF_PEAK) {
        act_sym = -cnt;
    } else {
        act_sym = cnt;
    }

    if (offset_type == SAO_CLASS_BO && act_sym) {
        if (biari_decode_symbol_eq_prob(p_aec)) { /* 符号位 */
            act_sym = -act_sym;
        }
    }

    return act_sym;
}

void aec_read_sao_offsets(avs2_aec *p_aec, aec_sao_param *p_sao_param, int *offset)
{
    int i;

    for (i = 0; i < 4; i++) {
        int offset_type;
        if (p_sao_param->typeIdc == SAO_TYPE_BO) {
            offset_type = SAO_CLASS_BO;
        } else {
            offset_type = (i >= 2) ? (i + 1) : i;
        }
        offset[i] = read_sao_offset(p_aec, offset_type);
    }
}

/* SAO 类型解码 */
static int read_sao_type(avs2_aec *p_aec, int act_ctx)
{
    int act_sym = 0;
    int golomb_order = 1;
    int length;

    if (act_ctx == 0) {
        length = NUM_SAO_EO_TYPES_LOG2;
    } else if (act_ctx == 1) {
        length = NUM_SAO_BO_CLASSES_LOG2;
    } else {
        length = NUM_SAO_BO_CLASSES_LOG2 - 1;
    }

    if (act_ctx == 2) {
        int temp;
        int rest;

        do {
            temp = biari_decode_symbol_eq_prob(p_aec);
            AEC_RETURN_ON_ERROR(-1);

            if (temp == 0) {
                act_sym += (1 << golomb_order);
                golomb_order++;
            }

            if (golomb_order == 4) {
                golomb_order = 0;
                temp = 1;
            }
        } while (temp != 1);

        rest = 0;
        while (golomb_order--) {
            temp = biari_decode_symbol_eq_prob(p_aec);
            if (temp == 1) {
                rest |= (temp << golomb_order);
            }
        }

        act_sym += rest;
    } else {
        int i;
        for (i = 0; i < length; i++) {
            act_sym = act_sym + (biari_decode_symbol_eq_prob(p_aec) << i);
        }
    }

    return act_sym;
}

int aec_read_sao_type(avs2_aec *p_aec, aec_sao_param *p_sao_param)
{
    int stBnd[2];

    if (p_sao_param->typeIdc == SAO_TYPE_BO) {
        stBnd[0] = read_sao_type(p_aec, 1);
        /* 读 BO 的 delta start band */
        stBnd[1] = read_sao_type(p_aec, 2) + 2;
        return (stBnd[0] + (stBnd[1] << NUM_SAO_BO_CLASSES_LOG2));
    } else {
        return read_sao_type(p_aec, 0);
    }
}

/**
 * ===========================================================================
 * 高层 AEC 解码函数 (依赖 aec_cu_t)
 * ===========================================================================
 */

/* 设置 TU split 类型 */
static int cu_set_tu_split_type(aec_cu_t *p_cu, int transform_split_flag, int enable_nsqt_sdip)
{
    static const int8_t TU_SPLIT_TYPE[MAX_PRED_MODES][2] = {
        { TU_SPLIT_CROSS,   TU_SPLIT_CROSS   }, /* 0: PRED_SKIP    */
        { TU_SPLIT_CROSS,   TU_SPLIT_CROSS   }, /* 1: PRED_2Nx2N   */
        { TU_SPLIT_CROSS,   TU_SPLIT_HOR     }, /* 2: PRED_2NxN    */
        { TU_SPLIT_CROSS,   TU_SPLIT_VER     }, /* 3: PRED_Nx2N    */
        { TU_SPLIT_CROSS,   TU_SPLIT_HOR     }, /* 4: PRED_2NxnU   */
        { TU_SPLIT_CROSS,   TU_SPLIT_HOR     }, /* 5: PRED_2NxnD   */
        { TU_SPLIT_CROSS,   TU_SPLIT_VER     }, /* 6: PRED_nLx2N   */
        { TU_SPLIT_CROSS,   TU_SPLIT_VER     }, /* 7: PRED_nRx2N   */
        { TU_SPLIT_NON,     TU_SPLIT_INVALID }, /* 8: PRED_I_2Nx2N */
        { TU_SPLIT_CROSS,   TU_SPLIT_CROSS   }, /* 9: PRED_I_NxN   */
        { TU_SPLIT_INVALID, TU_SPLIT_HOR     }, /*10: PRED_I_2Nxn  */
        { TU_SPLIT_INVALID, TU_SPLIT_VER     }  /*11: PRED_I_nx2N  */
    };
    int mode = p_cu->i_cu_type;
    int level = p_cu->i_cu_level;

    enable_nsqt_sdip = enable_nsqt_sdip && level > B8X8_IN_BIT;
    p_cu->i_trans_size = transform_split_flag ? TU_SPLIT_TYPE[mode][enable_nsqt_sdip] : TU_SPLIT_NON;

    return p_cu->i_trans_size;
}

/* intra CU 类型解码 */
int aec_read_intra_cu_type(avs2_aec *p_aec, aec_cu_t *p_cu, int b_sdip, int enable_nsqt)
{
    int cu_type = PRED_I_NxN;
    int b_tu_split = 0;
    int enable_sdip = b_sdip;  /* 保存原始值, cu_set_tu_split_type 需要 */
    b_sdip = (p_cu->i_cu_level == B16X16_IN_BIT || p_cu->i_cu_level == B32X32_IN_BIT) && b_sdip;

    /* 1, 读 intra cu split flag */
    if (p_cu->i_cu_level == B8X8_IN_BIT || b_sdip) {
        aec_ctx *p_ctx = p_aec->syn_ctx.transform_split_flag;
        b_tu_split = biari_decode_symbol(p_aec, p_ctx + 1 + b_sdip);
    }

    /* 2, 读 intra CU 划分类型 */
    if (!b_tu_split) {
        cu_type = PRED_I_2Nx2N;
    } else if (b_sdip) {
        aec_ctx *p_ctx = p_aec->syn_ctx.intra_pu_type_contexts;
        int symbol1 = biari_decode_symbol(p_aec, p_ctx);
        cu_type = symbol1 ? PRED_I_2Nxn : PRED_I_nx2N;
    }

    p_cu->i_cu_type = (int8_t)cu_type;
    /* intra 模式用 enable_sdip (对应 davs2: IS_INTRA_MODE ? enable_sdip : enable_nsqt) */
    cu_set_tu_split_type(p_cu, b_tu_split, enable_sdip);

    return cu_type;
}

/* CU 类型解码 */
int aec_read_cu_type(avs2_aec *p_aec, aec_cu_t *p_cu, int img_type, int b_amp,
                     int b_mhp, int b_wsm, int num_references)
{
    static const int MAP_CU_TYPE[2][7] = {
        {-1, 0, 1, 2, 3, -1, PRED_I_NxN},
        {-1, 0, 1, 2, 3, PRED_I_NxN}
    };

    int real_cu_type;

    if (img_type != AVS2_I_SLICE) {
        aec_ctx *p_ctx = p_aec->syn_ctx.cu_type_contexts;
        int bin_idx = 0;
        int act_ctx = 0;
        int act_sym = 0;
        int max_bit = 6 - (p_cu->i_cu_level == B8X8_IN_BIT);
        int symbol;

        while (act_sym < max_bit) {
            if ((bin_idx == 5) && (p_cu->i_cu_level != MIN_CU_SIZE_IN_BIT)) {
                symbol = biari_decode_final(p_aec);
            } else {
                symbol = biari_decode_symbol(p_aec, p_ctx + act_ctx);
            }

            AEC_RETURN_ON_ERROR(-1);
            bin_idx++;

            if (symbol == 0) {
                act_sym++;
                act_ctx = DAVS2_MIN(5, act_ctx + 1);
            } else {
                break;
            }
        }

        real_cu_type = MAP_CU_TYPE[p_cu->i_cu_level == B8X8_IN_BIT][act_sym];

        /* AMP */
        if (p_cu->i_cu_level >= B16X16_IN_BIT && b_amp && (real_cu_type == 2 || real_cu_type == 3)) {
            aec_ctx *p_ctx_amp = p_aec->syn_ctx.shape_of_partition_index;
            if (!biari_decode_symbol(p_aec, p_ctx_amp + 0)) {
                real_cu_type = real_cu_type * 2 + (!biari_decode_symbol(p_aec, p_ctx_amp + 1));
            }
        }
    } else {
        real_cu_type = PRED_I_NxN;     /* intra 模式 */
    }

    if (real_cu_type <= 0) {    /* Skip 模式 */
        int weighted_skipmode_fix = 0;
        int md_directskip_mode    = DS_NONE;

        if (img_type == AVS2_F_SLICE && b_wsm && num_references > 1) {
            weighted_skipmode_fix = aec_read_wpm(p_aec, num_references);
        }
        p_cu->i_weighted_skipmode = (int8_t)weighted_skipmode_fix;

        if ((weighted_skipmode_fix == 0) &&
            ((b_mhp && img_type == AVS2_F_SLICE) || img_type == AVS2_B_SLICE)) {
            md_directskip_mode = aec_read_dir_skip_mode(p_aec);
        } else {
            md_directskip_mode = DS_NONE;
        }

        p_cu->i_md_directskip_mode = (int8_t)md_directskip_mode;
    }

    return real_cu_type;
}

/* S 帧 CU 类型解码 */
int aec_read_cu_type_sframe(avs2_aec *p_aec)
{
    static const int MapSCUType[7] = {-1, PRED_SKIP, PRED_I_NxN};
    aec_ctx *p_ctx = p_aec->syn_ctx.cu_type_contexts;
    int act_ctx = 0;
    int cu_type = 0;

    for (;;) {
        if (biari_decode_symbol(p_aec, p_ctx + act_ctx) == 0) {
            cu_type++;
            act_ctx++;
        } else {
            break;
        }

        if (cu_type >= 2) {
            break;
        }
    }

    cu_type = MapSCUType[cu_type];

    return cu_type;
}

/* B 帧预测方向解码 */
static int aec_read_b_pdir(avs2_aec *p_aec, aec_cu_t *p_cu)
{
    static const int dir2offset[4][4] = {
        {  0,  2,  4,  9 },
        {  3,  1,  5, 10 },
        {  6,  7,  8, 11 },
        { 12, 13, 14, 15 }
    };

    int new_pdir[4] = { 3, 1, 0, 2 };
    aec_ctx *p_ctx = p_aec->syn_ctx.pu_type_index;
    int act_ctx = 0;
    int act_sym = 0;
    int pdir    = PDIR_FWD;
    int pdir0 = 0, pdir1 = 0;
    int symbol;

    if (p_cu->i_cu_type == PRED_2Nx2N) {
        act_sym = biari_decode_symbol_continu0_ext(p_aec, p_ctx, 32768, 2);
        if (act_sym == 2) {
            act_sym += (!biari_decode_symbol(p_aec, p_ctx + 2));
        }
        pdir = act_sym;
    } else if ((p_cu->i_cu_type >= PRED_2NxN && p_cu->i_cu_type <= PRED_nRx2N) && p_cu->i_cu_level == B8X8_IN_BIT) {
        p_ctx = p_aec->syn_ctx.b_pu_type_min_index;
        pdir0 = !biari_decode_symbol(p_aec, p_ctx + act_ctx);

        if (biari_decode_symbol(p_aec, p_ctx + act_ctx + 1)) {
            pdir1 = pdir0;
        } else {
            pdir1 = !pdir0;
        }

        pdir = dir2offset[pdir0][pdir1];
    } else if (p_cu->i_cu_type >= PRED_2NxN || p_cu->i_cu_type <= PRED_nRx2N) {
        act_sym = biari_decode_symbol_continu0_ext(p_aec, p_ctx + 3, 32768, 2);

        if (act_sym == 2) {
            act_sym += (!biari_decode_symbol(p_aec, p_ctx + 5));
        }
        pdir0 = act_sym;

        if (biari_decode_symbol(p_aec, p_ctx + 6)) {
            pdir1 = pdir0;
        } else {
            switch (pdir0) {
            case 0:
                if (biari_decode_symbol(p_aec, p_ctx + 7)) {
                    pdir1 = 1;
                } else {
                    symbol = biari_decode_symbol(p_aec, p_ctx + 8);
                    pdir1 = symbol ? 2 : 3;
                }
                break;
            case 1:
                if (biari_decode_symbol(p_aec, p_ctx + 9)) {
                    pdir1 = 0;
                } else {
                    symbol = biari_decode_symbol(p_aec, p_ctx + 10);
                    pdir1 = symbol ? 2 : 3;
                }
                break;
            case 2:
                if (biari_decode_symbol(p_aec, p_ctx + 11)) {
                    pdir1 = 0;
                } else {
                    symbol = biari_decode_symbol(p_aec, p_ctx + 12);
                    pdir1 = symbol ? 1 : 3;
                }
                break;
            case 3:
                if (biari_decode_symbol(p_aec, p_ctx + 13)) {
                    pdir1 = 0;
                } else {
                    symbol = biari_decode_symbol(p_aec, p_ctx + 14);
                    pdir1 = symbol ? 1 : 2;
                }
                break;
            default:
                break;
            }
        }

        pdir0 = new_pdir[pdir0];
        pdir1 = new_pdir[pdir1];
        pdir  = dir2offset[pdir0][pdir1];
    }

    return pdir;
}

/* P/F 帧 DHP 预测方向解码 */
static int aec_read_pdir_dhp(avs2_aec *p_aec, aec_cu_t *p_cu)
{
    static const int dir2offset[2][2] = {
        { 0, 1 },
        { 2, 3 }
    };

    aec_ctx *p_ctx = p_aec->syn_ctx.pu_type_index;
    int pdir = PDIR_FWD;
    int pdir0, pdir1;
    int symbol;

    if (p_cu->i_cu_type == PRED_2Nx2N) {
        pdir = pdir0 = biari_decode_symbol(p_aec, p_ctx);
    } else if (p_cu->i_cu_type >= PRED_2NxN || p_cu->i_cu_type <= PRED_nRx2N) {
        pdir0 = biari_decode_symbol(p_aec, p_ctx + 1);

        symbol = biari_decode_symbol(p_aec, p_ctx + 2);
        if (symbol) {
            pdir1 = pdir0;
        } else {
            pdir1 = 1 - pdir0;
        }

        pdir = dir2offset[pdir0][pdir1];
    }

    return pdir;
}

/* 设置 P/F 帧 CU 预测方向 */
static void cu_set_pdir_PFframe(aec_cu_t *p_cu, int pdir)
{
    static const int8_t pdir0[4] = { PDIR_FWD, PDIR_FWD, PDIR_DUAL, PDIR_DUAL };
    static const int8_t pdir1[4] = { PDIR_FWD, PDIR_DUAL, PDIR_FWD, PDIR_DUAL };
    int i_cu_type = p_cu->i_cu_type;
    int i;

    if (i_cu_type == PRED_2Nx2N) {
        pdir = (pdir == PDIR_FWD ? PDIR_FWD : PDIR_DUAL);
        for (i = 0; i < 4; i++) {
            p_cu->b8pdir[i] = (int8_t)pdir;
        }
    } else if (IS_HOR_PU_PART(i_cu_type)) {
        p_cu->b8pdir[0] = p_cu->b8pdir[2] = pdir0[pdir];
        p_cu->b8pdir[1] = p_cu->b8pdir[3] = pdir1[pdir];
    } else if (IS_VER_PU_PART(i_cu_type)) {
        p_cu->b8pdir[0] = p_cu->b8pdir[2] = pdir0[pdir];
        p_cu->b8pdir[1] = p_cu->b8pdir[3] = pdir1[pdir];
    } else {
        for (i = 0; i < 4; i++) {
            p_cu->b8pdir[i] = PDIR_INVALID;
        }
    }
}

/* 设置 B 帧 CU 预测方向 */
static void cu_set_pdir_Bframe(aec_cu_t *p_cu, int pdir)
{
    static const int8_t pdir0[16] = {
        PDIR_FWD, PDIR_BWD, PDIR_FWD, PDIR_BWD, PDIR_FWD, PDIR_BWD, PDIR_SYM, PDIR_SYM,
        PDIR_SYM, PDIR_FWD, PDIR_BWD, PDIR_SYM, PDIR_BID, PDIR_BID, PDIR_BID, PDIR_BID
    };
    static const int8_t pdir1[16] = {
        PDIR_FWD, PDIR_BWD, PDIR_BWD, PDIR_FWD, PDIR_SYM, PDIR_SYM, PDIR_FWD, PDIR_BWD,
        PDIR_SYM, PDIR_BID, PDIR_BID, PDIR_BID, PDIR_FWD, PDIR_BWD, PDIR_SYM, PDIR_BID
    };
    static const int8_t pdir2refidx[4][2] = {
        { B_FWD, INVALID_REF },  /* PDIR_FWD  */
        { INVALID_REF, B_BWD },  /* PDIR_BWD  */
        { B_FWD, B_BWD },        /* PDIR_SYM  */
        { B_FWD, B_BWD }         /* PDIR_BID  */
    };
    int i_cu_type = p_cu->i_cu_type;
    int8_t *b8pdir = p_cu->b8pdir;
    int i;

    if (i_cu_type == PRED_SKIP) {
        pdir = tab_pdir_bskip[p_cu->i_md_directskip_mode];
        for (i = 0; i < 4; i++) {
            b8pdir[i] = (int8_t)pdir;
        }
    } else if (i_cu_type == PRED_2Nx2N) {
        for (i = 0; i < 4; i++) {
            b8pdir[i] = (int8_t)pdir;
        }
    } else if (IS_HOR_PU_PART(i_cu_type)) {
        b8pdir[0] = b8pdir[2] = pdir0[pdir];
        b8pdir[1] = b8pdir[3] = pdir1[pdir];
    } else if (IS_VER_PU_PART(i_cu_type)) {
        b8pdir[0] = b8pdir[2] = pdir0[pdir];
        b8pdir[1] = b8pdir[3] = pdir1[pdir];
    } else {
        for (i = 0; i < 4; i++) {
            b8pdir[i] = PDIR_INVALID;
        }
    }

    for (i = 0; i < 4; i++) {
        const int8_t *p_idx = pdir2refidx[b8pdir[i]];
        p_cu->ref_idx[i].r[0] = p_idx[0];
        p_cu->ref_idx[i].r[1] = p_idx[1];
    }
}

/* 参考帧索引解码 */
static int aec_read_ref_frame(avs2_aec *p_aec, int num_of_references)
{
    aec_ctx *p_ctx = p_aec->syn_ctx.pu_reference_index;
    int act_sym;

    if (biari_decode_symbol(p_aec, p_ctx)) {
        act_sym = 0;
    } else {
        int act_ctx = 1;
        act_sym = 1;

        while ((act_sym != num_of_references - 1) && (!biari_decode_symbol(p_aec, p_ctx + act_ctx))) {
            act_sym++;
            act_ctx = DAVS2_MIN(2, act_ctx + 1);
        }
    }

    return act_sym;
}

/* 读参考帧索引 */
static int cu_read_references(avs2_aec *p_aec, aec_cu_t *p_cu, int num_references)
{
    int idx_pu;
    int num_pu = p_cu->i_cu_type == PRED_2Nx2N ? 1 : 2;

    for (idx_pu = 0; idx_pu < num_pu; idx_pu++) {
        int8_t ref_1st, ref_2nd;

        if (num_references > 1) {
            ref_1st = (int8_t)aec_read_ref_frame(p_aec, num_references);
            AEC_RETURN_ON_ERROR(-1);
        } else {
            ref_1st = 0;
        }

        if (p_cu->b8pdir[idx_pu] == PDIR_DUAL) {
            ref_2nd = !ref_1st;
        } else {
            ref_2nd = INVALID_REF;
        }

        p_cu->ref_idx[idx_pu].r[0] = ref_1st;
        p_cu->ref_idx[idx_pu].r[1] = ref_2nd;
    }

    return 0;
}

/* 解码帧间预测方向 */
void aec_read_inter_pred_dir(avs2_aec *p_aec, aec_cu_t *p_cu, int img_type,
                             int enable_dhp, int num_references)
{
    int pdir = PDIR_FWD;
    int real_cu_type = p_cu->i_cu_type;

    if (img_type == AVS2_B_SLICE) {  /* B 帧 */
        if (real_cu_type >= PRED_2Nx2N && real_cu_type <= PRED_nRx2N) {
            pdir = aec_read_b_pdir(p_aec, p_cu);
        }
        cu_set_pdir_Bframe(p_cu, pdir);
    } else {  /* 其它帧间 */
        if (IS_SKIP_MODE(real_cu_type)) {
            int i;
            if (p_cu->i_weighted_skipmode ||
                p_cu->i_md_directskip_mode == DS_DUAL_1ST ||
                p_cu->i_md_directskip_mode == DS_DUAL_2ND) {
                pdir = PDIR_DUAL;
            }
            for (i = 0; i < 4; i++) {
                p_cu->b8pdir[i] = (int8_t)pdir;
            }
        } else {
            if (img_type == AVS2_F_SLICE && num_references > 1 && enable_dhp) {
                if (!(p_cu->i_cu_level == B8X8_IN_BIT && real_cu_type >= PRED_2NxN && real_cu_type <= PRED_nRx2N)) {
                    pdir = aec_read_pdir_dhp(p_aec, p_cu);
                }
            }
            cu_set_pdir_PFframe(p_cu, pdir);
        }

        if (img_type != AVS2_S_SLICE && p_cu->i_cu_type != PRED_SKIP) {
            cu_read_references(p_aec, p_cu, num_references);
        }
    }
}

/* intra 色度预测模式解码 */
int aec_read_intra_pmode_c(avs2_aec *p_aec, int luma_mode, int c_ipred_mode_ctx)
{
    aec_ctx *p_ctx = p_aec->syn_ctx.intra_chroma_pred_mode;
    int act_ctx    = c_ipred_mode_ctx;
    int lmode      = tab_intra_mode_luma2chroma[luma_mode];
    int is_redundant = lmode >= 0;
    int act_sym;

    act_sym = !biari_decode_symbol(p_aec, p_ctx + act_ctx);
    if (act_sym != 0) {
        act_sym = unary_bin_max_decode(p_aec, p_ctx + 2, 0, 3) + 1;
        if (is_redundant && act_sym >= lmode) {
            if (act_sym == 4) {
                act_sym = 4;
            } else {
                act_sym++;
            }
        }
    }

    return act_sym;
}

/* 查找邻块 CBP 的亮度分量 (对应 davs2 get_neighbor_cbp_y + get_neighbor_cu_in_slice)
 * \param f       avs2_frame 指针
 * \param x_4x4   邻块在 4x4 单位下的 X 坐标
 * \param y_4x4   邻块在 4x4 单位下的 Y 坐标
 * \param p_cu    当前 CU 描述符 (用于判断邻块是否在当前 CU 内)
 * \return 邻块对应位置的 ctp_y (0 或 1)
 */
static int get_neighbor_cbp_y(avs2_frame *f, int x_4x4, int y_4x4,
                              aec_cu_t *p_cu)
{
    int scu_x = p_cu->scu_x;
    int scu_y = p_cu->scu_y;
    int cu_level = p_cu->i_cu_level;
    int cbp;
    int trans_size;
    int level;
    int cu_mask;
    int lx, ly;

    /* 当前 CU 在 4x4 单位下的左上角坐标和尺寸
     * shift_4x4 = MIN_CU_SIZE_IN_BIT - MIN_PU_SIZE_IN_BIT = 3 - 2 = 1
     * cur_x = scu_x << 1, cur_y = scu_y << 1
     * cu_size_4x4 = 1 << (cu_level - MIN_PU_SIZE_IN_BIT)
     */
    int cur_x_4x4 = scu_x << 1;
    int cur_y_4x4 = scu_y << 1;
    int cu_size_4x4 = 1 << (cu_level - MIN_PU_SIZE_IN_BIT);

    /* 边界检查 */
    if (x_4x4 < 0 || y_4x4 < 0 ||
        x_4x4 >= (f->w8 << 1) || y_4x4 >= (f->h8 << 1)) {
        return 0;
    }

    /* 判断邻块是否在当前 CU 内 (对应 davs2 get_neighbor_cu_in_slice 返回 p_cur) */
    if (x_4x4 >= cur_x_4x4 && x_4x4 < cur_x_4x4 + cu_size_4x4 &&
        y_4x4 >= cur_y_4x4 && y_4x4 < cur_y_4x4 + cu_size_4x4) {
        /* 邻块在当前 CU 内: 使用当前 CU 的 CBP */
        cbp = p_cu->i_cbp;
        trans_size = p_cu->i_trans_size;
    } else {
        /* 邻块在其他 CU 内: 查找 cu_grid */
        int bx = x_4x4 >> 1;
        int by = y_4x4 >> 1;
        avs2_cu *neighbor = &f->cu_grid[by * f->w8 + bx];

        if (!neighbor || neighbor->cu_level == 0) {
            return 0;
        }
        cbp = neighbor->i_cbp;
        trans_size = neighbor->i_tu_split;
        cu_level = neighbor->cu_level;
    }

    level = cu_level - MIN_PU_SIZE_IN_BIT;
    cu_mask = (1 << level) - 1;
    lx = x_4x4 & cu_mask;
    ly = y_4x4 & cu_mask;

    if (trans_size == TU_SPLIT_NON) {
        /* TU 不分割: 直接返回 cbp 的 bit 0 */
        return cbp & 1;
    } else if (trans_size == TU_SPLIT_VER) {
        /* 垂直分割: 4 个竖条 */
        lx >>= (level - 2);
        return (cbp >> lx) & 1;
    } else if (trans_size == TU_SPLIT_HOR) {
        /* 水平分割: 4 个横条 */
        ly >>= (level - 2);
        return (cbp >> ly) & 1;
    } else {
        /* 四块分割 (TU_SPLIT_CROSS) */
        lx >>= (level - 1);
        ly >>= (level - 1);
        return (cbp >> (lx + (ly << 1))) & 1;
    }
}

/* 完整版 ctp_y 解码 (含邻块 CBP 查找, 对应 davs2 aec_read_ctp_y) */
int aec_read_ctp_y(avs2_aec *p_aec, aec_cu_t *p_cu, int b8)
{
    aec_ctx *p_ctx;
    int b_hor = (p_cu->i_trans_size == TU_SPLIT_HOR);
    int b_ver = (p_cu->i_trans_size == TU_SPLIT_VER);
    int i_level = p_cu->i_cu_level;
    int cu_size = 1 << i_level;
    int a = 0, b = 0;
    int x, y;
    avs2_frame *f = (avs2_frame *)p_cu->p_frame;

    /* 当前 TB 在 CU 中的位置 */
    if (b_hor) {
        x = 0;
        y = ((cu_size * b8) >> 2);
    } else if (b_ver) {
        x = ((cu_size * b8) >> 2);
        y = 0;
    } else {
        x = ((cu_size * (b8 & 1)) >> 1);
        y = ((cu_size * (b8 >> 1)) >> 1);
    }

    /* TB 在图像中的位置 */
    x += (p_cu->scu_x << 3);
    y += (p_cu->scu_y << 3);
    /* 转换为 4x4 单位 */
    x >>= 2;
    y >>= 2;

    /* 左邻块 */
    if (b_ver && b8 > 0) {
        a = (p_cu->i_cbp >> (b8 - 1)) & 1;
    } else if (f) {
        a = get_neighbor_cbp_y(f, x - 1, y, p_cu);
    }

    /* 上邻块 */
    if (b_hor && b8 > 0) {
        b = (p_cu->i_cbp >> (b8 - 1)) & 1;
    } else if (f) {
        b = get_neighbor_cbp_y(f, x, y - 1, p_cu);
    }

    p_ctx = p_aec->syn_ctx.cbp_contexts + a + 2 * b;
    {
        int bit = biari_decode_symbol(p_aec, p_ctx);
        return bit;
    }
}

/* 简化版 ctp_y 解码 (无邻块 CBP 查找，使用上下文 0) */
int aec_read_ctp_y_simple(avs2_aec *p_aec, aec_cu_t *p_cu, int b8)
{
    aec_ctx *p_ctx;
    int a = 0, b = 0;

    /* 简化: 用已解码的 CBP 位作为邻块估计 */
    if (p_cu->i_trans_size == TU_SPLIT_VER && b8 > 0) {
        a = (p_cu->i_cbp >> (b8 - 1)) & 1;
    }
    if (p_cu->i_trans_size == TU_SPLIT_HOR && b8 > 0) {
        b = (p_cu->i_cbp >> (b8 - 1)) & 1;
    }

    p_ctx = p_aec->syn_ctx.cbp_contexts + a + 2 * b;
    return biari_decode_symbol(p_aec, p_ctx);
}

/* 简化版 CBP 解码 (无邻块查找) */
int aec_read_cbp_simple(avs2_aec *p_aec, aec_cu_t *p_cu, int chroma_format)
{
    int cbp = 0;
    int cbp_bit = 0;

    if (CU_IS_INTER(p_cu)) {
        if (IS_NOSKIP_INTER_MODE(p_cu->i_cu_type)) {
            cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 8); /* ctp_zero_flag */
        }
        if (cbp_bit == 0) {
            int b_tu_split = biari_decode_symbol(p_aec, p_aec->syn_ctx.transform_split_flag);
            cu_set_tu_split_type(p_cu, b_tu_split, 0);

            /* chroma */
            if (chroma_format != AVS2_CHROMA_400) {
                cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 4);
                if (cbp_bit) {
                    cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 5);
                    if (cbp_bit) {
                        cbp += 48;
                    } else {
                        cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 5);
                        cbp += (cbp_bit == 1) ? 32 : 16;
                    }
                }
            }

            /* luma */
            if (b_tu_split == 0) {
                if (cbp == 0) {
                    cbp = 1;
                } else {
                    cbp_bit = aec_read_ctp_y_simple(p_aec, p_cu, 0);
                    cbp    += cbp_bit;
                }
            } else {
                cbp_bit = aec_read_ctp_y_simple(p_aec, p_cu, 0);
                cbp    += cbp_bit;
                p_cu->i_cbp = (int8_t)cbp;

                cbp_bit = aec_read_ctp_y_simple(p_aec, p_cu, 1);
                cbp    += (cbp_bit << 1);
                p_cu->i_cbp = (int8_t)cbp;

                cbp_bit = aec_read_ctp_y_simple(p_aec, p_cu, 2);
                cbp    += (cbp_bit << 2);
                p_cu->i_cbp = (int8_t)cbp;

                cbp_bit = aec_read_ctp_y_simple(p_aec, p_cu, 3);
                cbp    += (cbp_bit << 3);
                p_cu->i_cbp = (int8_t)cbp;
            }
        } else {
            cu_set_tu_split_type(p_cu, 1, 0);
            p_cu->i_cbp = 0;
            cbp = 0;
        }
    } else {
        /* intra luma */
        if (p_cu->i_cu_type == PRED_I_2Nx2N) {
            cbp = aec_read_ctp_y_simple(p_aec, p_cu, 0);
        } else {
            cbp_bit = aec_read_ctp_y_simple(p_aec, p_cu, 0);
            cbp    += cbp_bit;
            p_cu->i_cbp = (int8_t)cbp;

            cbp_bit = aec_read_ctp_y_simple(p_aec, p_cu, 1);
            cbp    += (cbp_bit << 1);
            p_cu->i_cbp = (int8_t)cbp;

            cbp_bit = aec_read_ctp_y_simple(p_aec, p_cu, 2);
            cbp    += (cbp_bit << 2);
            p_cu->i_cbp = (int8_t)cbp;

            cbp_bit = aec_read_ctp_y_simple(p_aec, p_cu, 3);
            cbp    += (cbp_bit << 3);
            p_cu->i_cbp = (int8_t)cbp;
        }

        /* chroma */
        if (chroma_format != AVS2_CHROMA_400) {
            cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 6);
            if (cbp_bit) {
                cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 7);
                if (cbp_bit) {
                    cbp += 48;
                } else {
                    cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 7);
                    cbp += 16 << cbp_bit;
                }
            }
        }
    }

    return cbp;
}

/* 完整版 CBP 解码 (含邻块 CBP 查找, 对应 davs2 aec_read_cbp) */
int aec_read_cbp(avs2_aec *p_aec, aec_cu_t *p_cu, int chroma_format)
{
    int cbp = 0;
    int cbp_bit = 0;

    if (CU_IS_INTER(p_cu)) {
        if (IS_NOSKIP_INTER_MODE(p_cu->i_cu_type)) {
            cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 8); /* ctp_zero_flag */
        }
        if (cbp_bit == 0) {
            int b_tu_split = biari_decode_symbol(p_aec, p_aec->syn_ctx.transform_split_flag);
            cu_set_tu_split_type(p_cu, b_tu_split, 0);

            /* chroma */
            if (chroma_format != AVS2_CHROMA_400) {
                cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 4);
                if (cbp_bit) {
                    cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 5);
                    if (cbp_bit) {
                        cbp += 48;
                    } else {
                        cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 5);
                        cbp += (cbp_bit == 1) ? 32 : 16;
                    }
                }
            }

            /* luma (使用完整版 aec_read_ctp_y, 含邻块 CBP 查找) */
            if (b_tu_split == 0) {
                if (cbp == 0) {
                    cbp = 1;
                } else {
                    cbp_bit = aec_read_ctp_y(p_aec, p_cu, 0);
                    cbp    += cbp_bit;
                }
            } else {
                cbp_bit = aec_read_ctp_y(p_aec, p_cu, 0);
                cbp    += cbp_bit;
                p_cu->i_cbp = (int8_t)cbp;

                cbp_bit = aec_read_ctp_y(p_aec, p_cu, 1);
                cbp    += (cbp_bit << 1);
                p_cu->i_cbp = (int8_t)cbp;

                cbp_bit = aec_read_ctp_y(p_aec, p_cu, 2);
                cbp    += (cbp_bit << 2);
                p_cu->i_cbp = (int8_t)cbp;

                cbp_bit = aec_read_ctp_y(p_aec, p_cu, 3);
                cbp    += (cbp_bit << 3);
                p_cu->i_cbp = (int8_t)cbp;
            }
        } else {
            cu_set_tu_split_type(p_cu, 1, 0);
            p_cu->i_cbp = 0;
            cbp = 0;
        }
    } else {
        /* intra luma */
        if (p_cu->i_cu_type == PRED_I_2Nx2N) {
            cbp = aec_read_ctp_y(p_aec, p_cu, 0);
        } else {
            cbp_bit = aec_read_ctp_y(p_aec, p_cu, 0);
            cbp    += cbp_bit;
            p_cu->i_cbp = (int8_t)cbp;

            cbp_bit = aec_read_ctp_y(p_aec, p_cu, 1);
            cbp    += (cbp_bit << 1);
            p_cu->i_cbp = (int8_t)cbp;

            cbp_bit = aec_read_ctp_y(p_aec, p_cu, 2);
            cbp    += (cbp_bit << 2);
            p_cu->i_cbp = (int8_t)cbp;

            cbp_bit = aec_read_ctp_y(p_aec, p_cu, 3);
            cbp    += (cbp_bit << 3);
            p_cu->i_cbp = (int8_t)cbp;
        }

        /* chroma */
        if (chroma_format != AVS2_CHROMA_400) {
            cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 6);
            if (cbp_bit) {
                cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 7);
                if (cbp_bit) {
                    cbp += 48;
                } else {
                    cbp_bit = biari_decode_symbol(p_aec, p_aec->syn_ctx.cbp_contexts + 7);
                    cbp += 16 << cbp_bit;
                }
            }
        }
    }

    return cbp;
}

/**
 * ===========================================================================
 * 系数解码 (run-level, TU, EGK, CG 遍历)
 * ===========================================================================
 */

/* 读最后一个 CG 位置 */
static int aec_read_last_cg_pos(avs2_aec *p_aec, aec_ctx *p_ctx, aec_cu_t *p_cu,
                                int *CGx, int *CGy, int b_luma, int num_cg, int is_dc_diag,
                                int num_cg_x_minus1, int num_cg_y_minus1)
{
    int last_cg_x = 0;
    int last_cg_y = 0;
    int last_cg_idx = 0;

    if (b_luma && is_dc_diag) {
        DAVS2_SWAP(num_cg_x_minus1, num_cg_y_minus1);
    }

    if (num_cg == 4) {  /* 8x8 */
        last_cg_idx = 0;
        last_cg_idx += biari_decode_symbol_continu0_ext(p_aec, p_ctx, 2, 3);

        if (b_luma && p_cu->i_trans_size == TU_SPLIT_HOR) {
            last_cg_x = last_cg_idx;
            last_cg_y = 0;
        } else if (b_luma && p_cu->i_trans_size == TU_SPLIT_VER) {
            last_cg_x = 0;
            last_cg_y = last_cg_idx;
        } else {
            last_cg_x = last_cg_idx &  1;
            last_cg_y = last_cg_idx >> 1;
        }
    } else { /* 16x16 和 32x32 */
        int last_cg_bit;

        p_ctx += 3;
        last_cg_bit = biari_decode_symbol(p_aec, p_ctx);

        if (last_cg_bit == 0) {
            last_cg_x = 0;
            last_cg_y = 0;
            last_cg_idx  = 0;
        } else {
            p_ctx++;
            last_cg_x = biari_decode_symbol_continue0(p_aec, p_ctx, num_cg_x_minus1);

            p_ctx++;
            if (last_cg_x == 0) {
                if (num_cg_y_minus1 != 1) {
                    last_cg_y = biari_decode_symbol_continue0(p_aec, p_ctx, num_cg_y_minus1 - 1);
                }
                last_cg_y++;
            } else {
                last_cg_y = biari_decode_symbol_continue0(p_aec, p_ctx, num_cg_y_minus1);
            }
        }

        if (b_luma && is_dc_diag) {
            DAVS2_SWAP(last_cg_x, last_cg_y);
        }

        if (b_luma && p_cu->i_trans_size == TU_SPLIT_HOR) {
            last_cg_idx = raster2ZZ_2x8[last_cg_y * 8 + last_cg_x];
        } else if (b_luma && p_cu->i_trans_size == TU_SPLIT_VER) {
            last_cg_idx = raster2ZZ_8x2[last_cg_y * 2 + last_cg_x];
        } else if (num_cg == 16) {
            last_cg_idx = raster2ZZ_4x4[last_cg_y * 4 + last_cg_x];
        } else {
            last_cg_idx = raster2ZZ_8x8[last_cg_y * 8 + last_cg_x];
        }
    }

    *CGx = last_cg_x;
    *CGy = last_cg_y;
    return last_cg_idx;
}

/* 读 CG 内最后系数位置 */
static int aec_read_last_coeff_pos_in_cg(avs2_aec *p_aec, aec_ctx *p_ctx,
                                         int rank, int cg_x, int cg_y, int b_luma,
                                         int b_one_cg, int is_dc_diag)
{
    int xx, yy;
    int offset;

    /* AVS2-P2 标准 8.3.3.2.14: 确定 last_coeff_pos_x/y 的 ctxIdxInc */
    if (b_luma == 0) {
        offset = b_one_cg ? 0 : 4 + (rank == 0) * 4;
    } else if (b_one_cg) {
        offset = 40 + is_dc_diag * 4;
    } else if (cg_x != 0 && cg_y != 0) {
        offset = 32 + (rank == 0) * 4;
    } else {
        offset = (4 * (rank == 0) + 2 * (cg_x == 0 && cg_y == 0) + is_dc_diag) * 4;
    }

    p_ctx += offset;
    xx = biari_decode_symbol_continu0_ext(p_aec, p_ctx, 1, 3);

    p_ctx += 2;
    yy = biari_decode_symbol_continu0_ext(p_aec, p_ctx, 1, 3);

    if (cg_x == 0 && cg_y > 0 && is_dc_diag) {
        DAVS2_SWAP(xx, yy);
    }
    if (rank != 0) {
        xx = 3 - xx;
        if (is_dc_diag) {
            yy = 3 - yy;
        }
    }

    return tab_scan_coeff_pos_in_cg[yy][xx];
}

/* 计算最后若干系数的绝对值之和 */
static int get_abssum_of_n_last_coeffs(runlevel_pair_t *p_runlevel, int end_pair_pos, int start_pair_pos)
{
    int absSum5 = 0;
    int n = 0;
    int k;

    for (k = end_pair_pos - 1; k >= start_pair_pos; k--) {
        n += p_runlevel[k].run;
        if (n >= 6) {
            break;
        }
        absSum5 += DAVS2_ABS(p_runlevel[k].level);
        n++;
    }

    return absSum5;
}

/* run 解码函数指针 */
typedef int (*aec_read_run_f)(avs2_aec *p_aec, aec_ctx *p_ctx, int pos, int b_only_one_cg, int b_1st_cg);

/* luma run 解码 (非 DC_DIAG 扫描) */
static int aec_read_run_luma1(avs2_aec *p_aec, aec_ctx *p_ctx, int pos, int b_only_one_cg, int b_1st_cg)
{
    int ctxpos;
    int Run = 0;
    int offset = 0;

    b_only_one_cg = b_only_one_cg ? 0 : 4;

    for (ctxpos = 0; Run != pos; ctxpos++) {
        if (ctxpos < pos) {
            int moddiv; /* 0/1/2 */
            moddiv = (avs2_scan_4x4[pos - 1 - ctxpos][1] + 1) >> 1;
            offset = (b_1st_cg ? (pos == ctxpos + 1 ? 0 : (1 + moddiv)) : (4 + moddiv)) + b_only_one_cg;
        }

        if (biari_decode_symbol(p_aec, p_ctx + offset)) {
            break;
        }
        Run++;
    }

    return Run;
}

/* luma run 解码 (DC_DIAG 扫描) */
static int aec_read_run_luma2(avs2_aec *p_aec, aec_ctx *p_ctx, int pos, int b_only_one_cg, int b_1st_cg)
{
    int ctxpos;
    int Run = 0;
    int offset = 0;

    b_only_one_cg = b_only_one_cg ? 0 : 4;

    for (ctxpos = 0; Run != pos; ctxpos++) {
        if (ctxpos < pos) {
            int moddiv; /* 0/1/2 */
            moddiv = ((pos < ctxpos + 4) ? 0 : (pos < ctxpos + 11 ? 1 : 2));
            offset = (b_1st_cg ? (pos == ctxpos + 1 ? 0 : (1 + moddiv)) : (4 + moddiv)) + b_only_one_cg;
        }

        if (biari_decode_symbol(p_aec, p_ctx + offset)) {
            break;
        }
        Run++;
    }

    return Run;
}

/* chroma run 解码 */
static int aec_read_run_chroma(avs2_aec *p_aec, aec_ctx *p_ctx, int pos, int b_only_one_cg, int b_1st_cg)
{
    int ctxpos;
    int Run = 0;
    int offset = 0;

    b_only_one_cg = b_only_one_cg ? 0 : 3;

    for (ctxpos = 0; Run != pos; ctxpos++) {
        if (ctxpos < pos) {
            int moddiv = (pos >= 6 + ctxpos);
            offset = (b_1st_cg ? (pos == ctxpos + 1 ? 0 : (1 + moddiv)) : (3 + moddiv)) + b_only_one_cg;
        }

        if (biari_decode_symbol(p_aec, p_ctx + offset)) {
            break;
        }
        Run++;
    }

    return Run;
}

/* 完整 run-level 系数解码 */
int aec_read_run_level(avs2_aec *p_aec, aec_cu_t *p_cu, int num_cg, int b_luma, int is_dc_diag,
                       runlevel_t *runlevel, int scale, int shift)
{
    static const int numOfCoeffInCG = 16;
    (void)scale; (void)shift;  /* 反量化已移至 cu_get_block_coeffs 批量执行 */

    const uint8_t (*tab_cg_scan)[2]    = runlevel->cg_scan;
    aec_ctx (*ctxa_run)[NUM_MAP_CTX]   = runlevel->p_ctx_run;
    aec_ctx *p_ctx_level               = runlevel->p_ctx_level;
    aec_ctx *p_ctx_nonzero_cg_flag     = runlevel->p_ctx_sig_cg;
    aec_ctx *p_ctx_last_cg_pos         = runlevel->p_ctx_last_cg;
    aec_ctx *p_ctx_last_pos_in_cg      = runlevel->p_ctx_last_pos_in_cg;
    runlevel_pair_t *p_runlevel        = runlevel->run_level;
    int idx_cg;
    int cg_pos = 0;
    int CGx = 0;
    int CGy = 0;
    int b_only_one_cg = (num_cg == 1);
    int8_t dct_pattern = DCT_QUAD;
    int w_tr_half, w_tr_quad;
    int h_tr_half, h_tr_quad;
    int w_tr = runlevel->w_tr;
    int h_tr = runlevel->h_tr;
    int rank = 0;
    aec_read_run_f f_read_run = b_luma ? (!is_dc_diag ? aec_read_run_luma1 : aec_read_run_luma2) : aec_read_run_chroma;

    /* CG 位置边界 */
    if (w_tr == h_tr) {
        w_tr_half = w_tr >> 1;
        h_tr_half = h_tr >> 1;
        w_tr_quad = w_tr >> 2;
        h_tr_quad = h_tr >> 2;
    } else if (w_tr > h_tr) {
        w_tr_half = w_tr >> 1;
        h_tr_half = h_tr >> 0;
        w_tr_quad = w_tr >> 2;
        h_tr_quad = h_tr >> 0;
    } else {
        w_tr_half = w_tr >> 0;
        h_tr_half = h_tr >> 1;
        w_tr_quad = w_tr >> 0;
        h_tr_quad = h_tr >> 2;
    }
    /* 转换为 CG 位置的边界 */
    w_tr_half >>= 2;
    h_tr_half >>= 2;
    w_tr_quad >>= 2;
    h_tr_quad >>= 2;

    /* 1, 读最后一个 CG 位置 */
    if (num_cg > 1) {
        int num_cg_x_minus1 = tab_cg_scan[num_cg - 1][0];
        int num_cg_y_minus1 = tab_cg_scan[num_cg - 1][1];
        cg_pos = aec_read_last_cg_pos(p_aec, p_ctx_last_cg_pos, p_cu, &CGx, &CGy, b_luma, num_cg, is_dc_diag, num_cg_x_minus1, num_cg_y_minus1);
    }

    num_cg = cg_pos + 1;

    runlevel->num_nonzero_cg = num_cg;

    /* 2, 逐 CG 读系数 */
    for (idx_cg = 0; idx_cg < num_cg; idx_cg++) {
        int b_1st_cg = (cg_pos == 0);
        int nonzero_cg_flag = 1;

        /* 2.1, sig CG flag */
        if (rank > 0) {
            int ctx_sig_cg = (b_luma && cg_pos != 0);
            CGx = tab_cg_scan[cg_pos][0];
            CGy = tab_cg_scan[cg_pos][1];
            nonzero_cg_flag = biari_decode_symbol(p_aec, p_ctx_nonzero_cg_flag + ctx_sig_cg);
        }

        /* 2.2, CG 内系数 */
        if (nonzero_cg_flag) {
            int num_pairs_in_cg = 0;
            int i;

            /* CG 内最后位置 */
            int pos = aec_read_last_coeff_pos_in_cg(p_aec, p_ctx_last_pos_in_cg, rank, CGx, CGy, b_luma, b_only_one_cg, is_dc_diag);

            for (i = -numOfCoeffInCG; i != 0; i++) {
                int Run = 0;
                int Level = 1;
                int absSum5;
                aec_ctx *p_ctx;

                /* coeff_level_minus1_band */
                if (biari_decode_final(p_aec)) {
                    int golomb_order  = 0;
                    int binary_symbol = 0;

                    for (;;) {
                        int l = biari_decode_symbol_eq_prob(p_aec);
                        AEC_RETURN_ON_ERROR(-1);
                        if (l) {
                            break;
                        }
                        Level += (1 << golomb_order);
                        golomb_order++;
                    }

                    while (golomb_order--) {
                        int sig = biari_decode_symbol_eq_prob(p_aec);
                        binary_symbol |= (sig << golomb_order);
                    }

                    Level += binary_symbol;
                    Level += 32;
                } else {
                    int pairsInCGIdx = (num_pairs_in_cg + 1) >> 1;
                    pairsInCGIdx = DAVS2_MIN(2, pairsInCGIdx);
                    p_ctx = p_ctx_level;
                    p_ctx += 10 * (b_1st_cg && pos < 3) + DAVS2_MIN(rank, pairsInCGIdx + 2) + ((5 * pairsInCGIdx) >> 1);
                    Level += biari_decode_symbol_continue0(p_aec, p_ctx, 31);
                }

                AEC_RETURN_ON_ERROR(-1);
                absSum5 = get_abssum_of_n_last_coeffs(p_runlevel, num_pairs_in_cg, 0);
                absSum5 = (absSum5 + Level) >> 1;
                p_ctx = ctxa_run[DAVS2_MIN(absSum5, 2)];

                /* run */
                Run = 0;
                if (pos > 0) {
                    Run = f_read_run(p_aec, p_ctx, pos, b_only_one_cg, b_1st_cg);
                }
                AEC_RETURN_ON_ERROR(-1);

                p_runlevel[num_pairs_in_cg].level = (int16_t)Level;
                p_runlevel[num_pairs_in_cg].run   = (int16_t)Run;

                num_pairs_in_cg++;
                if (Level > T_Chr[rank]) {
                    rank = tab_rank[DAVS2_MIN(5, Level)];
                }
                if (Run == pos) {
                    break;
                }

                pos -= (Run + 1);
            } /* for (i = -numOfCoeffInCG; i != 0; i++) */

            /* level 符号 */
            for (i = 0; i < num_pairs_in_cg; i++) {
                if (biari_decode_symbol_eq_prob(p_aec)) {
                    p_runlevel[i].level = -p_runlevel[i].level;
                }
            }

            /* run-level 转换为 CG 内扫描系数 */
            {
                const int b_swap_xy  = runlevel->b_swap_xy;
                const int i_coeff    = runlevel->i_res;
                coeff_t *p_res = runlevel->p_res;
                int num_pairs  = num_pairs_in_cg;
                int coef_ctr   = -1;
                if (b_swap_xy) {
                    DAVS2_SWAP(CGx, CGy);
                }
                p_res += i_coeff * (CGy << 2) + (CGx << 2);
                while (num_pairs > 0) {
                    int x_in_cg, y_in_cg;
                    int level = p_runlevel[num_pairs - 1].level;
                    int run   = p_runlevel[num_pairs - 1].run;
                    num_pairs--;
                    if (run < 0 || run >= 16) {
                        return -1;
                    }
                    coef_ctr += run + 1;

                    x_in_cg = avs2_scan_4x4[coef_ctr][ b_swap_xy];
                    y_in_cg = avs2_scan_4x4[coef_ctr][!b_swap_xy];

                    /* 存原始 level, 反量化在 cu_get_block_coeffs 中批量执行 */
                    p_res[y_in_cg * i_coeff + x_in_cg] = (coeff_t)level;
                }

                if (CGy >= h_tr_half || CGx >= w_tr_half) {
                    dct_pattern = DCT_DEAULT;
                } else if ((CGy >= h_tr_quad || CGx >= w_tr_quad) && dct_pattern != DCT_DEAULT) {
                    dct_pattern = DCT_HALF;
                }
            }
        }  /* 读一个 CG 结束 */
        cg_pos--;
    }  /* 读所有 CG 结束 */

    return dct_pattern;
}

/* 取得一个块的系数 (封装 run-level 设置) */
int8_t cu_get_block_coeffs(avs2_aec *p_aec, runlevel_t *runlevel, aec_cu_t *p_cu,
                           coeff_t *p_res, int w_tr, int h_tr, int i_tu_level, int b_luma,
                           int intra_pred_class, int b_swap_xy,
                           int scale, int shift, int wq_size_id)
{
    int num_coeffs = w_tr * h_tr;
    int num_cg = num_coeffs >> 4;

    runlevel->p_res         = p_res;
    runlevel->i_res         = w_tr;
    runlevel->b_swap_xy     = b_swap_xy;
    runlevel->i_tu_level    = i_tu_level;
    runlevel->w_tr          = w_tr;
    runlevel->h_tr          = h_tr;
    (void)wq_size_id;

    int8_t dct_pattern = (int8_t)aec_read_run_level(p_aec, p_cu, num_cg, b_luma,
                                                     intra_pred_class == INTRA_PRED_DC_DIAG, runlevel, scale, shift);
    if (dct_pattern >= 0) {
        /* 批量反量化 (SIMD 加速) */
        avs2_dsp_table.dequant_block(p_res, num_coeffs, scale, shift);
    }
    return dct_pattern;
}
