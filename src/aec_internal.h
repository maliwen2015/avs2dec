#ifndef AVS2DEC_SRC_AEC_INTERNAL_H
#define AVS2DEC_SRC_AEC_INTERNAL_H

/*
 * AVS2 算术熵编码 (AEC) 解码器内部定义。
 *
 * 从 davs2 的 aec.cc / common.h 移植到 C。保持 AVS2 双域 R/LG 算法、
 * 上下文模型、概率转移表、MPS/LPS 更新逻辑以及完整的系数解码。
 *
 * 命名约定：小写字母加下划线，字段名与 davs2 原始 context_set_t 一致。
 */

#include <stdint.h>
#include "levels.h"

/* ---- clz (前导零计数) 辅助函数 ----
 * 用于 AEC 归一化优化: 一次性计算移位位数, 替代逐比特 while 循环.
 * clz32(0) 是未定义行为, 调用前必须检查 != 0. */
#if defined(_MSC_VER)
#include <intrin.h>
static inline int clz32(uint32_t mask) {
    unsigned long leading_zero = 0;
    _BitScanReverse(&leading_zero, mask);
    return 31 - (int)leading_zero;
}
#else
static inline int clz32(uint32_t mask) {
    return __builtin_clz(mask);
}
#endif

/* ---- AEC 算法常量 ---- */
#define LG_PMPS_SHIFTNO 2           /* LG_PMPS 存储值到使用值的移位 */
#define B_BITS          10          /* 编码位宽 */
#define QUARTER_SHIFT   (B_BITS - 2)
#define HALF            (1 << (B_BITS - 1))
#define QUARTER         (1 << (B_BITS - 2))
#define AEC_VALUE_BOUND 254         /* 保证 rs1 不会对 8 位溢出 */

/* 上下文打包宏: cycno(2) | MPS(1) | LG_PMPS(11) */
#define MAKE_CONTEXT(lg_pmps, mps, cycno) \
    (((uint16_t)(cycno) << 0) | ((uint16_t)(mps) << 2) | ((uint16_t)(lg_pmps) << 3))

/* ---- 上下文数量常量 (与 davs2 一致) ---- */
#define NUM_CUTYPE_CTX         6
#define NUM_SPLIT_CTX          3
#define NUM_INTRA_PU_TYPE_CTX  1
#define NUM_MVD_CTX            3
#define NUM_REF_NO_CTX         3
#define NUM_DELTA_QP_CTX       4
#define NUM_INTER_DIR_CTX      15
#define NUM_INTER_DIR_DHP_CTX  3
#define NUM_DMH_MODE_CTX       12
#define NUM_AMP_CTX            2
#define NUM_C_INTRA_MODE_CTX   3
#define NUM_CTP_CTX            9
#define NUM_INTRA_MODE_CTX     7
#define NUM_TU_SPLIT_CTX       3
#define WPM_NUM                3
#define NUM_DIR_SKIP_CTX       4
#define NUM_BLOCK_TYPES        3
#define NUM_MAP_CTX            11
#define NUM_LAST_CG_CTX_LUMA   6
#define NUM_LAST_CG_CTX_CHROMA 6
#define NUM_SIGCG_CTX_LUMA     2
#define NUM_SIGCG_CTX_CHROMA   1
#define NUM_LAST_POS_CTX_LUMA  48
#define NUM_LAST_POS_CTX_CHROMA 12
#define NUM_COEFF_LEVEL_CTX    40
#define NUM_SAO_MERGE_FLAG_CTX 3
#define NUM_SAO_MODE_CTX       1
#define NUM_SAO_OFFSET_CTX     2
#define NUM_INTER_DIR_MIN_CTX  2
#define NUM_ALF_LCU_CTX        4

/* ---- SAO 相关常量 (davs2 风格，仅供 AEC 内部使用) ---- */
#define SAO_MODE_OFF           0
#define SAO_MODE_NEW           1
#define SAO_MODE_MERGE         2
#define SAO_TYPE_EO_0          0
#define SAO_TYPE_BO            1
#define SAO_CLASS_EO_FULL_VALLEY 0
#define SAO_CLASS_EO_FULL_PEAK   1
#define SAO_CLASS_EO_HALF_PEAK   3
#define SAO_CLASS_EO_HALF_VALLEY 4
#define SAO_CLASS_BO             5
#define NUM_SAO_OFFSET           6
#define NUM_SAO_EO_TYPES_LOG2    2
#define NUM_SAO_BO_CLASSES_LOG2  5
#define NUM_SAO_BO_CLASSES_IN_BIT 5

/* ---- 预测模式相关宏 ---- */
#define MAX_PRED_MODES 12
#define INVALID_REF   -1
#define IS_SKIP_MODE(m)      ((m) == PRED_SKIP)
#define IS_NOSKIP_INTER_MODE(m) ((m) == PRED_2Nx2N || (m) == PRED_2NxN || (m) == PRED_Nx2N || \
                                 (m) == PRED_2NxnU || (m) == PRED_2NxnD || \
                                 (m) == PRED_nLx2N || (m) == PRED_nRx2N)
#define IS_HOR_PU_PART(m) ((m) == PRED_2NxN || (m) == PRED_2NxnU || (m) == PRED_2NxnD || (m) == PRED_I_2Nxn)
#define IS_VER_PU_PART(m) ((m) == PRED_Nx2N || (m) == PRED_nLx2N || (m) == PRED_nRx2N || (m) == PRED_I_nx2N)
#define IS_INTRA_MODE(m)  ((m) >= PRED_I_2Nx2N)
#define CU_IS_INTER(p_cu) (!IS_INTRA_MODE((p_cu)->i_cu_type))

/* 跳过模式直接方向 (对应 davs2 direct_skip_mode_e)
 * B 帧和 F 帧的常量值重叠, 由帧类型区分 */
enum {
    DS_NONE  = 0,       /* 无空间直接/skip 模式 */

    /* B 帧空间直接/skip 模式 */
    DS_B_BID = 1,       /* 双向 */
    DS_B_BWD = 2,       /* 后向 */
    DS_B_SYM = 3,       /* 对称 */
    DS_B_FWD = 4,       /* 前向 */

    /* F 帧空间直接/skip 模式 (值与 B 帧重叠) */
    DS_DUAL_1ST   = 1,  /* 双 1st */
    DS_DUAL_2ND   = 2,  /* 双 2nd */
    DS_SINGLE_1ST = 3,  /* 单 1st */
    DS_SINGLE_2ND = 4,  /* 单 2nd */

    DS_MAX_NUM    = 5
};

/* TU split 无效标记 (CU 类型不支持该方向时使用) */
#define TU_SPLIT_INVALID  (-1)

/* 32x32 块的 log2 尺寸 (levels.h 中未定义, AEC 内部使用) */
#define B32X32_IN_BIT  5

/* DCT 系数模式 */
enum {
    DCT_QUAD   = 0,
    DCT_HALF   = 1,
    DCT_DEAULT = 2
};

/* intra 预测分类 */
enum {
    INTRA_PRED_DC_DIAG = 0,
    INTRA_PRED_VER     = 1,
    INTRA_PRED_HOR     = 2
};

/* 16 位打包上下文: cycno(2) | MPS(1) | LG_PMPS(11) */
typedef union {
    uint16_t v;
    struct {
        uint16_t cycno   : 2;
        uint16_t mps     : 1;
        uint16_t lg_pmps : 11;  /* 直接存储 lg_pmps (0..1023)，使用时 >> LG_PMPS_SHIFTNO */
    } b;
} aec_ctx;

/* 上下文集合 (字段名与 davs2 context_set_t 一致) */
typedef struct {
    aec_ctx cu_type_contexts[NUM_CUTYPE_CTX];
    aec_ctx intra_pu_type_contexts[NUM_INTRA_PU_TYPE_CTX];
    aec_ctx cu_split_flag[NUM_SPLIT_CTX];
    aec_ctx transform_split_flag[NUM_TU_SPLIT_CTX];
    aec_ctx shape_of_partition_index[NUM_AMP_CTX];
    aec_ctx pu_reference_index[NUM_REF_NO_CTX];
    aec_ctx cbp_contexts[NUM_CTP_CTX];
    aec_ctx mvd_contexts[2][NUM_MVD_CTX];
    aec_ctx pu_type_index[NUM_INTER_DIR_CTX];
    aec_ctx b_pu_type_min_index[NUM_INTER_DIR_MIN_CTX];
    aec_ctx cu_subtype_index[NUM_DIR_SKIP_CTX];
    aec_ctx weighted_skip_mode[WPM_NUM];
    aec_ctx delta_qp_contexts[NUM_DELTA_QP_CTX];
    aec_ctx intra_luma_pred_mode[NUM_INTRA_MODE_CTX];
    aec_ctx intra_chroma_pred_mode[NUM_C_INTRA_MODE_CTX];
    aec_ctx coeff_run[2][NUM_BLOCK_TYPES][NUM_MAP_CTX];
    aec_ctx coeff_level[NUM_COEFF_LEVEL_CTX];
    aec_ctx last_cg_contexts[NUM_LAST_CG_CTX_LUMA + NUM_LAST_CG_CTX_CHROMA];
    aec_ctx sig_cg_contexts[NUM_SIGCG_CTX_LUMA + NUM_SIGCG_CTX_CHROMA];
    aec_ctx last_coeff_pos[NUM_LAST_POS_CTX_LUMA + NUM_LAST_POS_CTX_CHROMA];
    aec_ctx sao_mergeflag_context[NUM_SAO_MERGE_FLAG_CTX];
    aec_ctx sao_mode_context[NUM_SAO_MODE_CTX];
    aec_ctx sao_offset_context[NUM_SAO_OFFSET_CTX];
    aec_ctx alf_lcu_enable_scmodel[NUM_ALF_LCU_CTX * 3];
} aec_context_set;

/* run-level 对 */
typedef struct {
    int16_t level;
    int16_t run;
} runlevel_pair_t;

/* 系数类型 (10-bit 支持用 int16_t) */
typedef int16_t coeff_t;

/* run-level 解码工作集 (与 davs2 runlevel_t 对应) */
typedef struct {
    const uint8_t (*cg_scan)[2];   /* CG 扫描表 */
    aec_ctx (*p_ctx_run)[NUM_MAP_CTX];
    aec_ctx *p_ctx_level;
    aec_ctx *p_ctx_sig_cg;
    aec_ctx *p_ctx_last_cg;
    aec_ctx *p_ctx_last_pos_in_cg;
    runlevel_pair_t *run_level;    /* CG 内 run-level 缓冲 */
    coeff_t *p_res;                /* 系数输出缓冲 */
    int i_res;                     /* 输出缓冲宽度 */
    int b_swap_xy;
    int i_tu_level;
    int w_tr;
    int h_tr;
    int num_nonzero_cg;
} runlevel_t;

/* SAO 参数 (AEC 解码用) */
typedef struct {
    int modeIdc;
    int typeIdc;
    int startBand;
    int startBand2;
    int offset[8];
} aec_sao_param;

/* AEC 解码用的 CU 描述符 (字段名与 davs2 cu_t 对应) */
typedef struct {
    int8_t i_cu_type;
    int8_t i_cu_level;
    int8_t i_trans_size;
    int8_t b8pdir[4];
    struct { int8_t r[2]; } ref_idx[4];
    int8_t i_cbp;
    int8_t i_qp;
    int8_t i_weighted_skipmode;
    int8_t i_md_directskip_mode;
    /* 邻块 CBP 查找所需信息 */
    int    scu_x;        /* 当前 CU 的 SCU(8x8) X 坐标 */
    int    scu_y;        /* 当前 CU 的 SCU(8x8) Y 坐标 */
    void  *p_frame;      /* avs2_frame 指针, 用于访问 cu_grid */
} aec_cu_t;

/* AEC 解码器状态 (字段名与 davs2 aec_t 对应) */
struct avs2_aec {
    uint8_t *p_buffer;
    int      i_byte_pos;
    int      i_bytes;
    int8_t   i_bits_to_go;
    int      b_bit_error;
    uint64_t i_byte_buf;

    int      b_val_bound;   /* value < QUARTER 标志 */
    int      b_val_domain;  /* 1=R 域, 0=LG 域 */
    uint32_t i_s1;
    uint32_t i_t1;
    uint32_t i_value_s;
    uint32_t i_value_t;

    /* 上下文概率转移表 (查表优化用).
     * 每次 AEC 解码通过指针访问, 避免全局变量的 cache 不友好性.
     * 表在 avs2_aec_create 中初始化, 内容全局只读. */
    const uint16_t *tab_ctx_mps;  /* [4 * 2048] MPS 转移表 */
    const uint16_t *tab_ctx_lps;  /* [4 * 2048] LPS 转移表 */

    aec_context_set syn_ctx;
};

/* ---- 公共 API (与 internal.h 声明一致) ---- */
avs2_aec *avs2_aec_create(const uint16_t *aec_tab_ctx_mps, const uint16_t *aec_tab_ctx_lps);
void      avs2_aec_destroy(avs2_aec *aec);
void      avs2_aec_init_contexts(avs2_aec *aec, int slice_type);
int       avs2_aec_start_decoding(avs2_aec *aec, const uint8_t *buf, int sz, int bit_pos);
int       avs2_aec_decode_bin(avs2_aec *aec, void *ctx);
int       avs2_aec_decode_bin_eq_prob(avs2_aec *aec);
int       avs2_aec_decode_final(avs2_aec *aec);
unsigned  avs2_aec_decode_ue(avs2_aec *aec, void *ctx);
int       avs2_aec_decode_se(avs2_aec *aec, void *ctx);
int       avs2_aec_get_bits_read(avs2_aec *aec);

/* ---- 内部核心函数 ---- */
void aec_init_context_tab(uint16_t aec_tab_ctx_mps[4 * 2048], uint16_t aec_tab_ctx_lps[4 * 2048]);
void aec_new_slice(struct avs2_internal *c);
int  aec_bits_read(avs2_aec *p_aec);

/* ---- 高层 AEC 解码函数 (自包含，仅依赖 aec) ---- */
int  aec_read_split_flag(avs2_aec *p_aec, int i_level);
int  aec_read_dmh_mode(avs2_aec *p_aec, int i_cu_level);
void aec_read_mvds(avs2_aec *p_aec, avs2_mv *p_mvd);
int  aec_read_intra_pmode(avs2_aec *p_aec);
int  aec_read_cu_delta_qp(avs2_aec *p_aec, int i_last_dequant);
int  aec_startcode_follows(avs2_aec *p_aec, int eos_bit);
int  aec_read_alf_lcu_ctrl(avs2_aec *p_aec);
int  aec_read_sao_mergeflag(avs2_aec *p_aec, int mergeleft_avail, int mergeup_avail);
int  aec_read_sao_mode(avs2_aec *p_aec);
void aec_read_sao_offsets(avs2_aec *p_aec, aec_sao_param *p_sao_param, int *offset);
int  aec_read_sao_type(avs2_aec *p_aec, aec_sao_param *p_sao_param);

/* ---- 高层 AEC 解码函数 (依赖 aec_cu_t) ---- */
int  aec_read_intra_cu_type(avs2_aec *p_aec, aec_cu_t *p_cu, int b_sdip, int enable_nsqt);
int  aec_read_cu_type(avs2_aec *p_aec, aec_cu_t *p_cu, int img_type, int b_amp,
                      int b_mhp, int b_wsm, int num_references);
int  aec_read_cu_type_sframe(avs2_aec *p_aec);
void aec_read_inter_pred_dir(avs2_aec *p_aec, aec_cu_t *p_cu, int img_type,
                             int enable_dhp, int num_references);
int  aec_read_intra_pmode_c(avs2_aec *p_aec, int luma_mode, int c_ipred_mode_ctx);
int  aec_read_cbp_simple(avs2_aec *p_aec, aec_cu_t *p_cu, int chroma_format);
int  aec_read_ctp_y_simple(avs2_aec *p_aec, aec_cu_t *p_cu, int b8);
int  aec_read_ctp_y(avs2_aec *p_aec, aec_cu_t *p_cu, int b8);
int  aec_read_cbp(avs2_aec *p_aec, aec_cu_t *p_cu, int chroma_format);

/* ---- 系数解码 ---- */
int   aec_read_run_level(avs2_aec *p_aec, aec_cu_t *p_cu, int num_cg, int b_luma,
                         int is_dc_diag, runlevel_t *runlevel, int scale, int shift);
int8_t cu_get_block_coeffs(avs2_aec *p_aec, runlevel_t *runlevel, aec_cu_t *p_cu,
                           coeff_t *p_res, int w_tr, int h_tr, int i_tu_level,
                           int b_luma, int intra_pred_class, int b_swap_xy,
                           int scale, int shift, int wq_size_id);

#endif /* AVS2DEC_SRC_AEC_INTERNAL_H */
