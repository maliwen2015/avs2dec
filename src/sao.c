/*
 * SAO (Sample Adaptive Offset) 环路滤波 (从 davs2 sao.cc 移植到 C)。
 *
 * 完整算法:
 *   - EO (Edge Offset) 4 方向: EO_0(水平), EO_90(垂直), EO_135, EO_45;
 *     每像素根据左右/上下/对角邻居的符号差计算 edge_type (0..4),
 *     加上对应类别的偏移量。
 *   - BO (Band Offset) 32 带: 将像素值按 bit_depth-5 位移分为 32 个带,
 *     每个带加上对应偏移量。
 *   - 邻域可用性控制边界像素是否参与滤波。
 *
 * 命名: 小写加下划线。支持 8/10 位, 通过 bit_depth 分支。
 */

#include "internal.h"
#include "aec_internal.h"
#include <string.h>

/* 辅助宏 (本文件局部使用) */
#ifndef AVS2_MIN
#define AVS2_MIN(a, b)     ((a) < (b) ? (a) : (b))
#endif
#ifndef AVS2_CLIP3
#define AVS2_CLIP3(lo, hi, x) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))
#endif

/* --------------------------------------------------------------------------
 * 常量定义 (与 davs2 common.h / defines.h 一致)
 * -------------------------------------------------------------------------- */

#define SAO_SHIFT_PIX_NUM     4    /* SAO 滤波区域边界扩展像素数 */
#define MAX_NUM_SAO_CLASSES   32   /* BO 带数 + EO 类别数上限 */

/* SAO 邻域方向索引 (avail 数组) */
enum {
    SAO_NB_T  = 0,   /* top         */
    SAO_NB_D  = 1,   /* down        */
    SAO_NB_L  = 2,   /* left        */
    SAO_NB_R  = 3,   /* right       */
    SAO_NB_TL = 4,   /* top-left    */
    SAO_NB_TR = 5,   /* top-right   */
    SAO_NB_DL = 6,   /* down-left   */
    SAO_NB_DR = 7    /* down-right  */
};

/* SAO 滤波类型 (内部索引, 与 davs2 SAO_TYPE 一致)。
 * 注意: SAO_TYPE_EO_0 和 SAO_TYPE_BO 已在 aec_internal.h 中定义 (0 和 1),
 *   用于 AEC 读取阶段的哨兵值。这里定义的 SAO_FILT_* 用于 avs2_sao_param.type_idc,
 *   其中 BO 的内部值为 4 (避免与 EO_90=1 冲突)。 */
#define SAO_FILT_EO_0    0    /* 水平方向 */
#define SAO_FILT_EO_90   1    /* 垂直方向 */
#define SAO_FILT_EO_135  2
#define SAO_FILT_EO_45   3
#define SAO_FILT_BO      4

/* SAO EO 类别 (edge_type 值) */
#define SAO_CLASS_FULL_VALLEY  0
#define SAO_CLASS_HALF_VALLEY  1
#define SAO_CLASS_PLAIN        2
#define SAO_CLASS_HALF_PEAK    3
#define SAO_CLASS_FULL_PEAK    4

/* ------------------------------------------------------------------
 * SAO 滤波函数模板: 通过宏为 8 位 (uint8_t) 和 10 位 (uint16_t) 各生成
 * 一份实现, 再由分发函数按 bit_depth 选择调用。
 * stride 均为元素步长 (byte_stride / bytes_per_sample)。
 * ------------------------------------------------------------------ */

/* --------------------------------------------------------------------------
 * EO_0: 水平方向 (左右邻居)
 * -------------------------------------------------------------------------- */
#define DEFINE_SAO_EO_0(func_name, pel_t) \
static void func_name(pel_t *dst, int dst_stride, const pel_t *src, \
                      int src_stride, int w, int h, int bit_depth, \
                      const int *avail, const int *offset) \
{ \
    const int max_pel = (1 << bit_depth) - 1; \
    int left_sign, right_sign; \
    int edge_type; \
    int x, y; \
    int pel_diff; \
    int sx = avail[SAO_NB_L] ? 0 : 1; \
    int ex = avail[SAO_NB_R] ? w : (w - 1); \
    for (y = 0; y < h; y++) { \
        pel_diff = src[sx] - src[sx - 1]; \
        left_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
        for (x = sx; x < ex; x++) { \
            pel_diff = src[x] - src[x + 1]; \
            right_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
            edge_type = left_sign + right_sign + 2; \
            left_sign = -right_sign; \
            dst[x] = (pel_t)AVS2_CLIP3(0, max_pel, src[x] + offset[edge_type]); \
        } \
        src += src_stride; \
        dst += dst_stride; \
    } \
}

DEFINE_SAO_EO_0(sao_eo_0_c_8,  uint8_t)
DEFINE_SAO_EO_0(sao_eo_0_c_16, uint16_t)

void sao_eo_0_c(uint8_t *dst, int dst_stride, const uint8_t *src,
                int src_stride, int w, int h, int bit_depth,
                const int *avail, const int *offset)
{
    if (bit_depth > 8)
        sao_eo_0_c_16((uint16_t *)dst, dst_stride, (const uint16_t *)src,
                      src_stride, w, h, bit_depth, avail, offset);
    else
        sao_eo_0_c_8(dst, dst_stride, src, src_stride, w, h, bit_depth,
                     avail, offset);
}

/* --------------------------------------------------------------------------
 * EO_90: 垂直方向 (上下邻居)
 * -------------------------------------------------------------------------- */
#define DEFINE_SAO_EO_90(func_name, pel_t) \
static void func_name(pel_t *dst, int dst_stride, const pel_t *src, \
                      int src_stride, int w, int h, int bit_depth, \
                      const int *avail, const int *offset) \
{ \
    const int max_pel = (1 << bit_depth) - 1; \
    int edge_type; \
    int x, y; \
    int sy = avail[SAO_NB_T] ? 0 : 1; \
    int ey = avail[SAO_NB_D] ? h : (h - 1); \
    for (x = 0; x < w; x++) { \
        int pel_diff = src[sy * src_stride + x] - src[(sy - 1) * src_stride + x]; \
        int top_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
        for (y = sy; y < ey; y++) { \
            int pd = src[y * src_stride + x] - src[(y + 1) * src_stride + x]; \
            int down_sign = (pd > 0) ? 1 : ((pd < 0) ? -1 : 0); \
            edge_type = down_sign + top_sign + 2; \
            top_sign = -down_sign; \
            dst[y * dst_stride + x] = \
                (pel_t)AVS2_CLIP3(0, max_pel, src[y * src_stride + x] + offset[edge_type]); \
        } \
    } \
}

DEFINE_SAO_EO_90(sao_eo_90_c_8,  uint8_t)
DEFINE_SAO_EO_90(sao_eo_90_c_16, uint16_t)

void sao_eo_90_c(uint8_t *dst, int dst_stride, const uint8_t *src,
                 int src_stride, int w, int h, int bit_depth,
                 const int *avail, const int *offset)
{
    if (bit_depth > 8)
        sao_eo_90_c_16((uint16_t *)dst, dst_stride, (const uint16_t *)src,
                       src_stride, w, h, bit_depth, avail, offset);
    else
        sao_eo_90_c_8(dst, dst_stride, src, src_stride, w, h, bit_depth,
                      avail, offset);
}

/* --------------------------------------------------------------------------
 * EO_135: 135 度对角方向 (左上-右下邻居)
 * -------------------------------------------------------------------------- */
#define DEFINE_SAO_EO_135(func_name, pel_t) \
static void func_name(pel_t *dst, int dst_stride, const pel_t *src, \
                      int src_stride, int w, int h, int bit_depth, \
                      const int *avail, const int *offset) \
{ \
    int8_t sign_buf[AVS2_LCU_MAX + 32];  /* 上一行符号缓存 */ \
    int8_t *uprow_s = sign_buf + 16; \
    const int max_pel = (1 << bit_depth) - 1; \
    int reg = 0; \
    int sx, ex; \
    int sx_0, ex_0, sx_n, ex_n; \
    int top_sign, down_sign; \
    int edge_type; \
    int pel_diff; \
    int x, y; \
    sx = avail[SAO_NB_L] ? 0 : 1; \
    ex = avail[SAO_NB_R] ? w : (w - 1); \
    /* 初始化符号行缓存 */ \
    for (x = sx; x < ex; x++) { \
        pel_diff = src[src_stride + x + 1] - src[x]; \
        top_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
        uprow_s[x + 1] = (int8_t)top_sign; \
    } \
    /* 第一行 */ \
    sx_0 = avail[SAO_NB_TL] ? 0 : 1; \
    ex_0 = avail[SAO_NB_T] ? (avail[SAO_NB_R] ? w : (w - 1)) : 1; \
    for (x = sx_0; x < ex_0; x++) { \
        pel_diff = src[x] - src[-src_stride + x - 1]; \
        top_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
        edge_type = top_sign - uprow_s[x + 1] + 2; \
        dst[x] = (pel_t)AVS2_CLIP3(0, max_pel, src[x] + offset[edge_type]); \
    } \
    /* 中间行 */ \
    for (y = 1; y < h - 1; y++) { \
        src += src_stride; \
        dst += dst_stride; \
        for (x = sx; x < ex; x++) { \
            if (x == sx) { \
                pel_diff = src[x] - src[-src_stride + x - 1]; \
                top_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
                uprow_s[x] = (int8_t)top_sign; \
            } \
            pel_diff = src[x] - src[src_stride + x + 1]; \
            down_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
            edge_type = down_sign + uprow_s[x] + 2; \
            dst[x] = (pel_t)AVS2_CLIP3(0, max_pel, src[x] + offset[edge_type]); \
            uprow_s[x] = (int8_t)reg; \
            reg = -down_sign; \
        } \
    } \
    /* 最后一行 */ \
    sx_n = avail[SAO_NB_D] ? (avail[SAO_NB_L] ? 0 : 1) : (w - 1); \
    ex_n = avail[SAO_NB_DR] ? w : (w - 1); \
    src += src_stride; \
    dst += dst_stride; \
    for (x = sx_n; x < ex_n; x++) { \
        if (x == sx) { \
            pel_diff = src[x] - src[-src_stride + x - 1]; \
            top_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
            uprow_s[x] = (int8_t)top_sign; \
        } \
        pel_diff = src[x] - src[src_stride + x + 1]; \
        down_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
        edge_type = down_sign + uprow_s[x] + 2; \
        dst[x] = (pel_t)AVS2_CLIP3(0, max_pel, src[x] + offset[edge_type]); \
    } \
}

DEFINE_SAO_EO_135(sao_eo_135_c_8,  uint8_t)
DEFINE_SAO_EO_135(sao_eo_135_c_16, uint16_t)

void sao_eo_135_c(uint8_t *dst, int dst_stride, const uint8_t *src,
                  int src_stride, int w, int h, int bit_depth,
                  const int *avail, const int *offset)
{
    if (bit_depth > 8)
        sao_eo_135_c_16((uint16_t *)dst, dst_stride, (const uint16_t *)src,
                        src_stride, w, h, bit_depth, avail, offset);
    else
        sao_eo_135_c_8(dst, dst_stride, src, src_stride, w, h, bit_depth,
                       avail, offset);
}

/* --------------------------------------------------------------------------
 * EO_45: 45 度对角方向 (右上-左下邻居)
 * -------------------------------------------------------------------------- */
#define DEFINE_SAO_EO_45(func_name, pel_t) \
static void func_name(pel_t *dst, int dst_stride, const pel_t *src, \
                      int src_stride, int w, int h, int bit_depth, \
                      const int *avail, const int *offset) \
{ \
    int8_t sign_buf[AVS2_LCU_MAX + 32];  /* 上一行符号缓存 */ \
    int8_t *uprow_s = sign_buf + 16; \
    const int max_pel = (1 << bit_depth) - 1; \
    int sx, ex; \
    int sx_0, ex_0, sx_n, ex_n; \
    int top_sign, down_sign; \
    int edge_type; \
    int pel_diff; \
    int x, y; \
    sx = avail[SAO_NB_L] ? 0 : 1; \
    ex = avail[SAO_NB_R] ? w : (w - 1); \
    /* 初始化符号行缓存 */ \
    for (x = sx; x < ex; x++) { \
        pel_diff = src[src_stride + x - 1] - src[x]; \
        top_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
        uprow_s[x - 1] = (int8_t)top_sign; \
    } \
    /* 第一行 */ \
    sx_0 = avail[SAO_NB_T] ? (avail[SAO_NB_L] ? 0 : 1) : (w - 1); \
    ex_0 = avail[SAO_NB_TR] ? w : (w - 1); \
    for (x = sx_0; x < ex_0; x++) { \
        pel_diff = src[x] - src[-src_stride + x + 1]; \
        top_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
        edge_type = top_sign - uprow_s[x - 1] + 2; \
        dst[x] = (pel_t)AVS2_CLIP3(0, max_pel, src[x] + offset[edge_type]); \
    } \
    /* 中间行 */ \
    for (y = 1; y < h - 1; y++) { \
        src += src_stride; \
        dst += dst_stride; \
        for (x = sx; x < ex; x++) { \
            if (x == ex - 1) { \
                pel_diff = src[x] - src[-src_stride + x + 1]; \
                top_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
                uprow_s[x] = (int8_t)top_sign; \
            } \
            pel_diff = src[x] - src[src_stride + x - 1]; \
            down_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
            edge_type = down_sign + uprow_s[x] + 2; \
            dst[x] = (pel_t)AVS2_CLIP3(0, max_pel, src[x] + offset[edge_type]); \
            uprow_s[x - 1] = (int8_t)(-down_sign); \
        } \
    } \
    /* 最后一行 */ \
    sx_n = avail[SAO_NB_DL] ? 0 : 1; \
    ex_n = avail[SAO_NB_D] ? (avail[SAO_NB_R] ? w : (w - 1)) : 1; \
    src += src_stride; \
    dst += dst_stride; \
    for (x = sx_n; x < ex_n; x++) { \
        if (x == ex - 1) { \
            pel_diff = src[x] - src[-src_stride + x + 1]; \
            top_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
            uprow_s[x] = (int8_t)top_sign; \
        } \
        pel_diff = src[x] - src[src_stride + x - 1]; \
        down_sign = (pel_diff > 0) ? 1 : ((pel_diff < 0) ? -1 : 0); \
        edge_type = down_sign + uprow_s[x] + 2; \
        dst[x] = (pel_t)AVS2_CLIP3(0, max_pel, src[x] + offset[edge_type]); \
    } \
}

DEFINE_SAO_EO_45(sao_eo_45_c_8,  uint8_t)
DEFINE_SAO_EO_45(sao_eo_45_c_16, uint16_t)

void sao_eo_45_c(uint8_t *dst, int dst_stride, const uint8_t *src,
                 int src_stride, int w, int h, int bit_depth,
                 const int *avail, const int *offset)
{
    if (bit_depth > 8)
        sao_eo_45_c_16((uint16_t *)dst, dst_stride, (const uint16_t *)src,
                       src_stride, w, h, bit_depth, avail, offset);
    else
        sao_eo_45_c_8(dst, dst_stride, src, src_stride, w, h, bit_depth,
                      avail, offset);
}

/* --------------------------------------------------------------------------
 * BO: 带偏移 (32 带)
 * -------------------------------------------------------------------------- */
#define DEFINE_SAO_BO(func_name, pel_t) \
static void func_name(pel_t *dst, int dst_stride, const pel_t *src, \
                      int src_stride, int w, int h, int bit_depth, \
                      const int *offset) \
{ \
    const int max_pel = (1 << bit_depth) - 1; \
    const int band_shift = bit_depth - NUM_SAO_BO_CLASSES_IN_BIT; \
    int edge_type; \
    int x, y; \
    for (y = 0; y < h; y++) { \
        for (x = 0; x < w; x++) { \
            edge_type = src[x] >> band_shift; \
            dst[x] = (pel_t)AVS2_CLIP3(0, max_pel, src[x] + offset[edge_type]); \
        } \
        src += src_stride; \
        dst += dst_stride; \
    } \
}

DEFINE_SAO_BO(sao_bo_c_8,  uint8_t)
DEFINE_SAO_BO(sao_bo_c_16, uint16_t)

void sao_bo_c(uint8_t *dst, int dst_stride, const uint8_t *src,
              int src_stride, int w, int h, int bit_depth,
              const int *offset)
{
    if (bit_depth > 8)
        sao_bo_c_16((uint16_t *)dst, dst_stride, (const uint16_t *)src,
                    src_stride, w, h, bit_depth, offset);
    else
        sao_bo_c_8(dst, dst_stride, src, src_stride, w, h, bit_depth, offset);
}

/* --------------------------------------------------------------------------
 * 对一个块应用 SAO 滤波 (按 type_idc 分发到 EO 或 BO)
 * 对应 davs2 sao_lcu 中的单分量滤波部分
 * -------------------------------------------------------------------------- */
void avs2_sao_on_block(avs2_frame *dst_frm, avs2_frame *src_frm, int pl,
                       int pix_x, int pix_y, int blk_w, int blk_h,
                       int bit_depth, const int *avail, const avs2_sao_param *sp)
{
    const int bps = src_frm->bytes_per_sample;
    const ptrdiff_t src_stride = src_frm->stride[pl] / bps;
    const ptrdiff_t dst_stride = dst_frm->stride[pl] / bps;
    uint8_t *src = src_frm->data[pl] + (ptrdiff_t)pix_y * src_frm->stride[pl]
                   + (ptrdiff_t)pix_x * bps;
    uint8_t *dst = dst_frm->data[pl] + (ptrdiff_t)pix_y * dst_frm->stride[pl]
                   + (ptrdiff_t)pix_x * bps;

    if (sp->mode_idc == SAO_MODE_OFF) {
        return;
    }

    if (sp->type_idc == SAO_FILT_BO) {
        avs2_dsp_table.sao_bo(dst, (int)dst_stride, src, (int)src_stride,
                              blk_w, blk_h, bit_depth, sp->offset);
    } else {
        int eo_idx = sp->type_idc;  /* 0..3 */
        if (eo_idx >= 0 && eo_idx < 4) {
            avs2_dsp_table.sao_eo[eo_idx](dst, (int)dst_stride, src, (int)src_stride,
                                          blk_w, blk_h, bit_depth, avail, sp->offset);
        }
    }
}

/* --------------------------------------------------------------------------
 * 从 AEC 码流读取一个 LCU 的 SAO 参数 (对应 davs2 sao_read_lcu)
 * sao_param 为 3 个分量的参数数组 (Y, Cb, Cr)
 * -------------------------------------------------------------------------- */
void avs2_sao_read_param(avs2_frame_ctx *fc, struct avs2_internal *c,
                         int lcu_xy, const uint8_t *slice_sao_on,
                         avs2_sao_param *sao_param)
{
    avs2_frame *f = fc->fdec;
    const int w_in_lcu = f->w_lcu;
    const int lcu_x = lcu_xy % w_in_lcu;
    const int lcu_y = lcu_xy / w_in_lcu;
    avs2_aec *aec = fc->aec;
    int merge_mode = 0;
    int merge_top_avail, merge_left_avail;
    int i;

    (void)c;

    /* 邻域可用性 (avs2dec 无逐 CU slice 号, 简化为 LCU 位置判定) */
    merge_left_avail = (lcu_x != 0);
    merge_top_avail  = (lcu_y != 0);

    /* 读取合并标志 */
    if (merge_left_avail || merge_top_avail) {
        merge_mode = aec_read_sao_mergeflag(aec, merge_left_avail, merge_top_avail);
    }

    if (merge_mode) {
        /* 合并模式: 从左或上 LCU 复制参数 */
        int src_lcu_xy;
        if (merge_mode == 2) {
            src_lcu_xy = lcu_xy - 1;             /* 左 */
        } else {
            src_lcu_xy = lcu_xy - w_in_lcu;       /* 上 */
        }
        for (i = 0; i < 3; i++) {
            avs2_sao_param *src_sp = &f->sao_params[src_lcu_xy * 3 + i];
            sao_param[i] = *src_sp;
        }
    } else {
        /* 新参数模式: 逐分量读取 */
        for (i = 0; i < 3; i++) {
            if (!slice_sao_on[i]) {
                sao_param[i].mode_idc = SAO_MODE_OFF;
                sao_param[i].type_idc = SAO_FILT_EO_0;
                continue;
            }

            {
                aec_sao_param aec_sp;
                int sao_mode, sao_type;
                int offset[4];
                int st_bnd[2];
                int db_temp;

                aec_sp.modeIdc = SAO_MODE_OFF;
                aec_sp.typeIdc = SAO_TYPE_EO_0;

                sao_mode = aec_read_sao_mode(aec);
                switch (sao_mode) {
                case 0:
                    aec_sp.modeIdc = SAO_MODE_OFF;
                    break;
                case 1:
                    aec_sp.modeIdc = SAO_MODE_NEW;
                    aec_sp.typeIdc = SAO_TYPE_BO;
                    break;
                case 3:
                    aec_sp.modeIdc = SAO_MODE_NEW;
                    aec_sp.typeIdc = SAO_TYPE_EO_0;
                    break;
                default:
                    aec_sp.modeIdc = SAO_MODE_OFF;
                    break;
                }

                if (aec_sp.modeIdc == SAO_MODE_NEW) {
                    aec_read_sao_offsets(aec, &aec_sp, offset);
                    sao_type = aec_read_sao_type(aec, &aec_sp);

                    if (aec_sp.typeIdc == SAO_TYPE_BO) {
                        /* BO: 32 带偏移 */
                        memset(sao_param[i].offset, 0, MAX_NUM_SAO_CLASSES * sizeof(int));
                        db_temp = sao_type >> NUM_SAO_BO_CLASSES_LOG2;
                        st_bnd[0] = sao_type - (db_temp << NUM_SAO_BO_CLASSES_LOG2);
                        st_bnd[1] = (st_bnd[0] + db_temp) % 32;
                        sao_param[i].start_band  = (uint8_t)st_bnd[0];
                        sao_param[i].start_band2 = (uint8_t)st_bnd[1];
                        sao_param[i].offset[st_bnd[0]            ] = offset[0];
                        sao_param[i].offset[(st_bnd[0] + 1) % 32 ] = offset[1];
                        sao_param[i].offset[st_bnd[1]            ] = offset[2];
                        sao_param[i].offset[(st_bnd[1] + 1) % 32 ] = offset[3];
                        sao_param[i].type_idc = SAO_FILT_BO;
                    } else {
                        /* EO: 5 类偏移 (0..4) */
                        sao_param[i].type_idc = (uint8_t)sao_type;
                        sao_param[i].offset[SAO_CLASS_FULL_VALLEY] = offset[0];
                        sao_param[i].offset[SAO_CLASS_HALF_VALLEY] = offset[1];
                        sao_param[i].offset[SAO_CLASS_PLAIN      ] = 0;
                        sao_param[i].offset[SAO_CLASS_HALF_PEAK  ] = offset[2];
                        sao_param[i].offset[SAO_CLASS_FULL_PEAK  ] = offset[3];
                    }
                } else {
                    sao_param[i].type_idc = SAO_FILT_EO_0;
                }

                sao_param[i].mode_idc = (uint8_t)aec_sp.modeIdc;
            }
        }
    }

    /* 存储到帧的 sao_params 数组 (供后续 LCU 合并引用) */
    for (i = 0; i < 3; i++) {
        f->sao_params[lcu_xy * 3 + i] = sao_param[i];
    }
}

/* --------------------------------------------------------------------------
 * SAO DSP 初始化 (对应 davs2 davs2_sao_init)
 * -------------------------------------------------------------------------- */
void avs2_sao_init(void)
{
    avs2_dsp_table.sao_eo[SAO_FILT_EO_0  ] = sao_eo_0_c;
    avs2_dsp_table.sao_eo[SAO_FILT_EO_90 ] = sao_eo_90_c;
    avs2_dsp_table.sao_eo[SAO_FILT_EO_135] = sao_eo_135_c;
    avs2_dsp_table.sao_eo[SAO_FILT_EO_45 ] = sao_eo_45_c;
    avs2_dsp_table.sao_bo                  = sao_bo_c;
}
