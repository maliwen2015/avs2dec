/*
 * ipred.c
 *
 * 帧内预测实现 (从 davs2 intra.cc 移植为 C)。
 * 实现 AVS2 全部帧内预测模式:
 *   - DC 预测 (模式 0)
 *   - Plane 预测 (模式 1)
 *   - Bilinear 预测 (模式 2)
 *   - 30 种角度预测模式 (模式 3..32)
 *   - 参考样本准备 (intra edge fill)
 *
 * 参考: davs2 source/common/intra.cc, GB/T 33475.2
 */

#include "internal.h"
#include "tables.h"
#include <string.h>
#include <stdio.h>

/* ===================================================================
 * 类型与宏定义
 * =================================================================== */

/* 内联函数修饰符 (项目内统一未定义, 此处补充) */
#ifndef AVS2_INLINE
#define AVS2_INLINE inline
#endif

/* 帧内预测像素类型 (uint16_t 支持 10-bit) */
typedef uint16_t pel_t;
/* 帧内预测中间计算类型 (10-bit 时用 int32_t) */
typedef int32_t  itr_t;

/* DC 默认填充值, 对应 davs2 g_dc_value。
 * 在序列头解析后由 avs2_ipred_set_dc_value 设置为 1<<(bit_depth-1)。 */
int g_dc_value = 128;

#define UNUSED_PARAMETER(x)  (void)(x)

/* 栈缓冲区 32 字节对齐宏 (与 itx_simd.c 一致) */
#ifndef AVS2_ALIGN32
#if defined(_MSC_VER)
#define AVS2_ALIGN32(x) __declspec(align(32)) x
#else
#define AVS2_ALIGN32(x) x __attribute__((aligned(32)))
#endif
#endif

/* 邻域可用性标志位 */
#define MD_I_LEFT        0
#define MD_I_TOP         1
#define MD_I_LEFT_DOWN   2
#define MD_I_TOP_RIGHT   3
#define MD_I_TOP_LEFT    4
#define IS_NEIGHBOR_AVAIL(i_avai, md)  ((i_avai) & (1U << (md)))

/* 限幅宏 */
static AVS2_INLINE int intra_clip3(int low, int high, int v)
{
    return v < low ? low : (v > high ? high : v);
}

static AVS2_INLINE int intra_min(int a, int b) { return a < b ? a : b; }
static AVS2_INLINE int intra_max(int a, int b) { return a > b ? a : b; }

/* 按 bytes_per_sample 读取一个像素, 扩展为 pel_t (uint16_t) */
static AVS2_INLINE pel_t pel_read(const uint8_t *p, int bps)
{
    return bps > 1 ? *(const pel_t *)(const void *)p : (pel_t)*p;
}

/* 从帧缓冲 (bps 字节/像素, 连续) 复制 num 个像素到 EP (uint16_t) */
static AVS2_INLINE void copy_to_ep(pel_t *ep, const uint8_t *src, int num, int bps)
{
    int i;
    for (i = 0; i < num; i++) {
        ep[i] = pel_read(src + (ptrdiff_t)i * bps, bps);
    }
}


/* ===================================================================
 * 常量表 (从 davs2 intra.cc 移植)
 * =================================================================== */

/* 角度模式方向 dx 表 (模式 0..32) */
static const int8_t tab_auc_dir_dx[NUM_INTRA_MODE] = {
     0,  0,  0, 11,  2,
    11,  1,  8,  1,  4,
     1,  1,  0,  1,  1,
     4,  1,  8,  1, 11,
     2, 11,  4,  8,  0,
     8,  4, 11,  2, 11,
     1,  8,  1
};

/* 角度模式方向 dy 表 (模式 0..32) */
static const int8_t tab_auc_dir_dy[NUM_INTRA_MODE] = {
     0,   0,   0, -4,  -1,
    -8,  -1, -11, -2, -11,
    -4,  -8,   0,  8,   4,
    11,   2,  11,  1,   8,
     1,   4,   1,  1,   0,
    -1,  -1,  -4,  -1,  -8,
    -1, -11,  -2
};

/* 角度模式 dx/dy 和 dy/dx 分数表 (用于 get_context_pixel) */
const int8_t tab_auc_dir_dxdy[2][NUM_INTRA_MODE][2] = {
    {
        /* dx/dy */
        {  0, 0 }, {  0, 0 }, {  0, 0 }, { 11, 2 }, {  2, 0 },
        { 11, 3 }, {  1, 0 }, { 93, 7 }, {  1, 1 }, { 93, 8 },
        {  1, 2 }, {  1, 3 }, {  0, 0 }, {  1, 3 }, {  1, 2 },
        { 93, 8 }, {  1, 1 }, { 93, 7 }, {  1, 0 }, { 11, 3 },
        {  2, 0 }, { 11, 2 }, {  4, 0 }, {  8, 0 }, {  0, 0 },
        {  8, 0 }, {  4, 0 }, { 11, 2 }, {  2, 0 }, { 11, 3 },
        {  1, 0 }, { 93, 7 }, {  1, 1 },
    }, {
        /* dy/dx */
        {  0, 0 }, {  0, 0 }, {  0, 0 }, { 93, 8 }, {  1, 1 },
        { 93, 7 }, {  1, 0 }, { 11, 3 }, {  2, 0 }, { 11, 2 },
        {  4, 0 }, {  8, 0 }, {  0, 0 }, {  8, 0 }, {  4, 0 },
        { 11, 2 }, {  2, 0 }, { 11, 3 }, {  1, 0 }, { 93, 7 },
        {  1, 1 }, { 93, 8 }, {  1, 2 }, {  1, 3 }, {  0, 0 },
        {  1, 3 }, {  1, 2 }, { 93, 8 }, {  1, 1 }, { 93, 7 },
        {  1, 0 }, { 11, 3 }, {  2, 0 }
    }
};

/* log2 尺寸查表 (索引为块尺寸, 值为 log2) */
static const int8_t tab_log2size[MAX_CU_SIZE + 1] = {
    -1, -1, -1, -1,  2, -1, -1, -1,
     3, -1, -1, -1, -1, -1, -1, -1,
     4, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
     5, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1,
     6
};

/* 左下可用性表 (64x64 LCU, 16x16 索引) */
static const int8_t tab_dl_avail64[16 * 16] = {
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

static const int8_t tab_dl_avail32[8 * 8] = {
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0
};

static const int8_t tab_dl_avail16[4 * 4] = {
    1, 0, 1, 0,
    1, 0, 0, 0,
    1, 0, 1, 0,
    0, 0, 0, 0
};

static const int8_t tab_dl_avail8[2 * 2] = {
    1, 0,
    0, 0
};

/* 右上可用性表 */
static const int8_t tab_tr_avail64[16 * 16] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
};

static const int8_t tab_tr_avail32[8 * 8] = {
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 0
};

static const int8_t tab_tr_avail16[4 * 4] = {
    1, 1, 1, 1,
    1, 0, 1, 0,
    1, 1, 1, 0,
    1, 0, 1, 0
};

static const int8_t tab_tr_avail8[2 * 2] = {
    1, 1,
    1, 0
};

/* 可用性表指针数组 (索引为 LCU 尺寸的 log2, 3..6) */
static const int8_t *tab_dl_avails[MAX_CU_SIZE_IN_BIT + 1] = {
    NULL, NULL, NULL, tab_dl_avail8, tab_dl_avail16, tab_dl_avail32, tab_dl_avail64
};
static const int8_t *tab_tr_avails[MAX_CU_SIZE_IN_BIT + 1] = {
    NULL, NULL, NULL, tab_tr_avail8, tab_tr_avail16, tab_tr_avail32, tab_tr_avail64
};


/* ===================================================================
 * 邻域可用性查询 (供帧间 MVP / 空间直接模式复用)
 * =================================================================== */

/* 检查 TOPRIGHT 4x4 块在 LCU Z-scan 重建顺序下是否已重建。
 * \param pix_x, pix_y  当前块左上角绝对像素坐标
 * \param bsx           当前块宽度 (像素)
 * \param lcu_level     LCU 的 log2 尺寸 (通常为 6)
 * \return 1 表示 TOPRIGHT 邻居已重建 (可用), 0 表示不可用。
 * 对应 davs2 的 p_tab_TR_avail 查表逻辑 (cu_get_neighbors /
 * cu_get_neighbors_default_mvp 中的 block_available_TR 计算)。 */
int avs2_check_topright_avail(int pix_x, int pix_y, int bsx, int lcu_level)
{
    int log2_lcu_in_scu;
    int scu_mask;
    int x1_spu, y0_spu;
    int x4_tr, y4_tr;
    int idx;
    const int8_t *tab_tr;

    if (lcu_level < 3 || lcu_level > MAX_CU_SIZE_IN_BIT) {
        return 0;
    }

    log2_lcu_in_scu = lcu_level - B4X4_IN_BIT;   /* lcu_level - 2 */
    scu_mask = (1 << log2_lcu_in_scu) - 1;

    /* 当前块右下角 SPU 列 (绝对) 和顶部 SPU 行 (绝对) */
    x1_spu = (pix_x + bsx - 1) >> MIN_PU_SIZE_IN_BIT;
    y0_spu = pix_y >> MIN_PU_SIZE_IN_BIT;

    /* 转为 LCU 内相对坐标 (4x4 单位) */
    x4_tr = x1_spu & scu_mask;
    y4_tr = y0_spu & scu_mask;

    tab_tr = tab_tr_avails[lcu_level];
    if (tab_tr == NULL) {
        return 0;
    }

    idx = (y4_tr << log2_lcu_in_scu) + x4_tr;
    return tab_tr[idx] ? 1 : 0;
}


/* ===================================================================
 * 内存辅助函数
 * =================================================================== */

/* 将 val 重复写入 num 个像素位置 */
static AVS2_INLINE void mem_repeat_p(pel_t *dst, pel_t val, int num)
{
    while (num--) {
        *dst++ = val;
    }
}

/* 从 src 按步长 i_src 复制 num 个像素到 dst (垂直方向) */
static AVS2_INLINE void memcpy_vh_pp(pel_t *dst, pel_t *src, int i_src, int num)
{
    while (num--) {
        *dst++ = *src;
        src += i_src;
    }
}


/* ===================================================================
 * 基础预测函数
 * =================================================================== */

/* 垂直预测: 复制上方参考样本 */
void intra_pred_ver_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst, int mode,
                      int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t *p_src = src + 1;
    int y;
    UNUSED_PARAMETER(mode);
    UNUSED_PARAMETER(bit_depth);

    for (y = bsy; y != 0; y--) {
        memcpy(dst, p_src, (size_t)(bsx * sizeof(pel_t)));
        dst += i_dst;
    }
}

/* 水平预测: 复制左侧参考样本 */
void intra_pred_hor_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst, int mode,
                      int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t *p_src = src - 1;
    int x, y;
    UNUSED_PARAMETER(mode);
    UNUSED_PARAMETER(bit_depth);

    for (y = 0; y < bsy; y++) {
        for (x = 0; x < bsx; x++) {
            dst[x] = p_src[-y];
        }
        dst += i_dst;
    }
}

/* DC 预测: 取左侧和上方参考样本均值 */
/* mode 高 8 位为 b_top, 低 8 位为 b_left */
void intra_pred_dc_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst, int mode,
                     int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int b_top  = mode >> 8;
    int b_left = mode & 0xFF;
    pel_t *p_src = src - 1;
    int dc_value = 0;
    int max_val = (1 << bit_depth) - 1;
    int x, y;

    /* 计算 DC 值 */
    if (b_left) {
        for (y = 0; y < bsy; y++) {
            dc_value += p_src[-y];
        }
        p_src = src + 1;
        if (b_top) {
            for (x = 0; x < bsx; x++) {
                dc_value += p_src[x];
            }
            dc_value += ((bsx + bsy) >> 1);
            dc_value = (dc_value * (512 / (bsx + bsy))) >> 9;
        } else {
            dc_value += bsy / 2;
            dc_value /= bsy;
        }
    } else {
        p_src = src + 1;
        if (b_top) {
            for (x = 0; x < bsx; x++) {
                dc_value += p_src[x];
            }
            dc_value += bsx / 2;
            dc_value /= bsx;
        } else {
            dc_value = 1 << (bit_depth - 1);
        }
    }

    /* 填充块 */
    dc_value = intra_clip3(0, max_val, dc_value);
    for (y = 0; y < bsy; y++) {
        for (x = 0; x < bsx; x++) {
            dst[x] = (pel_t)dc_value;
        }
        dst += i_dst;
    }
}

/* Plane 预测: 双线性平面拟合 */
void intra_pred_plane_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst, int mode,
                               int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    static const int ib_mult[5]  = { 13, 17,  5, 11, 23 };
    static const int ib_shift[5] = {  7, 10, 11, 15, 19 };
    int im_h = ib_mult [tab_log2size[bsx] - 2];
    int im_v = ib_mult [tab_log2size[bsy] - 2];
    int is_h = ib_shift[tab_log2size[bsx] - 2];
    int is_v = ib_shift[tab_log2size[bsy] - 2];
    int iW2 = bsx >> 1;
    int iH2 = bsy >> 1;
    int iH = 0, iV = 0;
    int iA, iB, iC;
    int x, y;
    int iTmp, iTmp2;
    int max_val = (1 << bit_depth) - 1;
    pel_t *p_src;

    UNUSED_PARAMETER(mode);

    /* 计算水平梯度 iH */
    p_src = src + 1;
    p_src += (iW2 - 1);
    for (x = 1; x < iW2 + 1; x++) {
        iH += x * (p_src[x] - p_src[-x]);
    }

    /* 计算垂直梯度 iV */
    p_src = src - 1;
    p_src -= (iH2 - 1);
    for (y = 1; y < iH2 + 1; y++) {
        iV += y * (p_src[-y] - p_src[y]);
    }

    /* 计算平面参数 */
    p_src = src;
    iA = (p_src[-1 - (bsy - 1)] + p_src[1 + bsx - 1]) << 4;
    iB = ((iH << 5) * im_h + (1 << (is_h - 1))) >> is_h;
    iC = ((iV << 5) * im_v + (1 << (is_v - 1))) >> is_v;

    iTmp = iA - (iH2 - 1) * iC - (iW2 - 1) * iB + 16;
    for (y = 0; y < bsy; y++) {
        iTmp2 = iTmp;
        for (x = 0; x < bsx; x++) {
            dst[x] = (pel_t)intra_clip3(0, max_val, iTmp2 >> 5);
            iTmp2 += iB;
        }
        dst += i_dst;
        iTmp += iC;
    }
}

/* Bilinear 预测: 双线性插值 */
void intra_pred_bilinear_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst, int mode,
                                  int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    itr_t p_top[MAX_CU_SIZE];
    itr_t p_left[MAX_CU_SIZE];
    itr_t p_t[MAX_CU_SIZE];
    itr_t p_l[MAX_CU_SIZE];
    itr_t wy[MAX_CU_SIZE];
    int ishift_x  = tab_log2size[bsx];
    int ishift_y  = tab_log2size[bsy];
    int ishift    = intra_min(ishift_x, ishift_y);
    int ishift_xy = ishift_x + ishift_y + 1;
    int offset    = 1 << (ishift_x + ishift_y);
    int a, b, c, w, wxy, t;
    int predx;
    int x, y;
    int max_value = (1 << bit_depth) - 1;

    UNUSED_PARAMETER(mode);

    for (x = 0; x < bsx; x++) {
        p_top[x] = src[1 + x];
    }
    for (y = 0; y < bsy; y++) {
        p_left[y] = src[-1 - y];
    }

    a = p_top[bsx - 1];
    b = p_left[bsy - 1];
    c = (bsx == bsy) ? (a + b + 1) >> 1
                     : (((a << ishift_x) + (b << ishift_y)) * 13 + (1 << (ishift + 5))) >> (ishift + 6);
    w = (c << 1) - a - b;

    for (x = 0; x < bsx; x++) {
        p_t[x] = (itr_t)(b - p_top[x]);
        p_top[x] <<= ishift_y;
    }
    t = 0;
    for (y = 0; y < bsy; y++) {
        p_l[y] = (itr_t)(a - p_left[y]);
        p_left[y] <<= ishift_x;
        wy[y] = (itr_t)t;
        t += w;
    }

    for (y = 0; y < bsy; y++) {
        predx = p_left[y];
        wxy = -wy[y];
        for (x = 0; x < bsx; x++) {
            predx += p_l[y];
            wxy += wy[y];
            p_top[x] += p_t[x];
            dst[x] = (pel_t)intra_clip3(0, max_value,
                    (((predx << ishift_y) + (p_top[x] << ishift_x) + wxy + offset) >> ishift_xy));
        }
        dst += i_dst;
    }
}


/* ===================================================================
 * 角度预测辅助函数
 * =================================================================== */

/* 获取上下文像素位置和偏移量 */
static int get_context_pixel(int mode, int uiXYflag, int iTempD, int *offset)
{
    int imult = tab_auc_dir_dxdy[uiXYflag][mode][0];
    int ishift = tab_auc_dir_dxdy[uiXYflag][mode][1];
    int iTempDn = iTempD * imult >> ishift;

    *offset = ((iTempD * imult * 32) >> ishift) - iTempDn * 32;
    return iTempDn;
}

/* 通用角度预测 - 水平方向 (模式 3..11) */
void intra_pred_ang_x_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                               int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int iDx = tab_auc_dir_dx[dir_mode];
    int iDy = tab_auc_dir_dy[dir_mode];
    int iX;
    int c1, c2, c3, c4;
    int i, j;
    UNUSED_PARAMETER(iDx);
    UNUSED_PARAMETER(iDy);
    UNUSED_PARAMETER(bit_depth);

    for (j = 0; j < bsy; j++, iDy++) {
        iX = get_context_pixel(dir_mode, 0, j + 1, &c4);
        c1 = 32 - c4;
        c2 = 64 - c4;
        c3 = 32 + c4;

        i = 0;
        for (; i < bsx; i++) {
            dst[i] = (pel_t)((src[iX] * c1 + src[iX + 1] * c2 +
                              src[iX + 2] * c3 + src[iX + 3] * c4 + 64) >> 7);
            iX++;
        }

        dst += i_dst;
    }
}

/* 通用角度预测 - 垂直方向 (模式 25..32) */
void intra_pred_ang_y_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                               int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int offsets[64];
    int xsteps[64];
    int iDx = tab_auc_dir_dx[dir_mode];
    int iDy = tab_auc_dir_dy[dir_mode];
    int i, j;
    int offset;
    int iY;
    UNUSED_PARAMETER(iDx);
    UNUSED_PARAMETER(iDy);
    UNUSED_PARAMETER(bit_depth);

    for (i = 0; i < bsx; i++) {
        xsteps[i] = get_context_pixel(dir_mode, 1, i + 1, &offsets[i]);
    }

    for (j = 0; j < bsy; j++) {
        for (i = 0; i < bsx; i++) {
            int idx;
            iY = j + xsteps[i];
            idx = -iY;
            offset = offsets[i];
            dst[i] = (pel_t)((src[idx] * (32 - offset) + src[idx - 1] * (64 - offset) +
                              src[idx - 2] * (32 + offset) + src[idx - 3] * offset + 64) >> 7);
        }
        dst += i_dst;
    }
}

/* 通用角度预测 - 对角方向 (模式 13..23) */
void intra_pred_ang_xy_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int xoffsets[64];
    int xsteps[64];
    const int iDx = tab_auc_dir_dx[dir_mode];
    const int iDy = tab_auc_dir_dy[dir_mode];
    int i, j, iXx, iYy;
    int offsetx, offsety;
    UNUSED_PARAMETER(iDx);
    UNUSED_PARAMETER(iDy);
    UNUSED_PARAMETER(bit_depth);

    for (i = 0; i < bsx; i++) {
        xsteps[i] = get_context_pixel(dir_mode, 1, i + 1, &xoffsets[i]);
    }

    for (j = 0; j < bsy; j++) {
        iXx = -get_context_pixel(dir_mode, 0, j + 1, &offsetx);
        for (i = 0; i < bsx; i++) {
            iYy = j - xsteps[i];
            if (iYy <= -1) {
                dst[i] = (pel_t)((src[iXx + 2] * (32 - offsetx) + src[iXx + 1] * (64 - offsetx) +
                                  src[iXx] * (32 + offsetx) + src[iXx - 1] * offsetx + 64) >> 7);
            } else {
                offsety = xoffsets[i];
                dst[i] = (pel_t)((src[-iYy - 2] * (32 - offsety) + src[-iYy - 1] * (64 - offsety) +
                                  src[-iYy] * (32 + offsety) + src[-iYy + 1] * offsety + 64) >> 7);
            }
            iXx++;
        }
        dst      += i_dst;
    }
}


/* ===================================================================
 * 专用角度预测函数 - 水平方向 (模式 3..11)
 * =================================================================== */

/* 模式 3: 1/4 像素插值 (4 行一组) */
static void intra_pred_ang_x_3_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                 int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t first_line[(64 + 176) << 2];
    int line_size = bsx + (bsy >> 2) * 11 - 1;
    int aligned_line_size = 64 + 176;
    int i_dst4 = i_dst << 2;
    int i;
    pel_t *pfirst[4];
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    pfirst[0] = first_line;
    pfirst[1] = pfirst[0] + aligned_line_size;
    pfirst[2] = pfirst[1] + aligned_line_size;
    pfirst[3] = pfirst[2] + aligned_line_size;

    for (i = 0; i < line_size; i++, src++) {
        pfirst[0][i] = (pel_t)((    src[2] + 5 * src[3] + 7 * src[4] + 3 * src[5] + 8) >> 4);
        pfirst[1][i] = (pel_t)((    src[5] + 3 * src[6] + 3 * src[7] +     src[8] + 4) >> 3);
        pfirst[2][i] = (pel_t)((3 * src[8] + 7 * src[9] + 5 * src[10] +     src[11] + 8) >> 4);
        pfirst[3][i] = (pel_t)((    src[11] + 2 * src[12] +   src[13] + 0 * src[14] + 2) >> 2);
    }

    bsy >>= 2;
    for (i = 0; i < bsy; i++) {
        memcpy(dst,             pfirst[0] + i * 11, (size_t)(bsx * sizeof(pel_t)));
        memcpy(dst +     i_dst, pfirst[1] + i * 11, (size_t)(bsx * sizeof(pel_t)));
        memcpy(dst + 2 * i_dst, pfirst[2] + i * 11, (size_t)(bsx * sizeof(pel_t)));
        memcpy(dst + 3 * i_dst, pfirst[3] + i * 11, (size_t)(bsx * sizeof(pel_t)));
        dst += i_dst4;
    }
}

/* 模式 4: 1/2 像素插值 (2 行一组) */
static void intra_pred_ang_x_4_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                 int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t first_line[64 + 128];
    int line_size = bsx + ((bsy - 1) << 1);
    int iHeight2 = bsy << 1;
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    src += 3;
    for (i = 0; i < line_size; i++, src++) {
        first_line[i] = (pel_t)((src[-1] + src[0] * 2 + src[1] + 2) >> 2);
    }

    for (i = 0; i < iHeight2; i += 2) {
        memcpy(dst, first_line + i, (size_t)(bsx * sizeof(pel_t)));
        dst += i_dst;
    }
}

/* 模式 5: 1/8 像素插值 (8 行一组) */
static void intra_pred_ang_x_5_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                 int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    if ((bsy > 4) && (bsx > 8)) {
        pel_t first_line[(64 + 80) << 3];
        int line_size = bsx + (((bsy - 8) * 11) >> 3);
        int aligned_line_size = ((line_size + 15) >> 4) << 4;
        pel_t *pfirst[8];
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;

        pfirst[0] = first_line;
        pfirst[1] = pfirst[0] + aligned_line_size;
        pfirst[2] = pfirst[1] + aligned_line_size;
        pfirst[3] = pfirst[2] + aligned_line_size;
        pfirst[4] = pfirst[3] + aligned_line_size;
        pfirst[5] = pfirst[4] + aligned_line_size;
        pfirst[6] = pfirst[5] + aligned_line_size;
        pfirst[7] = pfirst[6] + aligned_line_size;

        for (i = 0; i < line_size; src++, i++) {
            pfirst[0][i] = (pel_t)((5 * src[1] + 13 * src[2] + 11 * src[3] + 3 * src[4] + 16) >> 5);
            pfirst[1][i] = (pel_t)((    src[2] +  5 * src[3] +  7 * src[4] + 3 * src[5] + 8) >> 4);
            pfirst[2][i] = (pel_t)((7 * src[4] + 15 * src[5] +  9 * src[6] +     src[7] + 16) >> 5);
            pfirst[3][i] = (pel_t)((    src[5] +  3 * src[6] +  3 * src[7] +     src[8] + 4) >> 3);
            pfirst[4][i] = (pel_t)((     src[6] +  9 * src[7]  + 15 * src[8]  +  7 * src[9]  + 16) >> 5);
            pfirst[5][i] = (pel_t)(( 3 * src[8] +  7 * src[9]  +  5 * src[10] +      src[11] +  8) >> 4);
            pfirst[6][i] = (pel_t)(( 3 * src[9] + 11 * src[10] + 13 * src[11] +  5 * src[12] + 16) >> 5);
            pfirst[7][i] = (pel_t)((    src[11] +  2 * src[12] +      src[13]                 + 2) >> 2);
        }

        bsy >>= 3;
        for (i = 0; i < bsy; i++) {
            memcpy(dst1, pfirst[0] + i * 11, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst2, pfirst[1] + i * 11, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst3, pfirst[2] + i * 11, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst4, pfirst[3] + i * 11, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst5, pfirst[4] + i * 11, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst6, pfirst[5] + i * 11, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst7, pfirst[6] + i * 11, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst8, pfirst[7] + i * 11, (size_t)(bsx * sizeof(pel_t)));
            dst1 = dst8 + i_dst; dst2 = dst1 + i_dst;
            dst3 = dst2 + i_dst; dst4 = dst3 + i_dst;
            dst5 = dst4 + i_dst; dst6 = dst5 + i_dst;
            dst7 = dst6 + i_dst; dst8 = dst7 + i_dst;
        }
    } else if (bsx == 16) {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        for (i = 0; i < bsx; i++, src++) {
            dst1[i] = (pel_t)((5 * src[1] + 13 * src[2] + 11 * src[3] + 3 * src[4] + 16) >> 5);
            dst2[i] = (pel_t)((    src[2] +  5 * src[3] +  7 * src[4] + 3 * src[5] + 8) >> 4);
            dst3[i] = (pel_t)((7 * src[4] + 15 * src[5] +  9 * src[6] +     src[7] + 16) >> 5);
            dst4[i] = (pel_t)((    src[5] +  3 * src[6] +  3 * src[7] +     src[8] + 4) >> 3);
        }
    } else if (bsx == 8) {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;

        for (i = 0; i < 8; src++, i++) {
            dst1[i] = (pel_t)((5 * src[1] + 13 * src[2] + 11 * src[3] + 3 * src[4] + 16) >> 5);
            dst2[i] = (pel_t)((    src[2] +  5 * src[3] +  7 * src[4] + 3 * src[5] + 8) >> 4);
            dst3[i] = (pel_t)((7 * src[4] + 15 * src[5] +  9 * src[6] +     src[7] + 16) >> 5);
            dst4[i] = (pel_t)((    src[5] +  3 * src[6] +  3 * src[7] +     src[8] + 4) >> 3);
            dst5[i] = (pel_t)((     src[6] +  9 * src[7]  + 15 * src[8]  +  7 * src[9]  + 16) >> 5);
            dst6[i] = (pel_t)(( 3 * src[8] +  7 * src[9]  +  5 * src[10] +      src[11] + 8) >> 4);
            dst7[i] = (pel_t)(( 3 * src[9] + 11 * src[10] + 13 * src[11] +  5 * src[12] + 16) >> 5);
            dst8[i] = (pel_t)((    src[11] +  2 * src[12] +      src[13]                 + 2) >> 2);
        }
        if (bsy == 32) {
            pel_t pad1 = src[8];
            int j;
            dst1 = dst8 + i_dst;
            for (j = 0; j < 24; j++) {
                for (i = 0; i < 8; i++) {
                    dst1[i] = pad1;
                }
                dst1 += i_dst;
            }
            dst1 = dst8 + i_dst;
            dst2 = dst1 + i_dst;
            dst3 = dst2 + i_dst;
            src += 4;
            dst1[0] = (pel_t)((5 * src[0] + 13 * src[1] + 11 * src[2] + 3 * src[3] + 16) >> 5);
            dst1[1] = (pel_t)((5 * src[1] + 13 * src[2] + 11 * src[3] + 3 * src[4] + 16) >> 5);
            dst1[2] = (pel_t)((5 * src[2] + 13 * src[3] + 11 * src[4] + 3 * src[5] + 16) >> 5);
            dst1[3] = (pel_t)((5 * src[3] + 13 * src[4] + 11 * src[5] + 3 * src[6] + 16) >> 5);
            dst2[0] = (pel_t)((src[1] + 5 * src[2] + 7 * src[3] + 3 * src[4] + 8) >> 4);
            dst2[1] = (pel_t)((src[2] + 5 * src[3] + 7 * src[4] + 3 * src[5] + 8) >> 4);
            dst2[2] = (pel_t)((src[3] + 5 * src[4] + 7 * src[5] + 3 * src[6] + 8) >> 4);
            dst3[0] = (pel_t)((7 * src[3] + 15 * src[4] +  9 * src[5] +     src[6] + 16) >> 5);
        }
    } else {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;

        for (i = 0; i < 4; i++, src++) {
            dst1[i] = (pel_t)((5 * src[1] + 13 * src[2] + 11 * src[3] + 3 * src[4] + 16) >> 5);
            dst2[i] = (pel_t)((    src[2] +  5 * src[3] +  7 * src[4] + 3 * src[5] + 8) >> 4);
            dst3[i] = (pel_t)((7 * src[4] + 15 * src[5] +  9 * src[6] +     src[7] + 16) >> 5);
            dst4[i] = (pel_t)((    src[5] +  3 * src[6] +  3 * src[7] +     src[8] + 4) >> 3);
        }
        if (bsy == 16) {
            pel_t *dst5 = dst4 + i_dst;
            int j;
            src += 4;
            {
                pel_t pad1 = src[0];
                for (j = 0; j < 12; j++) {
                    for (i = 0; i < 4; i++) {
                        dst5[i] = pad1;
                    }
                    dst5 += i_dst;
                }
            }
            dst5 = dst4 + i_dst;
            dst5[0] = (pel_t)((src[-2] + 9 * src[-1] + 15 * src[0] + 7 * src[1] + 16) >> 5);
            dst5[1] = (pel_t)((src[-1] + 9 * src[ 0] + 15 * src[1] + 7 * src[2] + 16) >> 5);
        }
    }
}

/* 模式 6: 1 像素偏移插值 */
static void intra_pred_ang_x_6_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                 int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t first_line[64 + 64];
    int line_size = bsx + bsy - 1;
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    for (i = 0; i < line_size; i++, src++) {
        first_line[i] = (pel_t)((src[1] + (src[2] << 1) + src[3] + 2) >> 2);
    }

    for (i = 0; i < bsy; i++) {
        memcpy(dst, first_line + i, (size_t)(bsx * sizeof(pel_t)));
        dst += i_dst;
    }
}

/* 模式 7: 1.5 像素插值 */
static void intra_pred_ang_x_7_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                 int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int i;
    pel_t *dst1 = dst;
    pel_t *dst2 = dst1 + i_dst;
    pel_t *dst3 = dst2 + i_dst;
    pel_t *dst4 = dst3 + i_dst;
    UNUSED_PARAMETER(bit_depth);

    if (bsy == 4) {
        for (i = 0; i < bsx; src++, i++) {
            dst1[i] = (pel_t)((src[0] *  9 + src[1] * 41 + src[2] * 55 + src[3] * 23 + 64) >> 7);
            dst2[i] = (pel_t)((src[1] *  9 + src[2] * 25 + src[3] * 23 + src[4] *  7 + 32) >> 6);
            dst3[i] = (pel_t)((src[2] * 27 + src[3] * 59 + src[4] * 37 + src[5] *  5 + 64) >> 7);
            dst4[i] = (pel_t)((src[2] *  3 + src[3] * 35 + src[4] * 61 + src[5] * 29 + 64) >> 7);
        }
    } else if (bsy == 8) {
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;
        for (i = 0; i < bsx; src++, i++) {
            dst1[i] = (pel_t)((src[0] *  9 + src[1] * 41 + src[2] * 55 + src[3] * 23 + 64) >> 7);
            dst2[i] = (pel_t)((src[1] *  9 + src[2] * 25 + src[3] * 23 + src[4] *  7 + 32) >> 6);
            dst3[i] = (pel_t)((src[2] * 27 + src[3] * 59 + src[4] * 37 + src[5] *  5 + 64) >> 7);
            dst4[i] = (pel_t)((src[2] *  3 + src[3] * 35 + src[4] * 61 + src[5] * 29 + 64) >> 7);
            dst5[i] = (pel_t)((src[3] *  3 + src[4] * 11 + src[5] * 13 + src[6] *  5 + 16) >> 5);
            dst6[i] = (pel_t)((src[4] * 21 + src[5] * 53 + src[6] * 43 + src[7] * 11 + 64) >> 7);
            dst7[i] = (pel_t)((src[5] * 15 + src[6] * 31 + src[7] * 17 + src[8] + 32)      >> 6);
            dst8[i] = (pel_t)((src[5] *  3 + src[6] * 19 + src[7] * 29 + src[8] * 13 + 32) >> 6);
        }
    } else {
        intra_pred_ang_x_c((uint8_t *)src, (uint8_t *)dst, i_dst, dir_mode, bsx, bsy, bit_depth);
    }
}

/* 模式 8: 2 像素插值 (2 行一组) */
static void intra_pred_ang_x_8_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                 int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t first_line[2 * (64 + 32)];
    int line_size = bsx + (bsy >> 1) - 1;
    int aligned_line_size = ((line_size + 15) >> 4) << 4;
    int i_dst2 = i_dst << 1;
    int i;
    pel_t *pfirst[2];
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    pfirst[0] = first_line;
    pfirst[1] = first_line + aligned_line_size;
    for (i = 0; i < line_size; i++, src++) {
        pfirst[0][i] = (pel_t)((src[0] + (src[1] + src[2]) * 3 + src[3] + 4) >> 3);
        pfirst[1][i] = (pel_t)((src[1] + (src[2] << 1)         + src[3] + 2) >> 2);
    }

    bsy >>= 1;
    for (i = 0; i < bsy; i++) {
        memcpy(dst,         pfirst[0] + i, (size_t)(bsx * sizeof(pel_t)));
        memcpy(dst + i_dst, pfirst[1] + i, (size_t)(bsx * sizeof(pel_t)));
        dst += i_dst2;
    }
}

/* 模式 9: 2.5 像素插值 */
static void intra_pred_ang_x_9_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                 int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int i;
    UNUSED_PARAMETER(bit_depth);

    if (bsy > 8) {
        intra_pred_ang_x_c((uint8_t *)src, (uint8_t *)dst, i_dst, dir_mode, bsx, bsy, bit_depth);
    } else if (bsy == 8) {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;
        for (i = 0; i < bsx; i++, src++) {
            dst1[i] = (pel_t)((21 * src[0] + 53 * src[1] + 43 * src[2] + 11 * src[3] + 64) >> 7);
            dst2[i] = (pel_t)((9  * src[0] + 41 * src[1] + 55 * src[2] + 23 * src[3] + 64) >> 7);
            dst3[i] = (pel_t)((15 * src[1] + 31 * src[2] + 17 * src[3] +      src[4] + 32) >> 6);
            dst4[i] = (pel_t)((9  * src[1] + 25 * src[2] + 23 * src[3] + 7  * src[4] + 32) >> 6);
            dst5[i] = (pel_t)((3  * src[1] + 19 * src[2] + 29 * src[3] + 13 * src[4] + 32) >> 6);
            dst6[i] = (pel_t)((27 * src[2] + 59 * src[3] + 37 * src[4] + 5  * src[5] + 64) >> 7);
            dst7[i] = (pel_t)((15 * src[2] + 47 * src[3] + 49 * src[4] + 17 * src[5] + 64) >> 7);
            dst8[i] = (pel_t)((3  * src[2] + 35 * src[3] + 61 * src[4] + 29 * src[5] + 64) >> 7);
        }
    } else {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        for (i = 0; i < bsx; i++, src++) {
            dst1[i] = (pel_t)((21 * src[0] + 53 * src[1] + 43 * src[2] + 11 * src[3] + 64) >> 7);
            dst2[i] = (pel_t)((9  * src[0] + 41 * src[1] + 55 * src[2] + 23 * src[3] + 64) >> 7);
            dst3[i] = (pel_t)((15 * src[1] + 31 * src[2] + 17 * src[3] +      src[4] + 32) >> 6);
            dst4[i] = (pel_t)((9  * src[1] + 25 * src[2] + 23 * src[3] + 7  * src[4] + 32) >> 6);
        }
    }
}

/* 模式 10: 3 像素插值 (4 行一组) */
static void intra_pred_ang_x_10_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                  int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t *dst1 = dst;
    pel_t *dst2 = dst1 + i_dst;
    pel_t *dst3 = dst2 + i_dst;
    pel_t *dst4 = dst3 + i_dst;
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    if (bsy != 4) {
        pel_t first_line[4 * (64 + 16)];
        int line_size = bsx + bsy / 4 - 1;
        int aligned_line_size = ((line_size + 15) >> 4) << 4;
        pel_t *pfirst[4];

        pfirst[0] = first_line;
        pfirst[1] = first_line + aligned_line_size;
        pfirst[2] = first_line + aligned_line_size * 2;
        pfirst[3] = first_line + aligned_line_size * 3;

        for (i = 0; i < line_size; i++, src++) {
            pfirst[0][i] = (pel_t)((src[0] * 3 +  src[1] * 7 + src[2]  * 5 + src[3]     + 8) >> 4);
            pfirst[1][i] = (pel_t)((src[0]     + (src[1]     + src[2]) * 3 + src[3]     + 4) >> 3);
            pfirst[2][i] = (pel_t)((src[0]     +  src[1] * 5 + src[2]  * 7 + src[3] * 3 + 8) >> 4);
            pfirst[3][i] = (pel_t)((src[1]     +  src[2] * 2 + src[3]                   + 2) >> 2);
        }

        bsy   >>= 2;
        i_dst <<= 2;
        for (i = 0; i < bsy; i++) {
            memcpy(dst1, pfirst[0] + i, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst2, pfirst[1] + i, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst3, pfirst[2] + i, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst4, pfirst[3] + i, (size_t)(bsx * sizeof(pel_t)));
            dst1 += i_dst; dst2 += i_dst;
            dst3 += i_dst; dst4 += i_dst;
        }
    } else {
        for (i = 0; i < bsx; i++, src++) {
            dst1[i] = (pel_t)((src[0] * 3 +  src[1] * 7 + src[2]  * 5 + src[3]     + 8) >> 4);
            dst2[i] = (pel_t)((src[0]     + (src[1]     + src[2]) * 3 + src[3]     + 4) >> 3);
            dst3[i] = (pel_t)((src[0]     +  src[1] * 5 + src[2]  * 7 + src[3] * 3 + 8) >> 4);
            dst4[i] = (pel_t)((src[1]     +  src[2] * 2 + src[3]                   + 2) >> 2);
        }
    }
}

/* 模式 11: 3.5 像素插值 (8 行一组) */
static void intra_pred_ang_x_11_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                  int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    if (bsy > 8) {
        pel_t first_line[(64 + 16) << 3];
        int line_size = bsx + (bsy >> 3) - 1;
        int aligned_line_size = ((line_size + 15) >> 4) << 4;
        int i_dst8 = i_dst << 3;
        pel_t *pfirst[8];

        pfirst[0] = first_line;
        pfirst[1] = pfirst[0] + aligned_line_size;
        pfirst[2] = pfirst[1] + aligned_line_size;
        pfirst[3] = pfirst[2] + aligned_line_size;
        pfirst[4] = pfirst[3] + aligned_line_size;
        pfirst[5] = pfirst[4] + aligned_line_size;
        pfirst[6] = pfirst[5] + aligned_line_size;
        pfirst[7] = pfirst[6] + aligned_line_size;
        for (i = 0; i < line_size; i++, src++) {
            pfirst[0][i] = (pel_t)((7 * src[0] + 15 * src[1] +  9 * src[2] +     src[3] + 16) >> 5);
            pfirst[1][i] = (pel_t)((3 * src[0] +  7 * src[1] +  5 * src[2] +     src[3] +  8) >> 4);
            pfirst[2][i] = (pel_t)((5 * src[0] + 13 * src[1] + 11 * src[2] + 3 * src[3] + 16) >> 5);
            pfirst[3][i] = (pel_t)((    src[0] +  3 * src[1] +  3 * src[2] +     src[3] +  4) >> 3);
            pfirst[4][i] = (pel_t)((3 * src[0] + 11 * src[1] + 13 * src[2] + 5 * src[3] + 16) >> 5);
            pfirst[5][i] = (pel_t)((    src[0] +  5 * src[1] +  7 * src[2] + 3 * src[3] +  8) >> 4);
            pfirst[6][i] = (pel_t)((    src[0] +  9 * src[1] + 15 * src[2] + 7 * src[3] + 16) >> 5);
            pfirst[7][i] = (pel_t)((    src[1] +  2 * src[2] +      src[3] + 0 * src[4] +  2) >> 2);
        }

        bsy >>= 3;
        for (i = 0; i < bsy; i++) {
            memcpy(dst,             pfirst[0] + i, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst +     i_dst, pfirst[1] + i, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst + 2 * i_dst, pfirst[2] + i, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst + 3 * i_dst, pfirst[3] + i, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst + 4 * i_dst, pfirst[4] + i, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst + 5 * i_dst, pfirst[5] + i, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst + 6 * i_dst, pfirst[6] + i, (size_t)(bsx * sizeof(pel_t)));
            memcpy(dst + 7 * i_dst, pfirst[7] + i, (size_t)(bsx * sizeof(pel_t)));
            dst += i_dst8;
        }
    } else if (bsy == 8) {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;
        for (i = 0; i < bsx; i++, src++) {
            dst1[i] = (pel_t)((7 * src[0] + 15 * src[1] +  9 * src[2] +     src[3] + 16) >> 5);
            dst2[i] = (pel_t)((3 * src[0] +  7 * src[1] +  5 * src[2] +     src[3] + 8) >> 4);
            dst3[i] = (pel_t)((5 * src[0] + 13 * src[1] + 11 * src[2] + 3 * src[3] + 16) >> 5);
            dst4[i] = (pel_t)((    src[0] +  3 * src[1] +  3 * src[2] +     src[3] + 4) >> 3);
            dst5[i] = (pel_t)((3 * src[0] + 11 * src[1] + 13 * src[2] + 5 * src[3] + 16) >> 5);
            dst6[i] = (pel_t)((    src[0] +  5 * src[1] +  7 * src[2] + 3 * src[3] +  8) >> 4);
            dst7[i] = (pel_t)((    src[0] +  9 * src[1] + 15 * src[2] + 7 * src[3] + 16) >> 5);
            dst8[i] = (pel_t)((    src[1] +  2 * src[2] +      src[3] + 0 * src[4] +  2) >> 2);
        }
    } else {
        for (i = 0; i < bsx; i++, src++) {
            pel_t *dst1 = dst;
            pel_t *dst2 = dst1 + i_dst;
            pel_t *dst3 = dst2 + i_dst;
            pel_t *dst4 = dst3 + i_dst;
            dst1[i] = (pel_t)(( 7 * src[0] + 15 * src[1] +  9 * src[2] +      src[3] + 16) >> 5);
            dst2[i] = (pel_t)(( 3 * src[0] +  7 * src[1] +  5 * src[2] +      src[3] +  8) >> 4);
            dst3[i] = (pel_t)(( 5 * src[0] + 13 * src[1] + 11 * src[2] +  3 * src[3] + 16) >> 5);
            dst4[i] = (pel_t)((     src[0] +  3 * src[1] +  3 * src[2] +      src[3] +  4) >> 3);
        }
    }
}


/* ===================================================================
 * 专用角度预测函数 - 对角方向 (模式 13..23)
 * =================================================================== */

/* 模式 13 */
static void intra_pred_ang_xy_13_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                   int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    if (bsy > 8) {
        pel_t first_line[(64 + 16) << 3];
        int line_size = bsx + (bsy >> 3) - 1;
        int left_size = line_size - bsx;
        int aligned_line_size = ((line_size + 15) >> 4) << 4;
        pel_t *pfirst[8];

        pfirst[0] = first_line;
        pfirst[1] = pfirst[0] + aligned_line_size;
        pfirst[2] = pfirst[1] + aligned_line_size;
        pfirst[3] = pfirst[2] + aligned_line_size;
        pfirst[4] = pfirst[3] + aligned_line_size;
        pfirst[5] = pfirst[4] + aligned_line_size;
        pfirst[6] = pfirst[5] + aligned_line_size;
        pfirst[7] = pfirst[6] + aligned_line_size;

        src -= bsy - 8;
        for (i = 0; i < left_size; i++, src += 8) {
            pfirst[0][i] = (pel_t)((src[6] + (src[7] << 1) + src[8] + 2) >> 2);
            pfirst[1][i] = (pel_t)((src[5] + (src[6] << 1) + src[7] + 2) >> 2);
            pfirst[2][i] = (pel_t)((src[4] + (src[5] << 1) + src[6] + 2) >> 2);
            pfirst[3][i] = (pel_t)((src[3] + (src[4] << 1) + src[5] + 2) >> 2);
            pfirst[4][i] = (pel_t)((src[2] + (src[3] << 1) + src[4] + 2) >> 2);
            pfirst[5][i] = (pel_t)((src[1] + (src[2] << 1) + src[3] + 2) >> 2);
            pfirst[6][i] = (pel_t)((src[0] + (src[1] << 1) + src[2] + 2) >> 2);
            pfirst[7][i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
        }
        for (; i < line_size; i++, src++) {
            pfirst[0][i] = (pel_t)((7 * src[2] + 15 * src[1] + 9 * src[0] + src[-1] + 16) >> 5);
            pfirst[1][i] = (pel_t)((3 * src[2] + 7 * src[1] + 5 * src[0] + src[-1] + 8) >> 4);
            pfirst[2][i] = (pel_t)((5 * src[2] + 13 * src[1] + 11 * src[0] + 3 * src[-1] + 16) >> 5);
            pfirst[3][i] = (pel_t)((src[2] + 3 * src[1] + 3 * src[0] + src[-1] + 4) >> 3);
            pfirst[4][i] = (pel_t)((3 * src[2] + 11 * src[1] + 13 * src[0] + 5 * src[-1] + 16) >> 5);
            pfirst[5][i] = (pel_t)((src[2] + 5 * src[1] + 7 * src[0] + 3 * src[-1] + 8) >> 4);
            pfirst[6][i] = (pel_t)((src[2] + 9 * src[1] + 15 * src[0] + 7 * src[-1] + 16) >> 5);
            pfirst[7][i] = (pel_t)((src[1] + 2 * src[0] + src[-1] + 2) >> 2);
        }

        pfirst[0] += left_size; pfirst[1] += left_size;
        pfirst[2] += left_size; pfirst[3] += left_size;
        pfirst[4] += left_size; pfirst[5] += left_size;
        pfirst[6] += left_size; pfirst[7] += left_size;

        bsy >>= 3;
        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst[0] - i, (size_t)(bsx * sizeof(pel_t)));  dst += i_dst;
            memcpy(dst, pfirst[1] - i, (size_t)(bsx * sizeof(pel_t)));  dst += i_dst;
            memcpy(dst, pfirst[2] - i, (size_t)(bsx * sizeof(pel_t)));  dst += i_dst;
            memcpy(dst, pfirst[3] - i, (size_t)(bsx * sizeof(pel_t)));  dst += i_dst;
            memcpy(dst, pfirst[4] - i, (size_t)(bsx * sizeof(pel_t)));  dst += i_dst;
            memcpy(dst, pfirst[5] - i, (size_t)(bsx * sizeof(pel_t)));  dst += i_dst;
            memcpy(dst, pfirst[6] - i, (size_t)(bsx * sizeof(pel_t)));  dst += i_dst;
            memcpy(dst, pfirst[7] - i, (size_t)(bsx * sizeof(pel_t)));  dst += i_dst;
        }
    } else if (bsy == 8) {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        pel_t *dst5 = dst4 + i_dst;
        pel_t *dst6 = dst5 + i_dst;
        pel_t *dst7 = dst6 + i_dst;
        pel_t *dst8 = dst7 + i_dst;
        for (i = 0; i < bsx; i++, src++) {
            dst1[i] = (pel_t)((7 * src[2] + 15 * src[1] + 9 * src[0] + src[-1] + 16) >> 5);
            dst2[i] = (pel_t)((3 * src[2] + 7 * src[1] + 5 * src[0] + src[-1] + 8) >> 4);
            dst3[i] = (pel_t)((5 * src[2] + 13 * src[1] + 11 * src[0] + 3 * src[-1] + 16) >> 5);
            dst4[i] = (pel_t)((src[2] + 3 * src[1] + 3 * src[0] + src[-1] + 4) >> 3);
            dst5[i] = (pel_t)((3 * src[2] + 11 * src[1] + 13 * src[0] + 5 * src[-1] + 16) >> 5);
            dst6[i] = (pel_t)((src[2] + 5 * src[1] + 7 * src[0] + 3 * src[-1] + 8) >> 4);
            dst7[i] = (pel_t)((src[2] + 9 * src[1] + 15 * src[0] + 7 * src[-1] + 16) >> 5);
            dst8[i] = (pel_t)((src[1] + 2 * src[0] + src[-1]  + 2) >> 2);
        }
    } else {
        for (i = 0; i < bsx; i++, src++) {
            pel_t *dst1 = dst;
            pel_t *dst2 = dst1 + i_dst;
            pel_t *dst3 = dst2 + i_dst;
            pel_t *dst4 = dst3 + i_dst;
            dst1[i] = (pel_t)((7 * src[2] + 15 * src[1] +  9 * src[0] +     src[-1] + 16) >> 5);
            dst2[i] = (pel_t)((3 * src[2] +  7 * src[1] +  5 * src[0] +     src[-1] + 8) >> 4);
            dst3[i] = (pel_t)((5 * src[2] + 13 * src[1] + 11 * src[0] + 3 * src[-1] + 16) >> 5);
            dst4[i] = (pel_t)((    src[2] +  3 * src[1] +  3 * src[0] +     src[-1] + 4) >> 3);
        }
    }
}

/* 模式 14 */
static void intra_pred_ang_xy_14_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                   int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    if (bsy != 4) {
        pel_t first_line[4 * (64 + 16)];
        int line_size = bsx + (bsy >> 2) - 1;
        int left_size = line_size - bsx;
        int aligned_line_size = ((line_size + 15) >> 4) << 4;
        pel_t *pfirst[4];

        pfirst[0] = first_line;
        pfirst[1] = first_line + aligned_line_size;
        pfirst[2] = first_line + aligned_line_size * 2;
        pfirst[3] = first_line + aligned_line_size * 3;

        src -= bsy - 4;
        for (i = 0; i < left_size; i++, src += 4) {
            pfirst[0][i] = (pel_t)((src[ 2] + (src[3] << 1) + src[4] + 2) >> 2);
            pfirst[1][i] = (pel_t)((src[ 1] + (src[2] << 1) + src[3] + 2) >> 2);
            pfirst[2][i] = (pel_t)((src[ 0] + (src[1] << 1) + src[2] + 2) >> 2);
            pfirst[3][i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
        }
        for (; i < line_size; i++, src++) {
            pfirst[0][i] = (pel_t)((src[-1]     +  src[0] * 5 + src[1]  * 7 + src[2] * 3 + 8) >> 4);
            pfirst[1][i] = (pel_t)((src[-1]     + (src[0]     + src[1]) * 3 + src[2]     + 4) >> 3);
            pfirst[2][i] = (pel_t)((src[-1] * 3 +  src[0] * 7 + src[1]  * 5 + src[2]     + 8) >> 4);
            pfirst[3][i] = (pel_t)((src[-1]     +  src[0] * 2 + src[1]                   + 2) >> 2);
        }

        pfirst[0] += left_size; pfirst[1] += left_size;
        pfirst[2] += left_size; pfirst[3] += left_size;

        bsy >>= 2;
        for (i = 0; i < bsy; i++) {
            memcpy(dst, pfirst[0] - i, (size_t)(bsx * sizeof(pel_t))); dst += i_dst;
            memcpy(dst, pfirst[1] - i, (size_t)(bsx * sizeof(pel_t))); dst += i_dst;
            memcpy(dst, pfirst[2] - i, (size_t)(bsx * sizeof(pel_t))); dst += i_dst;
            memcpy(dst, pfirst[3] - i, (size_t)(bsx * sizeof(pel_t))); dst += i_dst;
        }
    } else {
        pel_t *dst1 = dst;
        pel_t *dst2 = dst1 + i_dst;
        pel_t *dst3 = dst2 + i_dst;
        pel_t *dst4 = dst3 + i_dst;
        for (i = 0; i < bsx; i++, src++) {
            dst1[i] = (pel_t)((src[-1]     +  src[0] * 5 + src[1]  * 7 + src[2] * 3 + 8) >> 4);
            dst2[i] = (pel_t)((src[-1]     + (src[0]     + src[1]) * 3 + src[2]     + 4) >> 3);
            dst3[i] = (pel_t)((src[-1] * 3 +  src[0] * 7 + src[1]  * 5 + src[2]     + 8) >> 4);
            dst4[i] = (pel_t)((src[-1]     +  src[0] * 2 + src[1]                   + 2) >> 2);
        }
    }
}

/* 模式 16 */
static void intra_pred_ang_xy_16_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                   int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t first_line[2 * (64 + 32)];
    int line_size = bsx + (bsy >> 1) - 1;
    int left_size = line_size - bsx;
    int aligned_line_size = ((line_size + 15) >> 4) << 4;
    int i_dst2 = i_dst << 1;
    pel_t *pfirst[2];
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    pfirst[0] = first_line;
    pfirst[1] = first_line + aligned_line_size;

    src -= bsy - 2;
    for (i = 0; i < left_size; i++, src += 2) {
        pfirst[0][i] = (pel_t)((src[ 0] + (src[1] << 1) + src[2] + 2) >> 2);
        pfirst[1][i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
    }
    for (; i < line_size; i++, src++) {
        pfirst[0][i] = (pel_t)((src[-1] + (src[0]       + src[1]) * 3 + src[2] + 4) >> 3);
        pfirst[1][i] = (pel_t)((src[-1] + (src[0] << 1) + src[1]               + 2) >> 2);
    }

    pfirst[0] += left_size;
    pfirst[1] += left_size;

    bsy >>= 1;
    for (i = 0; i < bsy; i++) {
        memcpy(dst,         pfirst[0] - i, (size_t)(bsx * sizeof(pel_t)));
        memcpy(dst + i_dst, pfirst[1] - i, (size_t)(bsx * sizeof(pel_t)));
        dst += i_dst2;
    }
}

/* 模式 18 */
static void intra_pred_ang_xy_18_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                   int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t first_line[64 + 64];
    int line_size = bsx + bsy - 1;
    int i;
    pel_t *pfirst = first_line + bsy - 1;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    src -= bsy - 1;
    for (i = 0; i < line_size; i++, src++) {
        first_line[i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
    }

    for (i = 0; i < bsy; i++) {
        memcpy(dst, pfirst, (size_t)(bsx * sizeof(pel_t)));
        pfirst--;
        dst += i_dst;
    }
}

/* 模式 20 */
static void intra_pred_ang_xy_20_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                   int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t first_line[64 + 128];
    int left_size = ((bsy - 1) << 1) + 1;
    int top_size = bsx - 1;
    int line_size = left_size + top_size;
    int i;
    pel_t *pfirst = first_line + left_size - 1;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    src -= bsy;
    for (i = 0; i < left_size; i += 2, src++) {
        first_line[i    ] = (pel_t)((src[-1] + (src[0] +  src[1]) * 3  + src[2] + 4) >> 3);
        first_line[i + 1] = (pel_t)((           src[0] + (src[1] << 1) + src[2] + 2) >> 2);
    }
    i--;
    for (; i < line_size; i++, src++) {
        first_line[i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
    }

    for (i = 0; i < bsy; i++) {
        memcpy(dst, pfirst, (size_t)(bsx * sizeof(pel_t)));
        pfirst -= 2;
        dst    += i_dst;
    }
}

/* 模式 22 */
static void intra_pred_ang_xy_22_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                   int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    if (bsx != 4) {
        pel_t first_line[64 + 256];
        int left_size = ((bsy - 1) << 2) + 3;
        int top_size  = bsx - 3;
        int line_size = left_size + top_size;
        pel_t *pfirst = first_line + left_size - 3;
        int j;

        src -= bsy;
        for (i = 0; i < left_size; i += 4, src++) {
            first_line[i    ] = (pel_t)((src[-1] * 3 +  src[0] * 7 + src[1]  * 5 + src[2]     + 8) >> 4);
            first_line[i + 1] = (pel_t)((src[-1]     + (src[0]     + src[1]) * 3 + src[2]     + 4) >> 3);
            first_line[i + 2] = (pel_t)((src[-1]     +  src[0] * 5 + src[1]  * 7 + src[2] * 3 + 8) >> 4);
            first_line[i + 3] = (pel_t)((               src[0]     + src[1]  * 2 + src[2]     + 2) >> 2);
        }
        i--;
        for (; i < line_size; i++, src++) {
            first_line[i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
        }

        for (j = 0; j < bsy; j++) {
            memcpy(dst, pfirst, (size_t)(bsx * sizeof(pel_t)));
            dst    += i_dst;
            pfirst -= 4;
        }
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((src[-2] * 3 +  src[-1] * 7 + src[0]  * 5 + src[1]     + 8) >> 4);
            dst[1] = (pel_t)((src[-2]     + (src[-1]     + src[0]) * 3 + src[1]     + 4) >> 3);
            dst[2] = (pel_t)((src[-2]     +  src[-1] * 5 + src[0]  * 7 + src[1] * 3 + 8) >> 4);
            dst[3] = (pel_t)((               src[-1]     + src[0]  * 2 + src[1]     + 2) >> 2);
            dst += i_dst;
        }
    }
}

/* 模式 23 */
static void intra_pred_ang_xy_23_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                   int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    if (bsx > 8) {
        pel_t first_line[64 + 512];
        int left_size = (bsy << 3) - 1;
        int top_size = bsx - 7;
        int line_size = left_size + top_size;
        pel_t *pfirst = first_line + left_size - 7;
        int j;

        src -= bsy;
        for (i = 0; i < left_size; i += 8, src++) {
            first_line[i    ] = (pel_t)((7 * src[-1] + 15 * src[0] +  9 * src[1] +     src[2] + 16) >> 5);
            first_line[i + 1] = (pel_t)((3 * src[-1] +  7 * src[0] +  5 * src[1] +     src[2] +  8) >> 4);
            first_line[i + 2] = (pel_t)((5 * src[-1] + 13 * src[0] + 11 * src[1] + 3 * src[2] + 16) >> 5);
            first_line[i + 3] = (pel_t)((    src[-1] +  3 * src[0] +  3 * src[1] +     src[2] +  4) >> 3);
            first_line[i + 4] = (pel_t)((3 * src[-1] + 11 * src[0] + 13 * src[1] + 5 * src[2] + 16) >> 5);
            first_line[i + 5] = (pel_t)((    src[-1] +  5 * src[0] +  7 * src[1] + 3 * src[2] +  8) >> 4);
            first_line[i + 6] = (pel_t)((    src[-1] +  9 * src[0] + 15 * src[1] + 7 * src[2] + 16) >> 5);
            first_line[i + 7] = (pel_t)((    src[ 0] +  2 * src[1] +      src[2] + 0 * src[3] +  2) >> 2);
        }
        i--;
        for (; i < line_size; i++, src++) {
            first_line[i] = (pel_t)((src[1] + (src[0] << 1) + src[-1] + 2) >> 2);
        }

        for (j = 0; j < bsy; j++) {
            memcpy(dst, pfirst, (size_t)(bsx * sizeof(pel_t)));
            dst += i_dst;
            pfirst -= 8;
        }
    } else if (bsx == 8) {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((7 * src[-2] + 15 * src[-1] +  9 * src[0] +     src[1] + 16) >> 5);
            dst[1] = (pel_t)((3 * src[-2] +  7 * src[-1] +  5 * src[0] +     src[1] +  8) >> 4);
            dst[2] = (pel_t)((5 * src[-2] + 13 * src[-1] + 11 * src[0] + 3 * src[1] + 16) >> 5);
            dst[3] = (pel_t)((    src[-2] +  3 * src[-1] +  3 * src[0] +     src[1] +  4) >> 3);
            dst[4] = (pel_t)((3 * src[-2] + 11 * src[-1] + 13 * src[0] + 5 * src[1] + 16) >> 5);
            dst[5] = (pel_t)((    src[-2] +  5 * src[-1] +  7 * src[0] + 3 * src[1] +  8) >> 4);
            dst[6] = (pel_t)((    src[-2] +  9 * src[-1] + 15 * src[0] + 7 * src[1] + 16) >> 5);
            dst[7] = (pel_t)((    src[-1] +  2 * src[ 0] +      src[1] + 0 * src[2] +  2) >> 2);
            dst += i_dst;
        }
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((7 * src[-2] + 15 * src[-1] + 9 * src[0] + src[1] + 16) >> 5);
            dst[1] = (pel_t)((3 * src[-2] + 7 * src[-1] + 5 * src[0] + src[1] + 8) >> 4);
            dst[2] = (pel_t)((5 * src[-2] + 13 * src[-1] + 11 * src[0] + 3 * src[1] + 16) >> 5);
            dst[3] = (pel_t)((src[-2] + 3 * src[-1] + 3 * src[0] + src[1] + 4) >> 3);
            dst += i_dst;
        }
    }
}


/* ===================================================================
 * 专用角度预测函数 - 垂直方向 (模式 25..32)
 * =================================================================== */

/* 模式 25 */
static void intra_pred_ang_y_25_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                  int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    if (bsx > 8) {
        pel_t first_line[64 + (64 << 3)];
        int line_size = bsx + ((bsy - 1) << 3);
        int iHeight8 = bsy << 3;
        for (i = 0; i < line_size; i += 8, src--) {
            first_line[0 + i] = (pel_t)((src[0] * 7 + src[-1] * 15 + src[-2] *  9 + src[-3] * 1 + 16) >> 5);
            first_line[1 + i] = (pel_t)((src[0] * 3 + src[-1] * 7  + src[-2] *  5 + src[-3] * 1 + 8) >> 4);
            first_line[2 + i] = (pel_t)((src[0] * 5 + src[-1] * 13 + src[-2] * 11 + src[-3] * 3 + 16) >> 5);
            first_line[3 + i] = (pel_t)((src[0] * 1 + src[-1] * 3  + src[-2] *  3 + src[-3] * 1 + 4) >> 3);
            first_line[4 + i] = (pel_t)((src[0] * 3 + src[-1] * 11 + src[-2] * 13 + src[-3] * 5 + 16) >> 5);
            first_line[5 + i] = (pel_t)((src[0] * 1 + src[-1] *  5 + src[-2] *  7 + src[-3] * 3 + 8) >> 4);
            first_line[6 + i] = (pel_t)((src[0] * 1 + src[-1] *  9 + src[-2] * 15 + src[-3] * 7 + 16) >> 5);
            first_line[7 + i] = (pel_t)((             src[-1] *  1 + src[-2] *  2 + src[-3] * 1 + 2) >> 2);
        }
        for (i = 0; i < iHeight8; i += 8) {
            memcpy(dst, first_line + i, (size_t)(bsx * sizeof(pel_t)));
            dst += i_dst;
        }
    } else if (bsx == 8) {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((src[0] * 7 + src[-1] * 15 + src[-2] *  9 + src[-3] * 1 + 16) >> 5);
            dst[1] = (pel_t)((src[0] * 3 + src[-1] *  7 + src[-2] *  5 + src[-3] * 1 + 8) >> 4);
            dst[2] = (pel_t)((src[0] * 5 + src[-1] * 13 + src[-2] * 11 + src[-3] * 3 + 16) >> 5);
            dst[3] = (pel_t)((src[0] * 1 + src[-1] *  3 + src[-2] *  3 + src[-3] * 1 + 4) >> 3);
            dst[4] = (pel_t)((src[0] * 3 + src[-1] * 11 + src[-2] * 13 + src[-3] * 5 + 16) >> 5);
            dst[5] = (pel_t)((src[0] * 1 + src[-1] *  5 + src[-2] *  7 + src[-3] * 3 + 8) >> 4);
            dst[6] = (pel_t)((src[0] * 1 + src[-1] *  9 + src[-2] * 15 + src[-3] * 7 + 16) >> 5);
            dst[7] = (pel_t)((             src[-1] *  1 + src[-2] *  2 + src[-3] * 1 + 2) >> 2);
            dst += i_dst;
        }
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((src[0] * 7 + src[-1] * 15 + src[-2] *  9 + src[-3] * 1 + 16) >> 5);
            dst[1] = (pel_t)((src[0] * 3 + src[-1] *  7 + src[-2] *  5 + src[-3] * 1 + 8) >> 4);
            dst[2] = (pel_t)((src[0] * 5 + src[-1] * 13 + src[-2] * 11 + src[-3] * 3 + 16) >> 5);
            dst[3] = (pel_t)((src[0] * 1 + src[-1] *  3 + src[-2] *  3 + src[-3] * 1 + 4) >> 3);
            dst += i_dst;
        }
    }
}

/* 模式 26 */
static void intra_pred_ang_y_26_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                  int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    if (bsx != 4) {
        pel_t first_line[64 + 256];
        int line_size = bsx + ((bsy - 1) << 2);
        int iHeight4 = bsy << 2;

        for (i = 0; i < line_size; i += 4, src--) {
            first_line[i    ] = (pel_t)((src[ 0] * 3 +  src[-1] * 7 + src[-2]  * 5 + src[-3]     + 8) >> 4);
            first_line[i + 1] = (pel_t)((src[ 0]     + (src[-1]     + src[-2]) * 3 + src[-3]     + 4) >> 3);
            first_line[i + 2] = (pel_t)((src[ 0]     +  src[-1] * 5 + src[-2]  * 7 + src[-3] * 3 + 8) >> 4);
            first_line[i + 3] = (pel_t)((src[-1]     +  src[-2] * 2 + src[-3]                    + 2) >> 2);
        }
        for (i = 0; i < iHeight4; i += 4) {
            memcpy(dst, first_line + i, (size_t)(bsx * sizeof(pel_t)));
            dst += i_dst;
        }
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((src[ 0] * 3 +  src[-1] * 7 + src[-2]  * 5 + src[-3]     + 8) >> 4);
            dst[1] = (pel_t)((src[ 0]     + (src[-1]     + src[-2]) * 3 + src[-3]     + 4) >> 3);
            dst[2] = (pel_t)((src[ 0]     +  src[-1] * 5 + src[-2]  * 7 + src[-3] * 3 + 8) >> 4);
            dst[3] = (pel_t)((src[-1]     +  src[-2] * 2 + src[-3]                    + 2) >> 2);
            dst += i_dst;
        }
    }
}

/* 模式 27 */
static void intra_pred_ang_y_27_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                  int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int i;
    UNUSED_PARAMETER(bit_depth);

    if (bsx > 8) {
        intra_pred_ang_y_c((uint8_t *)src, (uint8_t *)dst, i_dst, dir_mode, bsx, bsy, bit_depth);
    } else if (bsx == 8) {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((21 * src[0] +  53 * src[-1] + 43 * src[-2] + 11 * src[-3] + 64) >> 7);
            dst[1] = (pel_t)(( 9 * src[0] +  41 * src[-1] + 55 * src[-2] + 23 * src[-3] + 64) >> 7);
            dst[2] = (pel_t)((15 * src[-1] + 31 * src[-2] + 17 * src[-3] +  1 * src[-4] + 32) >> 6);
            dst[3] = (pel_t)(( 9 * src[-1] + 25 * src[-2] + 23 * src[-3] +  7 * src[-4] + 32) >> 6);
            dst[4] = (pel_t)(( 3 * src[-1] + 19 * src[-2] + 29 * src[-3] + 13 * src[-4] + 32) >> 6);
            dst[5] = (pel_t)((27 * src[-2] + 59 * src[-3] + 37 * src[-4] +  5 * src[-5] + 64) >> 7);
            dst[6] = (pel_t)((15 * src[-2] + 47 * src[-3] + 49 * src[-4] + 17 * src[-5] + 64) >> 7);
            dst[7] = (pel_t)(( 3 * src[-2] + 35 * src[-3] + 61 * src[-4] + 29 * src[-5] + 64) >> 7);
            dst += i_dst;
        }
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((21 * src[0]  + 53 * src[-1] + 43 * src[-2] + 11 * src[-3] + 64) >> 7);
            dst[1] = (pel_t)(( 9 * src[0]  + 41 * src[-1] + 55 * src[-2] + 23 * src[-3] + 64) >> 7);
            dst[2] = (pel_t)((15 * src[-1] + 31 * src[-2] + 17 * src[-3] +  1 * src[-4] + 32) >> 6);
            dst[3] = (pel_t)(( 9 * src[-1] + 25 * src[-2] + 23 * src[-3] +  7 * src[-4] + 32) >> 6);
            dst += i_dst;
        }
    }
}

/* 模式 28 */
static void intra_pred_ang_y_28_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                  int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t first_line[64 + 128];
    int line_size = bsx + ((bsy - 1) << 1);
    int iHeight2 = bsy << 1;
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    for (i = 0; i < line_size; i += 2, src--) {
        first_line[i    ] = (pel_t)((src[ 0] + (src[-1] + src[-2]) * 3 + src[-3] + 4) >> 3);
        first_line[i + 1] = (pel_t)((src[-1] + (src[-2] << 1)          + src[-3] + 2) >> 2);
    }

    for (i = 0; i < iHeight2; i += 2) {
        memcpy(dst, first_line + i, (size_t)(bsx * sizeof(pel_t)));
        dst += i_dst;
    }
}

/* 模式 29 */
static void intra_pred_ang_y_29_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                  int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    int i;
    UNUSED_PARAMETER(bit_depth);

    if (bsx > 8) {
        intra_pred_ang_y_c((uint8_t *)src, (uint8_t *)dst, i_dst, dir_mode, bsx, bsy, bit_depth);
    } else if (bsx == 8) {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((src[0] * 9 + src[-1] * 41 + src[-2] * 55 + src[-3] * 23 + 64) >> 7);
            dst[1] = (pel_t)((src[-1] * 9 + src[-2] * 25 + src[-3] * 23 + src[-4] * 7 + 32) >> 6);
            dst[2] = (pel_t)((src[-2] * 27 + src[-3] * 59 + src[-4] * 37 + src[-5] * 5 + 64) >> 7);
            dst[3] = (pel_t)((src[-2] * 3 + src[-3] * 35 + src[-4] * 61 + src[-5] * 29 + 64) >> 7);
            dst[4] = (pel_t)((src[-3] * 3 + src[-4] * 11 + src[-5] * 13 + src[-6] * 5 + 16) >> 5);
            dst[5] = (pel_t)((src[-4] * 21 + src[-5] * 53 + src[-6] * 43 + src[-7] * 11 + 64) >> 7);
            dst[6] = (pel_t)((src[-5] * 15 + src[-6] * 31 + src[-7] * 17 + src[-8] + 32) >> 6);
            dst[7] = (pel_t)((src[-5] * 3 + src[-6] * 19 + src[-7] * 29 + src[-8] * 13 + 32) >> 6);
            dst += i_dst;
        }
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((src[0] * 9 + src[-1] * 41 + src[-2] * 55 + src[-3] * 23 + 64) >> 7);
            dst[1] = (pel_t)((src[-1] * 9 + src[-2] * 25 + src[-3] * 23 + src[-4] * 7 + 32) >> 6);
            dst[2] = (pel_t)((src[-2] * 27 + src[-3] * 59 + src[-4] * 37 + src[-5] * 5 + 64) >> 7);
            dst[3] = (pel_t)((src[-2] * 3 + src[-3] * 35 + src[-4] * 61 + src[-5] * 29 + 64) >> 7);
            dst += i_dst;
        }
    }
}

/* 模式 30 */
static void intra_pred_ang_y_30_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                  int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t first_line[64 + 64];
    int line_size = bsx + bsy - 1;
    int i;
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    src -= 2;
    for (i = 0; i < line_size; i++, src--) {
        first_line[i] = (pel_t)((src[-1] + (src[0] << 1) + src[1] + 2) >> 2);
    }

    for (i = 0; i < bsy; i++) {
        memcpy(dst, first_line + i, (size_t)(bsx * sizeof(pel_t)));
        dst += i_dst;
    }
}

/* 模式 31 */
static void intra_pred_ang_y_31_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                  int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t dst_tran[MAX_CU_SIZE * MAX_CU_SIZE];
    pel_t src_tran[MAX_CU_SIZE << 3];
    int i, j;
    UNUSED_PARAMETER(bit_depth);

    if (bsx >= bsy) {
        /* 转置后调用模式 5 */
        for (i = 0; i < (bsy + bsx * 11 / 8 + 3); i++) {
            src_tran[i] = src[-i];
        }
        intra_pred_ang_x_5_c((uint8_t *)src_tran, (uint8_t *)dst_tran, bsy, 5, bsy, bsx, bit_depth);
        for (i = 0; i < bsy; i++) {
            for (j = 0; j < bsx; j++) {
                dst[j + i_dst * i] = dst_tran[i + bsy * j];
            }
        }
    } else if (bsx == 8) {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((5 * src[-1] + 13 * src[-2] + 11 * src[-3] + 3 * src[-4] + 16) >> 5);
            dst[1] = (pel_t)((1 * src[-2] + 5 * src[-3] + 7 * src[-4] + 3 * src[-5] + 8) >> 4);
            dst[2] = (pel_t)((7 * src[-4] + 15 * src[-5] + 9 * src[-6] + 1 * src[-7] + 16) >> 5);
            dst[3] = (pel_t)((1 * src[-5] + 3 * src[-6] + 3 * src[-7] + 1 * src[-8] + 4) >> 3);
            dst[4] = (pel_t)((1 * src[-6] + 9 * src[-7] + 15 * src[-8] + 7 * src[-9] + 16) >> 5);
            dst[5] = (pel_t)((3 * src[-8] + 7 * src[-9] + 5 * src[-10] + 1 * src[-11] + 8) >> 4);
            dst[6] = (pel_t)((3 * src[-9] + 11 * src[-10] + 13 * src[-11] + 5 * src[-12] + 16) >> 5);
            dst[7] = (pel_t)((1 * src[-11] + 2 * src[-12] + 1 * src[-13] + 0 * src[-14] + 2) >> 2);
            dst += i_dst;
        }
    } else {
        for (i = 0; i < bsy; i++, src--) {
            dst[0] = (pel_t)((5 * src[-1] + 13 * src[-2] + 11 * src[-3] + 3 * src[-4] + 16) >> 5);
            dst[1] = (pel_t)((1 * src[-2] + 5 * src[-3] + 7 * src[-4] + 3 * src[-5] + 8) >> 4);
            dst[2] = (pel_t)((7 * src[-4] + 15 * src[-5] + 9 * src[-6] + 1 * src[-7] + 16) >> 5);
            dst[3] = (pel_t)((1 * src[-5] + 3 * src[-6] + 3 * src[-7] + 1 * src[-8] + 4) >> 3);
            dst += i_dst;
        }
    }
}

/* 模式 32 */
static void intra_pred_ang_y_32_c(uint8_t *src_u8, uint8_t *dst_u8, int i_dst,
                                  int dir_mode, int bsx, int bsy, int bit_depth)
{
    pel_t *src = (pel_t *)(void *)src_u8;
    pel_t *dst = (pel_t *)(void *)dst_u8;
    pel_t first_line[2 * (32 + 64)];
    int line_size = (bsy >> 1) + bsx - 1;
    int aligned_line_size = ((line_size + 15) >> 4) << 4;
    int i_dst2 = i_dst << 1;
    int i;
    pel_t *pfirst[2];
    UNUSED_PARAMETER(dir_mode);
    UNUSED_PARAMETER(bit_depth);

    pfirst[0] = first_line;
    pfirst[1] = first_line + aligned_line_size;

    src -= 3;
    for (i = 0; i < line_size; i++, src -= 2) {
        pfirst[0][i] = (pel_t)((src[1] + (src[ 0] << 1) + src[-1] + 2) >> 2);
        pfirst[1][i] = (pel_t)((src[0] + (src[-1] << 1) + src[-2] + 2) >> 2);
    }

    bsy >>= 1;
    for (i = 0; i < bsy; i++) {
        memcpy(dst,         pfirst[0] + i, (size_t)(bsx * sizeof(pel_t)));
        memcpy(dst + i_dst, pfirst[1] + i, (size_t)(bsx * sizeof(pel_t)));
        dst += i_dst2;
    }
}


/* ===================================================================
 * 参考样本填充函数 (init_intra_border / extend_intra_border)
 * =================================================================== */

/* LCU 左上角 PU: 上方和左侧均来自 LCU 边界缓存 */
static void fill_reference_samples_0_c(const uint8_t *pTL, int i_TL,
                                       const uint8_t *pLcuEP, uint8_t *EP_u8,
                                       uint32_t i_avai, int bsx, int bsy, int bps)
{
    pel_t *EP = (pel_t *)(void *)EP_u8;
    int num_padding;
    UNUSED_PARAMETER(pTL);
    UNUSED_PARAMETER(i_TL);

    /* 填充默认值 (1 << (bit_depth-1)) */
    mem_repeat_p(&EP[-(bsy << 1)], g_dc_value, ((bsy + bsx) << 1) + 1);

    /* 填充上方和右上像素 */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        copy_to_ep(&EP[1], pLcuEP + bps, bsx, bps);
    }
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_RIGHT)) {
        copy_to_ep(&EP[bsx + 1], pLcuEP + (ptrdiff_t)(bsx + 1) * bps, bsx, bps);
    } else {
        mem_repeat_p(&EP[bsx + 1], EP[bsx], bsx);
    }

    /* 填充额外像素 */
    num_padding = bsy * 11 / 4 - bsx + 4;
    if (num_padding > 0) {
        mem_repeat_p(&EP[2 * bsx + 1], EP[2 * bsx], num_padding);
    }

    /* 填充左侧和左下像素 */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        copy_to_ep(&EP[-bsy], pLcuEP - (ptrdiff_t)bsy * bps, bsy, bps);
    }
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT_DOWN)) {
        copy_to_ep(&EP[-2 * bsy], pLcuEP - (ptrdiff_t)(2 * bsy) * bps, bsy, bps);
    } else {
        mem_repeat_p(&EP[-(bsy << 1)], EP[-bsy], bsy);
    }

    /* 填充左上角像素 */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_LEFT)) {
        EP[0] = pel_read(pLcuEP, bps);
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        EP[0] = pel_read(pLcuEP + bps, bps);
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        EP[0] = pel_read(pLcuEP - bps, bps);
    }

    /* 填充左侧额外像素 */
    num_padding = bsx * 11 / 4 - bsy + 4;
    if (num_padding > 0) {
        mem_repeat_p(&EP[-2 * bsy - num_padding], EP[-2 * bsy], num_padding);
    }
}

/* LCU 左边界 PU: 左侧来自帧, 上方来自 LCU 缓存 */
static void fill_reference_samples_x_c(const uint8_t *pTL, int i_TL,
                                       const uint8_t *pLcuEP, uint8_t *EP_u8,
                                       uint32_t i_avai, int bsx, int bsy, int bps)
{
    pel_t *EP = (pel_t *)(void *)EP_u8;
    const uint8_t *pL = pTL + i_TL;
    int num_padding;

    mem_repeat_p(&EP[-(bsy << 1)], g_dc_value, ((bsy + bsx) << 1) + 1);

    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        copy_to_ep(&EP[1], pLcuEP + bps, bsx, bps);
    }
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_RIGHT)) {
        copy_to_ep(&EP[bsx + 1], pLcuEP + (ptrdiff_t)(bsx + 1) * bps, bsx, bps);
    } else {
        mem_repeat_p(&EP[bsx + 1], EP[bsx], bsx);
    }

    num_padding = bsy * 11 / 4 - bsx + 4;
    if (num_padding > 0) {
        mem_repeat_p(&EP[2 * bsx + 1], EP[2 * bsx], num_padding);
    }

    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        const uint8_t *p_l = pL;
        int y;
        for (y = 0; y < bsy; y++) {
            EP[-1 - y] = pel_read(p_l, bps);
            p_l += i_TL;
        }
    }
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT_DOWN)) {
        const uint8_t *p_l = pL + bsy * i_TL;
        int y;
        for (y = 0; y < bsy; y++) {
            EP[-bsy - 1 - y] = pel_read(p_l, bps);
            p_l += i_TL;
        }
    } else {
        mem_repeat_p(&EP[-(bsy << 1)], EP[-bsy], bsy);
    }

    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_LEFT)) {
        EP[0] = pel_read(pLcuEP, bps);
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        EP[0] = pel_read(pLcuEP + bps, bps);
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        EP[0] = pel_read(pL, bps);
    }

    num_padding = bsx * 11 / 4 - bsy + 4;
    if (num_padding > 0) {
        mem_repeat_p(&EP[-2 * bsy - num_padding], EP[-2 * bsy], num_padding);
    }
}

/* LCU 上边界 PU: 上方来自帧, 左侧来自 LCU 缓存 */
static void fill_reference_samples_y_c(const uint8_t *pTL, int i_TL,
                                       const uint8_t *pLcuEP, uint8_t *EP_u8,
                                       uint32_t i_avai, int bsx, int bsy, int bps)
{
    pel_t *EP = (pel_t *)(void *)EP_u8;
    const uint8_t *pT = pTL + bps;
    int num_padding;
    UNUSED_PARAMETER(i_TL);

    mem_repeat_p(&EP[-(bsy << 1)], g_dc_value, ((bsy + bsx) << 1) + 1);

    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        copy_to_ep(&EP[1], pT, bsx, bps);
    }
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_RIGHT)) {
        copy_to_ep(&EP[bsx + 1], pT + (ptrdiff_t)bsx * bps, bsx, bps);
    } else {
        mem_repeat_p(&EP[bsx + 1], EP[bsx], bsx);
    }

    num_padding = bsy * 11 / 4 - bsx + 4;
    if (num_padding > 0) {
        mem_repeat_p(&EP[2 * bsx + 1], EP[2 * bsx], num_padding);
    }

    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        copy_to_ep(&EP[-bsy], pLcuEP - (ptrdiff_t)bsy * bps, bsy, bps);
    }
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT_DOWN)) {
        copy_to_ep(&EP[-2 * bsy], pLcuEP - (ptrdiff_t)(2 * bsy) * bps, bsy, bps);
    } else {
        mem_repeat_p(&EP[-(bsy << 1)], EP[-bsy], bsy);
    }

    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_LEFT)) {
        EP[0] = pel_read(pLcuEP, bps);
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        EP[0] = pel_read(pT, bps);
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        EP[0] = pel_read(pLcuEP - bps, bps);
    }

    num_padding = bsx * 11 / 4 - bsy + 4;
    if (num_padding > 0) {
        mem_repeat_p(&EP[-2 * bsy - num_padding], EP[-2 * bsy], num_padding);
    }
}

/* LCU 顶部边界 PU: 上方来自 intra_border 缓存 (pre-deblock), 左侧来自帧
 * 对应 davs2 fill_reference_samples_x_c (xy=1: ctu_x!=0, ctu_y==0)
 * pLcuEP 指向 (x-1, y-1) 处的缓存像素, 索引方式: pLcuEP[0]=top-left, pLcuEP[1..bsx]=top, pLcuEP[bsx+1..2*bsx]=top-right */
static void fill_reference_samples_top_c(const uint8_t *pTL, int i_TL,
                                         const uint8_t *pLcuEP, uint8_t *EP_u8,
                                         uint32_t i_avai, int bsx, int bsy, int bps)
{
    pel_t *EP = (pel_t *)(void *)EP_u8;
    const uint8_t *pL = pTL + i_TL;
    int num_padding;

    mem_repeat_p(&EP[-(bsy << 1)], g_dc_value, ((bsy + bsx) << 1) + 1);

    /* TOP 和 TOP-RIGHT 来自 pLcuEP (pre-deblock 缓存) */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        copy_to_ep(&EP[1], pLcuEP + bps, bsx, bps);
    }
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_RIGHT)) {
        copy_to_ep(&EP[bsx + 1], pLcuEP + (ptrdiff_t)(bsx + 1) * bps, bsx, bps);
    } else {
        mem_repeat_p(&EP[bsx + 1], EP[bsx], bsx);
    }

    num_padding = bsy * 11 / 4 - bsx + 4;
    if (num_padding > 0) {
        mem_repeat_p(&EP[2 * bsx + 1], EP[2 * bsx], num_padding);
    }

    /* LEFT 和 LEFT-DOWN 来自 pTL (帧缓冲, 当前 LCU 行尚未 deblock) */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        const uint8_t *p_l = pL;
        int y;
        for (y = 0; y < bsy; y++) {
            EP[-1 - y] = pel_read(p_l, bps);
            p_l += i_TL;
        }
    }
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT_DOWN)) {
        const uint8_t *p_l = pL + bsy * i_TL;
        int y;
        for (y = 0; y < bsy; y++) {
            EP[-bsy - 1 - y] = pel_read(p_l, bps);
            p_l += i_TL;
        }
    } else {
        mem_repeat_p(&EP[-(bsy << 1)], EP[-bsy], bsy);
    }

    /* TOP-LEFT 来自 pLcuEP (pre-deblock 缓存) */
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_LEFT)) {
        EP[0] = pel_read(pLcuEP, bps);
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        EP[0] = pel_read(pLcuEP + bps, bps);
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        EP[0] = pel_read(pL, bps);
    }

    num_padding = bsx * 11 / 4 - bsy + 4;
    if (num_padding > 0) {
        mem_repeat_p(&EP[-2 * bsy - num_padding], EP[-2 * bsy], num_padding);
    }
}

/* LCU 内部 PU: 上方和左侧均来自帧 */
static void fill_reference_samples_xy_c(const uint8_t *pTL, int i_TL,
                                        const uint8_t *pLcuEP, uint8_t *EP_u8,
                                        uint32_t i_avai, int bsx, int bsy, int bps)
{
    pel_t *EP = (pel_t *)(void *)EP_u8;
    const uint8_t *pT = pTL + bps;
    const uint8_t *pL = pTL + i_TL;
    int num_padding;
    UNUSED_PARAMETER(pLcuEP);

    mem_repeat_p(&EP[-(bsy << 1)], g_dc_value, ((bsy + bsx) << 1) + 1);

    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        copy_to_ep(&EP[1], pT, bsx, bps);
    }
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_RIGHT)) {
        copy_to_ep(&EP[bsx + 1], pT + (ptrdiff_t)bsx * bps, bsx, bps);
    } else {
        mem_repeat_p(&EP[bsx + 1], EP[bsx], bsx);
    }

    num_padding = bsy * 11 / 4 - bsx + 4;
    if (num_padding > 0) {
        mem_repeat_p(&EP[2 * bsx + 1], EP[2 * bsx], num_padding);
    }

    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        const uint8_t *p_l = pL;
        int y;
        for (y = 0; y < bsy; y++) {
            EP[-1 - y] = pel_read(p_l, bps);
            p_l += i_TL;
        }
    }
    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT_DOWN)) {
        const uint8_t *p_l = pL + bsy * i_TL;
        int y;
        for (y = 0; y < bsy; y++) {
            EP[-bsy - 1 - y] = pel_read(p_l, bps);
            p_l += i_TL;
        }
    } else {
        mem_repeat_p(&EP[-(bsy << 1)], EP[-bsy], bsy);
    }

    if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP_LEFT)) {
        EP[0] = pel_read(pTL, bps);
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_TOP)) {
        EP[0] = pel_read(pT, bps);
    } else if (IS_NEIGHBOR_AVAIL(i_avai, MD_I_LEFT)) {
        EP[0] = pel_read(pL, bps);
    }

    num_padding = bsx * 11 / 4 - bsy + 4;
    if (num_padding > 0) {
        mem_repeat_p(&EP[-2 * bsy - num_padding], EP[-2 * bsy], num_padding);
    }
}


/* ===================================================================
 * 邻域可用性计算 (简化版, 基于帧边界)
 * =================================================================== */

/* 计算帧内预测邻域可用性 (含 LCU 重建顺序检查)
 * \param lcu_level  LCU 的 log2 尺寸 (通常为 6, 即 64x64)
 * 对应 davs2 get_intra_neighbors: 检查帧边界 + 同 slice + LCU 内 Z-scan 重建顺序。
 * 单线程解码下同 slice 检查等价于帧边界检查, 仅需额外的 tab_dl_avail/tab_tr_avail 表。
 */
static uint32_t get_intra_neighbors_simple(avs2_frame *f, int x, int y,
                                           int bsx, int bsy, int lcu_level)
{
    int b_left, b_top, b_top_left, b_left_down, b_top_right;
    int x_4x4, y_4x4;
    int x_lcu, y_lcu;
    int log2_lcu_in_scu;
    int scu_mask;
    const int8_t *tab_dl;
    const int8_t *tab_tr;

    /* 1. 帧边界检查 */
    b_left     = (x > 0);
    b_top      = (y > 0);
    b_top_left = (x > 0 && y > 0);
    b_left_down = (x > 0 && y + bsy < f->height);
    b_top_right = (y > 0 && x + bsx < f->width);

    /* 2. LCU 重建顺序检查 (tab_dl_avail / tab_tr_avail)
     * 这些表编码了 LCU 内 Z-scan 顺序下, 各 4x4 位置的左下/右上邻居是否已重建。
     * 对应 davs2 的 p_tab_DL_avail / p_tab_TR_avail 查表。 */
    x_4x4 = x >> MIN_PU_SIZE_IN_BIT;   /* 像素 → 4x4 单位 */
    y_4x4 = y >> MIN_PU_SIZE_IN_BIT;

    log2_lcu_in_scu = lcu_level - B4X4_IN_BIT;   /* lcu_level - 2 */
    scu_mask = (1 << log2_lcu_in_scu) - 1;

    /* LCU 相对坐标 (4x4 单位) */
    x_lcu = x_4x4 & scu_mask;
    y_lcu = y_4x4 & scu_mask;

    tab_dl = tab_dl_avails[lcu_level];
    tab_tr = tab_tr_avails[lcu_level];

    if (b_left_down && tab_dl) {
        int idx = ((y_lcu + (bsy >> 2) - 1) << log2_lcu_in_scu) + x_lcu;
        b_left_down = tab_dl[idx];
    }

    if (b_top_right && tab_tr) {
        int idx = (y_lcu << log2_lcu_in_scu) + (x_lcu + (bsx >> 2) - 1);
        b_top_right = tab_tr[idx];
    }

    return (b_left << MD_I_LEFT) | (b_top << MD_I_TOP) |
           (b_top_left << MD_I_TOP_LEFT) |
           (b_top_right << MD_I_TOP_RIGHT) |
           (b_left_down << MD_I_LEFT_DOWN);
}


/* ===================================================================
 * 帧内预测分发器 (predict_intra)
 * =================================================================== */

/* 帧内预测分发: 根据模式选择对应预测函数
 * src/dst 按 bytes_per_sample 寻址 (8-bit=uint8, 10-bit=uint16),
 * i_dst 为元素步长 (byte_stride / bytes_per_sample).
 * 注意: 内部预测函数统一按 uint16_t 处理, 8-bit 时调用方需提供 uint16_t 临时缓冲。 */
void avs2_intra_pred_dispatch(uint8_t *src, uint8_t *dst, int i_dst,
                              int dir_mode, int bsy, int bsx, int i_avail,
                              int bit_depth)
{
    if (dir_mode != DC_PRED) {
        avs2_dsp_table.ipred_mode[dir_mode](src, dst, i_dst, dir_mode,
                                            bsx, bsy, bit_depth);
    } else {
        /* DC 模式: 将可用性编码到 mode 参数中 */
        int b_top  = !!IS_NEIGHBOR_AVAIL(i_avail, MD_I_TOP);
        int b_left = !!IS_NEIGHBOR_AVAIL(i_avail, MD_I_LEFT);
        int mode_ex = ((b_top << 8) + b_left);
        avs2_dsp_table.ipred_mode[dir_mode](src, dst, i_dst, mode_ex,
                                            bsx, bsy, bit_depth);
    }
}


/* ===================================================================
 * 高层入口函数 (亮度 / 色度帧内预测)
 * =================================================================== */

/* 亮度帧内预测入口 */
void avs2_get_intra_pred(struct avs2_internal *c, avs2_frame *f, avs2_cu *cu,
                         int x, int y, int bsx, int bsy, int predmode)
{
    /* EP 参考样本数组: 左侧负索引, 右侧正索引 */
    AVS2_ALIGN32(pel_t ep_buf[(MAX_CU_SIZE << 3) + 1]);
    pel_t *EP = ep_buf + (MAX_CU_SIZE << 2) - 1;
    int bit_depth = c->bit_depth;
    int lcu_level = c->seq->log2_lcu_size;
    int bps = f->bytes_per_sample;
    int byte_stride = (int)f->stride[0];
    int i_elem = byte_stride / bps;  /* 预测 dst 元素步长 (pel_t 单位) */
    uint8_t *p_fdec = f->data[0] + y * byte_stride + (ptrdiff_t)x * bps;
    /* pTL 指向左上角像素 (x-1, y-1), 按字节寻址 */
    const uint8_t *pTL = p_fdec - byte_stride - bps;
    uint32_t avail;

    UNUSED_PARAMETER(cu);

    /* 计算邻域可用性 (含 LCU 重建顺序检查) */
    avail = get_intra_neighbors_simple(f, x, y, bsx, bsy, lcu_level);

    /* 填充参考样本:
     * - LCU 顶部边界 (y 对齐 LCU 行且非首行): TOP 参考来自 intra_border 缓存
     *   (pre-deblock), 因为上一 LCU 行已 deblock, 垂直边滤波修改了底行像素。
     *   对应 davs2 fill_reference_samples_x_c (ctu_y==0 时使用 pLcuEP)。
     * - LCU 内部: 上方和左侧均来自帧缓冲 (当前行尚未 deblock)。 */
    {
        int lcu_size = 1 << lcu_level;
        int at_lcu_top = (y > 0 && (y & (lcu_size - 1)) == 0);
        if (at_lcu_top && f->intra_border[0]) {
            /* pLcuEP 指向 (x-1, y-1) 处的缓存像素 (intra_border 为 uint8_t*) */
            const uint8_t *pLcuEP = f->intra_border[0] + (ptrdiff_t)(AVS2_PAD_LUMA + x - 1) * bps;
            fill_reference_samples_top_c(pTL, byte_stride, pLcuEP, (uint8_t *)EP,
                                         avail, bsx, bsy, bps);
        } else {
            fill_reference_samples_xy_c(pTL, byte_stride, pTL, (uint8_t *)EP,
                                        avail, bsx, bsy, bps);
        }
    }

    /* 预测写入帧缓冲:
     * - 10-bit: p_fdec 本身即 uint16_t, 直接作为 dst
     * - 8-bit:  内部预测函数按 uint16_t 处理, 需要临时缓冲再打包到帧 */
    if (bps > 1) {
        avs2_intra_pred_dispatch((uint8_t *)EP, p_fdec, i_elem, predmode,
                                 bsy, bsx, avail, bit_depth);
    } else {
        AVS2_ALIGN32(pel_t tmp_dst[MAX_CU_SIZE * MAX_CU_SIZE]);
        int i_tmp = bsx;
        int xx, yy;
        avs2_intra_pred_dispatch((uint8_t *)EP, (uint8_t *)tmp_dst, i_tmp,
                                 predmode, bsy, bsx, avail, bit_depth);
        for (yy = 0; yy < bsy; yy++) {
            for (xx = 0; xx < bsx; xx++) {
                p_fdec[yy * byte_stride + xx] = (uint8_t)tmp_dst[yy * i_tmp + xx];
            }
        }
    }
}

/* 色度帧内预测入口 */
void avs2_get_intra_pred_chroma(struct avs2_internal *c, avs2_frame *f,
                                avs2_cu *cu, int x_c, int y_c)
{
    /* 色度模式到实际模式映射表 */
    static const int tab_chroma_mode_to_real[5] = {
        DC_PRED, DC_PRED, HOR_PRED, VERT_PRED, BI_PRED
    };
    int bit_depth = c->bit_depth;
    int lcu_level = c->seq->log2_lcu_size;
    int bps = f->bytes_per_sample;
    int bsize_c = 1 << (cu->cu_level - 1);
    int luma_mode = cu->intra_pred_modes[0];
    int chroma_mode = cu->i_intra_mode_c;
    int real_mode = (chroma_mode == DM_PRED_C)
                    ? luma_mode
                    : tab_chroma_mode_to_real[chroma_mode];
    int byte_stride_c = (int)f->stride[1];
    int i_elem_c = byte_stride_c / bps;  /* 预测 dst 元素步长 (pel_t 单位) */
    int bsx = bsize_c, bsy = bsize_c;
    uint8_t *p_fdec_u = f->data[1] + y_c * byte_stride_c + (ptrdiff_t)x_c * bps;
    uint8_t *p_fdec_v = f->data[2] + y_c * f->stride[2] + (ptrdiff_t)x_c * bps;
    /* pTL 指向左上角像素 (x-1, y-1), 按字节寻址 */
    const uint8_t *pTL_u = p_fdec_u - byte_stride_c - bps;
    const uint8_t *pTL_v = p_fdec_v - byte_stride_c - bps;
    AVS2_ALIGN32(pel_t ep_u[(MAX_CU_SIZE << 3) + 1]);
    AVS2_ALIGN32(pel_t ep_v[(MAX_CU_SIZE << 3) + 1]);
    pel_t *EP_u = ep_u + (MAX_CU_SIZE << 2) - 1;
    pel_t *EP_v = ep_v + (MAX_CU_SIZE << 2) - 1;
    uint32_t avail;

    /* 计算邻域可用性 (色度坐标, 需转换为对应亮度边界) */
    avail = get_intra_neighbors_simple(f, x_c * 2, y_c * 2, bsx * 2, bsy * 2, lcu_level);

    /* 色度 LCU 顶部边界检查 (y_c 对齐色度 LCU 行且非首行) */
    {
        int lcu_size_c = 1 << (lcu_level - 1);  /* 色度 LCU 尺寸 (4:2:0 减半) */
        int at_lcu_top_c = (y_c > 0 && (y_c & (lcu_size_c - 1)) == 0);

        /* U 分量预测 */
        if (at_lcu_top_c && f->intra_border[1]) {
            const uint8_t *pLcuEP_u = f->intra_border[1] + (ptrdiff_t)(AVS2_PAD_CHROMA + x_c - 1) * bps;
            fill_reference_samples_top_c(pTL_u, byte_stride_c, pLcuEP_u,
                                         (uint8_t *)EP_u, avail, bsx, bsy, bps);
        } else {
            fill_reference_samples_xy_c(pTL_u, byte_stride_c, pTL_u,
                                        (uint8_t *)EP_u, avail, bsx, bsy, bps);
        }
        if (bps > 1) {
            avs2_intra_pred_dispatch((uint8_t *)EP_u, p_fdec_u, i_elem_c,
                                     real_mode, bsy, bsx, avail, bit_depth);
        } else {
            AVS2_ALIGN32(pel_t tmp_dst[MAX_CU_SIZE * MAX_CU_SIZE]);
            int i_tmp = bsx;
            int xx, yy;
            avs2_intra_pred_dispatch((uint8_t *)EP_u, (uint8_t *)tmp_dst, i_tmp,
                                     real_mode, bsy, bsx, avail, bit_depth);
            for (yy = 0; yy < bsy; yy++) {
                for (xx = 0; xx < bsx; xx++) {
                    p_fdec_u[yy * byte_stride_c + xx] = (uint8_t)tmp_dst[yy * i_tmp + xx];
                }
            }
        }

        /* V 分量预测 */
        if (at_lcu_top_c && f->intra_border[2]) {
            const uint8_t *pLcuEP_v = f->intra_border[2] + (ptrdiff_t)(AVS2_PAD_CHROMA + x_c - 1) * bps;
            fill_reference_samples_top_c(pTL_v, byte_stride_c, pLcuEP_v,
                                         (uint8_t *)EP_v, avail, bsx, bsy, bps);
        } else {
            fill_reference_samples_xy_c(pTL_v, byte_stride_c, pTL_v,
                                        (uint8_t *)EP_v, avail, bsx, bsy, bps);
        }
        if (bps > 1) {
            avs2_intra_pred_dispatch((uint8_t *)EP_v, p_fdec_v, i_elem_c,
                                     real_mode, bsy, bsx, avail, bit_depth);
        } else {
            AVS2_ALIGN32(pel_t tmp_dst[MAX_CU_SIZE * MAX_CU_SIZE]);
            int i_tmp = bsx;
            int xx, yy;
            avs2_intra_pred_dispatch((uint8_t *)EP_v, (uint8_t *)tmp_dst, i_tmp,
                                     real_mode, bsy, bsx, avail, bit_depth);
            for (yy = 0; yy < bsy; yy++) {
                for (xx = 0; xx < bsx; xx++) {
                    p_fdec_v[yy * byte_stride_c + xx] = (uint8_t)tmp_dst[yy * i_tmp + xx];
                }
            }
        }
    }
}


/* ===================================================================
 * DSP 接口包装函数 (兼容 cu.c 调用)
 * =================================================================== */

/* DC 预测包装: 从分离的 left/top 数组构建 EP 参考数组后调用内部函数 */
static void ipred_dc_c(uint8_t *dst, ptrdiff_t stride, const int16_t *left,
                       const int16_t *top, int w, int h, int bit_depth)
{
    AVS2_ALIGN32(pel_t ep_buf[(MAX_CU_SIZE << 3) + 1]);
    pel_t *EP = ep_buf + (MAX_CU_SIZE << 2) - 1;
    AVS2_ALIGN32(pel_t pred_buf[MAX_CU_SIZE * MAX_CU_SIZE]);
    int i, y;
    int b_top = (top != NULL);
    int b_left = (left != NULL);
    int mode_ex = ((b_top << 8) + b_left);

    /* 构建 EP 参考样本数组 (左侧用负索引, 上方用正索引) */
    EP[0] = b_top ? (pel_t)top[0]
                  : (b_left ? (pel_t)left[0] : (pel_t)(1 << (bit_depth - 1)));
    if (b_top) {
        for (i = 0; i < w; i++) {
            EP[1 + i] = (pel_t)top[i];
        }
        /* 填充 top-right 区域 (用末尾值复制) */
        for (i = 0; i < w; i++) {
            EP[1 + w + i] = (pel_t)top[w - 1];
        }
    } else {
        for (i = 0; i < 2 * w; i++) {
            EP[1 + i] = EP[0];
        }
    }
    if (b_left) {
        for (i = 0; i < h; i++) {
            EP[-1 - i] = (pel_t)left[i];
        }
        /* 填充 left-down 区域 (用末尾值复制) */
        for (i = 0; i < h; i++) {
            EP[-1 - h - i] = (pel_t)left[h - 1];
        }
    } else {
        for (i = 0; i < 2 * h; i++) {
            EP[-1 - i] = EP[0];
        }
    }

    /* 调用内部 DC 预测函数 */
    intra_pred_dc_c((uint8_t *)EP, (uint8_t *)pred_buf, w, mode_ex, w, h, bit_depth);

    /* 将预测结果写回 dst (uint8_t 截断高位) */
    for (y = 0; y < h; y++) {
        for (i = 0; i < w; i++) {
            dst[y * stride + i] = (uint8_t)pred_buf[y * w + i];
        }
    }
}


/* Plane 预测包装: 从分离的 left/top 数组构建 EP 参考数组后调用内部函数 */
static void ipred_plane_c(uint8_t *dst, ptrdiff_t stride, const int16_t *left,
                          const int16_t *top, int w, int h, int bit_depth)
{
    AVS2_ALIGN32(pel_t ep_buf[(MAX_CU_SIZE << 3) + 1]);
    pel_t *EP = ep_buf + (MAX_CU_SIZE << 2) - 1;
    AVS2_ALIGN32(pel_t pred_buf[MAX_CU_SIZE * MAX_CU_SIZE]);
    int b_top = (top != NULL);
    int b_left = (left != NULL);
    int i, y;

    /* 构建 EP 参考样本数组 */
    EP[0] = b_top ? (pel_t)top[0]
                  : (b_left ? (pel_t)left[0] : (pel_t)(1 << (bit_depth - 1)));
    if (b_top) {
        for (i = 0; i < w; i++) {
            EP[1 + i] = (pel_t)top[i];
        }
        for (i = 0; i < w; i++) {
            EP[1 + w + i] = (pel_t)top[w - 1];
        }
    } else {
        for (i = 0; i < 2 * w; i++) {
            EP[1 + i] = EP[0];
        }
    }
    if (b_left) {
        for (i = 0; i < h; i++) {
            EP[-1 - i] = (pel_t)left[i];
        }
        for (i = 0; i < h; i++) {
            EP[-1 - h - i] = (pel_t)left[h - 1];
        }
    } else {
        for (i = 0; i < 2 * h; i++) {
            EP[-1 - i] = EP[0];
        }
    }

    /* 调用内部 Plane 预测函数 */
    intra_pred_plane_c((uint8_t *)EP, (uint8_t *)pred_buf, w, 0, w, h, bit_depth);

    /* 将预测结果写回 dst */
    for (y = 0; y < h; y++) {
        for (i = 0; i < w; i++) {
            dst[y * stride + i] = (uint8_t)pred_buf[y * w + i];
        }
    }
}


/* 角度预测包装: 从 ref (上方参考) 构建 EP 参考数组后调用内部函数 */
static void ipred_angular_c(uint8_t *dst, ptrdiff_t stride, const int16_t *ref,
                            int w, int h, int mode, int bit_depth,
                            int above_avail, int left_avail)
{
    AVS2_ALIGN32(pel_t ep_buf[(MAX_CU_SIZE << 3) + 1]);
    pel_t *EP = ep_buf + (MAX_CU_SIZE << 2) - 1;
    AVS2_ALIGN32(pel_t pred_buf[MAX_CU_SIZE * MAX_CU_SIZE]);
    int b_top = above_avail && (ref != NULL);
    int b_left = left_avail;
    int i, y;

    UNUSED_PARAMETER(b_left);

    /* 构建 EP 参考样本数组 (cu.c 仅传入上方参考) */
    EP[0] = b_top ? (pel_t)ref[0] : (pel_t)(1 << (bit_depth - 1));
    if (b_top) {
        for (i = 0; i < w; i++) {
            EP[1 + i] = (pel_t)ref[i];
        }
        for (i = 0; i < w; i++) {
            EP[1 + w + i] = (pel_t)ref[w - 1];
        }
    } else {
        for (i = 0; i < 2 * w; i++) {
            EP[1 + i] = EP[0];
        }
    }
    /* 左侧参考样本: cu.c 未单独传入, 用 EP[0] 填充 */
    for (i = 0; i < 2 * h; i++) {
        EP[-1 - i] = EP[0];
    }

    /* 调用内部角度预测函数 (通过函数指针表分发) */
    if (avs2_dsp_table.ipred_mode[mode] != NULL) {
        avs2_dsp_table.ipred_mode[mode]((uint8_t *)EP, (uint8_t *)pred_buf,
                                        w, mode, w, h, bit_depth);
    } else {
        /* 未注册的模式退化为 DC */
        intra_pred_dc_c((uint8_t *)EP, (uint8_t *)pred_buf, w, 0, w, h, bit_depth);
    }

    /* 将预测结果写回 dst */
    for (y = 0; y < h; y++) {
        for (i = 0; i < w; i++) {
            dst[y * stride + i] = (uint8_t)pred_buf[y * w + i];
        }
    }
}


/* ===================================================================
 * DSP 初始化函数
 * =================================================================== */

/* 帧内预测 DSP 初始化: 注册所有预测函数到函数指针表 */
void avs2_ipred_init(void)
{
    /* 旧接口包装函数 (兼容 cu.c 调用) */
    avs2_dsp_table.ipred_dc       = ipred_dc_c;
    avs2_dsp_table.ipred_plane    = ipred_plane_c;
    avs2_dsp_table.ipred_angular  = ipred_angular_c;

    /* 新接口: 33 种亮度模式函数指针表 (模式 0..32) */
    avs2_dsp_table.ipred_mode[DC_PRED]     = intra_pred_dc_c;
    avs2_dsp_table.ipred_mode[PLANE_PRED]  = intra_pred_plane_c;
    avs2_dsp_table.ipred_mode[BI_PRED]     = intra_pred_bilinear_c;

    /* 垂直/水平模式 (12, 24) */
    avs2_dsp_table.ipred_mode[VERT_PRED]   = intra_pred_ver_c;
    avs2_dsp_table.ipred_mode[HOR_PRED]    = intra_pred_hor_c;

    /* 角度模式 - 水平方向 (3..11) */
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_3]  = intra_pred_ang_x_3_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_4]  = intra_pred_ang_x_4_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_5]  = intra_pred_ang_x_5_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_6]  = intra_pred_ang_x_6_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_7]  = intra_pred_ang_x_7_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_8]  = intra_pred_ang_x_8_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_9]  = intra_pred_ang_x_9_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_10] = intra_pred_ang_x_10_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_X_11] = intra_pred_ang_x_11_c;

    /* 角度模式 - 对角方向 (13..23, 其中 15/17/19/21 用通用函数) */
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_13] = intra_pred_ang_xy_13_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_14] = intra_pred_ang_xy_14_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_15] = intra_pred_ang_xy_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_16] = intra_pred_ang_xy_16_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_17] = intra_pred_ang_xy_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_18] = intra_pred_ang_xy_18_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_19] = intra_pred_ang_xy_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_20] = intra_pred_ang_xy_20_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_21] = intra_pred_ang_xy_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_22] = intra_pred_ang_xy_22_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_XY_23] = intra_pred_ang_xy_23_c;

    /* 角度模式 - 垂直方向 (25..32) */
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_25] = intra_pred_ang_y_25_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_26] = intra_pred_ang_y_26_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_27] = intra_pred_ang_y_27_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_28] = intra_pred_ang_y_28_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_29] = intra_pred_ang_y_29_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_30] = intra_pred_ang_y_30_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_31] = intra_pred_ang_y_31_c;
    avs2_dsp_table.ipred_mode[INTRA_ANG_Y_32] = intra_pred_ang_y_32_c;

    /* 参考样本填充函数: 4 种边界情况 (0=左上角, 1=左边界, 2=上边界, 3=内部) */
    avs2_dsp_table.fill_edge[0] = fill_reference_samples_0_c;
    avs2_dsp_table.fill_edge[1] = fill_reference_samples_x_c;
    avs2_dsp_table.fill_edge[2] = fill_reference_samples_y_c;
    avs2_dsp_table.fill_edge[3] = fill_reference_samples_xy_c;
}
