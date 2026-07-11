/*
 * mc.c
 *
 * 运动补偿 (运动补偿预测) —— 从 davs2 mc.cc 移植为 C。
 *
 * 实现 AVS2 帧间预测的插值滤波:
 *   - 亮度: 8 抽头插值滤波 (1/4 像素精度)
 *   - 色度: 4 抽头插值滤波 (1/8 像素精度)
 *   - 支持所有子像素位置 (亮度: 1/4, 1/2, 3/4; 色度: 1/8 ... 7/8)
 *   - 双向预测 (前向 + 后向平均)
 *   - 加权预测 (加权跳过模式)
 *
 * 滤波器系数表见 tables.c (avs2_intpl_filters / avs2_intpl_filters_c)。
 *
 * 参考: davs2 source/common/mc.cc, GB/T 33475.2
 */

#include "internal.h"
#include "tables.h"
#include <string.h>

/* ===================================================================
 * 类型定义
 * =================================================================== */

/* 像素类型 (uint16_t 支持 10-bit) */
typedef uint16_t pel_t;

/* 中间计算类型 (8 抽头滤波结果可能超过 16-bit 范围, 用 int32_t) */
typedef int32_t mct_t;

/* ===================================================================
 * 宏定义
 * =================================================================== */

/* 8 抽头亮度水平滤波: 对 src[i] 位置做 8 抽头卷积
 * 系数之和为 64, 输出需右移 6 位 */
#define FLT_8TAP_HOR(src, i, coef) (\
    (src)[(i) - 3] * (coef)[0] + \
    (src)[(i) - 2] * (coef)[1] + \
    (src)[(i) - 1] * (coef)[2] + \
    (src)[(i)    ] * (coef)[3] + \
    (src)[(i) + 1] * (coef)[4] + \
    (src)[(i) + 2] * (coef)[5] + \
    (src)[(i) + 3] * (coef)[6] + \
    (src)[(i) + 4] * (coef)[7])

/* 8 抽头亮度垂直滤波: i_src 为行步长 (以元素计) */
#define FLT_8TAP_VER(src, i, i_src, coef) (\
    (src)[(i) - 3 * (i_src)] * (coef)[0] + \
    (src)[(i) - 2 * (i_src)] * (coef)[1] + \
    (src)[(i) -     (i_src)] * (coef)[2] + \
    (src)[(i)            ] * (coef)[3] + \
    (src)[(i) +     (i_src)] * (coef)[4] + \
    (src)[(i) + 2 * (i_src)] * (coef)[5] + \
    (src)[(i) + 3 * (i_src)] * (coef)[6] + \
    (src)[(i) + 4 * (i_src)] * (coef)[7])

/* 4 抽头色度水平滤波 */
#define FLT_4TAP_HOR(src, i, coef) (\
    (src)[(i) - 1] * (coef)[0] + \
    (src)[(i)    ] * (coef)[1] + \
    (src)[(i) + 1] * (coef)[2] + \
    (src)[(i) + 2] * (coef)[3])

/* 4 抽头色度垂直滤波 */
#define FLT_4TAP_VER(src, i, i_src, coef) (\
    (src)[(i) -     (i_src)] * (coef)[0] + \
    (src)[(i)            ] * (coef)[1] + \
    (src)[(i) +     (i_src)] * (coef)[2] + \
    (src)[(i) + 2 * (i_src)] * (coef)[3])

/* 裁剪到 [lo, hi] 范围 */
#define AVS2_CLIP3(lo, hi, v) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))

/* 亮度插值中间缓冲行步长 (最大块宽 64) */
#define LUMA_TMP_STRIDE   64
/* 色度插值中间缓冲行步长 (最大色度块宽 32) */
#define CHROMA_TMP_STRIDE 32

/* 最大块尺寸 (用于栈缓冲区分配) */
#define MAX_BLK_W 64
#define MAX_BLK_H 64


/* ===================================================================
 * 第一部分: 基础函数 (块拷贝、像素平均) —— 通过宏生成 8/16-bit 版本
 * =================================================================== */

/* -------------------------------------------------------------------
 * 块拷贝: 整像素位置直接复制 (对应 davs2 mc_block_copy_c)
 * ------------------------------------------------------------------- */
#define DEFINE_BLOCK_COPY(SUFFIX, PTYPE) \
static void block_copy_##SUFFIX(PTYPE *dst, int i_dst, \
                                const PTYPE *src, int i_src, \
                                int w, int h) \
{ \
    while (h--) { \
        memcpy(dst, src, (size_t)w * sizeof(PTYPE)); \
        dst += i_dst; \
        src += i_src; \
    } \
}

/* -------------------------------------------------------------------
 * 像素平均: 双向预测时将两路预测结果取平均
 * (对应 davs2 davs2_pixel_average_c)
 * ------------------------------------------------------------------- */
#define DEFINE_PIXEL_AVERAGE(SUFFIX, PTYPE) \
static void pixel_average_##SUFFIX(PTYPE *dst, int i_dst, \
                                   const PTYPE *src0, int i_src0, \
                                   const PTYPE *src1, int i_src1, \
                                   int width, int height) \
{ \
    int i, j; \
    for (i = 0; i < height; i++) { \
        for (j = 0; j < width; j++) { \
            dst[j] = (PTYPE)((src0[j] + src1[j] + 1) >> 1); \
        } \
        dst  += i_dst; \
        src0 += i_src0; \
        src1 += i_src1; \
    } \
}


/* ===================================================================
 * 第二部分: 插值滤波函数 (ip_filter) —— 通过宏生成 8/16-bit 版本
 * =================================================================== */

/* -------------------------------------------------------------------
 * 亮度水平插值: 对整像素行做水平 8 抽头滤波
 * (对应 davs2 intpl_luma_block_hor_c)
 * src 需在块左侧留有 3 个像素的边界扩展
 * ------------------------------------------------------------------- */
#define DEFINE_IP_FILTER_LUMA_HOR(SUFFIX, PTYPE) \
static void ip_filter_luma_hor_##SUFFIX( \
        PTYPE *dst, int i_dst, const PTYPE *src, int i_src, \
        int width, int height, const int8_t *coeff, int bit_depth) \
{ \
    int x, y, v; \
    int max_val = (1 << bit_depth) - 1; \
    for (y = 0; y < height; y++) { \
        for (x = 0; x < width; x++) { \
            v = (FLT_8TAP_HOR(src, x, coeff) + 32) >> 6; \
            dst[x] = (PTYPE)AVS2_CLIP3(0, max_val, v); \
        } \
        src += i_src; \
        dst += i_dst; \
    } \
}

/* -------------------------------------------------------------------
 * 亮度垂直插值: 对整像素列做垂直 8 抽头滤波
 * (对应 davs2 intpl_luma_block_ver_c)
 * src 需在块上方留有 3 行的边界扩展
 * ------------------------------------------------------------------- */
#define DEFINE_IP_FILTER_LUMA_VER(SUFFIX, PTYPE) \
static void ip_filter_luma_ver_##SUFFIX( \
        PTYPE *dst, int i_dst, const PTYPE *src, int i_src, \
        int width, int height, const int8_t *coeff, int bit_depth) \
{ \
    int x, y, v; \
    int max_val = (1 << bit_depth) - 1; \
    for (y = 0; y < height; y++) { \
        for (x = 0; x < width; x++) { \
            v = (FLT_8TAP_VER(src, x, i_src, coeff) + 32) >> 6; \
            dst[x] = (PTYPE)AVS2_CLIP3(0, max_val, v); \
        } \
        src += i_src; \
        dst += i_dst; \
    } \
}

/* -------------------------------------------------------------------
 * 亮度双向插值: 先水平后垂直, 两级滤波
 * (对应 davs2 intpl_luma_block_ext_c)
 *
 * 两级移位策略 (防止溢出):
 *   第一级 (水平): shift1 = bit_depth - 8, 将结果缩减到约 8-bit 范围
 *   第二级 (垂直): shift2 = 20 - bit_depth, 最终右移到 bit_depth 位
 * ------------------------------------------------------------------- */
#define DEFINE_IP_FILTER_LUMA_EXT(SUFFIX, PTYPE) \
static void ip_filter_luma_ext_##SUFFIX( \
        PTYPE *dst, int i_dst, const PTYPE *src, int i_src, \
        int width, int height, \
        const int8_t *coeff_h, const int8_t *coeff_v, int bit_depth) \
{ \
    mct_t tmp_buf[(MAX_BLK_H + 7) * LUMA_TMP_STRIDE]; \
    mct_t *tmp = tmp_buf; \
    const int shift1 = bit_depth - 8; \
    const int add1   = (1 << shift1) >> 1; \
    const int shift2 = 20 - bit_depth; \
    const int add2   = 1 << (shift2 - 1); \
    int max_val = (1 << bit_depth) - 1; \
    int x, y, v; \
    /* 源指针上移 3 行, 使滤波覆盖 [-3, +4] 行范围 */ \
    src -= 3 * i_src; \
    /* 第一级: 水平 8 抽头滤波, 输出到中间缓冲 */ \
    for (y = -3; y < height + 4; y++) { \
        for (x = 0; x < width; x++) { \
            v = FLT_8TAP_HOR(src, x, coeff_h); \
            tmp[x] = (mct_t)((v + add1) >> shift1); \
        } \
        src += i_src; \
        tmp += LUMA_TMP_STRIDE; \
    } \
    /* 第二级: 垂直 8 抽头滤波, 从中间缓冲输出到目标 */ \
    tmp = tmp_buf + 3 * LUMA_TMP_STRIDE; \
    for (y = 0; y < height; y++) { \
        for (x = 0; x < width; x++) { \
            v = (FLT_8TAP_VER(tmp, x, LUMA_TMP_STRIDE, coeff_v) + add2) >> shift2; \
            dst[x] = (PTYPE)AVS2_CLIP3(0, max_val, v); \
        } \
        dst += i_dst; \
        tmp += LUMA_TMP_STRIDE; \
    } \
}

/* -------------------------------------------------------------------
 * 色度水平插值: 4 抽头滤波
 * (对应 davs2 intpl_chroma_block_hor_c)
 * src 需在块左侧留有 1 个像素的边界扩展
 * ------------------------------------------------------------------- */
#define DEFINE_IP_FILTER_CHROMA_HOR(SUFFIX, PTYPE) \
static void ip_filter_chroma_hor_##SUFFIX( \
        PTYPE *dst, int i_dst, const PTYPE *src, int i_src, \
        int width, int height, const int8_t *coeff, int bit_depth) \
{ \
    int x, y, v; \
    int max_val = (1 << bit_depth) - 1; \
    for (y = 0; y < height; y++) { \
        for (x = 0; x < width; x++) { \
            v = (FLT_4TAP_HOR(src, x, coeff) + 32) >> 6; \
            dst[x] = (PTYPE)AVS2_CLIP3(0, max_val, v); \
        } \
        src += i_src; \
        dst += i_dst; \
    } \
}

/* -------------------------------------------------------------------
 * 色度垂直插值: 4 抽头滤波
 * (对应 davs2 intpl_chroma_block_ver_c)
 * src 需在块上方留有 1 行的边界扩展
 * ------------------------------------------------------------------- */
#define DEFINE_IP_FILTER_CHROMA_VER(SUFFIX, PTYPE) \
static void ip_filter_chroma_ver_##SUFFIX( \
        PTYPE *dst, int i_dst, const PTYPE *src, int i_src, \
        int width, int height, const int8_t *coeff, int bit_depth) \
{ \
    int x, y, v; \
    int max_val = (1 << bit_depth) - 1; \
    for (y = 0; y < height; y++) { \
        for (x = 0; x < width; x++) { \
            v = (FLT_4TAP_VER(src, x, i_src, coeff) + 32) >> 6; \
            dst[x] = (PTYPE)AVS2_CLIP3(0, max_val, v); \
        } \
        src += i_src; \
        dst += i_dst; \
    } \
}

/* -------------------------------------------------------------------
 * 色度双向插值: 先水平后垂直, 两级 4 抽头滤波
 * (对应 davs2 intpl_chroma_block_ext_c)
 * ------------------------------------------------------------------- */
#define DEFINE_IP_FILTER_CHROMA_EXT(SUFFIX, PTYPE) \
static void ip_filter_chroma_ext_##SUFFIX( \
        PTYPE *dst, int i_dst, const PTYPE *src, int i_src, \
        int width, int height, \
        const int8_t *coeff_h, const int8_t *coeff_v, int bit_depth) \
{ \
    int32_t tmp_buf[(MAX_BLK_H + 3) * CHROMA_TMP_STRIDE]; \
    int32_t *tmp = tmp_buf; \
    const int shift1 = bit_depth - 8; \
    const int add1   = (1 << shift1) >> 1; \
    const int shift2 = 20 - bit_depth; \
    const int add2   = 1 << (shift2 - 1); \
    int max_val = (1 << bit_depth) - 1; \
    int x, y, v; \
    /* 源指针上移 1 行, 使滤波覆盖 [-1, +2] 行范围 */ \
    src -= i_src; \
    /* 第一级: 水平 4 抽头滤波 */ \
    for (y = -1; y < height + 2; y++) { \
        for (x = 0; x < width; x++) { \
            v = FLT_4TAP_HOR(src, x, coeff_h); \
            tmp[x] = (v + add1) >> shift1; \
        } \
        src += i_src; \
        tmp += CHROMA_TMP_STRIDE; \
    } \
    /* 第二级: 垂直 4 抽头滤波 */ \
    tmp = tmp_buf + CHROMA_TMP_STRIDE; \
    for (y = 0; y < height; y++) { \
        for (x = 0; x < width; x++) { \
            v = (FLT_4TAP_VER(tmp, x, CHROMA_TMP_STRIDE, coeff_v) + add2) >> shift2; \
            dst[x] = (PTYPE)AVS2_CLIP3(0, max_val, v); \
        } \
        dst += i_dst; \
        tmp += CHROMA_TMP_STRIDE; \
    } \
}


/* ===================================================================
 * 第三部分: 实例化 8-bit 和 16-bit 版本
 * =================================================================== */

/* 8-bit 版本 (像素类型 uint8_t) */
DEFINE_BLOCK_COPY(8, uint8_t)
DEFINE_PIXEL_AVERAGE(8, uint8_t)
DEFINE_IP_FILTER_LUMA_HOR(8, uint8_t)
DEFINE_IP_FILTER_LUMA_VER(8, uint8_t)
DEFINE_IP_FILTER_LUMA_EXT(8, uint8_t)
DEFINE_IP_FILTER_CHROMA_HOR(8, uint8_t)
DEFINE_IP_FILTER_CHROMA_VER(8, uint8_t)
DEFINE_IP_FILTER_CHROMA_EXT(8, uint8_t)

/* 16-bit 版本 (像素类型 uint16_t, 用于 10-bit) */
DEFINE_BLOCK_COPY(16, uint16_t)
DEFINE_PIXEL_AVERAGE(16, uint16_t)
DEFINE_IP_FILTER_LUMA_HOR(16, uint16_t)
DEFINE_IP_FILTER_LUMA_VER(16, uint16_t)
DEFINE_IP_FILTER_LUMA_EXT(16, uint16_t)
DEFINE_IP_FILTER_CHROMA_HOR(16, uint16_t)
DEFINE_IP_FILTER_CHROMA_VER(16, uint16_t)
DEFINE_IP_FILTER_CHROMA_EXT(16, uint16_t)


/* ===================================================================
 * 第四部分: 通用插值分发器 ip_filter
 * =================================================================== */

/* -------------------------------------------------------------------
 * ip_filter: 通用插值滤波入口
 * 根据 dx/dy 分发到对应的水平/垂直/双向插值函数
 * (对应 davs2 mc_luma / mc_chroma 中的分发逻辑)
 *
 * 参数:
 *   dst, i_dst    - 目标缓冲及步长 (以元素计)
 *   src, i_src    - 源缓冲及步长 (以元素计)
 *   width, height - 块尺寸
 *   dx, dy        - 子像素偏移 (亮度: 0..3, 色度: 0..7)
 *   is_luma       - 1=亮度, 0=色度
 *   bit_depth     - 位深 (8 或 10)
 * ------------------------------------------------------------------- */
#define DEFINE_IP_FILTER(SUFFIX, PTYPE) \
static void ip_filter_##SUFFIX( \
        PTYPE *dst, int i_dst, const PTYPE *src, int i_src, \
        int width, int height, int dx, int dy, \
        int is_luma, int bit_depth) \
{ \
    if (is_luma) { \
        /* 亮度: 8 抽头, 1/4 像素精度 (dx, dy: 0..3) */ \
        if (dx == 0 && dy == 0) { \
            block_copy_##SUFFIX(dst, i_dst, src, i_src, width, height); \
        } else if (dx == 0) { \
            ip_filter_luma_ver_##SUFFIX(dst, i_dst, src, i_src, \
                                        width, height, \
                                        avs2_intpl_filters[dy], bit_depth); \
        } else if (dy == 0) { \
            ip_filter_luma_hor_##SUFFIX(dst, i_dst, src, i_src, \
                                        width, height, \
                                        avs2_intpl_filters[dx], bit_depth); \
        } else { \
            ip_filter_luma_ext_##SUFFIX(dst, i_dst, src, i_src, \
                                        width, height, \
                                        avs2_intpl_filters[dx], \
                                        avs2_intpl_filters[dy], bit_depth); \
        } \
    } else { \
        /* 色度: 4 抽头, 1/8 像素精度 (dx, dy: 0..7) */ \
        if (dx == 0 && dy == 0) { \
            block_copy_##SUFFIX(dst, i_dst, src, i_src, width, height); \
        } else if (dx == 0) { \
            ip_filter_chroma_ver_##SUFFIX(dst, i_dst, src, i_src, \
                                          width, height, \
                                          avs2_intpl_filters_c[dy], bit_depth); \
        } else if (dy == 0) { \
            ip_filter_chroma_hor_##SUFFIX(dst, i_dst, src, i_src, \
                                          width, height, \
                                          avs2_intpl_filters_c[dx], bit_depth); \
        } else { \
            ip_filter_chroma_ext_##SUFFIX(dst, i_dst, src, i_src, \
                                          width, height, \
                                          avs2_intpl_filters_c[dx], \
                                          avs2_intpl_filters_c[dy], bit_depth); \
        } \
    } \
}

DEFINE_IP_FILTER(8, uint8_t)
DEFINE_IP_FILTER(16, uint16_t)


/* ===================================================================
 * 第五部分: 加权跳过模式 (weighted_skip)
 * =================================================================== */

/* -------------------------------------------------------------------
 * weighted_skip: 加权跳过模式的双向预测
 *
 * AVS2 加权跳过模式 (Weighted Skip Mode, WSM) 是一种特殊的跳过
 * 模式, 其后向参考图像索引由码流显式指示。在像素级别, 预测值为
 * 前向和后向预测的简单平均:
 *   dst = (src0 + src1 + 1) >> 1
 *
 * 参数:
 *   dst, i_dst    - 目标缓冲及步长 (以字节计)
 *   src0, i_src0  - 前向预测结果及步长
 *   src1, i_src1  - 后向预测结果及步长
 *   width, height - 块尺寸
 *   bit_depth     - 位深 (8 或 10)
 * ------------------------------------------------------------------- */
void weighted_skip(uint8_t *dst, ptrdiff_t i_dst,
                   const uint8_t *src0, ptrdiff_t i_src0,
                   const uint8_t *src1, ptrdiff_t i_src1,
                   int width, int height, int bit_depth)
{
    if (bit_depth > 8) {
        /* 10-bit: 转换为 16-bit 指针, 步长折半为元素步长 */
        pixel_average_16((uint16_t *)(void *)dst, (int)(i_dst / 2),
                         (const uint16_t *)(const void *)src0, (int)(i_src0 / 2),
                         (const uint16_t *)(const void *)src1, (int)(i_src1 / 2),
                         width, height);
    } else {
        /* 8-bit: 字节步长即元素步长 */
        pixel_average_8(dst, (int)i_dst,
                        src0, (int)i_src0,
                        src1, (int)i_src1,
                        width, height);
    }
}


/* ===================================================================
 * 第六部分: DSP 入口函数 (mc_luma_c / mc_chroma_c)
 * =================================================================== */

/* -------------------------------------------------------------------
 * mc_luma_c: 亮度运动补偿入口
 *
 * 根据子像素偏移 (mx, my) 选择整像素拷贝或 8 抽头插值滤波。
 * 滤波器系数表: avs2_intpl_filters[4][8] (索引 0=整像素, 1=1/4,
 * 2=1/2, 3=3/4)。
 *
 * 参数:
 *   src, sstride - 参考帧源指针及步长 (字节)
 *   dst, dstride - 目标帧指针及步长 (字节)
 *   w, h         - 块尺寸
 *   mx, my       - 子像素偏移 (亮度: 1/4 像素, mx&3 为分数部分)
 *   bit_depth    - 位深
 * ------------------------------------------------------------------- */
void mc_luma_c(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                      ptrdiff_t dstride, int w, int h, int mx, int my,
                      int bit_depth)
{
    int dx = mx & 3;   /* 亮度 1/4 像素精度 */
    int dy = my & 3;

    if (bit_depth > 8) {
        /* 10-bit: 转换为 16-bit 指针, 步长折半为元素步长 */
        ip_filter_16((uint16_t *)(void *)dst, (int)(dstride / 2),
                     (const uint16_t *)(const void *)src, (int)(sstride / 2),
                     w, h, dx, dy, 1, bit_depth);
    } else {
        /* 8-bit: 字节步长即元素步长 */
        ip_filter_8(dst, (int)dstride, src, (int)sstride,
                    w, h, dx, dy, 1, bit_depth);
    }
}

/* -------------------------------------------------------------------
 * mc_chroma_c: 色度运动补偿入口
 *
 * 根据子像素偏移 (mx, my) 选择整像素拷贝或 4 抽头插值滤波。
 * 滤波器系数表: avs2_intpl_filters_c[8][4] (索引 0=整像素,
 * 1..7 对应 1/8 ... 7/8)。
 *
 * 参数同 mc_luma_c, 但 mx/my 为 1/8 像素精度色度偏移。
 * ------------------------------------------------------------------- */
void mc_chroma_c(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                        ptrdiff_t dstride, int w, int h, int mx, int my,
                        int bit_depth)
{
    int dx = mx & 7;   /* 色度 1/8 像素精度 */
    int dy = my & 7;

    if (bit_depth > 8) {
        /* 10-bit: 转换为 16-bit 指针, 步长折半为元素步长 */
        ip_filter_16((uint16_t *)(void *)dst, (int)(dstride / 2),
                     (const uint16_t *)(const void *)src, (int)(sstride / 2),
                     w, h, dx, dy, 0, bit_depth);
    } else {
        /* 8-bit: 字节步长即元素步长 */
        ip_filter_8(dst, (int)dstride, src, (int)sstride,
                    w, h, dx, dy, 0, bit_depth);
    }
}


/* ===================================================================
 * 第七部分: DSP 初始化
 * =================================================================== */

/* 双向预测平均 C 回退: dst[i] = (dst[i] + pred2[i] + 1) >> 1 */
static void bi_avg_c(uint8_t *dst, ptrdiff_t dst_stride, const int16_t *pred2,
                     int pred2_stride, int w, int h, int bit_depth)
{
    int x, y;
    if (bit_depth > 8) {
        uint16_t *p16 = (uint16_t *)(void *)dst;
        int stride16 = (int)(dst_stride >> 1);
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                int v = p16[y * stride16 + x] + pred2[y * pred2_stride + x];
                p16[y * stride16 + x] = (uint16_t)((v + 1) >> 1);
            }
        }
    } else {
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                int v = dst[y * dst_stride + x] + (uint8_t)pred2[y * pred2_stride + x];
                dst[y * dst_stride + x] = (uint8_t)((v + 1) >> 1);
            }
        }
    }
}

/* 双向预测平均 (双源) C 回退: dst[i] = (pred1[i] + pred2[i] + 1) >> 1 */
static void bi_avg_2src_c(uint8_t *dst, ptrdiff_t dst_stride,
                          const int16_t *pred1, int pred1_stride,
                          const int16_t *pred2, int pred2_stride,
                          int w, int h, int bit_depth)
{
    int x, y;
    if (bit_depth > 8) {
        uint16_t *p16 = (uint16_t *)(void *)dst;
        int stride16 = (int)(dst_stride >> 1);
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                int v = pred1[y * pred1_stride + x] + pred2[y * pred2_stride + x];
                p16[y * stride16 + x] = (uint16_t)((v + 1) >> 1);
            }
        }
    } else {
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                int v = pred1[y * pred1_stride + x] + pred2[y * pred2_stride + x];
                dst[y * dst_stride + x] = (uint8_t)((v + 1) >> 1);
            }
        }
    }
}

/* 块填充 C 回退: dst[i] = fill_val */
static void fill_block_c(uint8_t *dst, ptrdiff_t dst_stride, int w, int h,
                         int fill_val, int bit_depth)
{
    int x, y;
    if (bit_depth > 8) {
        uint16_t *p16 = (uint16_t *)(void *)dst;
        int stride16 = (int)(dst_stride >> 1);
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                p16[y * stride16 + x] = (uint16_t)fill_val;
            }
        }
    } else {
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) {
                dst[y * dst_stride + x] = (uint8_t)fill_val;
            }
        }
    }
}

/* MC + 双向平均 C 回退: 先 MC 到 pred2, 再与 dst 平均 */
void mc_luma_avg_c(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                          ptrdiff_t dstride, int w, int h, int mx, int my,
                          int bit_depth)
{
    int16_t pred2[64 * 64];
    int x, y;
    mc_luma_c(src, sstride, (uint8_t *)pred2, w * sizeof(int16_t),
              w, h, mx, my, bit_depth);
    if (bit_depth > 8) {
        uint16_t *p16 = (uint16_t *)(void *)dst;
        int stride16 = (int)(dstride >> 1);
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                int v = p16[y * stride16 + x] + pred2[y * w + x];
                p16[y * stride16 + x] = (uint16_t)((v + 1) >> 1);
            }
    } else {
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                int v = dst[y * dstride + x] + (uint8_t)pred2[y * w + x];
                dst[y * dstride + x] = (uint8_t)((v + 1) >> 1);
            }
    }
}

void mc_chroma_avg_c(const uint8_t *src, ptrdiff_t sstride, uint8_t *dst,
                            ptrdiff_t dstride, int w, int h, int mx, int my,
                            int bit_depth)
{
    int16_t pred2[32 * 32];
    int x, y;
    mc_chroma_c(src, sstride, (uint8_t *)pred2, w * sizeof(int16_t),
                w, h, mx, my, bit_depth);
    if (bit_depth > 8) {
        uint16_t *p16 = (uint16_t *)(void *)dst;
        int stride16 = (int)(dstride >> 1);
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                int v = p16[y * stride16 + x] + pred2[y * w + x];
                p16[y * stride16 + x] = (uint16_t)((v + 1) >> 1);
            }
    } else {
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                int v = dst[y * dstride + x] + (uint8_t)pred2[y * w + x];
                dst[y * dstride + x] = (uint8_t)((v + 1) >> 1);
            }
    }
}

/* -------------------------------------------------------------------
 * avs2_mc_init: 注册运动补偿函数到 DSP 函数表
 * ------------------------------------------------------------------- */
void avs2_mc_init(void)
{
    avs2_dsp_table.mc_luma   = mc_luma_c;
    avs2_dsp_table.mc_chroma = mc_chroma_c;
    avs2_dsp_table.mc_luma_avg   = mc_luma_avg_c;
    avs2_dsp_table.mc_chroma_avg = mc_chroma_avg_c;
    avs2_dsp_table.bi_avg    = bi_avg_c;
    avs2_dsp_table.bi_avg_2src = bi_avg_2src_c;
    avs2_dsp_table.fill_block = fill_block_c;
}
