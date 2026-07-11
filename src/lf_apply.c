/*
 * 环路滤波应用入口 (从 davs2 移植到 C)。
 *
 * 对一个 LCU 行依次执行:
 *   1. 去块滤波 (deblock) - 逐 LCU 调用 avs2_loop_filter
 *   2. SAO 滤波 - 从临时帧 (解码帧副本) 滤波到解码帧
 *   3. ALF 滤波 - 从临时帧 (解码帧副本) 滤波到解码帧
 *
 * SAO 和 ALF 需要 src/dst 分离 (滤波时读取邻域像素, 原地写会污染源),
 * 因此分配临时帧作为输入, 解码帧作为输出。
 *
 * 命名: 小写加下划线。pel_t -> uint16_t。
 */

#include "internal.h"
#include "aec_internal.h"
#include <string.h>
#include <stdio.h>

/* 辅助宏 (本文件局部使用) */
#ifndef AVS2_MIN
#define AVS2_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#define LF_SAO_SHIFT_PIX 4   /* SAO 滤波区域边界扩展像素数 */

/* --------------------------------------------------------------------------
 * 分配与 src 尺寸相同的临时帧 (仅平面数据, 无 CU 网格/SAO/ALF 参数)
 * 平面带 64 行上下 padding + 左右 padding, 与 avs2_frame_alloc 一致
 * -------------------------------------------------------------------------- */
static avs2_frame *lf_tmp_alloc(const avs2_frame *src)
{
    avs2_frame *tmp = avs2_mem_allocz(sizeof(avs2_frame));
    if (!tmp) return NULL;

    tmp->width  = src->width;
    tmp->height = src->height;
    tmp->stride[0] = src->stride[0];
    tmp->stride[1] = src->stride[1];
    tmp->stride[2] = src->stride[2];
    tmp->bytes_per_sample = src->bytes_per_sample;
    tmp->bit_depth = src->bit_depth;
    tmp->chroma_format = src->chroma_format;

    {
        const ptrdiff_t sy = src->stride[0];
        const ptrdiff_t sc = src->stride[1];
        const int h  = src->height;
        const int ch = (h + 1) >> 1;
        const int bps = src->bytes_per_sample;
        uint8_t *p0 = avs2_mem_allocz((size_t)sy * (h  + 2 * 64));
        uint8_t *p1 = avs2_mem_allocz((size_t)sc * (ch + 2 * 64));
        uint8_t *p2 = avs2_mem_allocz((size_t)sc * (ch + 2 * 64));
        if (!p0 || !p1 || !p2) {
            avs2_mem_free(p0);
            avs2_mem_free(p1);
            avs2_mem_free(p2);
            avs2_mem_free(tmp);
            return NULL;
        }
        /* 跳过上 padding 行 + 左 padding 像素 (与 avs2_frame_alloc 一致) */
        tmp->data[0] = p0 + 64 * sy + AVS2_PAD_LUMA * bps;
        tmp->data[1] = p1 + 64 * sc + AVS2_PAD_CHROMA * bps;
        tmp->data[2] = p2 + 64 * sc + AVS2_PAD_CHROMA * bps;
    }
    return tmp;
}

/* --------------------------------------------------------------------------
 * 释放临时帧
 * -------------------------------------------------------------------------- */
static void lf_tmp_free(avs2_frame *tmp)
{
    int i;
    if (!tmp) return;
    for (i = 0; i < 3; i++) {
        if (tmp->data[i]) {
            ptrdiff_t s = tmp->stride[i];
            int pad_h = (i == 0) ? AVS2_PAD_LUMA : AVS2_PAD_CHROMA;
            int bps = tmp->bytes_per_sample;
            uint8_t *base = tmp->data[i] - 64 * s - pad_h * bps;
            avs2_mem_free(base);
            tmp->data[i] = NULL;
        }
    }
    avs2_mem_free(tmp);
}

/* --------------------------------------------------------------------------
 * 确保 fc->lf_tmp 已分配且尺寸匹配 (分辨率变化时重分配)
 * 返回 lf_tmp 指针, 失败返回 NULL
 * -------------------------------------------------------------------------- */
static avs2_frame *ensure_lf_tmp(avs2_frame_ctx *fc, const avs2_frame *src)
{
    if (fc->lf_tmp &&
        fc->lf_tmp_width == src->width &&
        fc->lf_tmp_height == src->height) {
        return fc->lf_tmp;  /* 复用已有缓冲区 */
    }
    /* 尺寸变化或首次分配: 释放旧的, 分配新的 */
    if (fc->lf_tmp) {
        lf_tmp_free(fc->lf_tmp);
        fc->lf_tmp = NULL;
    }
    fc->lf_tmp = lf_tmp_alloc(src);
    if (fc->lf_tmp) {
        fc->lf_tmp_width = src->width;
        fc->lf_tmp_height = src->height;
    }
    return fc->lf_tmp;
}

/* --------------------------------------------------------------------------
 * 只拷贝当前 LCU 行 + 边界扩展区域的数据 (src -> dst)
 * SAO/ALF 滤波半径为 4 像素, 只需拷贝 [lcu_y*lcu - 4, lcu_y*lcu + lcu + 4) 行
 * -------------------------------------------------------------------------- */
static void lf_copy_lcu_row(avs2_frame *dst, const avs2_frame *src,
                            int lcu_y, int lcu_size)
{
    const int bps = src->bytes_per_sample;
    const int border = LF_SAO_SHIFT_PIX;  /* 4 像素边界 */
    int pl;

    /* 亮度行范围 */
    int y_start = lcu_y * lcu_size;
    int y_end   = y_start + lcu_size;
    if (y_start > 0) y_start -= border;
    if (y_end < src->height) y_end += border;
    if (y_start < 0) y_start = 0;
    if (y_end > src->height) y_end = src->height;

    for (pl = 0; pl < 3; pl++) {
        const ptrdiff_t stride = src->stride[pl];
        int w = (pl == 0) ? src->width : (src->width + 1) >> 1;
        int ys, ye;
        const uint8_t *s;
        uint8_t *d;
        int y;
        size_t row_bytes;

        if (pl == 0) {
            ys = y_start;
            ye = y_end;
        } else {
            /* 420 色度: 行范围折半 */
            ys = y_start >> 1;
            ye = (y_end + 1) >> 1;
        }
        row_bytes = (size_t)w * bps;
        s = src->data[pl] + (ptrdiff_t)ys * stride;
        d = dst->data[pl] + (ptrdiff_t)ys * stride;
        for (y = ys; y < ye; y++) {
            memcpy(d, s, row_bytes);
            s += stride;
            d += stride;
        }
    }
}

/* --------------------------------------------------------------------------
 * 释放 fc->lf_tmp (供 avs2_close 调用)
 * -------------------------------------------------------------------------- */
void avs2_lf_tmp_free(avs2_frame_ctx *fc)
{
    if (fc->lf_tmp) {
        lf_tmp_free(fc->lf_tmp);
        fc->lf_tmp = NULL;
        fc->lf_tmp_width = 0;
        fc->lf_tmp_height = 0;
    }
}

/* --------------------------------------------------------------------------
 * 保存 LCU 行底边到 intra_border (在 deblock 之前)
 * 用于下一 LCU 行顶部边界块的 intra 预测 TOP 参考 (对应 davs2 h->intra_border)
 * 行级并行模式: 在 Pass 2 重建后、LF 前调用, 使下一行重建可并行启动
 * -------------------------------------------------------------------------- */
void avs2_save_intra_border(avs2_frame_ctx *fc, struct avs2_internal *c, int lcu_y)
{
    avs2_frame *f = fc->fdec;
    avs2_seq_header *s = c->seq;
    const int lcu   = 1 << s->log2_lcu_size;
    const int h_lcu = f->h_lcu;
    const int bps   = f->bytes_per_sample;

    if (lcu_y >= h_lcu - 1) return;

    const int y_bottom = (lcu_y + 1) * lcu - 1;
    const int y_bottom_c = (lcu_y + 1) * (lcu >> 1) - 1;
    /* stride 已为字节步长, 直接按字节寻址 */
    const ptrdiff_t stride_y = f->stride[0];
    const ptrdiff_t stride_c = f->stride[1];
    uint8_t *src_y = f->data[0] + y_bottom * stride_y;
    uint8_t *src_u = f->data[1] + y_bottom_c * stride_c;
    uint8_t *src_v = f->data[2] + y_bottom_c * stride_c;
    const int pad_luma = AVS2_PAD_LUMA;
    const int pad_chroma = AVS2_PAD_CHROMA;
    /* 保存底行 (含左右 padding 偏移, 便于 top-left/top-right 索引) */
    memcpy(f->intra_border[0] + pad_luma * bps, src_y,
           (size_t)f->width * bps);
    memcpy(f->intra_border[1] + pad_chroma * bps, src_u,
           (size_t)((f->width + 1) >> 1) * bps);
    memcpy(f->intra_border[2] + pad_chroma * bps, src_v,
           (size_t)((f->width + 1) >> 1) * bps);
}

/* --------------------------------------------------------------------------
 * 对一个 LCU 行应用环路滤波 (不含 intra_border 保存, 供行级并行 Pass 2 使用)
 * 顺序: 去块滤波 -> SAO -> ALF
 * -------------------------------------------------------------------------- */
void avs2_lf_apply_lcu_row_nosave(avs2_frame_ctx *fc, struct avs2_internal *c, int lcu_y)
{
    avs2_frame *f = fc->fdec;
    avs2_seq_header *s = c->seq;
    /* 使用 fc->pic_local 而非 c->pic: worker 线程中 c->pic 可能已被主线程覆盖 */
    avs2_pic_header *p = &fc->pic_local;
    const int lcu   = 1 << s->log2_lcu_size;
    const int bd    = c->bit_depth;
    const int w_lcu = f->w_lcu;
    const int h_lcu = f->h_lcu;
    int lcu_x;

    if (c->skip_loop_filter || p->loop_filter_disable) {
        return;
    }

    /* ---- 1. 去块滤波 (逐 LCU, 原地) ---- */
    for (lcu_x = 0; lcu_x < w_lcu; lcu_x++) {
        avs2_loop_filter(fc, c, f, lcu_x, lcu_y);
    }

    /* ---- 2. SAO 滤波 ---- */
    if (s->enable_sao &&
        (fc->slice_sao_on[0] || fc->slice_sao_on[1] || fc->slice_sao_on[2])) {
        avs2_frame *tmp = ensure_lf_tmp(fc, f);
        if (tmp) {
            lf_copy_lcu_row(tmp, f, lcu_y, lcu);  /* 只拷贝当前 LCU 行 + 边界 */
            for (lcu_x = 0; lcu_x < w_lcu; lcu_x++) {
                int idx = lcu_y * w_lcu + lcu_x;
                int avail[8];
                int pl;

                /* 邻域可用性 (与 davs2 sao_get_neighbor_avail 一致) */
                avail[0] = (lcu_y != 0);           /* top        */
                avail[1] = (lcu_y < h_lcu - 1);    /* down       */
                avail[2] = (lcu_x != 0);           /* left       */
                avail[3] = (lcu_x < w_lcu - 1);    /* right      */
                avail[4] = avail[0] && avail[2];   /* top-left   */
                avail[5] = avail[0] && avail[3];   /* top-right  */
                avail[6] = avail[1] && avail[2];   /* down-left  */
                avail[7] = avail[1] && avail[3];   /* down-right */

                for (pl = 0; pl < 3; pl++) {
                    avs2_sao_param *sp;
                    int pix_x, pix_y, blk_w, blk_h;
                    int max_w, max_h;

                    if (!fc->slice_sao_on[pl])
                        continue;
                    sp = &f->sao_params[idx * 3 + pl];
                    if (sp->mode_idc == SAO_MODE_OFF)
                        continue;

                    /* 计算滤波区域 (含边界扩展) */
                    if (pl == 0) {
                        pix_x = lcu_x * lcu;
                        pix_y = lcu_y * lcu;
                        blk_w = lcu;
                        blk_h = lcu;
                        max_w = f->width;
                        max_h = f->height;
                    } else {
                        pix_x = (lcu_x * lcu) >> 1;
                        pix_y = (lcu_y * lcu) >> 1;
                        blk_w = lcu >> 1;
                        blk_h = lcu >> 1;
                        max_w = (f->width  + 1) >> 1;
                        max_h = (f->height + 1) >> 1;
                    }

                    /* 边界扩展 (与 davs2 sao_get_neighbor_avail 一致) */
                    if (avail[2]) { pix_x -= LF_SAO_SHIFT_PIX; }
                    else          { blk_w -= LF_SAO_SHIFT_PIX; }
                    if (avail[0]) { pix_y -= LF_SAO_SHIFT_PIX; }
                    else          { blk_h -= LF_SAO_SHIFT_PIX; }
                    if (!avail[3]) { blk_w += LF_SAO_SHIFT_PIX; }
                    if (!avail[1]) { blk_h += LF_SAO_SHIFT_PIX; }

                    /* 限制在图像范围内 */
                    blk_w = AVS2_MIN(blk_w, max_w - pix_x);
                    blk_h = AVS2_MIN(blk_h, max_h - pix_y);

                    if (blk_w > 0 && blk_h > 0) {
                        /* dst=解码帧 (输出), src=临时帧 (输入) */
                        avs2_sao_on_block(f, tmp, pl, pix_x, pix_y,
                                          blk_w, blk_h, bd, avail, sp);
                    }
                }
            }
            /* tmp 由 fc->lf_tmp 管理, 不释放, 下次复用 */
        }
    }

    /* ---- 3. ALF 滤波 ---- */
    if (s->enable_alf &&
        (p->alf_pic_flag_y | p->alf_pic_flag_cb | p->alf_pic_flag_cr)) {
        avs2_frame *tmp = ensure_lf_tmp(fc, f);
        if (tmp) {
            lf_copy_lcu_row(tmp, f, lcu_y, lcu);  /* 只拷贝当前 LCU 行 + 边界 */
            for (lcu_x = 0; lcu_x < w_lcu; lcu_x++) {
                int idx = lcu_y * w_lcu + lcu_x;
                avs2_alf_param *ap = &f->alf_params[idx];
                int b_top_avail  = (lcu_y != 0);
                int b_down_avail = (lcu_y < h_lcu - 1);
                /* cross_loop_filter_flag 为假时需检查 slice 边界
                 * (avs2dec 无逐 CU slice 号, 简化: 等价于始终允许跨 slice) */

                /* dst=解码帧 (输出), src=临时帧 (输入) */
                avs2_alf_on_block(f, tmp, ap, lcu_x, lcu_y, lcu,
                                  f->width, f->height, bd,
                                  b_top_avail, b_down_avail);
            }
            /* tmp 由 fc->lf_tmp 管理, 不释放, 下次复用 */
        }
    }
}

/* --------------------------------------------------------------------------
 * 对一个 LCU 行应用环路滤波 (含 intra_border 保存, 单 pass 模式使用)
 * 顺序: 保存底边 -> 去块滤波 -> SAO -> ALF
 * -------------------------------------------------------------------------- */
void avs2_lf_apply_lcu_row(avs2_frame_ctx *fc, struct avs2_internal *c, int lcu_y)
{
    avs2_save_intra_border(fc, c, lcu_y);
    avs2_lf_apply_lcu_row_nosave(fc, c, lcu_y);
}

/* --------------------------------------------------------------------------
 * DSP 函数指针表全局定义 + 初始化
 * -------------------------------------------------------------------------- */
avs2_dsp avs2_dsp_table;

/* 是否禁用 SIMD (测试用, --no-simd 设置) */
int g_disable_simd = 0;

/* SIMD 注册函数 (x86/arm), 在对应源文件中定义.
 * 每个 init 函数注册其指令集的实现, 后调用者覆盖先调用者. */
extern void avs2_itx_init_simd(const avs2_cpu_flags *cpu);
extern void avs2_mc_init_simd(const avs2_cpu_flags *cpu);
extern void avs2_ipred_init_simd(const avs2_cpu_flags *cpu);
extern void avs2_lf_init_simd(const avs2_cpu_flags *cpu);
extern void avs2_sao_init_simd(const avs2_cpu_flags *cpu);
extern void avs2_alf_init_simd(const avs2_cpu_flags *cpu);
extern void avs2_quant_init_simd(const avs2_cpu_flags *cpu);

/* 全局 cpu flags (simd_init 中使用) */
avs2_cpu_flags g_cpu_flags;

void avs2_dsp_init(void)
{
    /* 1. 注册 C 参考实现 (基础回退) */
    avs2_ipred_init();
    avs2_mc_init();
    avs2_itx_init();
    avs2_quant_init();
    avs2_loopfilter_init();
    avs2_sao_init();
    avs2_alf_init();

    /* 2. 检测 CPU 并按优先级注册 SIMD 实现 (高优先级覆盖低优先级) */
    if (!g_disable_simd) {
        avs2_cpu_detect(&g_cpu_flags);

        /* 优先级从低到高: SSE4.1 -> AVX2 -> AVX512 (x86) / NEON (arm) */
        avs2_itx_init_simd(&g_cpu_flags);
        avs2_mc_init_simd(&g_cpu_flags);
        avs2_ipred_init_simd(&g_cpu_flags);
        avs2_lf_init_simd(&g_cpu_flags);
        avs2_sao_init_simd(&g_cpu_flags);
        avs2_alf_init_simd(&g_cpu_flags);
        avs2_quant_init_simd(&g_cpu_flags);
    }
}

