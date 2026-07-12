#include "internal.h"
#include "aec_internal.h"
#include <string.h>

/* 帧边界扩展 padding (对应 davs2 AVS2_PAD) */
#define PAD_LUMA   AVS2_PAD_LUMA    /* 64 */
#define PAD_CHROMA AVS2_PAD_CHROMA  /* 32 */
#define PAD_VERT   64               /* 上下 padding 行数 (亮度和色度均 64) */

/* 获取指定平面的水平 padding 像素数 */
static inline int frame_pad_h(int pl)
{
    return pl == 0 ? PAD_LUMA : PAD_CHROMA;
}

int avs2_frame_alloc(avs2_frame *f, struct avs2_internal *c)
{
    const int bd = c->bit_depth;
    const int bps = (bd > 8) ? 2 : 1;
    const int new_w = (int)c->seq->enc_width;
    const int new_h = (int)c->seq->enc_height;
    const int new_chroma = (int)c->seq->chroma_format;

    /* 帧缓冲池化: 检查是否可复用已分配的缓冲区 */
    int can_reuse = (f->alloc_width == new_w &&
                     f->alloc_height == new_h &&
                     f->alloc_bit_depth == bd &&
                     f->alloc_chroma == new_chroma &&
                     f->data[0] != NULL);

    if (can_reuse) {
        /* 复用: 仅重置状态字段, 不重新分配内存.
         * 需要清零元数据缓冲区 (cu_grid, mvbuf, deblock_flags, sao/alf_params, ipredmode) */
        f->bit_depth = bd;
        f->bytes_per_sample = bps;
        f->chroma_format = new_chroma;
        f->width = new_w;
        f->height = new_h;
        f->w8 = (new_w + 7) >> 3;
        f->h8 = (new_h + 7) >> 3;
        const int lcu = 1 << c->seq->log2_lcu_size;
        f->w_lcu = (new_w + lcu - 1) / lcu;
        f->h_lcu = (new_h + lcu - 1) / lcu;
        f->qp = (int)c->pic->picture_qp;

        /* 清零元数据缓冲区内容 */
        {
            const int w = f->width, h = f->height;
            const int cw = (w + 1) >> 1;
            (void)cw;
            /* 清零图像数据 (含 padding) */
            for (int i = 0; i < 3; i++) {
                ptrdiff_t s = f->stride[i];
                int pad_h = frame_pad_h(i);
                int bps_cur = f->bytes_per_sample;
                uint8_t *base = f->data[i] - PAD_VERT * s - pad_h * bps_cur;
                int plane_h = (i == 0) ? h : ((h + 1) >> 1);
                memset(base, 0, (size_t)s * (plane_h + 2 * PAD_VERT));
            }
        }
        memset(f->cu_grid, 0, sizeof(avs2_cu) * (size_t)f->w8 * f->h8);
        memset(f->mvbuf, 0, sizeof(avs2_mv) * (size_t)f->w8 * f->h8 * 4);
        if (f->refbuf) {
            memset(f->refbuf, INVALID_REF, (size_t)f->w8 * f->h8 * 4);
        }
        memset(f->deblock_flags[0], 0, (size_t)f->w8 * f->h8);
        memset(f->deblock_flags[1], 0, (size_t)f->w8 * f->h8);
        memset(f->sao_params, 0, sizeof(avs2_sao_param) * (size_t)f->w_lcu * f->h_lcu * 3);
        memset(f->alf_params, 0, sizeof(avs2_alf_param) * (size_t)f->w_lcu * f->h_lcu);
        {
            int h_spu = (f->height + 3) >> 2;
            memset(f->ipredmode_base, DC_PRED,
                   sizeof(int8_t) * (size_t)f->ipredmode_stride * (h_spu + 2));
        }

        /* 重置状态标志 */
        f->poc = 0;
        f->coi = 0;
        f->type = 0;
        f->structure = 0;
        f->referenced = 0;
        f->output = 0;
        f->pts = 0;
        f->dts = 0;
        f->done = 0;
        f->p2_started = 0;
        f->lf_row_done_count = 0;
        for (int i = 0; i < AVS2_MAX_H_LCU; i++) {
            f->lf_row_done[i] = 0;
            f->aec_row_done[i] = 0;
        }
        f->used = 1;
        f->ref_cnt = 1;
        return AVS2_OK;
    }

    /* 不能复用: 释放旧缓冲区后重新分配 */
    if (f->data[0]) {
        avs2_frame_free_buffers(f);
    }

    /* 保存原始 alloc 字段, memset 后恢复 */
    memset(f, 0, sizeof(*f));
    f->bit_depth = bd;
    f->bytes_per_sample = bps;
    f->chroma_format = new_chroma;
    f->width = new_w;
    f->height = new_h;
    f->w8 = (new_w + 7) >> 3;
    f->h8 = (new_h + 7) >> 3;
    const int lcu = 1 << c->seq->log2_lcu_size;
    f->w_lcu = (new_w + lcu - 1) / lcu;
    f->h_lcu = (new_h + lcu - 1) / lcu;
    f->qp = (int)c->pic->picture_qp;

    const int w = f->width;
    const int h = f->height;
    const int cw = (w + 1) >> 1;  /* 420 */
    const int ch = (h + 1) >> 1;

    /* allocate planes with padding (左/右/上/下)
     * stride 包含左右 padding: 亮度 2*64, 色度 2*32 */
    ptrdiff_t sy = ((w + 2 * PAD_LUMA + 63) & ~(ptrdiff_t)63) * bps;
    ptrdiff_t sc = ((cw + 2 * PAD_CHROMA + 63) & ~(ptrdiff_t)63) * bps;
    f->stride[0] = sy;
    f->stride[1] = f->stride[2] = sc;
    f->data[0] = avs2_mem_allocz((size_t)sy * (h + 2 * PAD_VERT));
    f->data[1] = avs2_mem_allocz((size_t)sc * (ch + 2 * PAD_VERT));
    f->data[2] = avs2_mem_allocz((size_t)sc * (ch + 2 * PAD_VERT));
    if (!f->data[0] || !f->data[1] || !f->data[2]) {
        avs2_frame_free_buffers(f);
        return AVS2_ERR_NOMEM;
    }
    /* offset into the padded region: 跳过上 padding 行 + 左 padding 像素 */
    f->data[0] += PAD_VERT * sy + PAD_LUMA * bps;
    f->data[1] += PAD_VERT * sc + PAD_CHROMA * bps;
    f->data[2] += PAD_VERT * sc + PAD_CHROMA * bps;

    /* 辅助缓冲区: 合并 refbuf/deblock_flags/intra_border/ipredmode_base
     * 为单次分配, 减少 alloc/free 次数和内存碎片.
     * 各子缓冲区按 32 字节对齐切分. */
    {
        int w_spu = (f->width + 3) >> 2;   /* width in 4x4 blocks */
        int h_spu = (f->height + 3) >> 2;  /* height in 4x4 blocks */
        f->ipredmode_stride = w_spu + 2;   /* 1 padding on each side */

        /* 计算各子缓冲区大小 (32 字节对齐) */
    #define AUX_ALIGN32(n) (((size_t)(n) + 31) & ~(size_t)31)
        size_t sz_refbuf  = AUX_ALIGN32((size_t)f->w8 * f->h8 * 4);
        size_t sz_dbf0    = AUX_ALIGN32((size_t)f->w8 * f->h8);
        size_t sz_dbf1    = AUX_ALIGN32((size_t)f->w8 * f->h8);
        size_t sz_ib0     = AUX_ALIGN32((size_t)(w + 2 * PAD_LUMA) * bps);
        size_t sz_ib1     = AUX_ALIGN32((size_t)(cw + 2 * PAD_CHROMA) * bps);
        size_t sz_ib2     = AUX_ALIGN32((size_t)(cw + 2 * PAD_CHROMA) * bps);
        size_t sz_ipm     = AUX_ALIGN32((size_t)f->ipredmode_stride * (h_spu + 2));
        size_t aux_total  = sz_refbuf + sz_dbf0 + sz_dbf1 +
                            sz_ib0 + sz_ib1 + sz_ib2 + sz_ipm;
    #undef AUX_ALIGN32

        f->aux_buf = avs2_mem_allocz(aux_total);
        if (!f->aux_buf) {
            avs2_frame_free_buffers(f);
            return AVS2_ERR_NOMEM;
        }
        uint8_t *p = f->aux_buf;
        f->refbuf           = p;  p += sz_refbuf;
        f->deblock_flags[0] = p;  p += sz_dbf0;
        f->deblock_flags[1] = p;  p += sz_dbf1;
        f->intra_border[0]  = p;  p += sz_ib0;
        f->intra_border[1]  = p;  p += sz_ib1;
        f->intra_border[2]  = p;  p += sz_ib2;
        f->ipredmode_base   = p;  p += sz_ipm;
        /* offset to first real position (skip top row + left column) */
        f->ipredmode = f->ipredmode_base + f->ipredmode_stride + 1;

        /* 初始化: refbuf=INVALID_REF, ipredmode=DC_PRED, 其余已 zeroed */
        memset(f->refbuf, INVALID_REF, (size_t)f->w8 * f->h8 * 4);
        memset(f->ipredmode_base, DC_PRED,
               sizeof(int8_t) * (size_t)f->ipredmode_stride * (h_spu + 2));
    }

    /* per-CU grid: one entry per 8x8 block */
    f->cu_grid = avs2_mem_allocz(sizeof(avs2_cu) * (size_t)f->w8 * f->h8);
    f->mvbuf = avs2_mem_allocz(sizeof(avs2_mv) * (size_t)f->w8 * f->h8 * 4);
    f->sao_params = avs2_mem_allocz(sizeof(avs2_sao_param) * (size_t)f->w_lcu * f->h_lcu * 3);
    f->alf_params = avs2_mem_allocz(sizeof(avs2_alf_param) * (size_t)f->w_lcu * f->h_lcu);
    /* 行级 LF 完成跟踪数组 (静态数组, 无需分配) */
    f->lf_row_done_count = 0;
    if (!f->cu_grid || !f->mvbuf || !f->sao_params || !f->alf_params) {
        avs2_frame_free_buffers(f);
        return AVS2_ERR_NOMEM;
    }

    /* 记录已分配尺寸, 供下次复用判断 */
    f->alloc_width = new_w;
    f->alloc_height = new_h;
    f->alloc_bit_depth = bd;
    f->alloc_chroma = new_chroma;

    f->used = 1;
    f->ref_cnt = 1;
    return AVS2_OK;
}

/* 释放帧的内部缓冲区 (仅在关闭或分辨率变化时调用) */
void avs2_frame_free_buffers(avs2_frame *f)
{
    if (!f) return;
    for (int i = 0; i < 3; i++) {
        if (f->data[i]) {
            /* undo the padding offset: 上 padding 行 + 左 padding 像素 */
            ptrdiff_t s = f->stride[i];
            int pad_h = frame_pad_h(i);
            int bps = f->bytes_per_sample;
            uint8_t *base = f->data[i] - PAD_VERT * s - pad_h * bps;
            avs2_mem_free(base);
            f->data[i] = NULL;
        }
    }
    /* aux_buf 包含 refbuf/deblock_flags/intra_border/ipredmode_base, 单次释放 */
    avs2_mem_free(f->aux_buf);       f->aux_buf = NULL;
    f->ipredmode_base = NULL; f->ipredmode = NULL;
    f->refbuf = NULL;
    f->deblock_flags[0] = NULL; f->deblock_flags[1] = NULL;
    f->intra_border[0] = NULL; f->intra_border[1] = NULL; f->intra_border[2] = NULL;
    avs2_mem_free(f->cu_grid);        f->cu_grid = NULL;
    avs2_mem_free(f->mvbuf);          f->mvbuf = NULL;
    avs2_mem_free(f->sao_params);     f->sao_params = NULL;
    avs2_mem_free(f->alf_params);     f->alf_params = NULL;
    f->alloc_width = 0;
    f->alloc_height = 0;
    f->alloc_bit_depth = 0;
    f->alloc_chroma = 0;
}

/* 帧缓冲池化: 仅标记为空闲, 不释放缓冲区 (供下次复用) */
void avs2_frame_free(avs2_frame *f)
{
    if (!f) return;
    f->used = 0;
}

void avs2_frame_ref(avs2_frame *dst, avs2_frame *src)
{
    *dst = *src;
    /* shallow copy shares the buffers; bump usage */
    dst->used = 1;
}

avs2_frame *avs2_dpb_get_free(struct avs2_internal *c)
{
    /* 1. 查找真正空闲的帧 (!used).
     * avs2_frame_free 仅由 avs2_dpb_get_free (此处) 和 avs2_picture_unref 调用,
     * 两者都在 task_lock 保护下检查 ref_cnt, 故 !used 帧的 ref_cnt 必定 <= 1. */
    for (int i = 0; i < c->n_dpb; i++) {
        if (!c->dpb[i] || !c->dpb[i]->used)
            return c->dpb[i];
    }
    /* 2. 尝试回收: 已输出、不被后续帧参考、无 worker/user 引用的帧.
     * 在扩展 DPB 之前尝试回收, 避免不必要的内存增长.
     * ref_cnt 检查确保帧未被任何正在解码的帧引用 (ref_cnt 在
     * avs2_submit_frame_task 的 task_lock 内递增, 此处也在 task_lock 内). */
    for (int i = 0; i < c->n_dpb; i++) {
        avs2_frame *f = c->dpb[i];
        if (f && f->used && !f->referenced && f->output && f->ref_cnt <= 1) {
            avs2_frame_free(f);
            return f;
        }
    }
    /* 3. grow DPB */
    if (c->n_dpb < AVS2_MAX_FRAME_DELAY) {
        c->dpb[c->n_dpb] = avs2_mem_allocz(sizeof(avs2_frame));
        return c->dpb[c->n_dpb++];
    }
    /* 4. 最后手段: 释放已完成、不被参考且无 worker/user 引用的最老帧 (即使未输出).
     * 此路径仅在 DPB 已达上限 (AVS2_MAX_FRAME_DELAY) 时触发, 属于极端内存压力
     * 下的损失性回收: 丢弃一帧以避免 NOMEM 停摆 (帧并行模式下显示顺序与解码顺序
     * 不同, 未输出但已不被参考的帧会在 DPB 中堆积, 输出停滞时可能涨至上限).
     *
     * 必须检查 f->done: ref_cnt<=1 仅表示无正在解码的帧引用此帧, 但帧自身可能
     * 仍在解码中 (refered_by_others=0 的帧解码期间 ref_cnt 保持 1). 回收未完成
     * 的帧会导致 worker 写入被 avs2_frame_alloc 复用/memset 的缓冲区. 原代码靠
     * f->output 隐式保证 done (MT 下 output=1 蕴含 done=1), 移除 output 条件后
     * 必须显式检查 done. */
    {
        int oldest_coi = 0x7fffffff;
        avs2_frame *oldest = NULL;
        for (int i = 0; i < c->n_dpb; i++) {
            avs2_frame *f = c->dpb[i];
            if (f && f->used && f->done && !f->referenced && f->ref_cnt <= 1 && f->coi < oldest_coi) {
                oldest_coi = f->coi;
                oldest = f;
            }
        }
        if (oldest) {
            avs2_frame_free(oldest);
            return oldest;
        }
    }
    return NULL;
}

void avs2_dpb_clear(struct avs2_internal *c)
{
    /* 关闭时真正释放所有 DPB 帧的缓冲区 */
    for (int i = 0; i < c->n_dpb; i++) {
        if (c->dpb[i]) {
            avs2_frame_free_buffers(c->dpb[i]);
            c->dpb[i]->used = 0;
        }
    }
}

/* ---------------------------------------------------------------------------
 * 帧边界扩展 (对应 davs2 pad_line_lcu)
 * 在每个 LCU 行解码 + 环路滤波后调用, 将图像边缘像素复制到 padding 区域,
 * 供后续帧的帧间预测在 MV 指向图像外时读取.
 *
 * 对每个平面:
 *   1. 填充当前 LCU 行内每行的左右 padding (用边缘像素值)
 *   2. 第一行 LCU: 填充图像上方 padding 行 (复制第 0 行)
 *   3. 最后一行 LCU: 填充图像下方 padding 行 (复制最后行)
 * -------------------------------------------------------------------------- */

/* 填充一行像素的左右 padding (用边缘像素值) */
static void pad_line_pixel(uint8_t *pix, int width, int num_pad, int bps)
{
    if (bps == 2) {
        uint16_t *p = (uint16_t *)(void *)pix;
        uint16_t left_val = p[0];
        uint16_t right_val = p[width - 1];
        int i;
        for (i = 0; i < num_pad; i++) {
            p[-1 - i] = left_val;
            p[width + i] = right_val;
        }
    } else {
        uint8_t *p = pix;
        uint8_t left_val = p[0];
        uint8_t right_val = p[width - 1];
        int i;
        for (i = 0; i < num_pad; i++) {
            p[-1 - i] = left_val;
            p[width + i] = right_val;
        }
    }
}

void avs2_pad_line_lcu(avs2_frame *f, int lcu_y, int lcu_size_log2)
{
    const int lcu_size = 1 << lcu_size_log2;
    const int bps = f->bytes_per_sample;
    int pl;

    for (pl = 0; pl < 3; pl++) {
        int chroma_shift = !!pl;
        int pad = frame_pad_h(pl);
        int start = (lcu_y * lcu_size) >> chroma_shift;
        int end   = ((lcu_y + 1) * lcu_size) >> chroma_shift;
        ptrdiff_t stride = f->stride[pl];
        int width  = (pl == 0) ? f->width  : (f->width  + 1) >> 1;
        int height = (pl == 0) ? f->height : (f->height + 1) >> 1;
        uint8_t *pix;
        int j;

        /* 重叠 4 行 (对应 davs2 SAO/ALF 边界处理) */
        if (lcu_y > 0) start -= 4;
        if (lcu_y < f->h_lcu - 1) end -= 4;

        /* 裁剪到有效范围 */
        if (start < 0) start = 0;
        if (end > height) end = height;

        /* 填充每行的左右 padding */
        for (j = start; j < end; j++) {
            pix = f->data[pl] + (ptrdiff_t)j * stride;
            pad_line_pixel(pix, width, pad, bps);
        }

        /* 第一行 LCU: 填充图像上方 padding 行 (复制第 0 行, 含左右 padding) */
        if (lcu_y == 0) {
            pix = f->data[pl] - pad * bps;  /* 第 0 行左 padding 起始 */
            for (j = 0; j < PAD_VERT; j++) {
                memcpy(pix - stride, pix, (size_t)stride);
                pix -= stride;
            }
        }

        /* 最后一行 LCU: 填充图像下方 padding 行 (复制最后行, 含左右 padding) */
        if (lcu_y == f->h_lcu - 1) {
            pix = f->data[pl] + (ptrdiff_t)(height - 1) * stride - pad * bps;
            for (j = 0; j < PAD_VERT; j++) {
                memcpy(pix + stride, pix, (size_t)stride);
                pix += stride;
            }
        }
    }
}
