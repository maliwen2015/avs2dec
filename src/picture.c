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
    memset(f, 0, sizeof(*f));
    const int bd = c->bit_depth;
    const int bps = (bd > 8) ? 2 : 1;
    f->bit_depth = bd;
    f->bytes_per_sample = bps;
    f->chroma_format = (int)c->seq->chroma_format;
    f->width = (int)c->seq->enc_width;
    f->height = (int)c->seq->enc_height;
    f->w8 = (f->width + 7) >> 3;
    f->h8 = (f->height + 7) >> 3;
    const int lcu = 1 << c->seq->log2_lcu_size;
    f->w_lcu = (f->width + lcu - 1) / lcu;
    f->h_lcu = (f->height + lcu - 1) / lcu;
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
        avs2_frame_free(f);
        return AVS2_ERR_NOMEM;
    }
    /* offset into the padded region: 跳过上 padding 行 + 左 padding 像素 */
    f->data[0] += PAD_VERT * sy + PAD_LUMA * bps;
    f->data[1] += PAD_VERT * sc + PAD_CHROMA * bps;
    f->data[2] += PAD_VERT * sc + PAD_CHROMA * bps;

    /* intra prediction mode buffer: 4x4 granularity, 1-block border padding */
    {
        int w_spu = (f->width + 3) >> 2;   /* width in 4x4 blocks */
        int h_spu = (f->height + 3) >> 2;  /* height in 4x4 blocks */
        f->ipredmode_stride = w_spu + 2;   /* 1 padding on each side */
        f->ipredmode_base = avs2_mem_allocz(sizeof(int8_t) *
                        (size_t)f->ipredmode_stride * (h_spu + 2));
        if (!f->ipredmode_base) {
            avs2_frame_free(f);
            return AVS2_ERR_NOMEM;
        }
        /* offset to first real position (skip top row + left column) */
        f->ipredmode = f->ipredmode_base + f->ipredmode_stride + 1;
        /* fill with DC_PRED (=0, already zeroed, but explicit for clarity) */
        memset(f->ipredmode_base, DC_PRED,
               sizeof(int8_t) * (size_t)f->ipredmode_stride * (h_spu + 2));
    }

    /* per-CU grid: one entry per 8x8 block */
    f->cu_grid = avs2_mem_allocz(sizeof(avs2_cu) * (size_t)f->w8 * f->h8);
    f->mvbuf = avs2_mem_allocz(sizeof(avs2_mv) * (size_t)f->w8 * f->h8 * 4);
    f->refbuf = avs2_mem_alloc((size_t)f->w8 * f->h8 * 4);
    if (f->refbuf) {
        /* davs2 在每帧开始时将 refbuf 初始化为 INVALID_REF (-1),
         * 使得 intra CU 位置的 refbuf 为 -1, 时域直接模式会回退到空间 MVP */
        memset(f->refbuf, INVALID_REF, (size_t)f->w8 * f->h8 * 4);
    }
    f->deblock_flags[0] = avs2_mem_allocz((size_t)f->w8 * f->h8);
    f->deblock_flags[1] = avs2_mem_allocz((size_t)f->w8 * f->h8);
    f->sao_params = avs2_mem_allocz(sizeof(avs2_sao_param) * (size_t)f->w_lcu * f->h_lcu * 3);
    f->alf_params = avs2_mem_allocz(sizeof(avs2_alf_param) * (size_t)f->w_lcu * f->h_lcu);
    /* intra border cache: 亮度 width, 色度 width/2 (各一行, 按字节寻址)
     * 8-bit 时每像素 1 字节, 10-bit 时每像素 2 字节 */
    f->intra_border[0] = avs2_mem_allocz((size_t)(w + 2 * PAD_LUMA) * bps);
    f->intra_border[1] = avs2_mem_allocz((size_t)(cw + 2 * PAD_CHROMA) * bps);
    f->intra_border[2] = avs2_mem_allocz((size_t)(cw + 2 * PAD_CHROMA) * bps);
    /* 行级 LF 完成跟踪数组 (2-pass 帧并行行级依赖) */
    f->lf_row_done = (volatile int *)avs2_mem_allocz(sizeof(int) * (size_t)f->h_lcu);
    f->lf_row_done_count = 0;
    f->aec_row_done = (volatile int *)avs2_mem_allocz(sizeof(int) * (size_t)f->h_lcu);
    if (!f->cu_grid || !f->mvbuf || !f->refbuf || !f->deblock_flags[0] || !f->deblock_flags[1] ||
        !f->sao_params || !f->alf_params ||
        !f->intra_border[0] || !f->intra_border[1] || !f->intra_border[2] ||
        !f->lf_row_done || !f->aec_row_done) {
        avs2_frame_free(f);
        return AVS2_ERR_NOMEM;
    }
    f->used = 1;
    f->ref_cnt = 1;
    return AVS2_OK;
}

void avs2_frame_free(avs2_frame *f)
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
    avs2_mem_free(f->ipredmode_base); f->ipredmode_base = NULL; f->ipredmode = NULL;
    avs2_mem_free(f->cu_grid);        f->cu_grid = NULL;
    avs2_mem_free(f->mvbuf);          f->mvbuf = NULL;
    avs2_mem_free(f->refbuf);         f->refbuf = NULL;
    avs2_mem_free(f->deblock_flags[0]); f->deblock_flags[0] = NULL;
    avs2_mem_free(f->deblock_flags[1]); f->deblock_flags[1] = NULL;
    avs2_mem_free(f->sao_params);     f->sao_params = NULL;
    avs2_mem_free(f->alf_params);     f->alf_params = NULL;
    avs2_mem_free(f->intra_border[0]); f->intra_border[0] = NULL;
    avs2_mem_free(f->intra_border[1]); f->intra_border[1] = NULL;
    avs2_mem_free(f->intra_border[2]); f->intra_border[2] = NULL;
    avs2_mem_free((void *)f->lf_row_done); f->lf_row_done = NULL;
    avs2_mem_free((void *)f->aec_row_done); f->aec_row_done = NULL;
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
    for (int i = 0; i < c->n_dpb; i++) {
        if (!c->dpb[i] || !c->dpb[i]->used)
            return c->dpb[i];
    }
    /* grow DPB */
    if (c->n_dpb < AVS2_MAX_FRAME_DELAY) {
        c->dpb[c->n_dpb] = avs2_mem_allocz(sizeof(avs2_frame));
        return c->dpb[c->n_dpb++];
    }
    /* DPB 满: 尝试释放已输出、不被参考、无用户引用的帧 */
    for (int i = 0; i < c->n_dpb; i++) {
        avs2_frame *f = c->dpb[i];
        if (f && f->used && !f->referenced && f->output && f->ref_cnt <= 1) {
            avs2_frame_free(f);
            return f;
        }
    }
    /* 最后手段: 释放不被参考且无用户引用的最老帧 (即使未输出) */
    {
        int oldest_coi = 0x7fffffff;
        avs2_frame *oldest = NULL;
        for (int i = 0; i < c->n_dpb; i++) {
            avs2_frame *f = c->dpb[i];
            if (f && f->used && !f->referenced && f->output && f->ref_cnt <= 1 && f->coi < oldest_coi) {
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
    for (int i = 0; i < c->n_dpb; i++) {
        if (c->dpb[i] && c->dpb[i]->used) {
            avs2_frame_free(c->dpb[i]);
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
