/*
 * header.c
 *
 * 头解析函数，从 davs2 的 header.cc 移植到 C。
 * 字段解析顺序与 davs2 完全一致。
 */

#include "internal.h"
#include "tables.h"

#include <string.h>
#include <assert.h>
#include <stdio.h>

/* =====================================================================
 * 常量定义
 * ===================================================================== */

#define DAVS2_MAX_FRAME_RATE_CODE 13
#define FRAME 1  /* 帧编码 (picture_structure 的值) */

/* 加权量化参数模式 */
#define WQ_UNDETAILED 0
#define WQ_DETAILED   1

/* 辅助宏 */
#define UNUSED_PARAMETER(x) (void)(x)
#define DAVS2_MIN(a, b)     ((a) < (b) ? (a) : (b))
#define DAVS2_CLIP3(lo, hi, x) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))

/* 默认 4x4 加权量化矩阵 (来自 davs2 quant.cc) */
static const int g_wqm_default_4x4[16] = {
    64, 64, 64, 68,
    64, 64, 68, 72,
    64, 68, 76, 80,
    72, 76, 84, 96
};

/* 默认 8x8 加权量化矩阵 (来自 davs2 quant.cc) */
static const int g_wqm_default_8x8[64] = {
    64,  64,  64,  64,  68,  68,  72,  76,
    64,  64,  64,  68,  72,  76,  84,  92,
    64,  64,  68,  72,  76,  80,  88,  100,
    64,  68,  72,  80,  84,  92,  100, 28,
    68,  72,  80,  84,  92,  104, 112, 128,
    76,  80,  84,  92,  104, 116, 132, 152,
    96,  100, 104, 116, 124, 140, 164, 188,
    104, 108, 116, 128, 152, 172, 192, 216
};

/* 获取默认加权量化矩阵 */
static const int *wq_get_default_matrix(int size_id)
{
    return (size_id == 0) ? g_wqm_default_4x4 : g_wqm_default_8x8;
}

/* =====================================================================
 * 局部辅助函数
 * ===================================================================== */

/* 检查 QP 是否有效 */
static int is_valid_qp(struct avs2_internal *c, int i_qp)
{
    return i_qp >= 0 && i_qp <= (63 + 8 * (c->bit_depth - 8));
}

/* -----------------------------------------------------------------------
 * RPS 解析 (refered_by_others, num_of_ref, ref_pic[], num_to_remove, remove_pic[], marker_bit)
 * 序列头和图像头共用此函数。
 */
static void parse_rps(avs2_bs *bs, avs2_rps *rps)
{
    uint32_t j;

    rps->refered_by_others = avs2_bs_get1(bs);
    rps->num_of_ref        = avs2_bs_get(bs, 3);

    for (j = 0; j < rps->num_of_ref; j++) {
        rps->delta_coi_of_ref_pic[j] = avs2_bs_get(bs, 6);
    }

    rps->num_of_removed_pic = avs2_bs_get(bs, 3);
    assert(rps->num_of_removed_pic <= 8);

    for (j = 0; j < rps->num_of_removed_pic; j++) {
        rps->delta_coi_of_removed_pic[j] = avs2_bs_get(bs, 6);
    }

    avs2_bs_get1(bs); /* marker_bit */
}

/* -----------------------------------------------------------------------
 * 加权量化参数解析 (图像级)
 * 对应 davs2 header.cc 中 pic_weight_quant_enable 为真时的内联解析。
 */
static void parse_wq_param(avs2_bs *bs, avs2_pic_header *p)
{
    p->pic_wq_data_index = avs2_bs_get(bs, 2);

    if (p->pic_wq_data_index == 1) {
        int i;

        avs2_bs_get1(bs); /* reserved_bits (mb_adapt_wq_disable) */

        p->wq_param = avs2_bs_get(bs, 2);
        p->wq_model = avs2_bs_get(bs, 2);

        if (p->wq_param == 1) {
            for (i = 0; i < 6; i++) {
                p->quant_param_undetail[i] =
                    (int16_t)avs2_bs_get_se(bs) + (int16_t)avs2_wq_param_default[WQ_UNDETAILED][i];
            }
        }

        if (p->wq_param == 2) {
            for (i = 0; i < 6; i++) {
                p->quant_param_detail[i] =
                    (int16_t)avs2_bs_get_se(bs) + (int16_t)avs2_wq_param_default[WQ_DETAILED][i];
            }
        }
    } else if (p->pic_wq_data_index == 2) {
        int x, y, size_id, ui_wq_m_size;
        int i;

        for (size_id = 0; size_id < 2; size_id++) {
            i = 0;
            ui_wq_m_size = DAVS2_MIN(1 << (size_id + 2), 8);

            for (y = 0; y < ui_wq_m_size; y++) {
                for (x = 0; x < ui_wq_m_size; x++) {
                    p->pic_user_wq_matrix[size_id][i++] = (int16_t)avs2_bs_get_ue(bs);
                }
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * ALF 参数解析 (对应 davs2 alf_read_param)
 * 读取顺序: alf_pic_flag_Y/Cb/Cr, 然后按 Y->U->V 顺序读取系数。
 */
static void parse_alf_param(struct avs2_internal *c, avs2_bs *bs)
{
    avs2_pic_header *p = c->pic;

    if (!c->seq->enable_alf) {
        p->alf_pic_flag_y  = 0;
        p->alf_pic_flag_cb = 0;
        p->alf_pic_flag_cr = 0;
        return;
    }

    p->alf_pic_flag_y  = avs2_bs_get1(bs);
    p->alf_pic_flag_cb = avs2_bs_get1(bs);
    p->alf_pic_flag_cr = avs2_bs_get1(bs);

    /* Y 分量: 先读 filter_number, 再读每组滤波器系数 */
    if (p->alf_pic_flag_y) {
        unsigned filter_number = avs2_bs_get_ue(bs);
        /* AVS2 规范中 filters_per_group <= 16, 超出视为码流错误, 防止 CPU 耗尽 DoS.
         * parse_alf_param 返回 void, 通过设置 bs->error 让调用方返回 AVS2_ERR_INVALID. */
        if (filter_number > 15 || bs->error) {
            bs->error = 1;
            return;
        }
        unsigned filters_per_group = filter_number + 1;
        unsigned f;

        for (f = 0; f < filters_per_group; f++) {
            if (f > 0) {
                if (filters_per_group != 16) {
                    avs2_bs_get_ue(bs); /* region_distance */
                }
            }
            {
                int i;
                for (i = 0; i < ALF_MAX_NUM_COEF; i++) {
                    avs2_bs_get_se(bs); /* luma ALF coefficients */
                }
            }
        }
    }

    /* U 分量: 9 个系数 */
    if (p->alf_pic_flag_cb) {
        int i;
        for (i = 0; i < ALF_MAX_NUM_COEF; i++) {
            avs2_bs_get_se(bs); /* chroma ALF coefficients */
        }
    }

    /* V 分量: 9 个系数 */
    if (p->alf_pic_flag_cr) {
        int i;
        for (i = 0; i < ALF_MAX_NUM_COEF; i++) {
            avs2_bs_get_se(bs); /* chroma ALF coefficients */
        }
    }
}

/* =====================================================================
 * 序列头解析 (B0)
 * 对应 davs2 parse_sequence_header
 * ===================================================================== */
int avs2_parse_sequence_header(struct avs2_internal *c, avs2_bs *bs)
{
    /* 使用栈上临时变量解析, 末尾比较后再决定是否写入 c->seq.
     * 多线程下同一序列的重复 sequence header 内容相同, 跳过写入可避免
     * memset(c->seq, 0, ...) 与 worker 读取 c->seq 的竞争, 无需等待 worker. */
    avs2_seq_header tmp_s;
    avs2_seq_header *s = &tmp_s;
    int i, j;
    int num_of_rps;
    unsigned bit_rate_lower, bit_rate_upper;

    avs2_bs_skip(bs, 32); /* 跳过起始码 */

    memset(s, 0, sizeof(*s));  /* 重置临时变量 */

    s->profile_id         = avs2_bs_get(bs, 8);
    s->level_id           = avs2_bs_get(bs, 8);
    s->progressive_sequence = avs2_bs_get1(bs);
    s->field_coded_sequence  = avs2_bs_get1(bs);

    s->horizontal_size = avs2_bs_get(bs, 14);
    s->vertical_size   = avs2_bs_get(bs, 14);

    if (s->horizontal_size < 16 || s->vertical_size < 16) {
        return AVS2_ERR_INVALID;
    }
    /* 上界检查: 防止恶意码流通过极大尺寸触发 OOM.
     * 实际上限由 h_lcu <= AVS2_MAX_H_LCU (256) 与 LCU <= 64 约束,
     * 即 16384 像素. 此处用 frame_size_limit (若用户设置) 或 16384 做硬上限. */
    {
        unsigned max_dim = c->frame_size_limit ? c->frame_size_limit : 16384u;
        if ((unsigned)s->horizontal_size > max_dim || (unsigned)s->vertical_size > max_dim) {
            avs2_error(c, "Frame size %dx%d exceeds limit %u\n",
                       s->horizontal_size, s->vertical_size, max_dim);
            return AVS2_ERR_INVALID;
        }
    }

    s->chroma_format = avs2_bs_get(bs, 2);

    if (s->chroma_format != AVS2_CHROMA_420 && s->chroma_format != AVS2_CHROMA_400) {
        return AVS2_ERR_INVALID;
    }
    if (s->chroma_format == AVS2_CHROMA_400) {
        avs2_warn(c, "Un-supported Chroma Format YUV400 as 0 for GB/T.\n");
    }

    /* 样值位深 */
    if (s->profile_id == AVS2_PROFILE_MAIN10) {
        s->sample_precision   = avs2_bs_get(bs, 3);
        s->encoding_precision = avs2_bs_get(bs, 3);
    } else {
        s->sample_precision   = avs2_bs_get(bs, 3);
        s->encoding_precision = 1;
    }

    if (s->sample_precision < 1 || s->sample_precision > 3 ||
        s->encoding_precision < 1 || s->encoding_precision > 3) {
        return AVS2_ERR_INVALID;
    }

    s->internal_bit_depth = 6 + (s->encoding_precision << 1);
    s->output_bit_depth   = 6 + (s->encoding_precision << 1);
    s->bytes_per_sample   = (s->output_bit_depth > 8) ? 2 : 1;

    /* 比特率相关信息 */
    s->aspect_ratio_information = avs2_bs_get(bs, 4);
    s->frame_rate_id            = avs2_bs_get(bs, 4);
    bit_rate_lower = avs2_bs_get(bs, 18);
    avs2_bs_get1(bs);  /* marker_bit */
    bit_rate_upper = avs2_bs_get(bs, 12);
    s->low_delay = avs2_bs_get1(bs);
    avs2_bs_get1(bs);  /* marker_bit */
    s->temporal_id_exist_flag = avs2_bs_get1(bs);
    avs2_bs_get(bs, 18);  /* bbv_buffer_size */

    s->log2_lcu_size = avs2_bs_get(bs, 3);

    if (s->log2_lcu_size < 4 || s->log2_lcu_size > 6) {
        avs2_error(c, "Invalid LCU size: %d\n", s->log2_lcu_size);
        return AVS2_ERR_INVALID;
    }

    /* 加权量化 */
    s->enable_weighted_quant = avs2_bs_get1(bs);

    if (s->enable_weighted_quant) {
        int load_seq_wquant_data_flag = (int)avs2_bs_get1(bs);
        int size_id, ui_wq_m_size;
        int x, y;

        for (size_id = 0; size_id < 2; size_id++) {
            ui_wq_m_size = DAVS2_MIN(1 << (size_id + 2), 8);

            if (load_seq_wquant_data_flag == 1) {
                for (y = 0; y < ui_wq_m_size; y++) {
                    for (x = 0; x < ui_wq_m_size; x++) {
                        s->seq_wq_matrix[size_id][y * ui_wq_m_size + x] =
                            (int16_t)avs2_bs_get_ue(bs);
                    }
                }
            } else if (load_seq_wquant_data_flag == 0) {
                const int *default_wqm = wq_get_default_matrix(size_id);
                for (i = 0; i < (ui_wq_m_size * ui_wq_m_size); i++) {
                    s->seq_wq_matrix[size_id][i] = (int16_t)default_wqm[i];
                }
            }
        }
    }

    /* 编码工具使能标志 */
    s->background_picture_disable = avs2_bs_get1(bs);  /* enable_background_picture = flag ^ 1 */
    s->enable_mhpskip    = avs2_bs_get1(bs);
    s->enable_dhp        = avs2_bs_get1(bs);
    s->enable_wsm        = avs2_bs_get1(bs);
    s->enable_amp        = avs2_bs_get1(bs);
    s->enable_nsqt       = avs2_bs_get1(bs);
    s->enable_sdip       = avs2_bs_get1(bs);
    s->enable_2nd_transform = avs2_bs_get1(bs);
    s->enable_sao        = avs2_bs_get1(bs);
    s->enable_alf        = avs2_bs_get1(bs);
    s->enable_pmvr       = avs2_bs_get1(bs);

    if (avs2_bs_get1(bs) != 1) {  /* marker_bit */
        avs2_error(c, "expected marker_bit 1 while received 0.\n");
    }

    /* RPS 表 */
    num_of_rps = (int)avs2_bs_get(bs, 6);
    if (num_of_rps > AVS2_MAX_RPS) {
        return AVS2_ERR_INVALID;
    }

    s->num_of_rps = num_of_rps;

    for (i = 0; i < num_of_rps; i++) {
        avs2_rps *rps = &c->sps_rps[i];
        memset(rps, 0, sizeof(*rps));
        parse_rps(bs, rps);
    }

    if (s->low_delay == 0) {
        s->picture_reorder_delay = avs2_bs_get(bs, 5);
    }

    s->cross_loop_filter_flag = avs2_bs_get1(bs);
    avs2_bs_get(bs, 2);  /* reserved_bits */

    avs2_bs_align(bs);  /* 字节对齐 */

    /* 派生值 */
    if (s->frame_rate_id < 1 || s->frame_rate_id > DAVS2_MAX_FRAME_RATE_CODE) {
        avs2_error(c, "Invalid frame_rate_code %d, valid range [1, %d].\n",
                   s->frame_rate_id, DAVS2_MAX_FRAME_RATE_CODE);
        s->frame_rate_id = DAVS2_CLIP3(1, DAVS2_MAX_FRAME_RATE_CODE, s->frame_rate_id);
    }

    s->bit_rate   = ((bit_rate_upper << 18) + bit_rate_lower) * 400;
    s->frame_rate = avs2_tab_frame_rate[s->frame_rate_id - 1];

    s->enc_width  = ((s->horizontal_size + MIN_CU_SIZE - 1) >> MIN_CU_SIZE_IN_BIT) << MIN_CU_SIZE_IN_BIT;
    s->enc_height = ((s->vertical_size   + MIN_CU_SIZE - 1) >> MIN_CU_SIZE_IN_BIT) << MIN_CU_SIZE_IN_BIT;

    /* force_8bit: 始终在原始位深空间解码, 仅在输出时降为 8-bit.
     * force_8bit 标志由输出路径 (yuv.c) 检查, 逐像素右移 >>2 输出. */

    /* 多线程优化: 比较新旧 sequence header.
     * 若内容相同 (同一序列的重复 sequence header), 跳过写入 c->seq/c->bit_depth/g_dc_value,
     * 避免 memset 竞争, 无需等待 worker.
     * 若内容不同 (序列切换), 需等待 worker 完成后再写入. */
    if (c->n_threads_active > 0 && c->seq->profile_id != 0 &&
        memcmp(s, c->seq, sizeof(*s)) == 0) {
        /* 内容相同, 不修改 c->seq */
    } else {
        /* 内容不同或首次: 多线程时等待 worker 完成后再修改,
         * 防止 worker 读取 c->seq/c->bit_depth/g_dc_value 时遇到中间状态 */
        if (c->n_threads_active > 0 && c->seq->profile_id != 0) {
            avs2_mutex_lock(&c->task_lock);
            while (c->n_pending > 0) {
                c->n_waiters_done++;
                avs2_cond_wait(&c->done_cond, &c->task_lock);
                c->n_waiters_done--;
            }
            avs2_mutex_unlock(&c->task_lock);
        }
        *c->seq = *s;
        c->bit_depth    = (int)s->internal_bit_depth;
        c->max_cu_size  = 1 << s->log2_lcu_size;
        c->min_cu_size  = MIN_CU_SIZE;

        /* 设置帧内预测 DC 默认填充值 (对应 davs2 header.cc:1062
         * g_dc_value = 1 << (g_bit_depth - 1)) */
        extern int g_dc_value;
        g_dc_value = 1 << (c->bit_depth - 1);
    }

    UNUSED_PARAMETER(j);

    /* 仅在首次或序列参数变化时打印日志 (避免重复序列头导致日志刷屏) */
    if (!c->seq_logged) {
        avs2_info(c, "Sequence: profile=%02x level=%02x %ux%u chroma=%u bd=%d lcu=%d\n",
                  s->profile_id, s->level_id, s->horizontal_size, s->vertical_size,
                  s->chroma_format, c->bit_depth, c->max_cu_size);
        avs2_info(c, "  tools: wqm=%d 2nd_t=%d nsqt=%d sdip=%d amp=%d mhpskip=%d wsm=%d dhp=%d sao=%d alf=%d pmvr=%d\n",
                  s->enable_weighted_quant, s->enable_2nd_transform, s->enable_nsqt, s->enable_sdip, s->enable_amp,
                  s->enable_mhpskip, s->enable_wsm, s->enable_dhp,
                  s->enable_sao, s->enable_alf, s->enable_pmvr);
        c->seq_logged = 1;
    }

    return bs->error ? AVS2_ERR_INVALID : AVS2_OK;
}

/* =====================================================================
 * 帧内图像头解析 (B3)
 * 对应 davs2 parse_picture_header_intra
 * ===================================================================== */
static int parse_picture_header_intra(struct avs2_internal *c, avs2_bs *bs)
{
    avs2_pic_header *p = c->pic;
    avs2_seq_header *s = c->seq;
    int time_code_flag;
    int progressive_frame;
    int predict;
    int i;

    p->picture_coding_type = AVS2_I_SLICE;

    avs2_bs_skip(bs, 32);  /* 跳过起始码 */

    avs2_bs_get(bs, 32);  /* bbv_delay */
    time_code_flag = (int)avs2_bs_get1(bs);

    if (time_code_flag) {
        avs2_bs_get(bs, 24);  /* time_code */
    }

    /* 背景图像 */
    if (!s->background_picture_disable) {  /* b_bkgnd_picture */
        int background_picture_flag = (int)avs2_bs_get1(bs);

        if (background_picture_flag) {
            int b_output = (int)avs2_bs_get1(bs);  /* background_picture_output_flag */
            if (b_output) {
                p->picture_coding_type = AVS2_G_SLICE;
            } else {
                p->picture_coding_type = AVS2_GB_SLICE;
            }
        }
    }

    p->coding_order = avs2_bs_get(bs, 8);

    if (s->temporal_id_exist_flag) {
        p->temporal_id = avs2_bs_get(bs, TEMPORAL_MAXLEVEL_BIT);
    }

    if (s->low_delay == 0) {
        p->picture_output_delay = avs2_bs_get_ue(bs);
        if (p->picture_output_delay >= 64) {
            avs2_error(c, "invalid picture output delay intra.\n");
            return AVS2_ERR_INVALID;
        }
    }

    /* RPS 预测 */
    predict = (int)avs2_bs_get1(bs);
    if (predict) {
        int index = (int)avs2_bs_get(bs, 5);
        if (index >= (int)s->num_of_rps) {
            avs2_error(c, "invalid rps index.\n");
            return AVS2_ERR_INVALID;
        }
        p->rps = c->sps_rps[index];
    } else {
        parse_rps(bs, &p->rps);
    }

    if (s->low_delay) {
        avs2_bs_get_ue(bs);  /* bbv_check_times */
    }

    progressive_frame = (int)avs2_bs_get1(bs);

    if (!progressive_frame) {
        p->picture_structure = avs2_bs_get1(bs);
    } else {
        p->picture_structure = FRAME;
    }

    p->top_field_first    = avs2_bs_get1(bs);
    p->repeat_first_field = avs2_bs_get1(bs);

    if (s->field_coded_sequence) {  /* b_field_coding */
        p->is_top_field = avs2_bs_get1(bs);
        avs2_bs_get1(bs);  /* reserved bit for interlace coding */
    }

    p->fixed_picture_qp = avs2_bs_get1(bs);
    p->picture_qp       = avs2_bs_get(bs, 7);

    /* 环路滤波 */
    p->loop_filter_disable = avs2_bs_get1(bs);  /* b_loop_filter = flag ^ 1 */

    if (!p->loop_filter_disable) {  /* b_loop_filter */
        int loop_filter_parameter_flag = (int)avs2_bs_get1(bs);

        if (loop_filter_parameter_flag) {
            p->alpha_offset = avs2_bs_get_se(bs);
            p->beta_offset  = avs2_bs_get_se(bs);
        } else {
            p->alpha_offset = 0;
            p->beta_offset  = 0;
        }
    }

    /* 色度量化参数 */
    p->chroma_quant_param_disable = avs2_bs_get1(bs);  /* enable_chroma_quant_param = !flag */
    if (!p->chroma_quant_param_disable) {
        p->chroma_quant_param_delta_cb = avs2_bs_get_se(bs);
        p->chroma_quant_param_delta_cr = avs2_bs_get_se(bs);
    } else {
        p->chroma_quant_param_delta_cb = 0;
        p->chroma_quant_param_delta_cr = 0;
    }

    /* 自适应频率加权量化
     * davs2 中此处先令 enable_weighted_quant = 0 再判断, 因此图像级加权量化
     * 解析实际不执行。为保持比特流消耗一致, 此处同样跳过。 */
    {
        int wq_enable = 0;  /* 对应 davs2: h->seq_info.enable_weighted_quant = 0; */
        if (wq_enable) {
            int pic_weight_quant_enable = (int)avs2_bs_get1(bs);
            if (pic_weight_quant_enable) {
                parse_wq_param(bs, p);
                p->pic_weight_quant_enable = 1;
            }
        }
    }

    /* ALF 参数 */
    parse_alf_param(c, bs);

    /* QP 校验 */
    {
        int i_qp = (int)p->picture_qp;
        if (!is_valid_qp(c, i_qp)) {
            avs2_error(c, "Invalid I Picture QP: %d\n", i_qp);
        }
    }

    avs2_bs_align(bs);  /* 字节对齐 */

    UNUSED_PARAMETER(i);
    return AVS2_OK;
}

/* =====================================================================
 * 帧间图像头解析 (B6)
 * 对应 davs2 parse_picture_header_inter
 * ===================================================================== */
static int parse_picture_header_inter(struct avs2_internal *c, avs2_bs *bs)
{
    avs2_pic_header *p = c->pic;
    avs2_seq_header *s = c->seq;
    int background_pred_flag;
    int progressive_frame;
    int predict;
    int pic_struct;
    int i;

    avs2_bs_skip(bs, 32);  /* 跳过起始码 */

    avs2_bs_get(bs, 32);  /* bbv_delay */

    /* picture_coding_type (2 bits): 1=P, 2=B, 3=G */
    pic_struct = (int)avs2_bs_get(bs, 2);

    /* 背景预测 */
    if (!s->background_picture_disable && (pic_struct == 1 || pic_struct == 3)) {
        if (pic_struct == 1) {
            background_pred_flag = (int)avs2_bs_get1(bs);
        } else {
            background_pred_flag = 0;
        }

        if (background_pred_flag == 0) {
            p->background_reference_enable = avs2_bs_get1(bs);
        } else {
            p->background_reference_enable = 0;
        }
    } else {
        background_pred_flag = 0;
        p->background_reference_enable = 0;
    }

    /* 确定帧类型 */
    if (pic_struct == 1 && background_pred_flag) {
        p->picture_coding_type = AVS2_S_SLICE;
    } else if (pic_struct == 1) {
        p->picture_coding_type = AVS2_P_SLICE;
    } else if (pic_struct == 3) {
        p->picture_coding_type = AVS2_F_SLICE;
    } else {
        p->picture_coding_type = AVS2_B_SLICE;
    }

    p->coding_order = avs2_bs_get(bs, 8);

    if (s->temporal_id_exist_flag) {
        p->temporal_id = avs2_bs_get(bs, TEMPORAL_MAXLEVEL_BIT);
    }

    if (s->low_delay == 0) {
        p->picture_output_delay = avs2_bs_get_ue(bs);
        if (p->picture_output_delay >= 64) {
            avs2_error(c, "invalid picture output delay inter.\n");
            return AVS2_ERR_INVALID;
        }
    }

    /* RPS 预测 */
    predict = (int)avs2_bs_get1(bs);
    if (predict) {
        int index = (int)avs2_bs_get(bs, 5);
        if (index >= (int)s->num_of_rps) {
            avs2_error(c, "invalid rps index.\n");
            return AVS2_ERR_INVALID;
        }
        p->rps = c->sps_rps[index];
    } else {
        parse_rps(bs, &p->rps);
    }

    if (s->low_delay) {
        avs2_bs_get_ue(bs);  /* bbv_check_times */
    }

    progressive_frame = (int)avs2_bs_get1(bs);

    if (!progressive_frame) {
        p->picture_structure = avs2_bs_get1(bs);
    } else {
        p->picture_structure = FRAME;
    }

    p->top_field_first    = avs2_bs_get1(bs);
    p->repeat_first_field = avs2_bs_get1(bs);

    if (s->field_coded_sequence) {  /* b_field_coding */
        p->is_top_field = avs2_bs_get1(bs);
        avs2_bs_get1(bs);  /* reserved bit for interlace coding */
    }

    p->fixed_picture_qp = avs2_bs_get1(bs);
    p->picture_qp       = avs2_bs_get(bs, 7);

    /* reserved_bit: 仅当 B 帧且帧编码时不读 */
    if (!(pic_struct == 2 && p->picture_structure == FRAME)) {
        avs2_bs_get1(bs);  /* reserved_bit */
    }

    p->random_access_decodable_flag = avs2_bs_get1(bs);

    /* 环路滤波 */
    p->loop_filter_disable = avs2_bs_get1(bs);  /* b_loop_filter = flag ^ 1 */

    if (!p->loop_filter_disable) {  /* b_loop_filter */
        int loop_filter_parameter_flag = (int)avs2_bs_get1(bs);

        if (loop_filter_parameter_flag) {
            p->alpha_offset = avs2_bs_get_se(bs);
            p->beta_offset  = avs2_bs_get_se(bs);
        } else {
            p->alpha_offset = 0;
            p->beta_offset  = 0;
        }
    }

    /* 色度量化参数 */
    p->chroma_quant_param_disable = avs2_bs_get1(bs);  /* enable_chroma_quant_param = !flag */
    if (!p->chroma_quant_param_disable) {
        p->chroma_quant_param_delta_cb = avs2_bs_get_se(bs);
        p->chroma_quant_param_delta_cr = avs2_bs_get_se(bs);
    } else {
        p->chroma_quant_param_delta_cb = 0;
        p->chroma_quant_param_delta_cr = 0;
    }

    /* 自适应频率加权量化
     * davs2 中此处先令 enable_weighted_quant = 0 再判断, 因此图像级加权量化
     * 解析实际不执行。为保持比特流消耗一致, 此处同样跳过。 */
    {
        int wq_enable = 0;  /* 对应 davs2: h->seq_info.enable_weighted_quant = 0; */
        if (wq_enable) {
            int pic_weight_quant_enable = (int)avs2_bs_get1(bs);
            if (pic_weight_quant_enable) {
                parse_wq_param(bs, p);
                p->pic_weight_quant_enable = 1;
            }
        }
    }

    /* ALF 参数 */
    parse_alf_param(c, bs);

    /* QP 校验 */
    {
        int i_qp = (int)p->picture_qp;
        if (!is_valid_qp(c, i_qp)) {
            avs2_error(c, "Invalid PB Picture QP: %d\n", i_qp);
        }
    }

    avs2_bs_align(bs);  /* 字节对齐 */

    UNUSED_PARAMETER(i);
    return AVS2_OK;
}

/* =====================================================================
 * 图像头解析 (分发 + POC 推导)
 * 对应 davs2 parse_picture_header
 * ===================================================================== */
int avs2_parse_picture_header(struct avs2_internal *c, avs2_bs *bs, int start_code)
{
    avs2_pic_header *p = c->pic;
    avs2_seq_header *s = c->seq;
    int ret;

    memset(p, 0, sizeof(*p));

    assert(start_code == AVS2_SC_INTRA_PICTURE || start_code == AVS2_SC_INTER_PICTURE);

    if (start_code == AVS2_SC_INTRA_PICTURE) {
        ret = parse_picture_header_intra(c, bs);
        if (ret < 0) {
            return ret;
        }
    } else {
        ret = parse_picture_header_inter(c, bs);
        if (ret < 0) {
            return ret;
        }
    }

    /* 场图像不支持 */
    if (p->picture_structure != FRAME) {
        avs2_error(c, "field is not supported.\n");
        return AVS2_ERR_INVALID;
    }

    /* COI 回绕扩展 (对应 davs2 header.cc:726-739)
     * coding_order 是 8 位值 (0~255), 需要扩展为单调递增值.
     * 当当前 COI 小于上一个 COI 时, 说明发生了回绕, 增加回绕计数. */
    if (c->i_prev_coi >= 0 && (int)p->coding_order < c->i_prev_coi) {
        c->i_tr_wrap_cnt++;
    }
    c->i_prev_coi = (int)p->coding_order;
    p->coding_order += (uint32_t)(c->i_tr_wrap_cnt * AVS2_COI_CYCLE);

    /* POC 推导 (对应 davs2 header.cc:741-745) */
    if (s->low_delay == 0) {
        p->poc = (int)p->coding_order + (int)p->picture_output_delay - (int)s->picture_reorder_delay;
    } else {
        p->poc = (int)p->coding_order;
    }

    /* 序列首帧 (I 帧): 设定输出 POC 起点 (对应 davs2 header.cc:749-755) */
    if (!c->out_initialized && start_code == AVS2_SC_INTRA_PICTURE) {
        c->out_next_poc = p->poc;
        c->out_initialized = 1;
    }

    return bs->error ? AVS2_ERR_INVALID : AVS2_OK;
}

/* =====================================================================
 * 条带头解析
 * 对应 davs2 parse_slice_header
 * ===================================================================== */
int avs2_parse_slice_header(struct avs2_internal *c, avs2_bs *bs, int start_code,
                            avs2_frame_ctx *fc)
{
    /* 使用 fc->pic_local 而非 c->pic: worker 线程中 c->pic 可能已被主线程覆盖.
     * 主线程路径下 fc->pic_local 在 process_start_code 中已从 c->pic 复制. */
    avs2_pic_header *p = &fc->pic_local;
    avs2_seq_header *s = c->seq;
    const int lcu_size = 1 << s->log2_lcu_size;

    int slice_vertical_position;
    int slice_vertical_position_extension = 0;
    int slice_horizontal_position;
    int slice_horizontal_position_extension;
    int mb_row;

    UNUSED_PARAMETER(start_code);

    avs2_bs_skip(bs, 24);  /* 跳过起始码 00 00 01 */

    slice_vertical_position = (int)avs2_bs_get(bs, 8);

    if ((int)s->enc_height > (144 * lcu_size)) {
        slice_vertical_position_extension = (int)avs2_bs_get(bs, 3);
    }

    if ((int)s->enc_height > (144 * lcu_size)) {
        mb_row = (slice_vertical_position_extension << 7) + slice_vertical_position;
    } else {
        mb_row = slice_vertical_position;
    }

    slice_horizontal_position = (int)avs2_bs_get(bs, 8);
    if ((int)s->enc_width > (255 * lcu_size)) {
        slice_horizontal_position_extension = (int)avs2_bs_get(bs, 2);
        UNUSED_PARAMETER(slice_horizontal_position_extension);
    }

    /* QP */
    if (!p->fixed_picture_qp) {
        int fixed_slice_qp = (int)avs2_bs_get1(bs);
        int slice_qp = (int)avs2_bs_get(bs, 7);

        fc->slice_qp = slice_qp;
        fc->b_dqp    = !fixed_slice_qp;  /* 对应 davs2 h->b_DQP = !h->b_fixed_slice_qp */
    } else {
        fc->slice_qp = (int)p->picture_qp;
        fc->b_dqp    = 0;  /* fixed_picture_qp 时无 DQP */
    }

    /* 色度量化参数偏移 (对应 davs2 h->chroma_quant_param_delta_u/v) */
    fc->chroma_quant_param_delta_cb = p->chroma_quant_param_delta_cb;
    fc->chroma_quant_param_delta_cr = p->chroma_quant_param_delta_cr;

    /* QP 校验: 越界 QP 会流入量化路径用作表索引, 必须拒绝而非仅警告 */
    {
        int i_qp = fc->slice_qp;
        if (!is_valid_qp(c, i_qp)) {
            avs2_error(c, "Invalid Slice QP: %d\n", i_qp);
            return AVS2_ERR_INVALID;
        }
    }

    /* SAO 条带级标志 */
    if (s->enable_sao) {
        fc->slice_sao_on[0] = (uint8_t)avs2_bs_get1(bs);  /* sao_slice_flag_Y */
        fc->slice_sao_on[1] = (uint8_t)avs2_bs_get1(bs);  /* sao_slice_flag_Cb */
        fc->slice_sao_on[2] = (uint8_t)avs2_bs_get1(bs);  /* sao_slice_flag_Cr */
    } else {
        fc->slice_sao_on[0] = 0;
        fc->slice_sao_on[1] = 0;
        fc->slice_sao_on[2] = 0;
    }

    /* 保存条带位置 */
    fc->lcu_x = slice_horizontal_position;
    fc->lcu_y = mb_row;

    /* 校验条带起始位置在帧范围内, 防止 lcu_y 越界用作行数组索引 */
    {
        int h_lcu = ((int)s->enc_height + lcu_size - 1) >> s->log2_lcu_size;
        int w_lcu = ((int)s->enc_width  + lcu_size - 1) >> s->log2_lcu_size;
        if (fc->lcu_y < 0 || fc->lcu_y >= h_lcu ||
            fc->lcu_x < 0 || fc->lcu_x >= w_lcu) {
            avs2_error(c, "Slice position (%d,%d) out of range (h_lcu=%d w_lcu=%d)\n",
                       fc->lcu_x, fc->lcu_y, w_lcu, h_lcu);
            return AVS2_ERR_INVALID;
        }
    }

    return bs->error ? AVS2_ERR_INVALID : AVS2_OK;
}

/* =====================================================================
 * 参考帧列表构建
 * 对应 davs2 task_get_references 中的参考帧查找逻辑
 * ===================================================================== */
void avs2_build_reference_list(struct avs2_internal *c, avs2_frame_ctx *fc)
{
    avs2_pic_header *p = c->pic;
    avs2_rps *rps = &p->rps;
    int i, j;

    /* 根据 RPS 的 remove_pic 列表, 标记不再需要的参考帧 (对应 davs2 header.cc
     * 中 update_dpb 逻辑). p->coding_order 已是回绕扩展后的单调值, 无需掩码. */
    for (i = 0; i < (int)rps->num_of_removed_pic; i++) {
        int remove_coi = (int)p->coding_order - (int)rps->delta_coi_of_removed_pic[i];

        for (j = 0; j < c->n_dpb; j++) {
            avs2_frame *f = c->dpb[j];
            if (f && f->used && f->coi == remove_coi) {
                f->referenced = 0;  /* 不再被后续帧参考 */
                /* 不在此处调用 avs2_frame_free: 当前帧可能仍在 ref_pic 列表中,
                 * ref_cnt++ 尚未执行 (在 avs2_submit_frame_task 中).
                 * 提前释放会使 avs2_dpb_get_free 的 !used 快速路径跳过 ref_cnt
                 * 检查, 导致帧在 worker 仍读取 mvbuf/refbuf 时被复用 memset.
                 * 释放交给 avs2_dpb_get_free 统一在 ref_cnt 安全时执行. */
                break;
            }
        }
    }

    fc->n_refs = 0;
    memset(fc->fref, 0, sizeof(fc->fref));

    /* 参考帧查找: p->coding_order 已是回绕扩展后的单调值 (对应 davs2 header.cc:1294) */
    for (i = 0; i < (int)rps->num_of_ref && i < AVS2_MAX_REFS; i++) {
        int target_coi = (int)p->coding_order - (int)rps->delta_coi_of_ref_pic[i];
        for (j = 0; j < c->n_dpb; j++) {
            avs2_frame *f = c->dpb[j];
            if (f && f->used && f->coi == target_coi) {
                fc->fref[fc->n_refs] = f;
                fc->n_refs++;
                break;
            }
        }
    }

    /* 一个帧可能同时在 remove_pic 和 ref_pic 列表中 (AVS2 常见场景):
     * 上面 remove_pic 循环将 referenced 置 0, 但此帧仍被当前帧参考.
     * 重新标记所有 fref 中的帧为 referenced=1, 防止 avs2_picture_unref
     * 在 ref_cnt 递增前释放它们. */
    for (i = 0; i < fc->n_refs; i++) {
        if (fc->fref[i]) {
            fc->fref[i]->referenced = 1;
        }
    }

    /* 计算距离索引 (对应 davs2 header.cc:962-981)
     * 用于时域直接模式的 MV 缩放. */
    {
        avs2_frame *fdec = fc->fdec;
        int cur_poc = p->poc;
        int frame_type = p->picture_coding_type;

        for (i = 0; i < AVS2_MAX_REFS; i++) {
            /* davs2 初始化为 -1, 使得未填充的槽位不会被误用 */
            fdec->dist_refs[i] = -1;
            fdec->dist_scale_refs[i] = -1;
        }

        if (frame_type == AVS2_B_SLICE) {
            /* B 帧: fref[0]=后向(B_BWD), fref[1]=前向(B_FWD) */
            if (fc->n_refs > 0 && fc->fref[B_BWD]) {
                int dist = 2 * (fc->fref[B_BWD]->poc - cur_poc);
                fdec->dist_refs[B_BWD] = AVS2_DISTANCE_INDEX(dist);
                if (fdec->dist_refs[B_BWD] > 0) {
                    fdec->dist_scale_refs[B_BWD] = MULTI / fdec->dist_refs[B_BWD];
                }
            }
            if (fc->n_refs > 1 && fc->fref[B_FWD]) {
                int dist = 2 * (cur_poc - fc->fref[B_FWD]->poc);
                fdec->dist_refs[B_FWD] = AVS2_DISTANCE_INDEX(dist);
                if (fdec->dist_refs[B_FWD] > 0) {
                    fdec->dist_scale_refs[B_FWD] = MULTI / fdec->dist_refs[B_FWD];
                }
            }
        } else {
            /* P/F 帧: 所有参考都是前向 */
            for (i = 0; i < fc->n_refs && i < AVS2_MAX_REFS; i++) {
                if (fc->fref[i]) {
                    int dist = 2 * (cur_poc - fc->fref[i]->poc);
                    fdec->dist_refs[i] = AVS2_DISTANCE_INDEX(dist);
                    if (fdec->dist_refs[i] > 0) {
                        fdec->dist_scale_refs[i] = MULTI / fdec->dist_refs[i];
                    }
                }
            }
        }
    }
}
