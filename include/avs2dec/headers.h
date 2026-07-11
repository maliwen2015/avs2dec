/*
 * avs2dec - high-performance AVS2 (GB/T 33475.2) video decoder
 *
 * Sequence/picture header descriptors exposed to the user
 */

#ifndef AVS2DEC_HEADERS_H
#define AVS2DEC_HEADERS_H

#include <stdint.h>

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AVS2_MAX_REFS 4
#define AVS2_MAX_RPS  32

typedef struct avs2_seq_header {
    uint32_t profile_id;
    uint32_t level_id;
    uint32_t progressive_sequence;
    uint32_t field_coded_sequence;

    uint32_t horizontal_size;
    uint32_t vertical_size;
    uint32_t chroma_format;
    uint32_t sample_precision;
    uint32_t encoding_precision;
    uint32_t internal_bit_depth;
    uint32_t output_bit_depth;
    uint32_t bytes_per_sample;

    uint32_t aspect_ratio_information;
    uint32_t frame_rate_id;
    uint32_t bit_rate;
    uint32_t low_delay;
    uint32_t temporal_id_exist_flag;
    uint32_t bbv_buffer_size;
    uint32_t log2_lcu_size;

    uint32_t enable_weighted_quant;
    uint32_t background_picture_disable;
    uint32_t enable_mhpskip;
    uint32_t enable_dhp;
    uint32_t enable_wsm;
    uint32_t enable_amp;
    uint32_t enable_nsqt;
    uint32_t enable_sdip;
    uint32_t enable_2nd_transform;
    uint32_t enable_sao;
    uint32_t enable_alf;
    uint32_t enable_pmvr;

    uint32_t num_of_rps;
    uint32_t picture_reorder_delay;
    uint32_t cross_loop_filter_flag;

    /* 序列级加权量化矩阵 [sizeId][coef] */
    int16_t seq_wq_matrix[2][64];

    /* derived */
    uint32_t enc_width;
    uint32_t enc_height;
    float frame_rate;
} avs2_seq_header;

typedef struct avs2_rps {
    uint32_t refered_by_others;
    uint32_t num_of_ref;
    uint32_t delta_coi_of_ref_pic[AVS2_MAX_REFS];
    uint32_t num_of_removed_pic;
    uint32_t delta_coi_of_removed_pic[8];
} avs2_rps;

typedef struct avs2_pic_header {
    uint32_t picture_coding_type; /* I/P/B/G/F/S */
    uint32_t bbv_delay;
    uint32_t time_code_flag;
    uint32_t time_code;
    uint32_t background_picture_flag;
    uint32_t background_picture_output_flag;
    uint32_t background_reference_enable;
    uint32_t coding_order;        /* COI */
    uint32_t temporal_id;
    uint32_t picture_output_delay;
    uint32_t use_rps_in_sps;
    uint32_t rps_index;
    avs2_rps rps;
    uint32_t progressive_frame;
    uint32_t picture_structure;
    uint32_t top_field_first;
    uint32_t repeat_first_field;
    uint32_t is_top_field;
    uint32_t fixed_picture_qp;
    uint32_t picture_qp;
    uint32_t loop_filter_disable;
    uint32_t loop_filter_parameter_flag;
    int32_t alpha_offset;
    int32_t beta_offset;
    uint32_t chroma_quant_param_disable;
    int32_t chroma_quant_param_delta_cb;
    int32_t chroma_quant_param_delta_cr;
    uint32_t random_access_decodable_flag;
    uint32_t pic_weight_quant_enable;

    /* 加权量化参数 */
    uint32_t pic_wq_data_index;
    uint32_t wq_param;
    uint32_t wq_model;
    int16_t quant_param_undetail[6];
    int16_t quant_param_detail[6];
    int16_t pic_user_wq_matrix[2][64];

    /* ALF picture-level flags */
    uint32_t alf_pic_flag_y;
    uint32_t alf_pic_flag_cb;
    uint32_t alf_pic_flag_cr;

    /* derived */
    int32_t poc;
} avs2_pic_header;

#ifdef __cplusplus
}
#endif

#endif /* AVS2DEC_HEADERS_H */
