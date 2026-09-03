/*
 * avs2dec - high-performance AVS2 (GB/T 33475.2) video decoder
 *
 * Single public library API header (dav1d-style push/pull interface).
 * Architecture inspired by dav1d. Algorithm reference: davs2 and
 * GB/T 33475.2-2024 (AVS2).
 *
 * Usage:
 *   avs2_settings s; avs2_default_settings(&s);
 *   avs2_ctx *c = avs2_open(&s);
 *   ...
 *   avs2_send_data(c, &data);        // push compressed data
 *   avs2_get_picture(c, &pic, &seq); // pull decoded picture
 *   ...
 *   avs2_close(&c);
 */

#ifndef AVS2DEC_AVS2DEC_H
#define AVS2DEC_AVS2DEC_H

#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =====================================================================
 * Build export macros
 * ===================================================================== */
#ifdef AVS2DEC_STATIC
#  define AVS2DEC_API
#elif defined(AVS2DEC_BUILD_EXPORTS)
#  ifdef _WIN32
#    define AVS2DEC_API __declspec(dllexport)
#  elif defined(__GNUC__) && __GNUC__ >= 4
#    define AVS2DEC_API __attribute__((visibility("default")))
#  else
#    define AVS2DEC_API
#  endif
#else
#  ifdef _WIN32
#    define AVS2DEC_API __declspec(dllimport)
#  else
#    define AVS2DEC_API
#  endif
#endif

/* =====================================================================
 * Version
 * ===================================================================== */
#define AVS2DEC_VERSION_MAJOR 1
#define AVS2DEC_VERSION_MINOR 1
#define AVS2DEC_VERSION_PATCH 0

#define AVS2DEC_API_MAJOR 1
#define AVS2DEC_API_MINOR 1
#define AVS2DEC_API_PATCH 0

#define AVS2DEC_VERSION_STR "1.1.0"

/* API version (0x00MMmmpp). */
#define AVS2DEC_API_VERSION                                                 \
    ((AVS2DEC_API_MAJOR << 16) | (AVS2DEC_API_MINOR << 8) | AVS2DEC_API_PATCH)

/* =====================================================================
 * Common enums & logger
 * ===================================================================== */

/* Log levels */
enum avs2_log_level_e {
    AVS2_LOG_ERROR   = 0,
    AVS2_LOG_WARNING = 1,
    AVS2_LOG_INFO    = 2,
    AVS2_LOG_DEBUG   = 3,
};

/* Picture types (AVS2) */
enum avs2_picture_type_e {
    AVS2_PIC_I = 0, /* Intra */
    AVS2_PIC_P = 1, /* Predictive */
    AVS2_PIC_B = 2, /* Bi-predictive */
    AVS2_PIC_G = 3, /* GOP background (GB) */
    AVS2_PIC_F = 4, /* F-frame (field-based forward) */
    AVS2_PIC_S = 5, /* S-frame (scene background) */
};

/* Chroma format */
enum avs2_chroma_format_e {
    AVS2_CHROMA_400 = 0,
    AVS2_CHROMA_420 = 1,
    AVS2_CHROMA_422 = 2,
    AVS2_CHROMA_444 = 3,
};

/* Error codes */
enum avs2_error_e {
    AVS2_OK            =  0,
    AVS2_ERR_AGAIN     = -1, /* need more data */
    AVS2_ERR_EOF       = -2, /* end of stream */
    AVS2_ERR_INVALID   = -3, /* invalid bitstream */
    AVS2_ERR_NOMEM     = -4, /* out of memory */
    AVS2_ERR_UNSUPPORTED = -5,
};

/* avs2_logger callback uses va_list; implement with vfprintf/vsnprintf. */
typedef struct avs2_logger {
    void *cookie;
    void (*callback)(void *cookie, int level, const char *fmt, va_list ap);
} avs2_logger;

/* =====================================================================
 * Sequence/picture headers
 * ===================================================================== */
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

/* =====================================================================
 * Picture descriptor
 * ===================================================================== */

/* Picture structure (1=frame, 2=top field, 3=bottom field) */
enum avs2_picture_structure_e {
    AVS2_PIC_STRUCTURE_FRAME       = 0,
    AVS2_PIC_STRUCTURE_TOP_FIELD   = 1,
    AVS2_PIC_STRUCTURE_BOT_FIELD   = 2,
};

typedef struct avs2_picture {
    /* picture data */
    uint8_t *data[3];       /* per-plane pixel data */
    ptrdiff_t stride[3];    /* per-plane stride in bytes */
    int width[3];           /* per-plane width */
    int height[3];          /* per-plane height */
    int p_w, p_h;           /* luma width/height */

    /* picture properties */
    int type;               /* avs2_picture_type_e */
    int poc;                /* picture order count */
    int qp;                 /* picture QP */
    int bit_depth;          /* sample bit depth */
    int bytes_per_sample;   /* 1 or 2 */
    int chroma_format;      /* avs2_chroma_format_e */
    int structure;          /* avs2_picture_structure_e */

    /* timestamps */
    int64_t pts;
    int64_t dts;

    void *dec_frame;        /* opaque decoder frame reference */
} avs2_picture;

/* =====================================================================
 * Input data wrapper
 * ===================================================================== */
typedef struct avs2_data {
    const uint8_t *data; /* pointer to bitstream data */
    size_t sz;           /* size in bytes */
    int64_t pts;         /* presentation timestamp */
    int64_t dts;         /* decoding timestamp */
} avs2_data;

/*
 * Wrap user-provided data. The buffer is copied into the decoder's internal
 * buffer during avs2_send_data(), so the caller may free or reuse it as soon
 * as the call returns (regardless of the return value).
 */
AVS2DEC_API void avs2_data_wrap(avs2_data *data, const uint8_t *buf,
                                size_t sz, int64_t pts, int64_t dts);

/* =====================================================================
 * Decoder context & settings
 * ===================================================================== */

#define AVS2_MAX_THREADS 256
#define AVS2_MAX_FRAME_DELAY 256

/* 线程模式: 帧级并行或行级并行(块并行) */
enum avs2_thread_mode {
    AVS2_THREAD_FRAME = 0,  /* 帧级并行 (默认): 多帧并行解码 */
    AVS2_THREAD_ROW   = 1,  /* 行级并行 (块并行): 2-pass, AEC串行+重建并行 */
};

typedef struct avs2_ctx avs2_ctx;

typedef struct avs2_settings {
    int n_threads;       /* 0 = auto (logical cores) */
    int max_frame_delay; /* 1 = low-latency; 0 = auto */
    int log_level;       /* avs2_log_level_e */
    unsigned frame_size_limit; /* 0 = 默认上限 16384; 否则为单边最大像素数 */
    avs2_logger logger;
    int skip_loop_filter;   /* 1 = skip all in-loop filters */
    int thread_mode;        /* avs2_thread_mode: 0=frame, 1=row */
} avs2_settings;

/* Initialize settings to defaults. */
AVS2DEC_API void avs2_default_settings(avs2_settings *s);

/* Library version string. */
AVS2DEC_API const char *avs2_version(void);

/* Open a decoder. Returns NULL on failure. */
AVS2DEC_API avs2_ctx *avs2_open(const avs2_settings *s);

/* Close and free the decoder. Sets ctx to NULL. */
AVS2DEC_API void avs2_close(avs2_ctx **ctx);

/*
 * Push compressed AVS2 bitstream data (Annex B start-code format).
 * Returns AVS2_OK on success (可能已解码若干帧, 调用 avs2_get_picture 取走),
 * AVS2_ERR_NOMEM 表示 DPB 满 (解码暂停, 数据已被缓冲, DPB 释放后自动继续;
 * 调用方应继续调用 avs2_get_picture 排空输出, 不要重发同一数据),
 * AVS2_ERR_INVALID 表示码流错误或参数无效.
 * 传入 data==NULL 表示 flush 信号 (排空解码器).
 * 数据在返回前已被复制进内部缓冲 (调用者可立即释放, 无论返回值如何).
 */
AVS2DEC_API int avs2_send_data(avs2_ctx *ctx, avs2_data *data);

/*
 * Retrieve a decoded picture (pull interface). On success, *pic is
 * filled and the caller must release it with avs2_picture_unref().
 * Returns AVS2_OK if a picture is available, AVS2_ERR_AGAIN if more
 * input is needed, AVS2_ERR_EOF at end of stream.
 */
AVS2DEC_API int avs2_get_picture(avs2_ctx *ctx, avs2_picture *pic,
                                 avs2_seq_header *seq);

/* Flush the decoder (e.g. on seek). */
AVS2DEC_API void avs2_flush(avs2_ctx *ctx);

/* Release a picture obtained from avs2_get_picture(). */
AVS2DEC_API void avs2_picture_unref(avs2_ctx *ctx, avs2_picture *pic);

#ifdef __cplusplus
}
#endif

#endif /* AVS2DEC_AVS2DEC_H */