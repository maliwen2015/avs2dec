/*
 * avs2dec - high-performance AVS2 (GB/T 33475.2) video decoder
 *
 * Public library API (dav1d-style push/pull interface)
 *
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

#include "common.h"
#include "data.h"
#include "headers.h"
#include "picture.h"
#include "version.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AVS2_MAX_THREADS 256
#define AVS2_MAX_FRAME_DELAY 256

/* 线程模式: 帧级并行或行级并行(块并行) */
enum avs2_thread_mode {
    AVS2_THREAD_FRAME = 0,  /* 帧级并行 (默认): 多帧并行解码 */
    AVS2_THREAD_ROW   = 1,  /* 行级并行 (块并行): 2-pass, AEC串行+重建并行 */
};

typedef struct avs2_ctx avs2_ctx;
typedef struct avs2_ref avs2_ref;

typedef struct avs2_settings {
    int n_threads;       /* 0 = auto (logical cores) */
    int max_frame_delay; /* 1 = low-latency; 0 = auto */
    int log_level;       /* avs2_log_level_e */
    unsigned frame_size_limit; /* 0 = 默认上限 16384; 否则为单边最大像素数 */
    avs2_picture_alloc allocator; /* 自定义帧分配器 (当前未实现, 保留) */
    avs2_logger logger;
    int strict_std_compliance;
    int skip_loop_filter;   /* 1 = skip all in-loop filters */
    int thread_mode;        /* avs2_thread_mode: 0=frame, 1=row */
    int force_8bit;         /* 1 = 强制 8-bit 解码 (有损, 提升性能) */
    uint8_t reserved[24];
} avs2_settings;

/* Initialize settings to defaults. */
AVS2DEC_API void avs2_default_settings(avs2_settings *s);

/* Library version string. */
AVS2DEC_API const char *avs2_version(void);

/* API version (0x00MMmmpp). */
AVS2DEC_API unsigned avs2_version_api(void);

/* Open a decoder. Returns NULL on failure. */
AVS2DEC_API avs2_ctx *avs2_open(const avs2_settings *s);

/* Close and free the decoder. Sets ctx to NULL. */
AVS2DEC_API void avs2_close(avs2_ctx **ctx);

/*
 * Push compressed AVS2 bitstream data (Annex B start-code format).
 * Returns AVS2_OK on success (可能已解码若干帧, 调用 avs2_get_picture 取走),
 * AVS2_ERR_NOMEM 表示 DPB 满 (调用 avs2_get_picture 输出帧后重试),
 * AVS2_ERR_INVALID 表示码流错误或参数无效.
 * 传入 data==NULL 表示 flush 信号 (排空解码器).
 * 数据在返回前被完全消费 (调用者可立即释放), 但 NOMEM 时当前帧未消费.
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
