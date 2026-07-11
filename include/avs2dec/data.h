/*
 * avs2dec - high-performance AVS2 (GB/T 33475.2) video decoder
 *
 * Input data wrapper
 */

#ifndef AVS2DEC_DATA_H
#define AVS2DEC_DATA_H

#include <stddef.h>
#include <stdint.h>

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct avs2_data {
    const uint8_t *data; /* pointer to bitstream data */
    size_t sz;           /* size in bytes */
    int64_t pts;         /* presentation timestamp */
    int64_t dts;         /* decoding timestamp */
    void *ref;           /* optional reference-tracking cookie */
    void (*free_cb)(const uint8_t *data, void *ref);
} avs2_data;

/*
 * Wrap user-provided data. The data pointer must remain valid until the
 * decoder consumes it (i.e. until avs2_send_data() returns AVS2_OK).
 */
AVS2DEC_API void avs2_data_wrap(avs2_data *data, const uint8_t *buf,
                                size_t sz, int64_t pts, int64_t dts);

/*
 * Wrap user-provided data with a custom free callback.
 */
AVS2DEC_API void avs2_data_wrap_with_cb(avs2_data *data, const uint8_t *buf,
                                        size_t sz, int64_t pts, int64_t dts,
                                        void *ref,
                                        void (*free_cb)(const uint8_t *, void *));

#ifdef __cplusplus
}
#endif

#endif /* AVS2DEC_DATA_H */
