/*
 * avs2dec - high-performance AVS2 (GB/T 33475.2) video decoder
 *
 * Decoded picture descriptor and allocator
 */

#ifndef AVS2DEC_PICTURE_H
#define AVS2DEC_PICTURE_H

#include <stddef.h>
#include <stdint.h>

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Picture structure (1=frame, 2=top field, 3=bottom field) */
enum avs2_picture_structure_e {
    AVS2_PIC_STRUCTURE_FRAME       = 0,
    AVS2_PIC_STRUCTURE_TOP_FIELD   = 1,
    AVS2_PIC_STRUCTURE_BOT_FIELD   = 2,
};

typedef struct avs2_picture_alloc {
    void *cookie;
    /*
     * Allocate a picture buffer. Returns 0 on success.
     *  data  - per-plane data pointers (set by allocator)
     *  stride- per-plane stride in bytes (set by allocator)
     */
    int (*alloc_picture)(void *cookie, uint8_t *data[3],
                         ptrdiff_t stride[3], void **pic_cookie);
    void (*release_picture)(void *cookie, void *pic_cookie);
} avs2_picture_alloc;

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

    void *pic_cookie;       /* allocator private */
    void *dec_frame;        /* opaque decoder frame reference */
} avs2_picture;

#ifdef __cplusplus
}
#endif

#endif /* AVS2DEC_PICTURE_H */
