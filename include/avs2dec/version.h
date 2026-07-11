/*
 * avs2dec - high-performance AVS2 (GB/T 33475.2) video decoder
 *
 * Public API version
 */

#ifndef AVS2DEC_VERSION_H
#define AVS2DEC_VERSION_H

#define AVS2DEC_VERSION_MAJOR 1
#define AVS2DEC_VERSION_MINOR 0
#define AVS2DEC_VERSION_PATCH 0

#define AVS2DEC_API_MAJOR 1
#define AVS2DEC_API_MINOR 0
#define AVS2DEC_API_PATCH 0

#define AVS2DEC_VERSION_STR "1.0.0"

#define AVS2DEC_API_VERSION                                                 \
    ((AVS2DEC_API_MAJOR << 16) | (AVS2DEC_API_MINOR << 8) | AVS2DEC_API_PATCH)

#endif /* AVS2DEC_VERSION_H */
