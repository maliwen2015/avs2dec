/*
 * avs2dec - high-performance AVS2 (GB/T 33475.2) video decoder
 *
 * Common public definitions
 */

#ifndef AVS2DEC_COMMON_H
#define AVS2DEC_COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Build export macros */
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

/* avs2_logger 回调使用 va_list (stdarg.h 已在文件头引入) */
typedef struct avs2_logger {
    void *cookie;
    /* 回调签名采用 va_list 风格 (与 vfprintf 一致), 便于正确转发可变参数.
     * 回调实现可用 vfprintf/vsnprintf 等处理 ap. */
    void (*callback)(void *cookie, int level, const char *fmt, va_list ap);
} avs2_logger;

#ifdef __cplusplus
}
#endif

#endif /* AVS2DEC_COMMON_H */
