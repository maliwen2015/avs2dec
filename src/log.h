#ifndef AVS2DEC_SRC_LOG_H
#define AVS2DEC_SRC_LOG_H

#include "avs2dec/common.h"

struct avs2_internal;

void avs2_log(struct avs2_internal *c, int level, const char *fmt, ...);

#define avs2_error(c, ...)   avs2_log(c, AVS2_LOG_ERROR, __VA_ARGS__)
#define avs2_warn(c, ...)    avs2_log(c, AVS2_LOG_WARNING, __VA_ARGS__)
#define avs2_info(c, ...)    avs2_log(c, AVS2_LOG_INFO, __VA_ARGS__)
#define avs2_debug(c, ...)   avs2_log(c, AVS2_LOG_DEBUG, __VA_ARGS__)

#endif
