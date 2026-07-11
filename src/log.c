#include "avs2dec/avs2dec.h"
#include "log.h"
#include "internal.h"

#include <stdio.h>
#include <stdarg.h>

void avs2_log(struct avs2_internal *c, int level, const char *fmt, ...)
{
    if (!c) return;
    if (level > c->log_level) return;
    if (!c->logger.callback) {
        /* default stderr logger */
        static const char *prefix[] = { "Error", "Warning", "Info", "Debug" };
        va_list ap;
        fprintf(stderr, "avs2dec [%s]: ", prefix[level & 3]);
        va_start(ap, fmt);
        vfprintf(stderr, fmt, ap);
        va_end(ap);
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    c->logger.callback(c->logger.cookie, level, fmt, ap);
    va_end(ap);
}
