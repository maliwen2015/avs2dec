/* Reference management is folded into picture.c; this file keeps the build
 * target list satisfied and holds DPB reference-list helpers. */
#include "internal.h"

/* scale_mv: scale a motion vector from one reference distance to another. */
int32_t avs2_scale_mv(int32_t mv, int dist_dst, int dist_src)
{
    int sign = mv < 0 ? -1 : 1;
    int abs_mv = mv < 0 ? -mv : mv;
    int scaled = (int)(((int64_t)abs_mv * dist_dst * dist_src + HALF_MULTI) >> OFFSET);
    return sign * scaled;
}

/* scale_mv_biskip variant (matches davs2 scale_mv_biskip). */
int32_t avs2_scale_mv_biskip(int32_t mv, int dist_dst, int dist_src)
{
    int sign = mv < 0 ? -1 : 1;
    int abs_mv = mv < 0 ? -mv : mv;
    int scaled = (int)(((int64_t)dist_src * (1 + abs_mv * dist_dst) - 1) >> OFFSET);
    return sign * scaled;
}
