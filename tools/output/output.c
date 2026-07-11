#include "output.h"
#include <string.h>
#include <stdio.h>

int avs2_output_open(avs2_output *out, const char *path, int is_y4m)
{
    memset(out, 0, sizeof(*out));
    out->is_y4m = is_y4m;
    if (!path || (path[0] == '-' && path[1] == 0))
        out->fp = stdout;
    else
        out->fp = fopen(path, "wb");
    if (!out->fp) return -1;
    return 0;
}

void avs2_output_close(avs2_output *out)
{
    if (out->fp && out->fp != stdout) fclose(out->fp);
    out->fp = NULL;
}
