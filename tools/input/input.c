#include "input.h"
#include <string.h>

int avs2_input_open(avs2_input *in, const char *path, int is_annexb)
{
    memset(in, 0, sizeof(*in));
    in->is_annexb = is_annexb;
    if (!path || (path[0] == '-' && path[1] == 0))
        in->fp = stdin;
    else
        in->fp = fopen(path, "rb");
    if (!in->fp) return -1;
    return 0;
}

int avs2_input_read(avs2_input *in, uint8_t *buf, int sz)
{
    return (int)fread(buf, 1, (size_t)sz, in->fp);
}

void avs2_input_close(avs2_input *in)
{
    if (in->fp && in->fp != stdin) fclose(in->fp);
    in->fp = NULL;
}
