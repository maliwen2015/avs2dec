#include "output.h"

int avs2_output_write_yuv(avs2_output *out, const avs2_picture *pic)
{
    const int bps = pic->bytes_per_sample;
    for (int pl = 0; pl < 3; pl++) {
        const uint8_t *d = pic->data[pl];
        ptrdiff_t s = pic->stride[pl];
        int w = pic->width[pl], h = pic->height[pl];
        size_t row_bytes = (size_t)w * bps;
        for (int y = 0; y < h; y++)
            fwrite(d + y * s, 1, row_bytes, out->fp);
    }
    return 0;
}
