#include "output.h"
#include <stdio.h>
#include <string.h>

int avs2_output_write_y4m(avs2_output *out, const avs2_picture *pic,
                          const avs2_seq_header *seq)
{
    if (!out->frame_written) {
        /* header */
        const char *cf = "420jpeg";
        if (seq && seq->chroma_format == 2) cf = "422";
        else if (seq && seq->chroma_format == 3) cf = "444";
        fprintf(out->fp, "YUV4MPEG2 W%d H%d F%d:1 C%s\n",
                pic->p_w, pic->p_h,
                seq ? (int)(seq->frame_rate + 0.5) : 25, cf);
        out->frame_written = 1;
        out->width = pic->p_w;
        out->height = pic->p_h;
    }
    fprintf(out->fp, "FRAME\n");
    for (int pl = 0; pl < 3; pl++) {
        const uint8_t *d = pic->data[pl];
        ptrdiff_t s = pic->stride[pl];
        int w = pic->width[pl], h = pic->height[pl];
        for (int y = 0; y < h; y++)
            fwrite(d + y * s, 1, (size_t)w, out->fp);
    }
    return 0;
}
