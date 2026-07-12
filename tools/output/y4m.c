#include "output.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

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

    /* force_8bit: 10-bit → 8-bit 逐行转换 */
    if (out->force_8bit && pic->bytes_per_sample == 2) {
        int max_w = 0;
        uint8_t *row8;
        int pl;

        for (pl = 0; pl < 3; pl++)
            if (pic->width[pl] > max_w) max_w = pic->width[pl];

        row8 = (uint8_t *)malloc(max_w);
        if (!row8) return -1;

        for (pl = 0; pl < 3; pl++) {
            const uint16_t *d16 = (const uint16_t *)(const void *)pic->data[pl];
            ptrdiff_t s = pic->stride[pl];
            int stride16 = (int)(s >> 1);
            int w = pic->width[pl], h = pic->height[pl];
            int y;

            for (y = 0; y < h; y++) {
                const uint16_t *src = d16 + y * stride16;
                int x;
                for (x = 0; x < w; x++)
                    row8[x] = (uint8_t)((src[x] + 2) >> 2);
                fwrite(row8, 1, w, out->fp);
            }
        }
        free(row8);
        return 0;
    }

    for (int pl = 0; pl < 3; pl++) {
        const uint8_t *d = pic->data[pl];
        ptrdiff_t s = pic->stride[pl];
        int w = pic->width[pl], h = pic->height[pl];
        for (int y = 0; y < h; y++)
            fwrite(d + y * s, 1, (size_t)w, out->fp);
    }
    return 0;
}
