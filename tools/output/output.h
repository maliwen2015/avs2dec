#ifndef AVS2DEC_OUTPUT_OUTPUT_H
#define AVS2DEC_OUTPUT_OUTPUT_H

#include <stdio.h>

#include "avs2dec/avs2dec.h"

typedef struct avs2_output {
    FILE *fp;
    int is_y4m;
    int frame_written;
    int width, height, bit_depth, chroma_format;
    int force_8bit;  /* 1 = 将 10-bit 解码输出降为 8-bit (逐像素 >>2) */
} avs2_output;

int avs2_output_open(avs2_output *out, const char *path, int is_y4m);
int avs2_output_write(avs2_output *out, const avs2_picture *pic,
                      const avs2_seq_header *seq);
void avs2_output_close(avs2_output *out);

#endif
