#ifndef AVS2DEC_INPUT_INPUT_H
#define AVS2DEC_INPUT_INPUT_H

#include <stdint.h>
#include <stdio.h>

typedef struct avs2_input {
    FILE *fp;
    int is_annexb; /* 1 = Annex B start code format, 0 = section 5 */
    /* read up to sz bytes into buf, returns bytes read */
    int (*read)(struct avs2_input *in, uint8_t *buf, int sz);
} avs2_input;

int avs2_input_open(avs2_input *in, const char *path, int is_annexb);
int avs2_input_read(avs2_input *in, uint8_t *buf, int sz);
void avs2_input_close(avs2_input *in);

#endif
