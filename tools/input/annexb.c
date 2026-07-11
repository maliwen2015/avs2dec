/* Annex B demuxer: passes raw bytes through (start codes are inline). */
#include "input.h"

int avs2_annexb_read(avs2_input *in, uint8_t *buf, int sz)
{
    return avs2_input_read(in, buf, sz);
}
