/* Section 5 demuxer: raw byte stream, same as Annex B for AVS2. */
#include "input.h"

int avs2_section5_read(avs2_input *in, uint8_t *buf, int sz)
{
    return avs2_input_read(in, buf, sz);
}
