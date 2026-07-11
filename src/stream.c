#include "internal.h"
#include <stdlib.h>

/*
 * Scan for an AVS2 start code (00 00 01 XX). Returns the byte offset of the
 * start code in *sc_pos and the start code id (XX) in *sc_id. Returns 1 if
 * found, 0 otherwise.
 */
int avs2_find_start_code(const uint8_t *data, int sz, int *sc_pos, int *sc_id)
{
    for (int i = 0; i + 3 < sz; i++) {
        if (data[i] == 0 && data[i+1] == 0 && data[i+2] == 1) {
            *sc_pos = i;
            *sc_id = data[i+3];
            return 1;
        }
    }
    return 0;
}

/*
 * 移除 AVS2 码流中的伪起始码 (pseudo start code emulation prevention).
 * 对应 davs2 bs_dispose_pseudo_code: 在 slice 数据中, 编码器会插入
 * 00 00 02 序列来防止 AEC 数据中出现 00 00 00/01 等起始码前缀.
 * 解码器需在 AEC 解码前移除这些防竞争比特 (每次移除 2 bit).
 *
 * 函数扫描整个 ES unit, 保留真正的起始码 (00 00 01 xx) 和非 slice
 * 起始码后的数据不变, 仅对 slice 起始码后的 AEC 数据做位级重排.
 */
int avs2_dispose_pseudo_code(uint8_t *dst, const uint8_t *src, int i_src)
{
    static const int BITMASK[] = { 0x00, 0x00, 0xc0, 0x00, 0xf0, 0x00, 0xfc, 0x00 };
    int b_found_start_code = 0;
    int leading_zeros  = 0;
    int last_bit_count = 0;
    int curr_bit_count = 0;
    int b_dispose = 0;
    int i_pos = 0;
    int i_dst = 0;
    uint8_t last_byte = 0;
    uint8_t curr_byte = 0;

    while (i_pos < i_src) {
        curr_byte = src[i_pos++];
        curr_bit_count = 8;
        switch (curr_byte) {
        case 0:
            if (b_found_start_code) {
                b_dispose          = 1; /* start code of first slice: [00 00 01 00] */
                b_found_start_code = 0;
            }
            leading_zeros++;
            break;
        case 1:
            if (leading_zeros >= 2) {
                /* find start code: [00 00 01] */
                b_found_start_code = 1;
                if (last_bit_count) {
                    /* terminate the fixing work before new start code */
                    last_bit_count = 0;
                    dst[i_dst++]   = 0; /* insert the dispose byte */
                }
            }
            leading_zeros = 0;
            break;
        case 2:
            if (b_dispose && leading_zeros == 2) {
                /* dispose the pseudo code, two bits */
                curr_bit_count = 6;
            }
            leading_zeros = 0;
            break;
        default:
            if (b_found_start_code) {
                if (curr_byte == 0xB0 || curr_byte == 0xB5 || curr_byte == 0xB1) {
                    /* SC_SEQUENCE_HEADER, SC_USER_DATA, SC_EXTENSION */
                    b_dispose = 0;
                } else {
                    b_dispose = 1;
                }
                b_found_start_code = 0;
            }
            leading_zeros = 0;
            break;
        }

        if (curr_bit_count == 8) {
            if (last_bit_count == 0) {
                dst[i_dst++] = curr_byte;
            } else {
                dst[i_dst++] = ((last_byte & BITMASK[last_bit_count]) | ((curr_byte & BITMASK[8 - last_bit_count]) >> last_bit_count));
                last_byte    = (curr_byte << (8 - last_bit_count)) & BITMASK[last_bit_count];
            }
        } else {
            if (last_bit_count == 0) {
                last_byte      = curr_byte;
                last_bit_count = curr_bit_count;
            } else {
                dst[i_dst++]   = ((last_byte & BITMASK[last_bit_count]) | ((curr_byte & BITMASK[8 - last_bit_count]) >> last_bit_count));
                last_byte      = (curr_byte << (8 - last_bit_count)) & BITMASK[last_bit_count - 2];
                last_bit_count = last_bit_count - 2;
            }
        }
    }

    if (last_bit_count != 0 && last_byte != 0) {
        dst[i_dst++] = last_byte;
    }

    return i_dst;
}
