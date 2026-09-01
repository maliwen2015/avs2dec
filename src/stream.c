#include "internal.h"
#include <stdlib.h>
#include <string.h>

/*
 * Scan for an AVS2 start code (00 00 01 XX). Returns the byte offset of the
 * start code in *sc_pos and the start code id (XX) in *sc_id. Returns 1 if
 * found, 0 otherwise.
 *
 * 优化: 起始码必以 0x00 开头, 用 memchr (glibc 内部为 SIMD 实现) 快速
 * 跳到下一个候选 0x00, 避免逐字节 3 次比较. 码流中 0x00 稀疏,
 * 平均每次 memchr 可跳过 ~256 字节.
 */
int avs2_find_start_code(const uint8_t *data, int sz, int *sc_pos, int *sc_id)
{
    const uint8_t *base;
    const uint8_t *p;
    const uint8_t *limit;

    if (sz < 4)
        return 0;

    /* 起始码前缀 (00 00 01) 的最大起始下标为 sz-4 */
    base  = data;
    limit = data + sz - 4;
    p     = data;

    for (;;) {
        const uint8_t *z = (const uint8_t *)memchr(p, 0, (size_t)(limit - p) + 1);
        if (!z)
            return 0;
        /* z+3 <= data+sz-1, 越界访问安全 */
        if (z[1] == 0 && z[2] == 1) {
            *sc_pos = (int)(z - base);
            *sc_id  = z[3];
            return 1;
        }
        p = z + 1;
    }
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
        /* 快路径: 不处于位重排状态 (last_bit_count==0)、无待决起始码
         * (b_found_start_code==0) 且前导零 < 2 时, 到下一个 0x00 之前的
         * 连续非零字节均原样拷贝 (状态机对它们不做任何特判):
         *   - 0x01 需要 leading_zeros>=2 才构成起始码;
         *   - 0x02 需要 b_dispose 且 leading_zeros==2 才触发位重排;
         * 均不满足, 逐字节状态机退化为纯拷贝.
         * 用 memchr 定位下一个 0x00 (AEC 数据中 0x00 稀疏),
         * 中间区间整体 memcpy, 消除逐字节 switch 分支树. */
        if (last_bit_count == 0 && !b_found_start_code && leading_zeros < 2) {
            const uint8_t *cur  = src + i_pos;
            const uint8_t *zero = (const uint8_t *)memchr(cur, 0, (size_t)(i_src - i_pos));
            if (!zero) {
                memcpy(dst + i_dst, cur, (size_t)(i_src - i_pos));
                i_dst += i_src - i_pos;
                break;
            }
            {
                int run = (int)(zero - cur);
                if (run > 0) {
                    memcpy(dst + i_dst, cur, (size_t)run);
                    i_dst += run;
                    i_pos += run;
                    leading_zeros = 0;
                }
            }
        }

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
