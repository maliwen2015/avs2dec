#include "getbits.h"

void avs2_bs_init(avs2_bs *bs, const uint8_t *buf, int sz)
{
    bs->buf = buf;
    bs->sz = sz;
    bs->pos = 0;
    bs->error = 0;
}

void avs2_bs_align(avs2_bs *bs)
{
    bs->pos = ((bs->pos + 7) >> 3) << 3;
}

int avs2_bs_left_bits(avs2_bs *bs)
{
    return (bs->sz << 3) - bs->pos;
}

int avs2_bs_byte_pos(avs2_bs *bs)
{
    return bs->pos >> 3;
}

/*
 * Read up to 32 bits MSB-first.
 * Mirrors davs2 read_bits(): bit_offset within a byte goes 7..0.
 */
unsigned avs2_bs_get(avs2_bs *bs, int n)
{
    unsigned val = 0;
    if (n <= 0) return 0;
    int total_bits = bs->sz << 3;
    if (bs->pos + n > total_bits) {
        bs->error = 1;
        /* clamp: return what we can */
        if (bs->pos >= total_bits) return 0;
        n = total_bits - bs->pos;
    }
    int byte_off = bs->pos >> 3;
    int bit_off  = 7 - (bs->pos & 7);
    bs->pos += n;
    while (n-- > 0) {
        val <<= 1;
        val |= (unsigned)((bs->buf[byte_off] >> bit_off) & 1);
        if (--bit_off < 0) {
            bit_off = 7;
            byte_off++;
        }
    }
    return val;
}

unsigned avs2_bs_get1(avs2_bs *bs)
{
    return avs2_bs_get(bs, 1);
}

/*
 * AVS2 ue(v): 标准指数哥伦布编码 order 0.
 * k 个前导 0, 然后 1, 然后读取 k 个 info 位.
 * value = (1 << k) + info - 1
 * 对应 davs2 vlc_ue_v: bit_counter=2k+1, (1<<(bit_counter>>1))+info-1 = (1<<k)+info-1
 */
unsigned avs2_bs_get_ue(avs2_bs *bs)
{
    int k = 0;
    /* count leading zeros */
    while (k < 31) {
        if (bs->pos >= (bs->sz << 3)) { bs->error = 1; return 0; }
        int byte_off = bs->pos >> 3;
        int bit_off = 7 - (bs->pos & 7);
        if (((bs->buf[byte_off] >> bit_off) & 1) != 0)
            break;
        bs->pos++;
        k++;
    }
    /* skip the '1' separator */
    bs->pos++;
    unsigned info = 0;
    if (k > 0)
        info = avs2_bs_get(bs, k);
    unsigned val = (1u << k) + info - 1;
    return val;
}

int avs2_bs_get_se(avs2_bs *bs)
{
    unsigned n = avs2_bs_get_ue(bs);
    int ret = (int)((n + 1) >> 1);
    if ((n & 1) == 0) ret = -ret;
    return ret;
}
