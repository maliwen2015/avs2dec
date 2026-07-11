#ifndef AVS2DEC_SRC_GETBITS_H
#define AVS2DEC_SRC_GETBITS_H

#include <stdint.h>

/*
 * AVS2 bitstream reader. Modelled after davs2: tracks a byte buffer with
 * an absolute bit position, MSB-first within each byte.
 *
 * Supports:
 *   u(n)   fixed-length
 *   flag   = u(1)
 *   ue(v)  AVS2 unsigned exp-golomb variant: (1 << (len>>1)) + info - 1
 *   se(v)  signed variant of ue(v)
 */

typedef struct {
    const uint8_t *buf;   /* bitstream buffer */
    int            sz;    /* buffer size in bytes */
    int            pos;   /* current bit position */
    int            error; /* set on overread */
} avs2_bs;

void avs2_bs_init(avs2_bs *bs, const uint8_t *buf, int sz);
void avs2_bs_align(avs2_bs *bs);
int  avs2_bs_left_bits(avs2_bs *bs);
int  avs2_bs_byte_pos(avs2_bs *bs);

unsigned avs2_bs_get(avs2_bs *bs, int n);   /* u(n), n in [0,32] */
unsigned avs2_bs_get1(avs2_bs *bs);         /* flag */
unsigned avs2_bs_get_ue(avs2_bs *bs);       /* AVS2 ue(v) */
int      avs2_bs_get_se(avs2_bs *bs);       /* AVS2 se(v) */

/* Skip n bits. */
static inline void avs2_bs_skip(avs2_bs *bs, int n) { bs->pos += n; }

#endif
