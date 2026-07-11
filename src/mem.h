#ifndef AVS2DEC_SRC_MEM_H
#define AVS2DEC_SRC_MEM_H

#include <stddef.h>
#include <stdint.h>

/* Aligned memory allocation helpers. */
void *avs2_mem_alloc(size_t sz);
void *avs2_mem_allocz(size_t sz);
void *avs2_mem_realloc(void *p, size_t sz);
void  avs2_mem_free(void *p);

/* Allocate a 2D-ish buffer of rows*cols with alignment. */
void *avs2_mem_alloc_2d(size_t rows, size_t cols, ptrdiff_t *stride);

/* Reference-counted allocation (dav1d-style). */
typedef struct avs2_ref {
    void *data;
    void (*free_cb)(void *data, void *user);
    void *user;
    int ref_cnt;
} avs2_ref;

avs2_ref *avs2_ref_alloc(size_t sz);
void avs2_ref_inc(avs2_ref *r);
void avs2_ref_dec(avs2_ref *r);

#endif
