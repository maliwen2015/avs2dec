#ifndef AVS2DEC_SRC_MEM_H
#define AVS2DEC_SRC_MEM_H

#include <stddef.h>
#include <stdint.h>

/* Aligned memory allocation helpers. */
void *avs2_mem_alloc(size_t sz);
void *avs2_mem_allocz(size_t sz);
void *avs2_mem_realloc(void *p, size_t sz);
void  avs2_mem_free(void *p);

/* 32 字节对齐修饰宏 (用于结构体成员和栈数组, SIMD 操作要求).
 * 用法: AVS2_ALIGNED_32(int16_t buf[N]); */
#if defined(_MSC_VER)
#define AVS2_ALIGNED_32(decl) __declspec(align(32)) decl
#else
#define AVS2_ALIGNED_32(decl) decl __attribute__((aligned(32)))
#endif

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
