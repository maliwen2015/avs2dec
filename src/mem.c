#include "mem.h"

#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <intrin.h>
#ifdef _MSC_VER
#pragma intrinsic(_InterlockedExchangeAdd)
#endif
#endif

/* SIMD 友好的对齐分配: 统一 32 字节对齐, 保证 AVX2 的 load/store 不崩溃.
 * 对齐分配/释放配对使用, free 端用对应的对齐释放函数. */
#define AVS2_MEM_ALIGN 32

void *avs2_mem_alloc(size_t sz)
{
    if (!sz) sz = 1;
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__) || defined(_WIN32)
    return _aligned_malloc(sz, AVS2_MEM_ALIGN);
#else
    void *p = NULL;
    if (posix_memalign(&p, AVS2_MEM_ALIGN, sz) != 0) return NULL;
    return p;
#endif
}

void *avs2_mem_allocz(size_t sz)
{
    void *p = avs2_mem_alloc(sz);
    if (p) memset(p, 0, sz ? sz : 1);
    return p;
}

void *avs2_mem_realloc(void *p, size_t sz)
{
    if (!sz) sz = 1;
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__) || defined(_WIN32)
    return _aligned_realloc(p, sz, AVS2_MEM_ALIGN);
#else
    /* posix 平台 realloc 通常保持对齐 (glibc 2.28+), 简化处理 */
    return realloc(p, sz);
#endif
}

void avs2_mem_free(void *p)
{
    if (!p) return;
#if defined(_MSC_VER) || defined(__MINGW32__) || defined(__MINGW64__) || defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

void *avs2_mem_alloc_2d(size_t rows, size_t cols, ptrdiff_t *stride)
{
    /* simple flat allocation with cache-line alignment */
    ptrdiff_t s = (ptrdiff_t)((cols + 63) & ~(size_t)63);
    *stride = s;
    return avs2_mem_allocz((rows ? rows : 1) * (size_t)s);
}

/* ---- reference counting ---- */

avs2_ref *avs2_ref_alloc(size_t sz)
{
    /* 结构体本身用对齐分配, data 紧跟其后.
     * 由于 AVS2_MEM_ALIGN >= sizeof(avs2_ref) 的最大对齐需求, data 也会对齐 */
    avs2_ref *r = (avs2_ref *)avs2_mem_alloc(sizeof(*r) + sz + AVS2_MEM_ALIGN);
    if (!r) return NULL;
    /* 对 data 指针做对齐修正 */
    uintptr_t base = (uintptr_t)((uint8_t *)r + sizeof(*r));
    uintptr_t aligned = (base + (AVS2_MEM_ALIGN - 1)) & ~((uintptr_t)AVS2_MEM_ALIGN - 1);
    r->data = (void *)aligned;
    r->free_cb = NULL;
    r->user = NULL;
    r->ref_cnt = 1;
    return r;
}

void avs2_ref_inc(avs2_ref *r)
{
    if (r) {
#ifdef _WIN32
        _InterlockedExchangeAdd((long *)&r->ref_cnt, 1);
#else
        __sync_fetch_and_add(&r->ref_cnt, 1);
#endif
    }
}

void avs2_ref_dec(avs2_ref *r)
{
    if (!r) return;
    int cnt;
#ifdef _WIN32
    cnt = _InterlockedExchangeAdd((long *)&r->ref_cnt, -1) - 1;
#else
    cnt = __sync_fetch_and_sub(&r->ref_cnt, 1) - 1;
#endif
    if (cnt == 0) {
        if (r->free_cb) r->free_cb(r->data, r->user);
        avs2_mem_free(r);
    }
}
