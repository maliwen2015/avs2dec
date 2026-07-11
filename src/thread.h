#ifndef AVS2DEC_SRC_THREAD_H
#define AVS2DEC_SRC_THREAD_H

/*
 * Minimal portable threading layer (mirrors dav1d's thread.h abstraction).
 * For AVS2 we mainly need frame-level worker threads.
 */

#if defined(_WIN32)
#include <windows.h>
typedef HANDLE avs2_thread_t;
typedef CRITICAL_SECTION avs2_mutex_t;
/* 使用 Vista+ 原生 CONDITION_VARIABLE 替代 auto-reset event.
 * 原实现用 auto-reset event 模拟条件变量, avs2_cond_broadcast 循环
 * SetEvent 无法真正广播 (auto-reset event 是电平触发, 多次 SetEvent
 * 只存储一个信号), 导致多线程场景下部分等待者永久阻塞. */
typedef CONDITION_VARIABLE avs2_cond_t;
typedef HANDLE avs2_sem_t;
#else
#include <pthread.h>
typedef pthread_t avs2_thread_t;
typedef pthread_mutex_t avs2_mutex_t;
typedef pthread_cond_t avs2_cond_t;
#include <semaphore.h>
typedef sem_t avs2_sem_t;
#endif

static inline void avs2_mutex_init(avs2_mutex_t *m) {
#if defined(_WIN32)
    InitializeCriticalSection(m);
#else
    pthread_mutex_init(m, NULL);
#endif
}
static inline void avs2_mutex_destroy(avs2_mutex_t *m) {
#if defined(_WIN32)
    DeleteCriticalSection(m);
#else
    pthread_mutex_destroy(m);
#endif
}
static inline void avs2_mutex_lock(avs2_mutex_t *m) {
#if defined(_WIN32)
    EnterCriticalSection(m);
#else
    pthread_mutex_lock(m);
#endif
}
static inline void avs2_mutex_unlock(avs2_mutex_t *m) {
#if defined(_WIN32)
    LeaveCriticalSection(m);
#else
    pthread_mutex_unlock(m);
#endif
}

static inline void avs2_cond_init(avs2_cond_t *c) {
#if defined(_WIN32)
    InitializeConditionVariable(c);
#else
    pthread_cond_init(c, NULL);
#endif
}
static inline void avs2_cond_destroy(avs2_cond_t *c) {
#if defined(_WIN32)
    /* CONDITION_VARIABLE 无需显式销毁 */
    (void)c;
#else
    pthread_cond_destroy(c);
#endif
}
static inline void avs2_cond_signal(avs2_cond_t *c) {
#if defined(_WIN32)
    WakeConditionVariable(c);
#else
    pthread_cond_signal(c);
#endif
}
static inline void avs2_cond_wait(avs2_cond_t *c, avs2_mutex_t *m) {
#if defined(_WIN32)
    SleepConditionVariableCS(c, m, INFINITE);
#else
    pthread_cond_wait(c, m);
#endif
}

/* 条件变量广播 (唤醒所有等待者). Win32 使用 WakeAllConditionVariable
 * 实现真正的广播语义 (不再需要 n_waiters 计数). */
static inline void avs2_cond_broadcast(avs2_cond_t *c, avs2_mutex_t *m, int n_waiters) {
#if defined(_WIN32)
    (void)m; (void)n_waiters;
    WakeAllConditionVariable(c);
#else
    (void)m; (void)n_waiters;
    pthread_cond_broadcast(c);
#endif
}

/* ===========================================================================
 * 原子操作 (用于 WPP 行级并行, 避免锁竞争)
 * =========================================================================== */
#if defined(_MSC_VER)
#include <intrin.h>
#define avs2_atomic_load(p)            (*(volatile int*)(p))
#define avs2_atomic_store(p, v)        (*(volatile int*)(p) = (v))
#define avs2_atomic_inc(p)             _InterlockedIncrement((volatile long*)(p))
#define avs2_atomic_cas(p, old, neu)   (_InterlockedCompareExchange((volatile long*)(p), (long)(neu), (long)(old)) == (long)(old))
#else
#define avs2_atomic_load(p)            __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define avs2_atomic_store(p, v)        __atomic_store_n((p), (v), __ATOMIC_RELEASE)
#define avs2_atomic_inc(p)             __atomic_add_fetch((p), 1, __ATOMIC_ACQ_REL)
#define avs2_atomic_cas(p, old, neu)   __atomic_compare_exchange_n((p), &(old), (neu), 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)
#endif

/* 轻量级 CPU relax (spin-wait 优化) */
#if defined(_MSC_VER)
#include <intrin.h>
#define avs2_cpu_relax() _mm_pause()
#elif defined(__GNUC__) || defined(__clang__)
#define avs2_cpu_relax() __builtin_ia32_pause()
#else
#define avs2_cpu_relax() ((void)0)
#endif

/* ===========================================================================
 * 线程创建/Join (帧级并行 worker 线程)
 * =========================================================================== */
typedef void *(*avs2_thread_func_t)(void *arg);

static inline int avs2_thread_create(avs2_thread_t *t, avs2_thread_func_t func, void *arg) {
#if defined(_WIN32)
    /* LPTHREAD_START_ROUTINE 返回 DWORD, avs2_thread_func_t 返回 void*.
     * avs2_thread_join 在 Windows 上不读取返回值, 故类型转换安全. */
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
    *t = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)func, arg, 0, NULL);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    return (*t != NULL) ? 0 : -1;
#else
    return pthread_create(t, NULL, func, arg);
#endif
}

static inline int avs2_thread_join(avs2_thread_t *t, void **retval) {
#if defined(_WIN32)
    DWORD r = WaitForSingleObject(*t, INFINITE);
    if (r != WAIT_OBJECT_0) return -1;
    if (retval) {
        /* Windows 不直接支持线程返回值, 这里不获取 */
    }
    CloseHandle(*t);
    *t = NULL;
    return 0;
#else
    return pthread_join(*t, retval);
#endif
}

#endif /* AVS2DEC_SRC_THREAD_H */
